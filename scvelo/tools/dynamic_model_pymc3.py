# -*- coding: utf-8 -*-
"""Dynamic model of splicing kinetics for scRNAseq data. Version 1
"""

import sys, ast, os
import time
import itertools
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import theano.tensor as tt
import pymc3 as pm
import pickle
import theano
import string

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
from functools import wraps
import matplotlib.cm as cm

from pymc3.distributions.dist_math import bound, logpow, factln, binomln

from scvelo.pymc3_model import Pymc3Model

# defining the model itself
class DynamicModel_V1(Pymc3Model):
    r"""NanostringWTA Generative Model:
    Dynamic model of splicing kinetics for scRNAseq data. Version 1
    
    Model parameters include: gene specific transcription rates, splicing rates, degradation rates,
    transcriptional state change time point (assuming just one state change for now), as well as cell-specific latent time
    """

    def __init__(
        self,
        u_counts: np.ndarray,
        s_counts: np.ndarray,
        data_type: str = 'float32',
        n_iter = 100000,
        learning_rate = 0.001,
        total_grad_norm_constraint = 200,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None,  sample_id=None):
        
        ############# Initialise parameters ###########
        super().__init__(u_counts, 0,
                         data_type, n_iter, 
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, sample_id)
        
        self.u_counts = u_counts
        self.s_counts = s_counts
        self.n_genes = np.shape(self.u_counts)[0]
        self.n_cells = np.shape(self.u_counts)[1]
        
        ############# Define the model ################
        self.model = pm.Model()
        with self.model:
            
            # Parameter priors:
            self.t_c = pm.Uniform('t_c', lower = 0, upper = 1, shape = (1, self.n_cells))
            self.PK_gk = pm.Dirichlet('PK_gk', a = (1,1,1,1), shape = (self.n_genes, 4)
            self.t0_0g = pm.Uniform('t0_0g', lower = 0, upper = 1, shape = (self.n_genes, 1))
            self.t0_2g = pm.Uniform('t0_2g', lower = 0, upper = 1, shape = (self.n_genes, 1))
            self.alpha_g = pm.Gamma('alpha_g', alpha = 1, beta = 1, shape = (self.n_genes, 1))
            self.beta_g = pm.Gamma('beta_g', alpha = 1, beta = 1, shape = (self.n_genes, 1))
            self.gamma_g = pm.Gamma('gamma_g', alpha = 1, beta = 1, shape = (self.n_genes, 1))
            
            # Calculate expected abundance of unspliced and spliced counts in each of the four states:
                                       
            # 0: Induction                            
            self.tau_0 = self.t_c - self.t0_g
            self.mu.u_0g = self.alpha_g/self.beta_g * (1 - tt.exp(-self.beta_g*self.tau_0))
            self.mu.s_0g = self.alpha_g/self.gamma_g * (1 - tt.exp(-self.gamma_g*self.tau_0)) + (self.alpha_g -self.beta_g*self.u0_kg)/ (self.gamma_g - self.beta_g) * (tt.exp(-self.gamma*self.tau_0) - tt.exp(-self.beta*self.tau_0)   
            
            # 1: Induction steady state
            self.mu.u_1g = self.alpha_g/self.beta_g
            self.mu.s_1g = self.alpha_g/self.gamma_g
            
            # 2: Repression
            self.tau_2 = self.t_c - self.t2_g
            self.mu.u_2g = self.mu.u_1g*tt.exp(-self.beta_g*self.tau_2)
            self.mu.s_2g = self.mu.s_1g*tt.exp(-self.gamma_g*self.tau_2) - self.beta_g*self.mu.u_1g/(self.gamma_g - self.beta_g) * (tt.exp(-self.gamma*self.tau_2) - tt.exp(-self.beta*self.tau_2)   
                                                                                                  
            # 3: Repression steady state
            self.tau = self.t_c - self.t0_k                                                                                      
            self.mu.u_3g = 0
            self.mu.s_3g = 0                        

            # And then integrate over outcomes of transcriptional states:                          
                                       
            self.mu.u_g = self.PK_gk[:,0] * self.mu.u_0g + self.PK_gk[:,1] * self.mu.u_1g + self.PK_gk[:,2] * self.mu.u_2g + self.PK_gk[:,3] * self.mu.u_3g
            self.mu.s_g = self.PK_gk[:,0] * self.mu.s_0g + self.PK_gk[:,1] * self.mu.s_1g + self.PK_gk[:,2] * self.mu.s_2g + self.PK_gk[:,3] * self.mu.s_3g
                                                                                                   
            # Alpha for NB distribution:
            self.phi_hyp_u_g = pm.Gamma('phi_hyp_u_g', mu=3,
                                    sigma=1, shape=(1, 1))
            self.gene_E_u_g = pm.Exponential('gene_E_u_g', self.phi_hyp_u_g, shape=(self.n_genes, 1))
            self.alpha_u_g = 1 / (self.gene_E_u_g.T * self.gene_E_u_g.T)                                                                           
            self.phi_hyp_s_g = pm.Gamma('phi_hyp_s_g', mu=3,
                                    sigma=1, shape=(1, 1))
            self.gene_E_s_g = pm.Exponential('gene_E_s_g', self.phi_hyp_s_g, shape=(self.n_genes, 1))
            self.alpha_s_g = 1 / (self.gene_E_s_g.T * self.gene_E_s_g.T)                                                                        
            
            # Concatenate mean and alpha for Negative Binomial Distribution:
            self.mu = tt.concatenate([self.mu.u_g, self.mu.s_g], axis = 1)
            self.alpha = tt.concatenate([self.alpha.u_g, self.alpha.s_g], axis = 1)
            
            # Calculate NB log probability density:
            self.data_target = pm.NegativeBinomial('data_target', mu= self.mu,
                                                   alpha= self.alpha,
                                                   observed=tt.concatenate([self.u_counts, self.s_counts], axis = 1))

        
        
        
        