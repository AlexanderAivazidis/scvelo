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
            self.k0_g = pm.Bernoulli('k_g', p = 0.5, shape = (self.n_genes, 1)
            self.t0_g = pm.Uniform('t0_g', lower = 1/self.n_cells, upper = 1, shape = (self.n_genes, 1))
            self.alpha_kg = pm.Gamma('alpha_g', alpha = 1, beta = 1, shape = (2, self.n_genes))
            self.beta_g = pm.Gamma('beta_g', alpha = 1, beta = 1, shape = (1, self.n_genes))
            self.gamma_g = pm.Gamma('gamma_g', alpha = 1, beta = 1, shape = (1, self.n_genes))
            
            # Dynamic model equations (Mean of NB distribution):
            self.u0_kg = 
            self.s0_kg =                          
            self.tau = self.t_c - self.t0_k
            self.mu.u = self.u0_kg*tt.exp(-self.beta_g*self.tau) + self.alpha_kg/self.beta_g * (1 - tt.exp(-self.beta_g*self_tau))
            self.mu.s = self.s0_kg*tt.exp(-self.gamma_g*self.tau) + self.alpha_kg/self.gamma_g * (1 - tt.exp(-self.gamma_g*self_tau))
                        + (self.alpha_kg - self.beta_g*self.u0_kg)/(self.gamma_g - self.beta_g) * (tt.exp(-self.gamma*self.tau) - tt.exp(-self.beta*self.tau)
                                                                                                   
            # Alpha for NB distribution:
            self.phi_hyp_u = pm.Gamma('phi_hyp_u', mu=3,
                                    sigma=1, shape=(1, 1))
            self.gene_E_u = pm.Exponential('gene_E_u', self.phi_hyp_u, shape=(self.n_genes, 1))
            self.alpha_u = 1 / (self.gene_E_u.T * self.gene_E_u.T)                                                                           
            self.phi_hyp_s = pm.Gamma('phi_hyp_s', mu=3,
                                    sigma=1, shape=(1, 1))
            self.gene_E_s = pm.Exponential('gene_E_s', self.phi_hyp_s, shape=(self.n_genes, 1))
            self.alpha_s = 1 / (self.gene_E_s.T * self.gene_E_s.T)                                                                        
            
            # Concatenate mean and alpha for Negative Binomial Distribution:
            self.mu = tt.concatenate([self.mu.u, self.mu.s], axis = 1)
            self.alpha = tt.concatenate([self.alpha.u, self.alpha.s], axis = 1)
            
            # Calculate NB log probability density:
            self.data_target = pm.NegativeBinomial('data_target', mu= self.mu,
                                                   alpha= self.alpha,
                                                   observed=tt.concatenate([self.u_counts, self.s_counts], axis = 1))

        
        
        
        