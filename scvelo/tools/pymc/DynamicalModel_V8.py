# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import theano
import matplotlib.pyplot as plt
import matplotlib
import os
import dill as pickle
# import sys

from scvelo.tools.pymc.pymc3_model import Pymc3Model

from pymc3.math import logsumexp
from pymc3.distributions.dist_math import (
    betaln,
    binomln,
    bound,
    factln,
    logpow,
    normal_lccdf,
    normal_lcdf,
)

def logp_NB(value, mu, alpha):
    negbinom = bound(
        binomln(value + alpha - 1, value)
        + logpow(mu / (mu + alpha), value)
        + logpow(alpha / (mu + alpha), alpha),
        value >= 0,
        mu > 0,
        alpha > 0,
    )
    return negbinom

def logp_nbmix(mus, pi, alpha):
    def logp_(value):

        logps = [tt.log(pi[i]) + logp_NB(value, mus[i], alpha) for i in range(len(mus))]
        
        return tt.sum(logsumexp(tt.stacklists(logps), axis=0))

    return logp_

def classProbs_nbmix(mus, pi, alpha):
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_NB(value, mus[i], alpha) for i in range(len(mus))]
        classProbs = tt.sum(tt.stacklists(logps), axis = 3)
        normFactor = logsumexp(classProbs, axis=0)
        return tt.swapaxes(tt.swapaxes(tt.exp(classProbs-normFactor), 0, 2), 0, 1)
    return logp_

def classProbs_nbmix_extraDim(mus, pi, alpha):
    def logp_(value):
        logps = [tt.log(pi[:,i]).reshape((mus[0].shape[0],1,1,1)) + logp_NB(value, mus[i], alpha) for i in range(len(mus))]
        classProbs = tt.sum(tt.stacklists(logps), axis = 4)
        normFactor = logsumexp(classProbs, axis=0)
        return tt.swapaxes(tt.swapaxes(tt.exp(classProbs-normFactor), 0, 2), 1, 3)
    return logp_

def approx_tau_inf(alpha, alpha0, beta, gamma, u0, s0, eta = 0.999, kappa = 1):
    ''' Finds the approximate time at which spliced mRNA abundance reaches a fraction 'eta'
    of steady state abundance. This depends on rate parameters and initial states. Optionally multiplies with kappa.
    alpha0 is transcription rate in on state, alpha is transcription rate in current state'''
    
    argument = tt.switch(tt.lt(beta,gamma),

          (alpha0/gamma*eta-alpha/gamma)*(beta-gamma)/(alpha-beta*u0),

          (alpha0/gamma*eta-alpha/gamma)/(s0 - alpha/gamma + (alpha - beta*u0)/(gamma-beta)))
    
    rate = tt.switch(tt.lt(beta, gamma), beta, gamma)
    argument = tt.switch(tt.lt(0.99999, argument), 0.9999, argument)
    argument = tt.switch(tt.lt(argument, 0.00001), 0.0001, argument)
        
    return -1/rate * tt.log(argument)
    
    return kappa*tau_inf

def repression_initialState(alpha_0g, alpha_1g, beta_g, gamma_g, t0_g, n_cells, add_axis = 0):
    ''' Calculate initial state for repression phase, based on rates and switch time from induction state.'''

    # 3: Induction initial state, equal to repression steady state (Often set to 0 via alpha_0g = 0):
    mu_u_3cg = tt.extra_ops.repeat(alpha_0g/beta_g, n_cells, axis = 0 + add_axis)
    mu_s_3cg = tt.extra_ops.repeat(alpha_0g/gamma_g, n_cells, axis = 0 + add_axis)
    
    # 1: Induction steady state:
    mu_u_1cg = tt.extra_ops.repeat(alpha_1g/beta_g, n_cells, axis = 0 + add_axis)
    mu_s_1cg = tt.extra_ops.repeat(alpha_1g/gamma_g, n_cells, axis = 0 + add_axis)
    
    # 0: Induction transient state:
    tau = tt.extra_ops.repeat(t0_g, n_cells, axis = 0 + add_axis)
    u0 = mu_u_3cg*tt.exp(-beta_g*tau) + mu_u_1cg * (1 - tt.exp(-beta_g*tau))
    s0 = mu_s_3cg*tt.exp(-gamma_g*tau) + mu_s_1cg * (1 - tt.exp(-gamma_g*tau)) + (tt.extra_ops.repeat(alpha_1g, n_cells, axis = 0 + add_axis) - beta_g * mu_u_3cg)/(gamma_g - beta_g) * (tt.exp(-gamma_g*tau) - tt.exp(-beta_g*tau))
    
    return u0, s0

def mu_mRNA(alpha_0g, alpha_1g, beta_g, gamma_g, tau0_cg, tau2_cg, t0_g, l_c, n_cells, add_axis = 0):
    ''' Calculates expected value of mRNA count in each of four possible states (Induction transient sate, induction steady state,
    repression transient state, repression steady state. Needs rates, latent times, switch time and sequencing depth parameter.'''
    
    # 3: Induction initial state, equal to repression steady state (Often set to 0 via alpha_0g = 0):
    mu_u_3cg = tt.extra_ops.repeat(alpha_0g/beta_g, n_cells, axis = 0 + add_axis)
    mu_s_3cg = tt.extra_ops.repeat(alpha_0g/gamma_g, n_cells, axis = 0 + add_axis)

    # 1: Induction steady state:
    mu_u_1cg = tt.extra_ops.repeat(alpha_1g/beta_g, n_cells, axis = 0 + add_axis)
    mu_s_1cg = tt.extra_ops.repeat(alpha_1g/gamma_g, n_cells, axis = 0 + add_axis)

    # 0: Induction transient state:
    mu_u_0cg = mu_u_3cg*tt.exp(-beta_g*tau0_cg) + mu_u_1cg * (1 - tt.exp(-beta_g*tau0_cg))
    mu_s_0cg = mu_s_3cg*tt.exp(-gamma_g*tau0_cg) + mu_s_1cg * (1 - tt.exp(-gamma_g*tau0_cg)) + (tt.extra_ops.repeat(alpha_1g, n_cells, axis = 0 + add_axis) - beta_g * mu_u_3cg)/(gamma_g - beta_g) * (tt.exp(-gamma_g*tau0_cg) - tt.exp(-beta_g*tau0_cg))

    # 2: Repression transient state:
    u0, s0 = repression_initialState(alpha_0g, alpha_1g, beta_g, gamma_g, t0_g, n_cells, add_axis = add_axis)
    mu_u_2cg = u0*tt.exp(-beta_g*tau2_cg) + mu_u_3cg * (1 - tt.exp(-beta_g*tau2_cg))
    mu_s_2cg = s0*tt.exp(-gamma_g*tau2_cg) + mu_s_3cg * (1 - tt.exp(-gamma_g*tau2_cg)) + (tt.extra_ops.repeat((alpha_0g - beta_g), n_cells, axis = 0 + add_axis) * u0)/(gamma_g - beta_g) * (tt.exp(-gamma_g*tau2_cg) - tt.exp(-beta_g*tau2_cg))

    # Stack values for unspliced and spliced counts and multiply with sequencing depth parameter:
    mu0 = tt.stack([mu_u_0cg, mu_s_0cg], axis = 2 + add_axis)*l_c
    mu1 = tt.stack([mu_u_1cg, mu_s_1cg], axis = 2 + add_axis)*l_c
    mu2 = tt.stack([mu_u_2cg, mu_s_2cg], axis = 2 + add_axis)*l_c
    mu3 = tt.stack([mu_u_3cg, mu_s_3cg], axis = 2 + add_axis)*l_c
    
    return [mu0, mu1, mu2, mu3]

def mu_mRNA_oneState(alpha_0g, alpha_1g, beta_g, gamma_g, tau0_cg, tau2_cg, t0_g, l_c, n_cells, add_axis = 0):
    ''' Calculates expected value of mRNA count in each of four possible states (Induction transient sate, induction steady state,
    repression transient state, repression steady state. Needs rates, latent times, switch time and sequencing depth parameter.'''
    
    # State specific latent times:
    tau0_cg = tt.clip(tau0_cg, 0, 10**(6))
    tau2_cg = tt.clip(tau2_cg, 0, 10**(6))
    
    # 3: Induction initial state, equal to repression steady state (Often set to 0 via alpha_0g = 0):
    mu_u_3cg = tt.extra_ops.repeat(alpha_0g/beta_g, n_cells, axis = 0 + add_axis)
    mu_s_3cg = tt.extra_ops.repeat(alpha_0g/gamma_g, n_cells, axis = 0 + add_axis)

    # 1: Induction steady state:
    mu_u_1cg = tt.extra_ops.repeat(alpha_1g/beta_g, n_cells, axis = 0 + add_axis)
    mu_s_1cg = tt.extra_ops.repeat(alpha_1g/gamma_g, n_cells, axis = 0 + add_axis)

    # 0: Induction transient state:
    mu_u_0cg = mu_u_3cg*tt.exp(-beta_g*tau0_cg) + mu_u_1cg * (1 - tt.exp(-beta_g*tau0_cg))
    mu_s_0cg = mu_s_3cg*tt.exp(-gamma_g*tau0_cg) + mu_s_1cg * (1 - tt.exp(-gamma_g*tau0_cg)) + (tt.extra_ops.repeat(alpha_1g, n_cells, axis = 0 + add_axis) - beta_g * mu_u_3cg)/(gamma_g - beta_g) * (tt.exp(-gamma_g*tau0_cg) - tt.exp(-beta_g*tau0_cg))

    # 2: Repression transient state:
    u0, s0 = repression_initialState(alpha_0g, alpha_1g, beta_g, gamma_g, t0_g, n_cells, add_axis = add_axis)
    mu_u_2cg = u0*tt.exp(-beta_g*tau2_cg) + mu_u_3cg * (1 - tt.exp(-beta_g*tau2_cg))
    mu_s_2cg = s0*tt.exp(-gamma_g*tau2_cg) + mu_s_3cg * (1 - tt.exp(-gamma_g*tau2_cg)) + (tt.extra_ops.repeat((alpha_0g - beta_g), n_cells, axis = 0 + add_axis) * u0)/(gamma_g - beta_g) * (tt.exp(-gamma_g*tau2_cg) - tt.exp(-beta_g*tau2_cg))

    # Stack values for unspliced and spliced counts and multiply with sequencing depth parameter:
    mu0 = tt.stack([mu_u_0cg, mu_s_0cg], axis = 2 + add_axis)*l_c
    mu2 = tt.stack([mu_u_2cg, mu_s_2cg], axis = 2 + add_axis)*l_c 
    
    return tt.stack([tt.switch(tt.lt(tt.extra_ops.repeat(t0_g, n_cells, axis = 0), tau_cg), mu2[:,:,0], mu0[:,:,0]),
           tt.switch(tt.lt(tt.extra_ops.repeat(t0_g, n_cells, axis = 0), tau_cg), mu2[:,:,1], mu0[:,:,1])], axis = 2) 
        
class DynamicalModel_V8(Pymc3Model):
    r"""
    Dynamic model of splicing kinetics for scRNAseq/snRNAseq data. Version 8
    
    Infers time using just one latent time parameter and two switch times.
    Has a different overdispersion parameter for each gene.
    
    Model parameters include: gene specific transcription rates, splicing rates, degradation rates,
    transcriptional state change time point, as well as cell-specific latent time.
    """

    def __init__(
        self,
        u_counts: np.ndarray,
        s_counts: np.ndarray,
        data_type: str = 'float32',
        n_iter = 20000,
        learning_rate = 0.01,
        total_grad_norm_constraint = 200,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None,  sample_id=None,
        prior_weights = np.array((1,1,1,1))):
        
        ############# Initialise parameters ###########
        super().__init__(u_counts,
                         data_type, n_iter, 
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, sample_id)
        
        self.u_counts = u_counts
        self.s_counts = s_counts
        self.n_cells = np.shape(self.u_counts)[0]
        self.n_genes = np.shape(self.u_counts)[1]
        self.prior_weights = prior_weights
        
        ############# Define the model ################
        self.model = pm.Model()
        with self.model:
            
            # Normalization and state weights:
            self.l_hyp = pm.Gamma('l_hyp', mu = 1, sigma = 0.005)
            self.l_c = pm.Gamma('l_c', mu = self.l_hyp, sigma = self.l_hyp*0.005, shape = (self.n_cells,1,1))
            
            # Hierarchical prior for rates:
            self.alpha_mu = pm.Gamma('alpha_mu', mu = 1, sd = 1, shape = 1)
            self.alpha_sd = pm.Gamma('alpha_sd', mu = 1, sd = 1, shape = 1)        
            self.alpha_0g = pm.Gamma('alpha_0g', mu = self.alpha_mu, sd = self.alpha_sd, shape = (1, self.n_genes))
            
            self.beta_mu = pm.Gamma('beta_mu', mu = 0.25, sd = 0.25, shape = 1)
            self.beta_sd = pm.Gamma('beta_sd', mu = 1, sd = 1, shape = 1)
            self.beta_g = pm.Gamma('beta_g', mu = self.beta_mu, sd = self.beta_sd, shape = (1, self.n_genes))
            
            self.gamma_mu = pm.Gamma('gamma_mu', mu = 0.18, sd = 1, shape = 1)
            self.gamma_sd = pm.Gamma('gamma_sd', mu = 1, sd = 1, shape = 1)
            self.gamma_g = pm.Gamma('gamma_g', mu = self.gamma_mu, sd = self.gamma_sd, shape = (1, self.n_genes))
            
            self.alpha_2g = theano.shared(np.repeat(10**(-5), self.n_genes).reshape(1,self.n_genes), broadcastable=(True,False))
            
            # Time:
            tmax0_mu = pm.Deterministic('tmax0_mu', approx_tau_inf(self.alpha_0g, self.alpha_0g, self.beta_g, self.gamma_g,
                                   self.alpha_2g/self.beta_g, self.alpha_2g/self.gamma_g,
                                   eta = 0.999))
            self.tmax0 = pm.Gamma('tmax0', mu = tmax0_mu, sd = tmax0_mu*0.05, shape = (1,self.n_genes))  
            self.t0_g = pm.Uniform('t0_g', lower = 0, upper = self.tmax0, shape = (1, self.n_genes))
            u0, s0 = repression_initialState(self.alpha_2g, self.alpha_0g, self.beta_g, self.gamma_g, self.t0_g, self.n_cells)
            tmax2_mu = pm.Deterministic('tmax2_mu', approx_tau_inf(self.alpha_2g, self.alpha_0g, self.beta_g, self.gamma_g,
                                   u0, s0, eta = 0.001))
            self.tmax2 = pm.Gamma('tmax2', mu = tmax2_mu[0,:], sd = tmax2_mu[0,:]*0.05, shape = (1,self.n_genes))  
            self.tau_cg = pm.Uniform('tau_cg', lower = 0, upper = self.t0_g + self.tmax2, shape = (self.n_cells, self.n_genes))         
              
#             self.tau_max = pm.Gamma('tau_max', mu = 100, sd = 50, shape = (1, self.n_genes))
#             self.t0_g = pm.Uniform('t0_g', lower = 0, upper = self.tau_max, shape = (1, self.n_genes))
#             self.tau_cg = pm.Uniform('tau_cg', lower = -self.tau_min, upper = self.tau_max, shape = (self.n_cells, self.n_genes))
            
            # Abundance of unspliced and spliced counts:
            self.mu = pm.Deterministic('mu', mu_mRNA_oneState(self.alpha_2g, self.alpha_0g, self.beta_g, self.gamma_g, self.tau_cg, self.t0_g, self.l_c, self.n_cells))

            # Overdispersion for Negative Binomial distribution:
            phi_hyp_g = pm.Gamma('phi_hyp_g', mu=1000,
                                    sigma=10, shape=1)
            gene_E_g = pm.Exponential('gene_E_g', phi_hyp_g, shape=(self.n_genes, 1))
            
            self.a_u_g = pm.Deterministic('a_u_g', 10 + 1 / (gene_E_g.T * gene_E_g.T)) 
            self.a_s_g = pm.Deterministic('a_s_g', 10 + 1 / (gene_E_g.T * gene_E_g.T))                     
            
            self.a_g = tt.stack([self.a_u_g, self.a_s_g], axis = 2)
            
            # Negative Binomial Likelihood:
            self.data_target = pm.NegativeBinomial("data_target", self.mu, self.a_g,
                                           observed=tt.stack([self.u_counts, self.s_counts], axis = 2))
                
    def add_mus_to_adata(self, adata):
        """Add expected values for relevant model parameters to adata"""
    
        adata.var['fit_alpha'] = self.samples['post_sample_means']['alpha_0g'].flatten()                                   
        adata.var['fit_beta'] = self.beta_g
        adata.var['fit_gamma'] = self.samples['post_sample_means']['gamma_g'].flatten()     

        adata.var['fit_u0'] = 0
        adata.var['fit_s0'] = 0

        adata.layers['fit_tau'] = self.samples['post_sample_means']['tau0_cg']
        adata.var['fit_t_'] = self.samples['post_sample_means']['t0_g'].flatten()
        adata.layers['fit_tau_'] = self.samples['post_sample_means']['t0_g'] + self.samples['post_sample_means']['tau2_cg']
        
        adata.layers['fit_t'] = np.zeros((self.n_cells, self.n_genes))
        ind = adata.layers['state'] == 0
        adata.layers['fit_t'][ind] =  adata.layers['fit_tau'][ind]
        rep = adata.layers['state'] == 1
        adata.layers['fit_t'][rep] =  adata.layers['fit_tau_'][rep]
            
    def compute_velocity(self, adata = None):
        """Calculates velocity based on estimated dynamic model parameters."""   
        self.mu_s = np.zeros((self.mus[0].shape.eval()[0], self.n_cells, self.n_genes))
        self.mu_u = np.zeros((self.mus[0].shape.eval()[0], self.n_cells, self.n_genes))
        k = [(self.samples['post_samples']['state'] == i) for i in range(4)]
        for i in range(4):
            self.mu_s[k[i]] = self.mus[i].eval()[:,:,:,1][k[i]]
            self.mu_u[k[i]] = self.mus[i].eval()[:,:,:,0][k[i]]
        self.velocity = self.beta_g * self.mu_u + self.samples['post_samples']['gamma_g']*self.mu_s
        
        