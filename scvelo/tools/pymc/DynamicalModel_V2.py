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

def mu_mRNA_twoState(alpha_0g, alpha_1g, beta_g, gamma_g, tau0_cg, tau2_cg, t0_g, l_c, n_cells, add_axis = 0):
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
    mu2 = tt.stack([mu_u_2cg, mu_s_2cg], axis = 2 + add_axis)*l_c
    
    return [mu0, mu2]
        
class DynamicalModel_V2(Pymc3Model):
    r"""
    Dynamic model of splicing kinetics for scRNAseq/snRNAseq data. Version 3
    
    Infers time, using a uniform prior that depends on rates. Constrains alpha of NB distribution.
    
    Model parameters include: gene specific transcription rates, splicing rates, degradation rates,
    transcriptional state change time point, as well as cell-specific latent time
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
        prior_weights = np.array((1,1))):
        
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
            self.l_hyp = pm.Gamma('l_hyp', mu = 1, sigma = 0.001)
            self.l_c = pm.Gamma('l_c', mu = self.l_hyp, sigma = self.l_hyp*0.001, shape = (self.n_cells,1,1))
            self.PK_k = pm.Dirichlet('PK_k', a = self.prior_weights, shape = 2)
            # Rates:
            self.beta_g = pm.Gamma('beta_g', mu = 1, sd = 10, shape = (1, self.n_genes))
            limits = pm.Deterministic('limits', pm.math.switch(pm.Beta('d',1,1) > 0.5, (0.01,0.99), (1.01,100)))
#             self.gamma_g = pm.Uniform('gamma_g', limits[0], upper = limits[1], shape = (1, self.n_genes))
#             ss_level = pm.Uniform('ss_level', lower = 0.01, upper = 10**4, shape = (1, self.n_genes))
#             self.alpha_0g = pm.Deterministic('alpha_0g', ss_level/(1/self.beta_g + 1/self.gamma_g))
            self.gamma_g = pm.Uniform('gamma_g', lower = 1, upper = 10, shape = (1, self.n_genes))
            self.alpha_0g = pm.Uniform('alpha_0g', lower = 0.01, upper = 100, shape = (1, self.n_genes))
            self.alpha_2g = pm.Uniform('alpha_2g', lower = self.alpha_0g*10**(-5), upper = self.alpha_0g*2*10**(-5), shape = (1, self.n_genes))
            # Time:
            tmax0 = approx_tau_inf(self.alpha_0g, self.alpha_0g, self.beta_g, self.gamma_g,
                                   self.alpha_2g/self.beta_g, self.alpha_2g/self.gamma_g,
                                   eta = 0.999)
            self.t0_g = pm.Uniform('t0_g', lower = 0.01*tmax0, upper = tmax0, shape = (1, self.n_genes))
            self.tau0_cg = pm.Uniform('tau0_cg', lower = 0, upper = self.t0_g, shape = (self.n_cells, self.n_genes))
            u0, s0 = repression_initialState(self.alpha_2g, self.alpha_0g, self.beta_g, self.gamma_g, self.t0_g, self.n_cells)
            tmax2 = approx_tau_inf(self.alpha_2g, self.alpha_0g, self.beta_g, self.gamma_g,
                                   u0, s0, eta = 0.001)
            self.tau2_cg = pm.Uniform('tau2_cg', lower = 0, upper = tmax2,
                                      shape = (self.n_cells, self.n_genes))
        
            # Abundance of unspliced and spliced counts in each state:
            self.mus = mu_mRNA_twoState(self.alpha_2g, self.alpha_0g, self.beta_g, self.gamma_g, self.tau0_cg,
                               self.tau2_cg, self.t0_g, self.l_c, self.n_cells)

            # Overdispersion for Negative Binomial distribution:
#             phi_hyp_u_g = pm.Gamma('phi_hyp_u_g', mu=10,
#                                     sigma=1, shape=1)
#             gene_E_u_g = pm.Exponential('gene_E_u_g', phi_hyp_u_g, shape=(self.n_genes, 1))
#             self.a_u_g = pm.Deterministic('a_u_g', 1 / (gene_E_u_g.T * gene_E_u_g.T))
#             phi_hyp_s_g = pm.Gamma('phi_hyp_s_g', mu=10,
#                                     sigma=1, shape=1)
#             gene_E_s_g = pm.Exponential('gene_E_s_g', phi_hyp_s_g, shape=(self.n_genes, 1))
#             self.a_s_g = pm.Deterministic('a_s_g', 1 / (gene_E_s_g.T * gene_E_s_g.T))          
            
            self.a_u_g = pm.Uniform('a_u_g', lower = 100, upper = 101, shape = (1, self.n_genes))
            self.a_s_g = pm.Uniform('a_s_g', lower = 100, upper = 101, shape = (1, self.n_genes))
            
            self.a_g = tt.stack([self.a_u_g, self.a_s_g], axis = 2)
            
            # Negative Binomial mixture:
            self.data_target = pm.DensityDist("data_target", logp_nbmix(self.mus, self.PK_k, self.a_g),
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
        
    def compute_mean_expression(self):
        "Computes posterior mean expression of each cell in each state, given posterior of time and rate parameters."

        alpha_0g = theano.shared(self.samples['post_samples']['alpha_0g'])
        alpha_2g = theano.shared(self.samples['post_samples']['alpha_2g'])
        beta_g = self.beta_g
        gamma_g = self.samples['post_samples']['gamma_g']    
        tau0_cg = self.samples['post_samples']['tau0_cg']
        tau2_cg = self.samples['post_samples']['tau2_cg']
        l_c = self.samples['post_samples']['l_c']
        t0_g = theano.shared(self.samples['post_samples']['t0_g'])

        # Abundance of unspliced and spliced counts in each state:
        self.mus = mu_mRNA_twoState(alpha_2g, alpha_0g, beta_g, gamma_g, tau0_cg, tau2_cg, t0_g, l_c, self.n_cells, add_axis = 1)
        
    def compute_latent_state_probs(self, adata = None, posterior_samples = 10):
        """Calculate probability for latent states, given unspliced+spliced counts and taking 
        into account uncertainties in estimated dynamic model parameters."""        
        
        input_data = tt.stack([self.u_counts, self.s_counts], axis = 2)
        PK_k = self.samples['post_samples']['PK_k']
        a_s_g = self.samples['post_samples']['a_s_g']
        a_u_g = self.samples['post_samples']['a_u_g']
        a_g = tt.stack([a_u_g, a_s_g], axis = 3)

        probs = classProbs_nbmix_extraDim(self.mus, PK_k, a_g)(input_data).eval()
        probs = np.mean(probs, axis = 3)
        
        self.samples['post_samples']['state'] = pm.Categorical.dist(probs, shape = (probs.shape[0], probs.shape[1])).random(point = probs, size = posterior_samples)
        self.samples['post_sample_means']['state'] = np.argmax(probs, axis = 2)
        
        if adata:
            adata.layers['state'] = self.samples['post_sample_means']['state']
            
    def compute_velocity(self, adata = None):
        """Calculates velocity based on estimated dynamic model parameters."""   
        self.mu_s = np.zeros((self.mus[0].shape.eval()[0], self.n_cells, self.n_genes))
        self.mu_u = np.zeros((self.mus[0].shape.eval()[0], self.n_cells, self.n_genes))
        k = [(self.samples['post_samples']['state'] == i) for i in range(4)]
        for i in range(4):
            self.mu_s[k[i]] = self.mus[i].eval()[:,:,:,1][k[i]]
            self.mu_u[k[i]] = self.mus[i].eval()[:,:,:,0][k[i]]
        self.velocity = self.beta_g * self.mu_u + self.samples['post_samples']['gamma_g']*self.mu_s
        
        