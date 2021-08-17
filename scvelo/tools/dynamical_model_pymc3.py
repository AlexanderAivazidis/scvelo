# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import theano
import matplotlib.pyplot as plt

from scvelo.tools.pymc3_model import Pymc3Model

# defining the model itself
class DynamicModel_V1(Pymc3Model):
    r"""
    Dynamic model of splicing kinetics for scRNAseq/snRNAseq data. Version 1
    
    Model parameters include: gene specific transcription rates, splicing rates, degradation rates,
    transcriptional state change time point, as well as cell-specific latent time
    """

    def __init__(
        self,
        u_counts: np.ndarray,
        s_counts: np.ndarray,
        data_type: str = 'float32',
        n_iter = 20000,
        learning_rate = 0.001,
        total_grad_norm_constraint = 200,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None,  sample_id=None):
        
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
        
        ############# Define the model ################
        self.model = pm.Model()
        with self.model:
            
            # Parameter priors:
            t_max = 100
            self.tau_0 = pm.Exponential('tau_0', lam = 0.02, shape = (self.n_cells, self.n_genes))
            self.tau_2 = pm.Exponential('tau_2', lam = 0.02, shape = (self.n_cells, self.n_genes))
            self.PK_cgk = pm.Dirichlet('PK_cgk', a = np.ones((self.n_cells, self.n_genes, 2)), shape = (self.n_cells, self.n_genes, 2))
            self.alpha_g = pm.Gamma('alpha_g', alpha = 1, beta = 1, shape = (1, self.n_genes))
            self.beta_g = pm.Gamma('beta_g', alpha = 1, beta = 1, shape = (1, self.n_genes))
            self.gamma_g = pm.Gamma('gamma_g', alpha = 1, beta = 1, shape = (1, self.n_genes))
            
            # Calculate expected abundance of unspliced and spliced counts in each of the four states:
                
            # 0: Induction:                                   
            self.mu_u_0cg = self.alpha_g/self.beta_g * (1 - tt.exp(-self.beta_g*self.tau_0))
            self.mu_s_0cg = self.alpha_g/self.gamma_g * (1 - tt.exp(-self.gamma_g*self.tau_0)) + self.alpha_g/(self.gamma_g - self.beta_g) * (tt.exp(-self.gamma_g*self.tau_0) - tt.exp(-self.beta_g*self.tau_0))
            
            # 1: Induction steady state (needed as initial condition for repression state):
#             self.mu_u_1cg = self.alpha_g/self.beta_g * (1 - tt.exp(-self.beta_g*t_max))
#             self.mu_s_1cg = self.alpha_g/self.gamma_g * (1 - tt.exp(-self.gamma_g*t_max)) + self.alpha_g/(self.gamma_g - self.beta_g) * (tt.exp(-self.gamma_g*t_max) - tt.exp(-self.beta_g*t_max))
            
            self.mu_u_1cg = tt.extra_ops.repeat(self.alpha_g/self.beta_g, self.n_cells, axis = 0)
            self.mu_s_1cg = tt.extra_ops.repeat(self.alpha_g/self.gamma_g, self.n_cells, axis = 0)
            
            # 2: Repression:
            self.mu_u_2cg = self.mu_u_1cg*tt.exp(-self.beta_g*self.tau_2)
            self.mu_s_2cg = self.mu_s_1cg*tt.exp(-self.gamma_g*self.tau_2) - self.beta_g*self.mu_u_1cg/(self.gamma_g - self.beta_g) * (tt.exp(-self.gamma_g*self.tau_2) - tt.exp(-self.beta_g*self.tau_2))   
                                                                                                  
            # 3: Repression steady state:                                                                                 
#             self.mu_u_3cg = tt.zeros((self.n_cells, self.n_genes))
#             self.mu_s_3cg = tt.zeros((self.n_cells, self.n_genes))
            
            # Put all states into one tensor:
            self.mu_u_cgk = tt.stack([self.mu_u_0cg,  self.mu_u_2cg], axis = 2)
            self.mu_s_cgk = tt.stack([self.mu_s_0cg,  self.mu_s_2cg], axis = 2)
            
            # And then integrate over outcomes of transcriptional states:                                            
            self.mu_u_cg = tt.sum(self.PK_cgk * self.mu_u_cgk, axis = 2)
            self.mu_s_cg = tt.sum(self.PK_cgk * self.mu_s_cgk, axis = 2)
                                                                                                   
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
            self.mu = tt.concatenate([self.mu_u_cg, self.mu_s_cg], axis = 1)
            self.alpha = tt.concatenate([self.alpha_u_g, self.alpha_s_g], axis = 1) 
            
            # Calculate NB log probability density:
            self.data_target = pm.NegativeBinomial('data_target', mu= self.mu,
                                                   alpha= self.alpha,
                                                   observed=tt.concatenate([self.u_counts, self.s_counts], axis = 1))
                                                                                                                                    
    def add_mus_to_adata(self, adata):
        """Add expected values for relevant model parameters to adata"""
    
        adata.var['fit_alpha'] = self.samples['post_sample_means']['alpha_g'].flatten()                                   
        adata.var['fit_beta'] = self.samples['post_sample_means']['beta_g'].flatten()
        adata.var['fit_gamma'] = self.samples['post_sample_means']['gamma_g'].flatten()     
        adata.var['fit_t_'] = 0 #self.samples['post_sample_means']['t0'].flatten()
        adata.var['fit_u0'] = (self.samples['post_sample_means']['alpha_g']/self.samples['post_sample_means']['beta_g']).flatten()
        adata.var['fit_s0'] = (self.samples['post_sample_means']['alpha_g']/self.samples['post_sample_means']['gamma_g']).flatten()
        
        adata.layers['fit_t'] = self.samples['post_sample_means']['tau_0']
        adata.layers['fit_tau'] = self.samples['post_sample_means']['tau_0']
        adata.layers['fit_tau_'] = self.samples['post_sample_means']['tau_2']
                                                                                                                                    
# -*- coding: utf-8 -*-


def recover_dynamics_pymc3(adata,
                     total_iterations = 1000,
                     learning_rate = 0.01,
                     posterior_samples = 10,
                     verbose = True):
        r"""Wrapper function to fit pymc3 dynamic model. Makes it fit in with scvelo API"""
                                                                                                                                                                                
        if verbose:

            print('Initializing model...')

        model = DynamicModel_V1(
                u_counts = adata.layers['spliced'].toarray(),
                s_counts = adata.layers['unspliced'].toarray())

        if verbose:

            print('Fitting model ...')

        model.fit_advi_iterative(n_iter = total_iterations, learning_rate = learning_rate, n=1, method='advi')

        if verbose:

            model.plot_history()
            plt.show()
            model.plot_history(iter_start = int(np.round(total_iterations - (total_iterations*0.1))),
                               iter_end = int(total_iterations))
            plt.show()

            print('Sampling from posterior distribution...')

        model.sample_posterior(node='all', n_samples=posterior_samples, save_samples=True);

        model.add_mus_to_adata(adata)
#         model.add_sds_to_adata(adata)
        
        adata.uns['recover_dynamics'] = {'fit_connected_states': False, 'fit_basal_transcription': None, 'use_raw': False}
        adata.var['fit_scaling'] = 1
        adata.var['fit_std_u'], adata.var['fit_std_s'] = np.std(adata.layers['spliced'].toarray()), np.std(adata.layers['unspliced'].toarray())
        
        # We use correlation here instead of likelihood to find genes that fit well to the model:
        data_sample = model.mean_field['init_1'].sample_node(model.data_target, size=1).eval()
        data_sample = data_sample.squeeze()
        adata.var['fit_likelihood'] = [(np.corrcoef(data_sample[:,:int(np.shape(data_sample)[1]/2)][:,i], adata.layers['unspliced'].toarray()[:,i])[0,1] +
 np.corrcoef(data_sample[:,int(np.shape(data_sample)[1]/2):][:,i], adata.layers['spliced'].toarray()[:,i])[0,1])/2
 for i in range(int(np.shape(data_sample)[1]/2))]
        
        if verbose:

            print('Done.')
        return model