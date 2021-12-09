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

from scvelo.tools.pymc.DynamicalModel_V3 import DynamicalModel_V3
from scvelo.tools.pymc.DynamicalModel_V4 import DynamicalModel_V4

def recover_dynamics_pymc3(adata,
                     total_iterations = 1000,
                     learning_rate = 0.005,
                     verbose = True,
                     sample_prior = True,
                     diagnostic_plots = True,
                     posterior_samples = 10,
                     model_version = 'V3',
                     prior_samples = 10,
                     save = False):
        r"""Wrapper function to fit pymc3 dynamic model. Makes it fit in with scvelo API"""                                                         
        if save:
            
            directoryName = save + 'DynamicModel_' + model_version + '-' + str(np.shape(adata)[0]) + 'Cells' + '-' + str(np.shape(adata)[1]) + 'Genes' + '-' + str(total_iterations) + 'Iterations' + '-' + str(learning_rate) + 'LearningRate'     
            os.mkdir(directoryName)
        
        if verbose:
            
            if sample_prior:
            
                print('Initializing model and sampling from prior...')
                
            else:
                
                print('Initializing model...')
                
        versions = np.array(['V3', 'V4'])
        chosen_model = np.array([DynamicalModel_V3, DynamicalModel_V4])[versions == model_version][0]
        
        if verbose:
            
            print(chosen_model)
        
        model = chosen_model(
                u_counts = adata.layers['unspliced'].toarray(),
                s_counts = adata.layers['spliced'].toarray())
        
        if save:
            
            model.directoryName = directoryName   
        
        if sample_prior:
            
            print(np.shape(model.prior_sample))
            
            plt.hist2d(np.log10(model.prior_sample['data_target'][0,:,:int(np.shape(model.prior_sample['data_target'])[2]/2)].flatten()+1),
                       np.log10(model.u_counts.flatten()+1),
                       bins=50, norm=matplotlib.colors.LogNorm())
            plt.xlabel('Data, log10(nUMI)')
            plt.ylabel('Prior sample, log10(nUMI)')
            plt.title('Unspliced UMI counts (all cells, all genes)')
            plt.tight_layout()
            plt.show()

            plt.hist2d(np.log10(model.prior_sample['data_target'][0,:,int(np.shape(model.prior_sample['data_target'])[2]/2):].flatten()+1),
                       np.log10(model.s_counts.flatten()+1),
                       bins=50, norm=matplotlib.colors.LogNorm())
            plt.xlabel('Data, log10(nUMI)')
            plt.ylabel('Prior sample, log10(nUMI)')
            plt.title('Spliced UMI counts (all cells, all genes)')
            plt.show()
        
        if verbose:

            print('Fitting model ...')

        model.fit_advi_iterative(n_iter = total_iterations, learning_rate = learning_rate, n=1, method='advi')

        if verbose:

            model.plot_history()
            if save:
                plt.savefig(directoryName + '/OptimizationLossAll.pdf')
            plt.show()
            model.plot_history(iter_start = int(np.round(total_iterations - (total_iterations*0.25))),
                               iter_end = int(total_iterations))
            if save:
                plt.savefig(directoryName + '/OptimizationLossLast25Percent.pdf')
            plt.show()

            print('Sampling from posterior distribution...')
        
        model.sample_posterior(node='all', n_samples=posterior_samples, save_samples=True);
        
        if verbose:
            
            print('Calculating mean expression ...')
            
        model.compute_mean_expression()
        
        if verbose:

            print('Calculating latent state probabilities ...')
        
        model.compute_latent_state_probs(adata, posterior_samples = posterior_samples)
        
        if verbose:

            print('Calculating velocity ...')
        
        model.compute_velocity()
        
        model.add_mus_to_adata(adata)
        
        adata.uns['recover_dynamics'] = {'fit_connected_states': False, 'fit_basal_transcription': None, 'use_raw': False}
        adata.var['fit_scaling'] = 1
        adata.var['fit_std_u'], adata.var['fit_std_s'] = np.std(adata.layers['spliced'].toarray()), np.std(adata.layers['unspliced'].toarray())
        
        # We use correlation here instead of likelihood to find genes that fit well to the model:
        data_sample = model.mean_field['init_1'].sample_node(model.data_target, size=1).eval()
        data_sample = data_sample.squeeze()
        adata.var['fit_likelihood'] = [(np.corrcoef(data_sample[:,:,0][:,i], adata.layers['unspliced'].toarray()[:,i])[0,1] +
         np.corrcoef(data_sample[:,:,1][:,i], adata.layers['spliced'].toarray()[:,i])[0,1])/2
         for i in range(int(np.shape(data_sample)[1]))]
        
        if diagnostic_plots:
            
            if verbose:

                print('Producing diagnostic plots ...')
            
            # Plot 1:
            
            data_sample = model.mean_field['init_1'].sample_node(model.data_target, size=1).eval()
            
            data_sample = data_sample.squeeze()
            
            plt.hist2d(np.log10(data_sample[:,:,0].flatten()+1), np.log10(model.u_counts.flatten()+1),
                       bins=50, norm=matplotlib.colors.LogNorm())
            plt.xlabel('Data, log10(nUMI)')
            plt.ylabel('Posterior sample, log10(nUMI)')
            plt.title('Unspliced UMI counts (all cells, all genes)')
            plt.tight_layout()
            if save:
                plt.savefig(directoryName + '/DataVsPosterior_Unspliced.pdf')
            plt.show()

            plt.hist2d(np.log10(data_sample[:,:,1].flatten()+1), np.log10(model.s_counts.flatten()+1),
                       bins=50, norm=matplotlib.colors.LogNorm())
            plt.xlabel('Data, log10(nUMI)')
            plt.ylabel('Posterior sample, log10(nUMI)')
            plt.title('Spliced UMI counts (all cells, all genes)')
            plt.tight_layout()
            if save:
                plt.savefig(directoryName + '/DataVsPosterior_Spliced.pdf')
            plt.show()
            
            # Plot 2:
            
            fig, ax = plt.subplots(4,4, figsize = (12,9))
            
            if sample_prior:
                
                ax[0,0].hist(model.prior_sample['alpha_kg'][:,0,:].flatten(), density = True, label = 'Prior', bins = 100)
                ax[0,1].hist(model.prior_sample['beta_g'].flatten(), density = True, label = 'Prior', bins = 100)
                ax[0,2].hist(model.prior_sample['gamma_g'].flatten(), density = True, label = 'Prior', bins = 100)
                ax[1,2].hist(model.prior_sample['tau0_cg'].flatten(), density = True, label = 'Prior', bins = 100)
                ax[1,1].hist(model.prior_sample['tau2_cg'].flatten(), density = True, label = 'Prior', bins = 100)
                ax[2,0].hist(model.prior_sample['PK_k'][:,0].flatten(), density = True, label = 'Prior', bins = 100)
                ax[2,2].hist(model.prior_sample['PK_k'][:,1].flatten(), density = True, label = 'Prior', bins = 100)
                
            ax[0,0].hist(model.samples['post_samples']['alpha_0g'].flatten(), density = True, label = 'Posterior', bins = 100)
            ax[0,0].legend()
            ax[0,0].set_xlabel('Alpha')
            ax[0,0].set_ylabel('Density')

            ax[0,1].hist(model.samples['post_samples']['beta_g'].flatten(), density = True, label = 'Posterior ', bins = 100)
            ax[0,1].legend()
            ax[0,1].set_xlabel('Beta')
            ax[0,1].set_ylabel('Density')

            ax[0,2].hist(model.samples['post_samples']['gamma_g'].flatten(), density = True, label = 'Posterior', bins = 100)
            ax[0,2].legend()
            ax[0,2].set_xlabel('Gamma')
            ax[0,2].set_ylabel('Density')

            ax[0,3].hist(model.samples['post_samples']['l_hyp'].flatten(), density = True, label = 'Posterior')
            ax[0,3].legend()
            ax[0,3].set_xlabel('l_hyp')
            ax[0,3].set_ylabel('Density')
            
            ax[1,0].hist(model.samples['post_samples']['t0_g'].flatten(), density = True, label = 'Posterior', bins = 100)
            ax[1,0].legend()
            ax[1,0].set_xlabel('t0_g')
            ax[1,0].set_ylabel('Density')
            
            ax[1,1].hist(model.samples['post_samples']['tau2_cg'].flatten(), density = True, label = 'Posterior', bins = 100)
            ax[1,1].legend()
            ax[1,1].set_xlabel('tau2_cg')
            ax[1,1].set_ylabel('Density')
            
            ax[1,2].hist(model.samples['post_samples']['tau0_cg'].flatten(), density = True, label = 'Posterior', bins = 100)
            ax[1,2].legend()
            ax[1,2].set_xlabel('tau0_cg')
            ax[1,2].set_ylabel('Density')
            
            ax[1,3].hist(model.samples['post_samples']['l_c'].flatten(), density = True, label = 'Posterior')
            ax[1,3].legend()
            ax[1,3].set_xlabel('l_c')
            ax[1,3].set_ylabel('Density')
            
            ax[2,0].hist(model.samples['post_samples']['PK_k'][:,0].flatten(), density = True, label = 'Posterior')
            ax[2,0].legend()
            ax[2,0].set_xlabel('Probability for state 0')
            ax[2,0].set_ylabel('Density')
            
            ax[2,1].hist(model.samples['post_samples']['PK_k'][:,1].flatten(), density = True, label = 'Posterior')
            ax[2,1].legend()
            ax[2,1].set_xlabel('Probability for state 1')
            ax[2,1].set_ylabel('Density')
            
            ax[2,2].hist(model.samples['post_samples']['PK_k'][:,2].flatten(), density = True, label = 'Posterior')
            ax[2,2].legend()
            ax[2,2].set_xlabel('Probability for state 2')
            ax[2,2].set_ylabel('Density')
            
            ax[2,3].hist(model.samples['post_samples']['PK_k'][:,3].flatten(), density = True, label = 'Posterior')
            ax[2,3].legend()
            ax[2,3].set_xlabel('Probability for state 3')
            ax[2,3].set_ylabel('Density')
            
            ax[3,1].hist(model.samples['post_samples']['alpha_2g'].flatten(), density = True, label = 'Posterior', bins = 100)
            ax[3,1].legend()
            ax[3,1].set_xlabel('Alpha')
            ax[3,1].set_ylabel('Density')
            
            ax[3,2].hist(model.samples['post_samples']['a_u_g'].flatten(), density = True, label = 'Posterior')
            ax[3,2].legend()
            ax[3,2].set_xlabel('a_u_g')
            ax[3,2].set_ylabel('Density')
            
            ax[3,3].hist(model.samples['post_samples']['a_s_g'].flatten(), density = True, label = 'Posterior')
            ax[3,3].legend()
            ax[3,3].set_xlabel('a_s_g')
            ax[3,3].set_ylabel('Density')

            plt.tight_layout()
            if save:
                plt.savefig(directoryName + '/ParameterPosteriors.pdf')
            plt.show()
            
            # Plot 3:
            
            cv_tau0 = [np.sqrt(np.var(model.samples['post_samples']['tau0_cg'][:,:,i], axis = 0)/np.mean(model.samples['post_samples']['tau0_cg'][:,:,i], axis = 0))for i in range(model.n_genes)]
            mean_tau0 = [np.mean(model.samples['post_samples']['tau0_cg'][:,:,i], axis = 0) for i in range(model.n_genes)]
            cv_tau2 = [np.sqrt(np.var(model.samples['post_samples']['tau2_cg'][:,:,i], axis = 0)/np.mean(model.samples['post_samples']['tau0_cg'][:,:,i], axis = 0))for i in range(model.n_genes)]
            mean_tau2 = [np.mean(model.samples['post_samples']['tau2_cg'][:,:,i], axis = 0) for i in range(model.n_genes)]
            cv_alpha0 = [np.sqrt(np.var(model.samples['post_samples']['alpha_0g'][:,:,i], axis = 0)/np.mean(model.samples['post_samples']['alpha_0g'][:,:,i], axis = 0))for i in range(model.n_genes)]
            mean_alpha0 = [np.mean(model.samples['post_samples']['alpha_0g'][:,:,i], axis = 0) for i in range(model.n_genes)]
            cv_beta = [np.sqrt(np.var(model.samples['post_samples']['beta_g'][:,:,i], axis = 0)/np.mean(model.samples['post_samples']['beta_g'][:,:,i], axis = 0)) for i in range(model.n_genes)]
            mean_beta = [np.mean(model.samples['post_samples']['beta_g'][:,:,i], axis = 0) for i in range(model.n_genes)]
            cv_gamma = [np.sqrt(np.var(model.samples['post_samples']['gamma_g'][:,:,i], axis = 0)/np.mean(model.samples['post_samples']['gamma_g'][:,:,i], axis = 0)) for i in range(model.n_genes)]
            mean_gamma = [np.mean(model.samples['post_samples']['gamma_g'][:,:,i], axis = 0) for i in range(model.n_genes)]
            cv_t0 = [np.sqrt(np.var(model.samples['post_samples']['t0_g'][:,:,i], axis = 0)/np.mean(model.samples['post_samples']['t0_g'][:,:,i], axis = 0)) for i in range(model.n_genes)]
            mean_t0 = [np.mean(model.samples['post_samples']['t0_g'][:,:,i], axis = 0) for i in range(model.n_genes)]

            gene = 'HOPX'

            fig, ax = plt.subplots(6,3, figsize = (10,6*2.5))

            ax[0,0].hist(np.array(cv_tau0)[adata.var_names == gene,:].flatten(), bins = 100, density = True)
            ax[0,0].set_xlabel('Uncertainty (CV)')
            ax[0,0].set_ylabel('Density')
            ax[0,0].set_title('tau_0' + ' (' + gene + ')')
            ax[0,1].scatter(np.array(mean_tau0)[adata.var_names == gene,:].flatten(),
                        np.array(cv_tau0)[adata.var_names == gene,:].flatten(),
                       alpha = 0.1)
            ax[0,1].set_xlabel('Mean')
            ax[0,1].set_ylabel('Uncertainty (CV)')
            ax[0,1].set_title('tau_0' + ' (' + gene + ')')
            ax[0,2].hist(np.array(mean_tau0)[adata.var_names == gene,:].flatten(), bins = 100, density = True)
            ax[0,2].set_xlabel('Mean')
            ax[0,2].set_ylabel('Density')
            ax[0,2].set_title('tau_0' + ' (' + gene + ')')

            ax[1,0].hist(np.array(cv_tau2)[adata.var_names == gene,:].flatten(), bins = 100, density = True)
            ax[1,0].set_xlabel('Uncertainty (CV)')
            ax[1,0].set_ylabel('Density')
            ax[1,0].set_title('tau_2' + ' (' + gene + ')')
            ax[1,1].scatter(np.array(mean_tau2)[adata.var_names == gene,:].flatten(),
                        np.array(cv_tau2)[adata.var_names == gene,:].flatten(),
                       alpha = 0.1)
            ax[1,1].set_xlabel('Mean')
            ax[1,1].set_ylabel('Uncertainty (CV)')
            ax[1,1].set_title('tau_2' + ' (' + gene + ')')
            ax[1,2].hist(np.array(mean_tau2)[adata.var_names == gene,:].flatten(), bins = 100, density = True)
            ax[1,2].set_xlabel('Mean')
            ax[1,2].set_ylabel('Density')
            ax[1,2].set_title('tau_2' + ' (' + gene + ')')

            ax[2,0].hist(np.array(cv_alpha0).flatten(), bins = 100, density = True)
            ax[2,0].set_xlabel('Uncertainty (CV)')
            ax[2,0].set_ylabel('Density')
            ax[2,0].set_title('alpha_0g')
            ax[2,1].scatter(np.array(mean_alpha0).flatten(),
                        np.array(cv_alpha0).flatten(),
                       alpha = 0.1)
            ax[2,1].set_xlabel('Mean')
            ax[2,1].set_ylabel('Uncertainty (CV)')
            ax[2,1].set_title('alpha_0g')
            ax[2,2].hist(np.array(mean_alpha0).flatten(), bins = 100, density = True)
            ax[2,2].set_xlabel('Mean')
            ax[2,2].set_ylabel('Density')
            ax[2,2].set_title('alpha_0g')

            ax[3,0].hist(np.array(cv_t0).flatten(), bins = 100, density = True)
            ax[3,0].set_xlabel('Uncertainty (CV)')
            ax[3,0].set_ylabel('Density')
            ax[3,0].set_title('t0_g')
            ax[3,1].scatter(np.array(mean_t0).flatten(),
                        np.array(cv_t0).flatten(),
                       alpha = 0.1)
            ax[3,1].set_xlabel('Mean')
            ax[3,1].set_ylabel('Uncertainty (CV)')
            ax[3,1].set_title('t0_g')
            ax[3,2].hist(np.array(mean_t0).flatten(), bins = 100, density = True)
            ax[3,2].set_xlabel('Mean')
            ax[3,2].set_ylabel('Density')
            ax[3,2].set_title('t0_g')

            ax[4,0].hist(np.array(cv_beta).flatten(), bins = 100, density = True)
            ax[4,0].set_xlabel('Uncertainty (CV)')
            ax[4,0].set_ylabel('Density')
            ax[4,0].set_title('beta_g')
            ax[4,1].scatter(np.array(mean_beta).flatten(),
                        np.array(cv_beta).flatten(),
                       alpha = 0.1)
            ax[4,1].set_xlabel('Mean')
            ax[4,1].set_ylabel('Uncertainty (CV)')
            ax[4,1].set_title('beta_g')
            ax[4,2].hist(np.array(mean_beta).flatten(), bins = 100, density = True)
            ax[4,2].set_xlabel('Mean')
            ax[4,2].set_ylabel('Density')
            ax[4,2].set_title('beta_g')

            ax[5,0].hist(np.array(cv_gamma).flatten(), bins = 100, density = True)
            ax[5,0].set_xlabel('Uncertainty (CV)')
            ax[5,0].set_ylabel('Density')
            ax[5,0].set_title('gamma_g')
            ax[5,1].scatter(np.array(mean_gamma).flatten(),
                        np.array(cv_gamma).flatten(),
                       alpha = 0.1)
            ax[5,1].set_xlabel('Mean')
            ax[5,1].set_ylabel('Uncertainty (CV)')
            ax[5,1].set_title('gamma_g')
            ax[5,2].hist(np.array(mean_gamma).flatten(), bins = 100, density = True)
            ax[5,2].set_xlabel('Mean')
            ax[5,2].set_ylabel('Density')
            ax[5,2].set_title('gamma_g')

            plt.tight_layout()
            if save:
                plt.savefig(directoryName + '/ParameterPosteriorUncertainties.pdf')
            plt.show()
            
        if save:
            if verbose:
                print('Saving model output...')
#             sys.setrecursionlimit(50000)
            pickle.dump({'posterior_samples': model.samples, 'AnnData': adata}, open(directoryName + "/model_output.p", "wb" ) )
            
        if verbose:

            print('Done.')
            
        return model