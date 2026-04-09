#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:13:06 2026

@author: mrutala
"""
import gpflow
import numpy as np
import datetime as dt
import time
import copy
from tqdm.autonotebook import tqdm
import pandas as pd
import tensorflow as tf

from sklearn.pipeline import Pipeline
# from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import multiprocessing as mp
from joblib import Parallel, delayed, wrap_non_picklable_objects
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
import sklearn


class EnsembleGPR:
    def __init__(self, X_list, Y_list, k_list, 
                 X_scaler_list=None, Y_scaler_list=None):
        
        self.X_list = X_list
        self.Y_list = Y_list
        self.k_list = k_list
        
        # In the future, implement auto-scaling (MinMaxScaler)?
        self.X_scaler_list = X_scaler_list
        if self.X_scaler_list is None:
            print("Not supported-- set X scalers!")
            breakpoint()
        self.Y_scaler_list = Y_scaler_list
        if self.Y_scaler_list is None:
            print("Not supported-- set Y scalers!")
            breakpoint()
        
        self.GPR = True
        self.SGPR = False
        
        self.mean_function = None
        self.noise_variance = None
        self.likelihood = None
        
        self.sigma_list = None
        
        # Track individual models in the ensemble
        self.model_list = []
        
        return
    
    def optimize(self):
        
        n_GPs = len(self.X_list)

        for X, Y, k in tqdm(zip(self.X_list, self.Y_list, self.k_list), desc='Optimizing {} GP Models'.format(n_GPs), total=n_GPs):
            # print("Optimizing GP model #{} with {} points".format(i+1, len(X)))
            # t1 = time.time()
            
            pbar = tqdm(desc='Optimizing Model', leave=False)
            def tqdm_callback(j):
                pbar.update(1)
            
            # Copy kernel so the model is freshly solved each loop
            kernel = copy.deepcopy(k)
            
            if self.sigma_list is not None:
                try:
                    breakpoint()
                    variance = self.sigma_list[i]**2
                    variance[variance < 1e-5] = 1e-5 # Avoid <1e-6
                    likelihood = gpflow.likelihoods.Gaussian(variance)
                    gpflow.utilities.set_trainable(likelihood, False)
                except:
                    breakpoint()
            else:
                likelihood = None
                
            
            try:
                import tensorflow_probability as tfp
                custom_config = gpflow.config.Config(jitter=1e-5)
                with gpflow.config.as_context(custom_config):
                    model = gpflow.models.GPR((X, Y),
                                              kernel=kernel,
                                              mean_function=self.mean_function,
                                              noise_variance=self.noise_variance,
                                              likelihood=self.likelihood
                                              )
                    model.likelihood.variance.assign(np.float64(0.01))
                    model.likelihood.variance.prior = tfp.distributions.HalfNormal(np.float64(0.05))
            except:
                breakpoint()
                    
            
            try:
                opt = gpflow.optimizers.Scipy()
                opt.minimize(model.training_loss, model.trainable_variables, callback=tqdm_callback)
            except:
                breakpoint()
            
            # #++++++++++++++++++++++++++++++++++++++++++++++++++
            if model.likelihood.variance > 0.5:
                # Model is broken. Fix?
                import matplotlib.pyplot as plt
                Ymu, Ysigma2 = model.predict_f(X)
                Ymu, Ysigma = Ymu.numpy(), np.sqrt(Ysigma2)
                
                fig, axs = plt.subplots(nrows=2)
                axs[0].scatter(X.flatten(), Y.flatten(), color='black', marker='.', s=10, lw=0)
                
                axs[0].plot(X.flatten(), Ymu.flatten(), color='xkcd:coral')
                axs[0].fill_between(X.flatten(), (Ymu-np.sqrt(Ysigma2)).flatten(), (Ymu+np.sqrt(Ysigma2)).flatten(), color='xkcd:coral', lw=1, alpha=0.33)
    
                axs[1].scatter(X.flatten(), Y.flatten() - Ymu.flatten(), color='xkcd:coral')
                plt.show()
                breakpoint()
            # #--------------------------------------------------
            
            self.model_list.append(model)
            
            
        return
    
    def predict_f(self, scaled_X=None, unscaled_X=None, chunk_size=None, cpu_fraction=None):
        
        # For the time being, assume X is datascaled
        if unscaled_X is None:
            print("Warning! Only data-scaled X is currently supported!")
            return
        elif (scaled_X is None) & (unscaled_X is not None):
            X = unscaled_X
        
        
        # Parse keywords for parallelizing
        if cpu_fraction is None:
            cpu_fraction = 0.50
        n_jobs = int(cpu_fraction * mp.cpu_count())
            
        if chunk_size is None:
            chunk_size = np.ceil(unscaled_X.shape[0] / n_jobs)

        # Get the weights for all X and all models
        # weights = self._getWeights(X)

        # Chunk the data
        n_chunks = np.ceil(X.shape[0] / chunk_size)
        X_chunked = np.array_split(X, n_chunks, axis=0)
        # W_chunked = np.array_split(weights, n_chunks, axis=0)
        
        # Define an internal function to interact with gpflow
        def _predict_f(EnsembleGPR, _X):
            result_mu       = np.full(_X.shape, 0, dtype='float64')
            result_sigma2   = np.full(_X.shape, 0, dtype='float64')   
            
            weights = EnsembleGPR._getWeights(_X)
            
            for i in range(len(EnsembleGPR.model_list)):
                model = EnsembleGPR.model_list[i]
                X_scalers = EnsembleGPR.X_scaler_list[i]
                Y_scaler = EnsembleGPR.Y_scaler_list[i]
                
                _X_scaled = X_scalers.transform(_X)
                # weights = EnsembleGPR._getWeights(_X_scaled)
                
                f_mu, f_sigma2          = model.predict_f(_X_scaled)
                result_mu       += Y_scaler.inverse_transform(f_mu) * weights[i][:,None]
                result_sigma2  += (Y_scaler.scale_**2 * f_sigma2) * weights[i][:,None]
                
            return result_mu, result_sigma2
        
        # Avoid the parallelization overhead if chunk_size == len(X_new)
        if n_chunks > 1:
            generator = Parallel(return_as='generator', n_jobs=n_jobs)(
                delayed(_predict_f)(self, X_chunk) for X_chunk in X_chunked)
        
            results = list(tqdm(generator, total=len(X_chunked), desc="Predicting f(X)"))
        else:
            results = [_predict_f(self, X_chunked)]
        
        # Return mu, variance (sigma^2), and weights as lists of arrays
        results_mu = np.concatenate([r[0] for r in results], axis=0)
        results_sigma2 = np.concatenate([r[1] for r in results], axis=0)
        
        return results_mu, results_sigma2
    
    def predict_f_samples(self, scaled_X=None, unscaled_X=None, chunk_size=None, cpu_fraction=None, num_samples=1, full_cov=True):
        
        # For the time being, assume X is datascaled
        if unscaled_X is None:
            print("Warning! Only data-scaled X is currently supported!")
            return
        elif (scaled_X is None) & (unscaled_X is not None):
            X = unscaled_X
        
        # Parse keywords for parallelizing
        if cpu_fraction is None:
            cpu_fraction = 0.50
        n_jobs = int(cpu_fraction * mp.cpu_count())
            
        if chunk_size is None:
            chunk_size = np.ceil(unscaled_X.shape[0] / n_jobs)

        # Chunk the data
        n_chunks = np.ceil(X.shape[0] / chunk_size)
        
        # Custom chunking logic
        # Chunk the model data, avoiding gaps, so that samples are continuous
        # Then find overlaps with the input X
        model_X = []
        for X_scaler, model in zip(self.X_scaler_list, self.model_list):
            
            model_X.append(X_scaler.inverse_transform(model.data[0]))
            
        all_model_X = np.concatenate(model_X, axis=0)   
        uniq_model_X = np.unique(all_model_X, axis=0)
        model_X_chunks = np.array_split(uniq_model_X, n_chunks, axis=0)
        
        # Just use the last times in each model chunk to subset X
        # In case of missing data/different data cadence, this prevents gaps
        X_chunked = []
        for i, c in enumerate(model_X_chunks):
            start = X[0] if i == 0 else stop # Use the last stop value (noqa)
            stop = X[-1]+1 if i == n_chunks-1 else c[-1]
            overlap_index = (X >= start) & (X < stop)
            X_chunked.append(X[overlap_index][:,None])
        
        # Define an internal function to interact with gpflow
        def _predict_f(EnsembleGPR, _X):
            result_samples = np.full((num_samples, *_X.shape), 0, dtype='float64')
            
            weights = EnsembleGPR._getWeights(_X)
            
            for i in range(len(EnsembleGPR.model_list)):
                model = EnsembleGPR.model_list[i]
                X_scalers = EnsembleGPR.X_scaler_list[i]
                Y_scaler = EnsembleGPR.Y_scaler_list[i]
                shaped_weights = np.tile(weights[i][:,None], (num_samples,1,1))
                                         
                _X_scaled = X_scalers.transform(_X)
                # weights = EnsembleGPR._getWeights(_X_scaled)
                
                f_samples = model.predict_f_samples(_X_scaled, num_samples=num_samples, full_cov=full_cov)
                result_samples += np.array([Y_scaler.inverse_transform(f_sample) for f_sample in f_samples]) * shaped_weights
                
            return result_samples
        
        # Avoid the parallelization overhead if chunk_size == len(X_new)
        if n_chunks > 1:
            generator = Parallel(return_as='generator', n_jobs=n_jobs)(
                delayed(_predict_f)(self, X_chunk) for X_chunk in X_chunked)
        
            results = list(tqdm(generator, total=len(X_chunked), desc="Sampling f(X)"))
        else:
            results = [_predict_f(self, X_chunked)]
        
        # Return mu, variance (sigma^2), and weights as lists of arrays
        results = np.concatenate(results, axis=1)

        return results
    
    def _getWeights(self, unscaled_X):
        from scipy.spatial.distance import cdist
        from scipy.special          import softmax
        
        # Calculate distance from all models
        dists = []
        for X_scalers, model in zip(self.X_scaler_list, self.model_list):
            # Scaled the input X
            X_scaled = X_scalers.transform(unscaled_X)
            dist_matrix = cdist(model.data[0], X_scaled)
            dist_matrix[dist_matrix > 100] = 100
            min_dists = dist_matrix.min(axis=0)
            
            dists.append(min_dists)
            
        dists = np.array(dists)
        weights = softmax(100 - dists, axis=0)
        
        return weights
    
            
    # def predict_f_samples(self, X_new, num_samples=1):
        
    #     weights = self.calculate_weights(X_new)
        
    #     result = np.full((num_samples, len(X_new), 1), 0, dtype='float64')
    #     for w, model in zip(weights, self.model_list):
    #         f_samples = model.predict_f_samples(X_new, num_samples)
            
    #         result += np.tile(w[:,None], (num_samples, 1, 1)) * f_samples.numpy()
        
    #     return result
    
    
    
    # def predict_f_samples(self, X_new_list, num_samples=1, chunk_size=None, cpu_fraction=None):
    #     """
    #     Predict the values of f, the underlying function of GP regression, 
    #     without measurement errors.
    #     If chunksize is supplied, do the prediction in parallel.

    #     """
    #     # Parse keywords for parallelizing
    #     if chunk_size is None:
    #         chunk_size = len(X_new_list[0])
    #     if cpu_fraction is None:
    #         cpu_fraction = 0.50
        
    #     n_jobs = int(cpu_fraction * mp.cpu_count())
    #     breakpoint()
    #     # Chunk the data
    #     X_new_arr = np.array(X_new_list)
    #     n_chunks = np.ceil(X_new_arr.shape[1] / chunk_size)
    #     X_new_chunked = np.array_split(X_new_arr, n_chunks, axis=1)

    #     # Define an internal function to interact with gpflow
    #     def _predict_f_samples(GPFlowEnsemble, _X):
    #         result_sample             = np.full((num_samples, *_X.shape), 0, dtype='float64')          
    #         for i, (x, model) in enumerate(zip(_X, GPFlowEnsemble.model_list)):
    #             f_sample              = model.predict_f_samples(x, num_samples)
    #             result_sample[:,i,:,:] += f_sample.numpy()
    #         return result_sample
        
    #     # Avoid the parallelization overhead if chunk_size == len(X_new)
    #     if n_chunks > 1:
    #         generator = Parallel(return_as='generator', n_jobs=n_jobs)(
    #             delayed(_predict_f_samples)(self, X_chunk) for X_chunk in X_new_chunked)
        
    #         results = list(tqdm(generator, total=len(X_new_chunked)))
    #     else:
    #         results = [_predict_f_samples(self, X_new_chunked)]
        
    #     # Return mu, variance (sigma^2), and weights as lists of arrays
    #     results_sample = np.concatenate([r for r in results], axis=2)
    #     weights = np.concatenate([self.getWeights(_X) for _X in X_new_chunked], axis=1)[:,:,None]
    #     weights = np.repeat(weights[np.newaxis,:], 50, axis=0)
        
    #     return results_sample, weights
    
    
    
    


# class GPFlowEnsemble:
#     def __init__(self, kernel, X_list, Y_list, 
#                  Y_sigma_list=None, noise_variance=None, weight_scaling=60, SGPR=1, interpolate_mean=None, overlap=15):
    
#         if SGPR==1:
#             self.type = 'GPR'
#         else:
#             self.type = 'SGPR'
#             self.inducing_point_fraction = SGPR
    
#         # Given variables
#         self.kernel = kernel
#         self.X_list = X_list
#         self.Y_list = Y_list
#         self.sigma_list = Y_sigma_list
#         self.noise_variance = noise_variance
#         self.weight_scaling = weight_scaling
#         self.overlap = overlap
        
#         if interpolate_mean is not None:
#             # self.mean_function = CustomMeanFunction(*interpolate_mean)
#             breakpoint()
#         else:
#             self.mean_function = None
        
#         # Derived variables
#         self.nChunks = len(X_list)
#         self.model_list = []
#         self.optimize_models()
    
    
#     def optimize_models(self):
        
#         print("Optimizing {} GP models".format(self.nChunks))
#         print("Current time: {}".format(dt.datetime.now().strftime("%H:%M:%S")))
#         t0 = time.time()
    
#         for i, (X, Y) in enumerate(zip(self.X_list, self.Y_list)):
#             print("Optimizing GP model #{} with {} points".format(i+1, len(X)))
#             t1 = time.time()
            
#             # Copy kernel so the model is freshly solved each loop
#             kernel = copy.deepcopy(self.kernel)
#             # kernel = self.kernel
            
#             if self.sigma_list is not None:
#                 try:
#                     variance = self.sigma_list[i]**2
#                     variance[variance < 1e-5] = 1e-5 # Avoid <1e-6
#                     likelihood = gpflow.likelihoods.Gaussian(variance)
#                     gpflow.utilities.set_trainable(likelihood, False)
#                 except:
#                     breakpoint()
#             else:
#                 likelihood = None
                
#             if self.type == 'GPR':
#                 try:
#                     model = gpflow.models.GPR((X, Y),
#                                               kernel=kernel,
#                                               mean_function=self.mean_function,
#                                               noise_variance=self.noise_variance,
#                                               likelihood=likelihood
#                                               )
#                 except:
#                     breakpoint()
                    
#             elif self.type == 'SGPR':
#                 # aim for 20 points
#                 stepsize = int(np.round(1/self.inducing_point_fraction))
#                 print("Step size for SGPR: {}".format(stepsize))
#                 model = gpflow.models.SGPR((X, Y),
#                                            kernel=kernel,
#                                            # noise_variance=self.noise_variance,
#                                            inducing_variable=X[::stepsize,:],
#                                            likelihood=likelihood
#                                            )
#             else:
#                 breakpoint()
            
#             try:
#                 opt = gpflow.optimizers.Scipy()
#                 opt.minimize(model.training_loss, model.trainable_variables)
#             except:
#                 breakpoint()
            
#             self.model_list.append(model)
            
#             if i == 0: first_kernel_iter = copy.deepcopy(self.kernel)
            
#             print("Completed in {:.1f} s".format(time.time() - t1))
            
#         print("All GP models optimized in {:.1f} s".format(time.time() - t0))
#         return

    
#     def predict_f(self, X_new_list, chunk_size=None, cpu_fraction=None):
#         """
#         Predict the values of f, the underlying function of GP regression, 
#         without measurement errors.
#         If chunksize is supplied, do the prediction in parallel.

#         """
#         # Parse keywords for parallelizing
#         if chunk_size is None:
#             chunk_size = len(X_new_list[0])
#         if cpu_fraction is None:
#             cpu_fraction = 0.50
        
#         n_jobs = int(cpu_fraction * mp.cpu_count())
        
#         # Chunk the data
#         X_new_arr = np.array(X_new_list)
#         n_chunks = np.ceil(X_new_arr.shape[1] / chunk_size)
#         X_new_chunked = np.array_split(X_new_arr, n_chunks, axis=1)

#         # Define an internal function to interact with gpflow
#         def _predict_f(GPFlowEnsemble, _X):
#             result_mu       = np.full(_X.shape, 0, dtype='float64')
#             result_sigma2   = np.full(_X.shape, 0, dtype='float64')            
#             for i, (x, model) in enumerate(zip(_X, GPFlowEnsemble.model_list)):
#                 f_mu, f_sigma2          = model.predict_f(x)
#                 result_mu[i,:,:]       += f_mu.numpy()
#                 result_sigma2 [i,:,:]  += f_sigma2.numpy()
#             return result_mu, result_sigma2
        
#         # Avoid the parallelization overhead if chunk_size == len(X_new)
#         if n_chunks > 1:
#             generator = Parallel(return_as='generator', n_jobs=n_jobs)(
#                 delayed(_predict_f)(self, X_chunk) for X_chunk in X_new_chunked)
        
#             results = list(tqdm(generator, total=len(X_new_chunked)))
#         else:
#             results = [_predict_f(self, X_new_chunked)]
        
#         # Return mu, variance (sigma^2), and weights as lists of arrays
#         results_mu = np.concatenate([r[0] for r in results], axis=1)
#         results_sigma2 = np.concatenate([r[1] for r in results], axis=1)
#         weights = np.concatenate([self.getWeights(_X) for _X in X_new_chunked], axis=1)[:,:,None]
        
#         return results_mu, results_sigma2, weights
    
#     def getWeights(self, X_new_list):
                                    
#         from scipy.spatial.distance import cdist
#         from scipy.special          import softmax
        
#         # The minimum distance characterizes which model is 'closest' to the new X
#         dist_list = []
#         for X_new, model in zip(X_new_list, self.model_list):
#             dist_matrix = cdist(model.data[0], X_new)
#             dist_matrix[dist_matrix > 100] = 100
#             dist_list.append(dist_matrix.min(axis=0))
        
#         # Convert distances to an array and inverse scale them
#         dist_arr = np.array(dist_list)
#         initial_weight_arr = 1 - (dist_arr/100)
        
#         # Get the weights, and ditch small numbers
#         # It would be ideal if we could scale these weights dynamically...
#         # i.e., ensure that the weights are > 0.05 within 300 hours of another point
#         # But for now, these weights have effectively no cross-talk
#         weight_arr = softmax(initial_weight_arr*self.weight_scaling, axis=0)
#         weight_arr[weight_arr < 1e-10] = 0
        
#         # As a check, calculate the average number of data points overlapping
#         total_overlap = ((weight_arr > 0.05) & (weight_arr < 0.95)).sum()
#         number_of_overlaps = (len(X_new_list) - 2) * 2 + 2
#         average_overlap = total_overlap / number_of_overlaps
        
#         # Return the weights as a list
#         weight_list = [weight_arr[i] for i in range(len(weight_arr))]
        
#         self.weight_average_overlap = average_overlap
        
#         return weight_list
            
#     # def predict_f_samples(self, X_new, num_samples=1):
        
#     #     weights = self.calculate_weights(X_new)
        
#     #     result = np.full((num_samples, len(X_new), 1), 0, dtype='float64')
#     #     for w, model in zip(weights, self.model_list):
#     #         f_samples = model.predict_f_samples(X_new, num_samples)
            
#     #         result += np.tile(w[:,None], (num_samples, 1, 1)) * f_samples.numpy()
        
#     #     return result
    
    
    
#     def predict_f_samples(self, X_new_list, num_samples=1, chunk_size=None, cpu_fraction=None):
#         """
#         Predict the values of f, the underlying function of GP regression, 
#         without measurement errors.
#         If chunksize is supplied, do the prediction in parallel.

#         """
#         # Parse keywords for parallelizing
#         if chunk_size is None:
#             chunk_size = len(X_new_list[0])
#         if cpu_fraction is None:
#             cpu_fraction = 0.50
        
#         n_jobs = int(cpu_fraction * mp.cpu_count())
#         breakpoint()
#         # Chunk the data
#         X_new_arr = np.array(X_new_list)
#         n_chunks = np.ceil(X_new_arr.shape[1] / chunk_size)
#         X_new_chunked = np.array_split(X_new_arr, n_chunks, axis=1)

#         # Define an internal function to interact with gpflow
#         def _predict_f_samples(GPFlowEnsemble, _X):
#             result_sample             = np.full((num_samples, *_X.shape), 0, dtype='float64')          
#             for i, (x, model) in enumerate(zip(_X, GPFlowEnsemble.model_list)):
#                 f_sample              = model.predict_f_samples(x, num_samples)
#                 result_sample[:,i,:,:] += f_sample.numpy()
#             return result_sample
        
#         # Avoid the parallelization overhead if chunk_size == len(X_new)
#         if n_chunks > 1:
#             generator = Parallel(return_as='generator', n_jobs=n_jobs)(
#                 delayed(_predict_f_samples)(self, X_chunk) for X_chunk in X_new_chunked)
        
#             results = list(tqdm(generator, total=len(X_new_chunked)))
#         else:
#             results = [_predict_f_samples(self, X_new_chunked)]
        
#         # Return mu, variance (sigma^2), and weights as lists of arrays
#         results_sample = np.concatenate([r for r in results], axis=2)
#         weights = np.concatenate([self.getWeights(_X) for _X in X_new_chunked], axis=1)[:,:,None]
#         weights = np.repeat(weights[np.newaxis,:], 50, axis=0)
        
#         return results_sample, weights
    
    
    
    
    
    
    
#     # def predict_f_samples(self, X_new, num_samples=1, chunk_size=None, cpu_fraction=None):
#     #     """
#     #     Predict the values of f, the underlying function of GP regression, 
#     #     without measurement errors.
#     #     If chunksize is supplied, do the prediction in parallel.

#     #     """
#     #     if chunk_size is None:
#     #         chunk_size = len(X_new)
#     #     if cpu_fraction is None:
#     #         cpu_fraction = 0.50
        
#     #     n_jobs = int(cpu_fraction * mp.cpu_count())
        
#     #     X_new_chunked = [X_new[pos:pos + chunk_size] for pos in range(0, len(X_new), chunk_size)]
        
#     #     def _predict_f_samples(GPFlowEnsemble, _X):
#     #         weights = GPFlowEnsemble.calculate_weights(_X)
#     #         result = np.full((num_samples, len(_X), 1), 0, dtype='float64')
#     #         for w, model in zip(weights, GPFlowEnsemble.model_list):
#     #             f_samples = model.predict_f_samples(_X, num_samples)
#     #             result += np.tile(w[:,None], (num_samples, 1, 1)) * f_samples.numpy()
                
#     #         return result
        
#     #     # Avoid the parallelization overhead if chunk_size == len(X_new)
#     #     if len(X_new_chunked) > 1:
#     #         generator = Parallel(return_as='generator', n_jobs=n_jobs)(
#     #             delayed(_predict_f_samples)(self, X_chunk) for X_chunk in X_new_chunked)
        
#     #         results = list(tqdm(generator, total=len(X_new_chunked)))
#     #     else:
#     #         results = [_predict_f_samples(self, X_new)]
        
#     #     results = np.concatenate(results, axis=1)
        
#     #     return results
    
#     # def predict_y(self, X_new):
        
#     #     weights = self.calculate_weights(X_new)
        
#     #     result_mu = np.full((len(X_new), 1), 0, dtype='float64')
#     #     result_sigma2 = np.full((len(X_new), 1), 0, dtype='float64')
#     #     for w, model in zip(weights, self.model_list):
#     #         y_mu, y_sigma2 = model.predict_y(X_new)
            
#     #         result_mu += w[:,None] * y_mu.numpy()
#     #         result_sigma2 += w[:,None] * y_sigma2.numpy()
            
#     #     return result_mu, result_sigma2
    
#     # def predict_y(self, X_new, chunk_size=None, cpu_fraction=None):
#     #     if chunk_size is None:
#     #         chunk_size = len(X_new)
#     #     if cpu_fraction is None:
#     #         cpu_fraction = 0.50
        
#     #     n_jobs = int(cpu_fraction * mp.cpu_count())
        
#     #     X_new_chunked = [X_new[pos:pos + chunk_size] for pos in range(0, len(X_new), chunk_size)]
        
#     #     def _predict_y(GPFlowEnsemble, _X):
#     #         weights = GPFlowEnsemble.calculate_weights(_X)
#     #         result_mu = np.full((len(_X), 1), 0, dtype='float64')
#     #         result_sigma2 = np.full((len(_X), 1), 0, dtype='float64')
#     #         for w, model in zip(weights, GPFlowEnsemble.model_list):
#     #             y_mu, y_sigma2 = model.predict_y(_X)
                
#     #             result_mu += w[:,None] * y_mu.numpy()
#     #             result_sigma2 += w[:,None] * y_sigma2.numpy()
                
#     #         return result_mu, result_sigma2
        
#     #     # Avoid the parallelization overhead if chunk_size == len(X_new)
#     #     if len(X_new_chunked) > 1:
#     #         generator = Parallel(return_as='generator', n_jobs=n_jobs)(
#     #             delayed(_predict_y)(self, X_chunk) for X_chunk in X_new_chunked)
        
#     #         results = list(tqdm(generator, total=len(X_new_chunked)))
#     #     else:
#     #         results = [_predict_y(self, X_new)]
        
#     #     results_mu = np.vstack([r[0] for r in results])
#     #     results_sigma2 = np.vstack([r[1] for r in results])

#     #     return results_mu, results_sigma2
    
#     # def calculate_weights(self, X_new):
#     #     import scipy
#     #     from scipy.spatial.distance import cdist
        
#     #     # X_centers = [np.mean(X, axis=0) for X in self.X_list]
        
#     #     # # Distances are n_chunks by n_X_new
#     #     # distances = [np.linalg.norm(X_new - X_center, axis=1) for X_center in X_centers]
#     #     # distances = np.stack(distances) 
        
#     #     # weights = scipy.special.softmax(-distances, axis=0)
        
#     #     min_distances = []
#     #     for model in self.model_list:
#     #         # Get only the X dimensions of the model data
#     #         data = model.data[0]
            
#     #         dist_matrix = cdist(data, X_new)
#     #         min_dists = np.min(dist_matrix, axis=0)
            
#     #         min_distances.append(min_dists)
            
#     #     min_distances = np.array(min_distances)
        
#     #     # Normalize min_distances to the distance expected after 
#     #     norm_min_distances = min_distances / (1/len(self.X_list))
#     #     norm_min_distances[norm_min_distances > 1] = 1
        
#     #     weights = scipy.special.softmax(self.weight_scaling*(1-norm_min_distances), axis=0)
        
#     #     return weights
    
#     def print_summary(self):
#         import gpflow
#         for model in self.model_list:
#             gpflow.utilities.print_summary(model, 'simple')
            
#         df = pd.DataFrame()
#         for i, model in enumerate(self.model_list):
#             d = gpflow.utilities.parameter_dict(model)
            
#             for key, value in d.items():
#                 df.loc[i, key] = value.numpy()
            
#         return df


# # def _process_sample(df_sample, method_sample):
# #     sf_sample_copy = df_sample.copy(deep=True)
# #     insitu_df_copy['V'] = U_sample
# #     return map_vBoundaryInwards(source, insitu_df_copy, method_sample)

# # # %%
# # if __name__ == '__main__':
# #     import generate_external_input
# #     # =========================================================================
# #     # THIS SHOULD ALL BE MOVED TO A NOTEBOOK WHEN WORKING!
# #     # =========================================================================
    
# #     # ========================================================================
# #     # Initialize an MSIR inputs object
# #     # =========================================================================
# #     start = dt.datetime(2012, 1, 1)
# #     stop = dt.datetime(2012, 7, 1)
# #     rmax = 10 # AU
# #     latmax = 15
    
# #     inputs = multihuxt_inputs(start, stop, rmax=rmax, latmax=latmax)
# #     # =============================================================================
# #     # Search for available background SW and transient data
# #     # =============================================================================
# #     inputs.get_availableBackgroundData()
# #     inputs.filter_availableBackgroundData()
# #     # inputs.sort_availableSources('rad_HGI')
    
# #     # Get ICME/IPS data for all available source
# #     inputs.get_availableTransientData()
    
# #     # =============================================================================
# #     # Generate background and boundary distributions:
# #     #   - Remove ICMEs
# #     #   - GP interpolate 1D in-situ time series
# #     #   - Backmap to 21.5 RS
# #     #   - GP interpolate 3D (time, lon, lat) source model
# #     # =============================================================================
    
# #     # Generate an input CME distribution
# #     inputs.generate_cmeDistribution()
    
# #     inputs.generate_backgroundDistributions()
    
# #     inputs.generate_boundaryDistributions(nSamples=16, constant_sig=0)
    
# #     # Either choose one boundary distribution, or do a 3D GP interpolation
# #     # inputs.generate_boundaryDistribution3D(nLat=32, extend='omni', GP=False)
# #     inputs.generate_boundaryDistribution3D(nLat=32, GP=True)
    

    
# #     breakpoint()

# #     # Add Saturn SKR Data
# #     saturn_df = generate_external_input.Cassini_SKR(inputs.availableBackgroundData.index)
# #     inputs.availableBackgroundData = pd.merge(inputs.availableBackgroundData, 
# #                                               saturn_df,
# #                                               left_index=True, right_index=True)
    
# #     nSamples = 16
# #     weights = [1/nSamples]*nSamples
    
# #     # for source in ...
# #     source = 'saturn'
    
# #     boundarySamples, cmeSamples = inputs.sample3D(weights, at=source)
    
# #     ensemble = inputs.predict2(boundarySamples, cmeSamples, source)
    
# #     # Save as checkpoint
# #     with open('/Users/mrutala/projects/OHTransients/inputs_checkpoint.pkl', 'wb') as f:
# #         pickle.dump(inputs, f)
         
# #     with open('/Users/mrutala/projects/OHTransients/inputs_checkpoint.pkl', 'rb') as f:
# #         inputs = pickle.load(f)
    
# #     # CIME interaction time @ Saturn (Palmerio+ 2021)
# #     interaction_time = dt.datetime(2012, 6, 12, 00, 00)
    