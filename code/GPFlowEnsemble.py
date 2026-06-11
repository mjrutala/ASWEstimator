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
# from tqdm.autonotebook import tqdm
from tqdm import tqdm
import pandas as pd
import tensorflow as tf

from sklearn.pipeline import Pipeline
# from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import multiprocessing as mp
from joblib import Parallel, delayed, wrap_non_picklable_objects
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
import sklearn
import tensorflow_probability as tfp

class EnsembleGPR:
    def __init__(self, X_list, Y_list, k_lists, 
                 X_scaler_list=None, Y_scaler_list=None, 
                 Y_variance_list=None, Y_variance_scaler_list=None, variance_kernels=None,
                 mean_function_list=None, jitter=1e-5):
        
        self.jitter = jitter
        
        self.X_list = X_list
        self.Y_list = Y_list
        self.k_lists = k_lists # list of lists, potentially
        
        # In the future, implement auto-scaling (MinMaxScaler)?
        self.X_scaler_list = X_scaler_list
        if self.X_scaler_list is None:
            print("Not supported-- set X scalers!")
            breakpoint()
        self.Y_scaler_list = Y_scaler_list
        if self.Y_scaler_list is None:
            print("Not supported-- set Y scalers!")
            breakpoint()
        
        self.Y_variance_list = Y_variance_list
        self.Y_variance_scaler_list = Y_variance_scaler_list
        self.variance_kernels = variance_kernels
        
        # self.GPR = True
        # self.SGPR = False
        
        self.mean_function_list = mean_function_list
        # self.noise_variance = None
        # self.likelihood = None
        
        # self.sigma_list = None
        
        # Track individual models in the ensemble
        self.model_list = []
        self.variance_model_list = []
        
        return
    
    def optimize(self):
        
        n_GPs = len(self.X_list)

        for i, (X, Y, k_list) in tqdm(enumerate(zip(self.X_list, self.Y_list, self.k_lists)), 
                              desc='Optimizing {} GP Models'.format(n_GPs), total=n_GPs):
            # print("Optimizing GP model #{} with {} points".format(i+1, len(X)))
            # t1 = time.time()
            
            pbar = tqdm(desc='Optimizing Model', leave=False)
            def tqdm_callback(j):
                pbar.update(1)
            
            # Set up likelihood
            likelihood = gpflow.likelihoods.Gaussian(variance=0.02**2)
            likelihood.variance.prior = tfp.distributions.HalfNormal(np.float64(0.02**2))
            
            if self.mean_function_list is not None:
                mean_function = self.mean_function_list[i]
            else:
                mean_function = None
            
            # Try successively simpler kernels
            if type(k_list) != list:
                k_list = [k_list]
            for kernel_ in k_list:
                
                # # Copy kernel so the model is freshly solved each loop
                kernel = copy.deepcopy(kernel_)
                
                custom_config = gpflow.config.Config(jitter=self.jitter)
                with gpflow.config.as_context(custom_config):
                    model = gpflow.models.GPR((X, Y),
                                              kernel=kernel,
                                              mean_function=mean_function,
                                              # noise_variance=self.noise_variance,
                                              likelihood=likelihood,
                                              )
            
                opt = gpflow.optimizers.Scipy()
                status = opt.minimize(
                    model.training_loss,
                    model.trainable_variables, 
                    method='L-BFGS-B', 
                    callback=tqdm_callback)
                
                if status.success:
                    break
                
            self.model_list.append(model)
            
            #
            if self.Y_variance_list is not None:
                for variance_kernel_ in self.variance_kernels:
                    variance_kernel = copy.deepcopy(variance_kernel_)
                    
                    custom_config = gpflow.config.Config(jitter=self.jitter)
                    with gpflow.config.as_context(custom_config):
                        variance_model = gpflow.models.GPR(
                            (X, self.Y_variance_list[i]), kernel = variance_kernel, likelihood=likelihood)
                        
                        v_opt = gpflow.optimizers.Scipy()
                        v_status = v_opt.minimize(
                            variance_model.training_loss, 
                            variance_model.trainable_variables, 
                            method='L-BFGS-B')
                        
                        if v_status.success:
                            break
                        
                self.variance_model_list.append(variance_model)
            
            if not status.success:
                print("All kernel options failed to optimize!")
                breakpoint()
                
            
            
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
        n_chunks = np.ceil(X.shape[0] / chunk_size)    
        
        mu, var = self._predict_f(X, n_chunks, n_jobs)
        
        # If there is a variance model, add it to the mean model variance 
        if len(self.variance_model_list) > 0:
            var_mu, var_var = self._predict_f(X, n_chunks, n_jobs, analyze_variance=True)
            var = var + var_mu
        
        return mu, var
    
    def _predict_f(self, X, n_chunks, n_jobs, analyze_variance=False, full_cov=False):
        
        if analyze_variance:
            model_list = self.variance_model_list
            scaler_list = self.Y_variance_scaler_list
        else:
            model_list = self.model_list
            scaler_list = self.Y_scaler_list
        
        # Chunk the data
        X_chunked = np.array_split(X, n_chunks, axis=0)
        
        # Define an internal function to interact with gpflow
        def _parallel_predict_f(EnsembleGPR, _X, model_list, scaler_list):
            result_mu       = np.full((_X.shape[0],1), 0, dtype='float64')
            result_sigma2   = np.full((_X.shape[0],1), 0, dtype='float64')   
            
            weights = EnsembleGPR._getWeights(_X)
            
            for i in range(len(model_list)):
                model = model_list[i]
                X_scalers = EnsembleGPR.X_scaler_list[i]
                Y_scaler = scaler_list[i]
                
                # Get X, even if multiple scalers
                _X_scaled = np.zeros(_X.shape)
                for xi, X_scaler in enumerate(X_scalers):
                    _X_scaled[:,xi:xi+1] = X_scaler.transform(_X[:,xi][:,None])
                
                f_mu, f_sigma2          = model.predict_f(_X_scaled)
                result_mu       += Y_scaler.inverse_transform(f_mu) * weights[i][:,None]
                
                # Backing variance out of a Pipeline is difficult and unnecessary
                try:
                    result_sigma2  += (Y_scaler.scale_**2 * f_sigma2) * weights[i][:,None]
                except:
                    result_sigma2 += f_sigma2 * np.nan
                
            return result_mu, result_sigma2
        
        # Avoid the parallelization overhead if chunk_size == len(X_new)
        if n_chunks > 1:
            generator = Parallel(return_as='generator', n_jobs=n_jobs)(
                delayed(_parallel_predict_f)(self, X_chunk, model_list, scaler_list) for X_chunk in X_chunked)
        
            results = list(tqdm(generator, total=len(X_chunked), desc="Predicting f(X)"))
        else:
            results = [_parallel_predict_f(self, X, model_list, scaler_list)]
        
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
        n_chunks = np.ceil(X.shape[0] / chunk_size)
        
        samples = self._predict_f_samples(X, n_chunks, n_jobs, num_samples)
        
        # If there is a variance model, add it to the mean model variance 
        if len(self.variance_model_list) > 0:
            
            # This should probably draw the full covariance matrix in real space, then sample that using numpy
            # But converting the covariance matrix throught the Pipeline is difficult
            var_samples = self._predict_f_samples(X, n_chunks, n_jobs, num_samples, analyze_variance=True)
            
            rng = np.random.default_rng()
            for i in range(len(samples)):
                samples[i] = samples[i] + rng.normal(loc=var_samples[i]*0.0, scale=var_samples[i])
        
        return samples
    
    def _predict_f_samples(self, X, n_chunks, n_jobs, num_samples, analyze_variance=False, full_cov=False):
        
        if analyze_variance:
            model_list = self.variance_model_list
            scaler_list = self.Y_variance_scaler_list
        else:
            model_list = self.model_list
            scaler_list = self.Y_scaler_list
            
        # Chunk the data
        X_chunked = np.array_split(X, n_chunks, axis=0)
        
        # Define an internal function to interact with gpflow
        def _internal_predict_f_samples(EnsembleGPR, _X, model_list, scaler_list):
            result_samples = np.full((num_samples, _X.shape[0], 1), 0, dtype='float64')  
            
            weights = EnsembleGPR._getWeights(_X)
            
            for i in range(len(model_list)):
                model = model_list[i]
                X_scalers = EnsembleGPR.X_scaler_list[i]
                Y_scaler = scaler_list[i]
                
                # Get X, even if multiple scalers
                _X_scaled = np.zeros(_X.shape)
                for xi, X_scaler in enumerate(X_scalers):
                    _X_scaled[:,xi:xi+1] = X_scaler.transform(_X[:,xi][:,None])
                
                f_samples        = model.predict_f_samples(_X_scaled, num_samples, full_cov=full_cov)
                for j in range(num_samples):
                    result_samples[j:j+1,:,:] += Y_scaler.inverse_transform(f_samples[j]) * weights[i][:,None]
                
            return result_samples
        
        # Avoid the parallelization overhead if chunk_size == len(X_new)
        if n_chunks > 1:
            generator = Parallel(return_as='generator', n_jobs=n_jobs)(
                delayed(_internal_predict_f_samples)(self, X_chunk, model_list, scaler_list) for X_chunk in X_chunked)
        
            results = list(tqdm(generator, total=len(X_chunked), desc="Predicting f(X)"))
        else:
            results = [_internal_predict_f_samples(self, X, model_list, scaler_list)]
        
        # Return
        results = np.concatenate(results, axis=1)
        
        return results
    
    def predict_y(self, scaled_X=None, unscaled_X=None, chunk_size=None, cpu_fraction=None):
        
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
        X_chunked = np.array_split(X, n_chunks, axis=0)
        # W_chunked = np.array_split(weights, n_chunks, axis=0)
        
        # Define an internal function to interact with gpflow
        def _predict_y(EnsembleGPR, _X):
            result_mu       = np.full((_X.shape[0],1), 0, dtype='float64')
            result_sigma2   = np.full((_X.shape[0],1), 0, dtype='float64')   
            
            weights = EnsembleGPR._getWeights(_X)
            
            for i in range(len(EnsembleGPR.model_list)):
                model = EnsembleGPR.model_list[i]
                X_scalers = EnsembleGPR.X_scaler_list[i]
                Y_scaler = EnsembleGPR.Y_scaler_list[i]
                
                # Get X, even if multiple scalers
                # _X_scaled = X_scalers.transform(_X)
                _X_scaled = np.zeros(_X.shape)
                for xi, X_scaler in enumerate(X_scalers):
                    _X_scaled[:,xi:xi+1] = X_scaler.transform(_X[:,xi][:,None])
                
                y_mu, y_sigma2          = model.predict_y(_X_scaled)
                result_mu       += Y_scaler.inverse_transform(y_mu) * weights[i][:,None]
                result_sigma2  += (Y_scaler.scale_**2 * y_sigma2) * weights[i][:,None]
                
            return result_mu, result_sigma2
        
        # Avoid the parallelization overhead if chunk_size == len(X_new)
        if n_chunks > 1:
            generator = Parallel(return_as='generator', n_jobs=n_jobs)(
                delayed(_predict_y)(self, X_chunk) for X_chunk in X_chunked)
        
            results = list(tqdm(generator, total=len(X_chunked), desc="Predicting f(X)"))
        else:
            results = [_predict_y(self, X)]
        
        # Return mu, variance (sigma^2), and weights as lists of arrays
        results_mu = np.concatenate([r[0] for r in results], axis=0)
        results_sigma2 = np.concatenate([r[1] for r in results], axis=0)
        
        return results_mu, results_sigma2
    
    def _getWeights(self, unscaled_X):
        from scipy.spatial.distance import cdist
        from scipy.special          import softmax
        
        dist_cutoff = 100
        softmax_scale = 10
        
        # Calculate distance from all models
        dists = []
        for X_scalers, model in zip(self.X_scaler_list, self.model_list):
            
            # Scaled the input X
            X_scaled = np.zeros(unscaled_X.shape)
            for xi, X_scaler in enumerate(X_scalers):
                X_scaled[:,xi:xi+1] = X_scaler.transform(unscaled_X[:,xi][:,None])
            # X_scaled = X_scalers.transform(unscaled_X)
            
            dist_matrix = cdist(model.data[0], X_scaled)
            dist_matrix[dist_matrix > dist_cutoff] = dist_cutoff
            min_dists = dist_matrix.min(axis=0)
            
            dists.append(min_dists)
            
        dists = np.array(dists)
        weights = softmax(softmax_scale * (dist_cutoff - dists), axis=0)
        
        
        return weights
    
            
    def print_summary(self):
        import gpflow
        # for model in self.model_list:
        #     gpflow.utilities.print_summary(model, 'simple')
            
        df = pd.DataFrame()
        for i, model in enumerate(self.model_list):
            d = gpflow.utilities.parameter_dict(model)
            
            for key, value in d.items():
                df.loc[i, key] = value.numpy()
        
        print(df)
        
        return df


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

from check_shapes import inherit_check_shapes


class FixedVarianceOfMean(gpflow.functions.Function):
    def __init__(self, Y_var: gpflow.base.AnyNDArray):
        super().__init__
        self.var = Y_var

    @inherit_check_shapes
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.var
    
    #@inherit_check_shapes
    def variance_at(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.var
