#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:09:53 2025

@author: mrutala
"""
import astropy.units as u
from astropy.time import Time
import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import copy
import tensorflow as tf
import pickle
import tensorflow_probability  as     tfp

import sys
sys.path.append('/Users/mrutala/projects/HUXt/code/')
sys.path.append('/Users/mrutala/projects/ASWEstimator/code/')
import huxt as H
# import huxt_analysis as HA
import huxt_inputs as Hin
# import huxt_atObserver as hao
import multihuxt_readers as mr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer


import gpflow
import GPFlowEnsemble as gpflowf

# import huxt_inputs_wsa as Hin_wsa
import queryDONKI

try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

"""
Notes:
    - self.solar_wind is a misnomer. This is actually in-situ 
    solar wind data, not background (e.g. non-Transient) data specifically
    
Overview: 
    multihuxt_inputs keeps track of the:
        (in-situ) data:
        transients:
        background:
        
"""
    
# %%

class ASWEstimator:
    def __init__(self, start, stop, 
                 rmax=1, latmax=10):
        self.start = start
        self.stop = stop
        self.radmax = rmax * u.AU
        self.latmax = latmax * u.deg
        self.innerbound= 215 * u.solRad # 21.5 * u.solRad
        
        self.usw_minimum = 200 * u.km/u.s
        self.SiderealCarringtonRotation = 27.28 * u.day
        self.SynodicCarringtonRotation = 25.38 * u.day
        
        # These keywords can only be set AFTER object initialization
        
        # ICME parameters
        self._icme_duration = 4.0 * u.day # conservative duration (Richardson & Cane 2010)
        self._icme_duration_buffer = 1.0 * u.day # conservative buffer (Richardson & Cane 2010)
        self._icme_interp_buffer = 1.0 * u.day
        
        # Required initializations
        # Other methods check that these are None (or have value) before 
        # continuing, so they must be intialized here
        self._availableSources = None
        self._boundarySources = None
        self._ephemeris = {}
        
        # Input data initialization
        cols = ['t_mu', 't_sig', 'lon_mu', 'lon_sig', 'lat_mu', 'lat_sig',
                'width_mu', 'width_sig', 'speed_mu', 'speed_sig', 
                'thickness_mu', 'thickness_sig', 'innerbound']
        self.cmeDistribution = pd.DataFrame(columns = cols)
        
        
        
        return
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 
    # ----------------------------------------------------------------------
    def copy(self):
        return copy.deepcopy(self)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Ease-of-use time properties
    # ----------------------------------------------------------------------
    @property
    def starttime(self):
        return Time(self.start)
    
    @property
    def stoptime(self):
        return Time(self.stop)
    
    @property
    def simpadding(self):
        n = np.ceil((self.radmax / self.usw_minimum).to(u.day) / (27*u.day))
        return (n * 27 * u.day, 27 * u.day)
    
    @property 
    def simstart(self):
        return self.start - datetime.timedelta(days=self.simpadding[0].to(u.day).value)
    
    @property 
    def simstop(self):
        return self.stop + datetime.timedelta(days=self.simpadding[1].to(u.day).value)
    
    @property
    def simstarttime(self):
        return self.starttime - self.simpadding[0]
    
    @property
    def simstoptime(self):
        return self.stoptime + self.simpadding[1]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Methods relating to managing data sources
    # ----------------------------------------------------------------------
    @property
    def supported_sources(self):
        supported_sources = [
            'omni', 'parker solar probe', 'stereo a', 'stereo b', 'ulysses', 
            'voyager 1', 'voyager 2'
            ]
        return supported_sources
    @property
    def availableSources(self):
        if self._availableSources is None:
            availableSources = set(self.solar_wind.columns.get_level_values(0))
            availableSources = set(availableSources) - {'mjd'}
            self._availableSources = sorted(availableSources)
        return self._availableSources
    
    @availableSources.setter
    def availableSources(self, addedSources):
        self._availableSources.extend(addedSources)
        self._availableSources = sorted(self._availableSources)
        
    @property
    def boundarySources(self):
        if self._boundarySources is None:
            self._boundarySources = ['omni', 'stereo a', 'stereo b']
        return self._boundarySources
    
    @boundarySources.setter
    def boundarySources(self, boundarySources):
        self._boundarySources = boundarySources
        
    # def _identify_source(self, source):  
    #     source_aliases = {'omni': ['omni'],
    #                       'parker solar probe': ['parkersolarprobe', 'psp', 'parker solar probe'],
    #                       'stereo a': ['stereoa', 'stereo a', 'sta'],
    #                       'stereo b': ['stereob', 'stereo b', 'stb'],
    #                       # 'helios1': ['helios1', 'helios 1'],
    #                       # 'helios2': ['helios2', 'helios 2'],
    #                       'ulysses': ['ulysses', 'uy'],
    #                       # 'maven': ['maven'],
    #                       'voyager 1': ['voyager1', 'voyager 1'],
    #                       'voyager 2': ['voyager2', 'voyager 2']}
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Methods relating to (in-situ, unprocessed) data
    # ----------------------------------------------------------------------
    def getSolarWind(self, sources=None):
        
        # Check if sources are specified; if not, use them all
        if sources is None:
            sources = self.supported_sources
        else:
            breakpoint()
            #!!!! Add functionality to search alias dictionary
        
        # Read each source into a dictionary
        available_sources = []
        available_data_dict = {}
        for source in sources:
            print(source)
            print('----------------------------')
            data_df = mr.SolarWindData(source, self.simstart, self.simstop).data
            if not data_df.isna().all().all(): 
                available_sources.append(source)
                available_data_dict[source] = data_df
                
        available_data_df = pd.concat(available_data_dict, axis='columns')
        available_data_df['mjd'] = Time(available_data_df.index).mjd
        
        self.solar_wind = available_data_df
        
        return
    
    def filterSolarWind(self):
        
        sources_to_remove = []
        for source in self.availableSources:
            
            # Where is the source out of radial and latitudinal range?
            out_of_range = (np.abs(self.solar_wind[(source, 'lat_HGI')]) > np.abs(self.latmax)) &\
                           (self.solar_wind[(source, 'rad_HGI')] > self.radmax)
            
            # Set these as NaNs
            self.solar_wind.loc[out_of_range, source] = np.nan
            
            # If no data is in range, delete the source and columns entirely
            if out_of_range.all() == True:
                sources_to_remove.append(source)
                self.solar_wind.drop(columns = source, level = 0, inplace = True)
                          
        # for source in sources_to_remove:
        #     self.availableSources.remove(source)
            
        return
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Handling transients
    # ----------------------------------------------------------------------
    def getTransients(self, sources=None):
        
        location_aliases = {'omni': 'Earth',
                            'stereo a': 'STEREO%20A',
                            'stereo b': 'STEREO%20B',
                            'maven': 'Mars'}
        
        all_sources = list(location_aliases.keys())
        
        # Parse which sources to lookup transients for
        if sources is None:
            # Either use the sources we have data for, or all of them
            if len(self.availableSources) > 0:
                sources = list(set(all_sources).intersection(set(self.availableSources)))  
            else:
                sources = all_sources
        else:
            breakpoint()
            #!!!! Add functionality to search alias dictionary
        
        # Lookup ICMEs for each source
        availableTransientData_list = []
        for source in sources:
            location = location_aliases[source]
            icmes = queryDONKI.ICME(self.simstart, 
                                    self.simstop, 
                                    location = location, 
                                    duration = self._icme_duration,
                                    ensureCME = True) 

            icmes['affiliated_source'] = source
            
            availableTransientData_list.append(icmes)
        
        availableTransientData_list = [df for df in availableTransientData_list if not df.empty]
        availableTransientData_df = pd.concat(availableTransientData_list, axis='rows')
        availableTransientData_df.reset_index(inplace=True, drop=True)
        if len(availableTransientData_df) > 0:
            availableTransientData_df['mjd'] = Time(availableTransientData_df['eventTime'])

        self.transients = availableTransientData_df 
        
        # Add ICMEs to background data
        self.set_ICMEs()
        
        return
    
    def set_ICMEs(self, icme_df = None):
        
        # Default to the icme_df attribute
        if icme_df is None:
            icme_df = self.transients
            
        # Drop ICME columns already assigned to self.solar_wind
        if 'ICME' in self.solar_wind.columns.get_level_values(1):
            self.solar_wind.drop('ICME', axis=1, level=1, inplace=True)
        
        for source in self.availableSources:
            
            # Format insitu data for HUXt's remove_ICMEs function
            insitu = self.solar_wind[source].copy()
            insitu.loc[:, 'mjd'] = self.solar_wind.loc[:, 'mjd']
            
            # Format ICME data for HUXt's remove_ICMEs function
            icmes = icme_df.query('affiliated_source == @source')
            icmes.reset_index(inplace=True, drop=True)
            if 'eventTime' in icmes.columns: 
                icmes = icmes.rename(columns = {'eventTime': 'Shock_time'})
                icmes['ICME_end'] = [row['Shock_time'] + datetime.timedelta(days=(row['duration'])) 
                                     for _, row in icmes.iterrows()]
            
            # Interpolate over existing data gaps (NaNs), so they aren't caught as ICMEs
            insitu.interpolate(method='linear', axis='columns', limit_direction='both', inplace=True)
            
            # Extract the timesteps during which there is an ICME
            if len(icmes) > 0:
                insitu_noicme = Hin.remove_ICMEs(insitu, icmes, 
                                                 params=['U'], 
                                                 interpolate = False, 
                                                 icme_buffer = self._icme_duration_buffer, 
                                                 interp_buffer = self._icme_interp_buffer, 
                                                 fill_vals = np.nan)
                
                icme_series = insitu_noicme['U'].isna().to_numpy()
                
            else:
                insitu_noicme = insitu
                
                icme_series = [None] * len(insitu)
                
            # Add ICME indices to background data
            idx = self.solar_wind.columns.get_loc((source, insitu.columns[-2]))
            self.solar_wind.insert(idx+1, (source, 'ICME'), icme_series)
                          
        return insitu_noicme['U'].isna()
    
    @property
    def ephemeris(self):
        from astropy.time import Time
        # If this hasn't been run before, run for all 
        if len(self._ephemeris) == 0:
            print("No ephemeris loaded. Now generating...")
            for source in self.availableSources:
                eph = H.Observer(source, Time(self.solar_wind.index))
                self._ephemeris[source] = eph
                    
        return self._ephemeris
    
    
    def get_carringtonPeriod(self, distance):
                   
        # source speed, approximated as circular
        kepler_const = ((1 * u.year).to(u.day))/((1 * u.au)**(3/2))
        source_period = kepler_const * distance**(3/2)
        source_speed = (2 * np.pi * u.rad) / (source_period.to(u.day))
        
        # sun speed
        sun_speed = (2 * np.pi * u.rad)/(25.38 * u.day)
        
        synodic_period = 1/(sun_speed - source_speed) * (2 * np.pi * u.rad)
        
        return synodic_period
    
    # =============================================================================
    #     
    # =============================================================================
    def makeBackgroundDistribution(self,
                                   # inducing_variable=True,
                                   GP = False, interpolate = False,
                                   # target_noise = 1e-2,
                                   # max_chunk_length = 1024,
                                   n_samples = 1):
        target_variables = ['U']
        
        # summary holds summary statistics (mean, standard deviation)
        all_summary = {}
        # samples holds individual samples drawn from the full covariance
        all_scalers = {}
        all_models = {}
        # Set up dictionaries to hold results
        
        # 
        for source in self.boundarySources:
            
            # Get a copy of the insitu data
            insitu_df = self.solar_wind.loc[:, source].copy()
            insitu_df['mjd'] = self.solar_wind.loc[:, 'mjd']
            
            # Set all ICME rows to NaNs
            data_columns = list(set(insitu_df.columns) - set(['ICME', 'mjd']))
            insitu_df.loc[insitu_df['ICME'], data_columns] = np.nan
            
            # Send the data to the correct parser
            if GP is True:
                # self._backgroundDistributionMethod = 'GP'
                
                carrington_period = self.get_carringtonPeriod(self.ephemeris[source].r.mean())
                
                summary, models = self._imputeBackgroundDistribution(
                    insitu_df, carrington_period, target_variables=target_variables)
                
                # all_scalers.update({source: scalers})
                all_models.update({source: models})
                
            elif interpolate is True:
                # self._backgroundDistributionMethod = 'extend'
                
                summary = self._extendBackgroundDistributions(
                    insitu_df, target_variables=target_variables)
                
                all_scalers.update({})
                all_models.update({})
                
            else:
                print("Cannot have extend=str and GP=True!")
                breakpoint()
            
            all_summary.update({source: summary})
        
        # Convert all_summary into a df for return
        
        self.backgroundDistributions = pd.concat(all_summary, axis=1)
        self.backgroundDistributions['mjd'] = self.solar_wind['mjd']
        
        # Assign scalers and models to attributes
        self._backgroundScalers = all_scalers
        self._backgroundModels = all_models
        
        # For convenience, draw samples here
        self.sampleBackgroundDistributions(n_samples=n_samples)
        
        return 
    
    def sampleBackgroundDistributions(self, n_samples=1, chunk_size=2000, cpu_fraction=0.75):
        
        df = self.backgroundDistributions.copy()
        samples = [self.backgroundDistributions.copy() for _ in range(n_samples)]
        
        if len(self._backgroundModels.keys()) == 0:
            # Background is linearly interpolated, without uncertainty
            # All samples are identical
            for i in range(n_samples):
                samples[i] = samples[i].rename(columns={'U_mu': 'U', 'Br_mu': 'Br'})
                if 'U_sigma' in samples[i].columns.get_level_values(1):
                    samples[i] = samples[i].drop(columns='U_sigma', level=1)
                if 'Br_sigma' in samples[i].columns.get_level_values(1):
                    samples[i] = samples[i].drop(columns='Br_sigma', level=1)

        else:
            # Background is found with 1D Gaussian Process regression
            for source in self._backgroundModels.keys():
                
                # Scale MJD
                # X_scaler_list = self._backgroundScalers[source]['mjd']
                # X = [scaler.transform(df['mjd'].to_numpy()[:,None]) for scaler in X_scaler_list]
                mjd = df['mjd'].to_numpy()[:,None]
                
                for var in self._backgroundModels[source].keys():
                    
                    # Y_scaler_list = self._backgroundScalers[source][var]
                    
                    # Draw samples
                    results = self._backgroundModels[source][var].predict_f_samples(
                        unscaled_X=mjd, num_samples=n_samples, 
                        chunk_size=chunk_size, cpu_fraction=cpu_fraction)
                    
                    for i in range(n_samples):
                        samples[i][(source, var)] = results[i]
                        samples[i] = samples[i].drop(columns=[(source, var+'_mu'), (source, var+'_sigma')])
                    
        self.backgroundSamples = samples
        
        return         
    
    def _extendBackgroundDistributions(self, input_df,
                                        target_variables = ['U'],
                                        # noise_constant = 0.0,
                                        # n_samples = 0
                                        ):
        
        
        # Use df, which already has NaNs where ICMEs are present
        
        df = input_df.copy()
        
        # Simulate HUXt ICME removal:
        # Define a window twice as wide as the interp buffer, then truncate the
        # rolling window where the ICME is
        # The last and first values surrounding the ICME are thus a window-length mean
        # Then interpolate these, and fill back in for the original df, only where the ICME is present
        window = datetime.timedelta(days=2*self._icme_interp_buffer.to(u.day).value)
        test = df[target_variables].rolling(window, center=True).mean()
        test[df['ICME']] = np.nan
        
        smooth_interp = test.interpolate('linear', limit_direction='both')
        
        for var in target_variables:
            df.loc[:, var+'_mu'] = df.loc[:, var]
            df.loc[df['ICME'], var+'_mu'] = smooth_interp.loc[df['ICME'], var]
            df.loc[:, var+'_sigma'] = 0.0
            
            df.drop(columns=var, inplace=True)

        return df
    
    def _imputeBackgroundDistribution(self, df, carrington_period,
                                        target_variables = ['U'],
                                        #target_noise = 1e-2,
                                        #max_chunk_length = 1024
                                        ):
        
        # Physically motivated data chunking
        # Do this before fitting so each chunk may have an appropriate mean function
        df_chunks = self._getChunksInTime(df, delta=90 * u.day)
        
        # Initialize objects to hold results from looping over target_variables
        bgDistribution_df = pd.DataFrame(index=df.index)
        bgScalers = {}
        bgGPModels = {}
        
        for target_var in target_variables:
            bgScalers[target_var] = {'mjd': [], 'val': []}
            bgGPModels[target_var] = {}
            
            X_list, Y_list, k_list = [], [], []
            for df_chunk in df_chunks:
                
                # Map MJD onto the interval [0,10]
                time_scaler = MinMaxScaler(feature_range=(0,10))
                time_scaler.fit(df_chunk['mjd'].to_numpy()[:,None])
                
                X_all = time_scaler.transform(df_chunk['mjd'].to_numpy()[:,None])
                
                bgScalers[target_var]['mjd'].append(time_scaler)
                # X_scaler_list.append(time_scaler)
                
                # Map the target variable onto a centered normal distribution
                val_scaler = StandardScaler()
                val_scaler.fit(df_chunk[target_var].to_numpy()[:,None])
                
                # bgScalers.update({target_var: val_scaler})
                bgScalers[target_var]['val'].append(val_scaler)
                # Y_scaler_list.append(val_scaler)
                
                Y_all = val_scaler.transform(df_chunk[target_var].to_numpy()[:,None])
                
                # Remove NaNs in Y from both X & Y
                valid_index = ~df_chunk[target_var].isna().to_numpy()
                X_train = X_all[valid_index,:]
                Y_train = Y_all[valid_index,:]
                
                # =================================================================
                # Define kernel for each dimension separately, then altogether
                # =================================================================
                period_rescaled = np.float64(carrington_period.to(u.day).value * time_scaler.scale_[0])
                period_gp = gpflow.Parameter(period_rescaled, trainable=False)
                
                # Only predict 1 Carrington Rotation forward
                min_x = np.float64(0)
                mid_x = period_rescaled
                max_x = np.float64(10) # 4*period_rescaled
                
                lengthscale_gp = gpflow.Parameter(mid_x, 
                    transform = tfp.bijectors.SoftClip(min_x, max_x))
                
                base_kernel = gpflow.kernels.RationalQuadratic(lengthscales = lengthscale_gp)
                amplitude_kernel = gpflow.kernels.SquaredExponential(lengthscales = lengthscale_gp)
                period_kernel = gpflow.kernels.Periodic(
                    gpflow.kernels.SquaredExponential(lengthscales=period_gp),
                    period=period_gp)
                
                kernel = base_kernel + amplitude_kernel * period_kernel
                # kernel = base_kernel + period_kernel
                
                # =============================================================================
                # ~Fancy~ Chunking  
                # =============================================================================
                # Xc, Yc, optimized_noise = self._optimize_clustering(X, Y, target_noise_variance=target_noise)
                # XYc = np.column_stack([Xc, Yc])
                
                # n_chunks = int(np.ceil(len(Xc)/max_chunk_length))
                
                # sort = np.argsort(XYc[:,0]) # sort by MJD
                # XYc_chunks = np.array_split(XYc[sort,:], n_chunks)
                # Xc_chunks = [chunk[:,0][:,None] for chunk in XYc_chunks]
                # Yc_chunks = [chunk[:,1][:,None] for chunk in XYc_chunks]
                
                #!!!!!!! Change this subsampling
                X_list.append(X_train)
                Y_list.append(Y_train)
                k_list.append(kernel)
              
            # =============================================================================
            # Plug into the ensemble GP model
            # =============================================================================
            model = gpflowf.EnsembleGPR(X_list, Y_list, k_list, bgScalers[target_var]['mjd'], bgScalers[target_var]['val'])
            model.optimize()
            
            # model = gpflowf.GPFlowEnsemble(kernel, X_list, Y_list, noise_variance=0.05) # optimized_noise)
            bgGPModels[target_var] = model
            
            # =================================================================
            # Get predictions for all MJD (filling in gaps)
            # and inverse transform
            # =================================================================
            
            # Xo_list = [scaler.transform(df['mjd'].to_numpy()[:,None]) for scaler in bgScalers['mjd']]
            X = df['mjd'].to_numpy()[:,None]
            
            # These custom wrappers return data-scaled Y given data-scaled X
            fo_mu, fo_sigma2 = model.predict_f(unscaled_X=X, cpu_fraction=0.75, chunk_size=2000)
            # fo_samples = model.predict_f_samples(unscaled_X=X, cpu_fraction=0.75, chunk_size=2000, num_samples=100)
            
            bgDistribution_df['mjd'] = df['mjd'].to_numpy()
            bgDistribution_df[target_var+'_mu'] = fo_mu.mean(axis=1)
            bgDistribution_df[target_var+'_sigma'] = np.sqrt(fo_sigma2.mean(axis=1))
        
        # Cast res and samples into full dfs
        bgDistribution_full_df = df.copy(deep=True)
        bgDistribution_full_df.drop(columns=target_variables, inplace=True)
        for target_var in target_variables:
            bgDistribution_full_df[target_var+'_mu'] = bgDistribution_df[target_var+'_mu']
            bgDistribution_full_df[target_var+'_sigma'] = bgDistribution_df[target_var+'_sigma']
        
        return bgDistribution_full_df, bgGPModels
    
    def generate_boundaryDistributions(self, constant_percent_error=0.0):
        from tqdm import tqdm
        # from dask.distributed import Client, as_completed, LocalCluster
        import multiprocessing as mp
        import logging
        from joblib import Parallel, delayed
        from tqdm import tqdm
        
        nCores = int(0.75 * mp.cpu_count()) 

        rng = np.random.default_rng()
        
        # methodOptions = ['forward', 'back', 'both']
        methodOptions = ['both']
        
        boundaryDistributions_d = {}
        boundarySamples_d = {}
        for source in self.boundarySources:
        
            # Format the insitu df (backgroundDistribution) as HUXt expects it
            insitu_df = self.backgroundDistributions[source].copy(deep=True)
            insitu_df['BX_GSE'] =  -insitu_df['Br_mu']
            insitu_df['V'] = insitu_df['U_mu']
            insitu_df['datetime'] = insitu_df.index
            insitu_df = insitu_df.reset_index()
        
            # Map inwards once to get the appropriate dimensions, etc.
            # t, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(self.simstart, self.simstop, omni_input=insitu_df)
            t, vcarr, bcarr = Hin.generate_vCarr_from_insitu(self.simstart, self.simstop, 
                                                             insitu_source=source, insitu_input=insitu_df)
            
            # Sample the velocity distribution and assign random mapping directions (method)
            # Randomly assigning these is equivalent to performing each mapping for each sample (for large numbers of samples)
            # Having a single random population should be better mathematically
            
            dfSamples = [df[source] for df in self.backgroundSamples]
            methodSamples = rng.choice(methodOptions, len(dfSamples))
            
            func = _map_vBoundaryInwards
            funcGenerator = Parallel(return_as='generator', n_jobs=nCores)(
                delayed(func)(self.simstart, self.simstop, source, df_sample, method_sample, self.ephemeris[source], self.innerbound)
                for df_sample, method_sample in zip(dfSamples, methodSamples))
            
            result_tuples = list(tqdm(funcGenerator, total=len(dfSamples)))
            
            vcarr_results = [result_tuple[0] for result_tuple in result_tuples]
            bcarr_results = [result_tuple[1] for result_tuple in result_tuples]
            
            # Characterize the resulting samples as one distribution
            vcarr_mu = np.nanmean(vcarr_results, axis=0)
            vcarr_sig = np.sqrt(np.nanstd(vcarr_results, axis=0)**2 + (vcarr_mu * constant_percent_error)**2)
            
            bcarr_mu = np.nanmean(bcarr_results, axis=0)
            bcarr_sig = np.sqrt(np.nanstd(bcarr_results, axis=0)**2 + (bcarr_mu * constant_percent_error)**2)
            
            # Get the left edges of longitude bins
            lons = np.linspace(0, 360, vcarr_mu.shape[0]+1)[:-1]
            
            boundaryDistributions_d[source] = {'t_grid': t,
                                               'lon_grid': lons, 
                                               'U_mu_grid': vcarr_mu,
                                               'U_sigma_grid': vcarr_sig,
                                               'Br_mu_grid': bcarr_mu,
                                               'Br_sigma_grid': bcarr_sig}
            
            # For completeness, add boundarySamples here
            boundarySamples_d[source] = []
            for result_tuple in result_tuples:
                boundarySamples_d[source].append({'t_grid': t,
                                                  'lon_grid': lons, 
                                                  'U_grid': result_tuple[0],
                                                  'B_grid': result_tuple[1]})
        
        self.boundaryDistributions = boundaryDistributions_d
        self.boundarySamples = boundarySamples_d
        
        # # =============================================================================
        # # Visualization 
        # # =============================================================================
        # fig, axs = plt.subplots(figsize=(6,4.5), ncols=2)
        
        # mu_img = axs[0].imshow(vcarr_mu, 
        #                        extent=[self.simstarttime.mjd, self.simstoptime.mjd, 0, 360], 
        #                        origin='lower', aspect=0.2)
        # axs[0].set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        # fig.colorbar(mu_img, ax=axs[0])
        
        # sig_img = axs[1].imshow(vcarr_sig, 
        #                         extent=[self.simstarttime.mjd, self.simstoptime.mjd, 0, 360], 
        #                         origin='lower', aspect=0.2)
        # axs[1].set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        # fig.colorbar(sig_img, ax=axs[1])
        
        # axs[0].set(ylabel='Heliolongitude [deg.]', xlabel='Date [MJD]')
        # axs[1].set(xlabel='Date [MJD]')
        
        # plt.show()
        
        return
    
    def generate_boundaryDistribution3D(self, nLat=16, extend=None, GP=True, 
                                        num_samples=0, 
                                        
                                        **kwargs):
                                        # max_chunk_length=1024,
                                        # target_reduction = None, target_noise = None,
                                        # SGPR=0.1):
        
        # Get dimensions from OMNI boundary distribution, which *must* exist
        nLon, nTime = self.boundaryDistributions['omni']['U_mu_grid'].shape
        
        # Coordinates = (lat, lon, time)
        # Values = boundary speed, magnetic field* (*not implemented fully)
        lat_for3d = np.linspace(-self.latmax.value, self.latmax.value, nLat)
        lon_for3d = np.linspace(0, 360, nLon+1)[:-1]
        mjd_for3d = self.boundaryDistributions['omni']['t_grid']
        
        if (type(extend) == str) & (GP == True):
            print("Cannot have extend=str and GP=True!")
            return
        if type(extend) == str:
            summary = self._extend_boundaryDistributions(nLat, extend)
            
            self._assign_boundaryDistributions3D(
                mjd_for3d, lon_for3d, lat_for3d,
                summary['U_mu'], summary['U_sigma'], summary['Br_mu'], summary['Br_sigma'])
            self._boundaryScalers = {}
            self._boundaryModels = {}
            
        elif GP is True:
            summary, scalers, models = self._impute_boundaryDistributions(
                lat_for3d, lon_for3d, mjd_for3d, num_samples=num_samples, **kwargs)
            
            self._assign_boundaryDistributions3D(
                mjd_for3d, lon_for3d, lat_for3d,
                summary['U_mu'], summary['U_sigma'], summary['Br_mu'], summary['Br_sigma'])
            self._boundaryScalers = scalers
            self._boundaryModels = models
            
        return
    
    def _assign_boundaryDistributions3D(self, t_grid, lon_grid, lat_grid, U_mu_grid, U_sig_grid, Br_mu_grid, Br_sig_grid):
        """
        This method is independent of generate_boundaryDistributions3D to allow
        assignment to attribute within the _extend and _impute methods, and 
        thus to allow easier testing

        Parameters
        ----------
        t_grid : TYPE
            DESCRIPTION.
        lon_grid : TYPE
            DESCRIPTION.
        lat_grid : TYPE
            DESCRIPTION.
        U_mu_grid : TYPE
            DESCRIPTION.
        U_sig_grid : TYPE
            DESCRIPTION.
        B_grid : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.boundaryDistributions3D = {'t_grid': t_grid,
                                        'lon_grid': lon_grid,
                                        'lat_grid': lat_grid,
                                        'U_mu_grid': U_mu_grid,
                                        'U_sigma_grid': U_sig_grid,
                                        'Br_mu_grid': Br_mu_grid,
                                        'Br_sigma_grid': Br_sig_grid,
                                        }
        
    def _extend_boundaryDistributions(self, nLat, name):
        
        U_mu_3d = np.tile(self.boundaryDistributions[name]['U_mu_grid'], 
                          (nLat, 1, 1))
        U_sigma_3d = np.tile(self.boundaryDistributions[name]['U_sigma_grid'], 
                          (nLat, 1, 1))
        # B_3d = np.tile(self.boundaryDistributions[name]['B_grid'], 
        #                   (nLat, 1, 1))
        Br_mu_3d = np.tile(self.boundaryDistributions[name]['Br_mu_grid'], 
                          (nLat, 1, 1))
        Br_sigma_3d = np.tile(self.boundaryDistributions[name]['Br_sigma_grid'], 
                          (nLat, 1, 1))
        
        summaries = {'U_mu': U_mu_3d, 
                     'U_sigma': U_sigma_3d,
                     'Br_mu': Br_mu_3d,
                     'Br_sigma': Br_sigma_3d}
        return summaries
    
    def _interpolate_boundaryDistributions(self, lat_for3d, lon_for3d, mjd_for3d):
        
        breakpoint()
        
        return
        
    def _impute_boundaryDistributions(self, lat_for3d, lon_for3d, mjd_for3d,
                                      maximum_span = 60*u.day, 
                                      **kwargs):
        import gpflow
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
        from sklearn.pipeline import Pipeline
        # from scipy.cluster.vq import kmeans
        from sklearn.cluster import KMeans
        import multiprocessing as mp
        from joblib import Parallel, delayed
        from sklearn.cluster import MiniBatchKMeans
        
        # Get dimensions from OMNI boundary distribution, which *must* exist
        nLat = len(lat_for3d)
        nLon = len(lon_for3d)
        nMjd = len(mjd_for3d)
        
        all_summaries = {}
        all_samples = {}
        all_scalers = {}
        all_models = {}
        
        # Setup normalizations ahead of time
        # Normalizations are error-normalized to prevent issues in matrix decomposition
        lat_scaler = StandardScaler() # MinMaxScaler((-1,1))
        lat_scaler.fit(lat_for3d[:,None])
        
        lon_scaler = StandardScaler() # MinMaxScaler((-1,1))
        lon_scaler.fit(lon_for3d[:,None])
        
        mjd_scaler = StandardScaler() # MinMaxScaler((-1,1))
        mjd_scaler.fit(mjd_for3d[:,None])
        
        # Assign these dependent variables to all_scalers
        all_scalers.update({'lat_grid': lat_scaler, 'lon_grid': lon_scaler, 't_grid': mjd_scaler})
        
        # Extract variables and fit dependent 
        for target_var in ['U', 'Br']:
            
            # Initialize value scalers for mean (mu) and standard deviation (sigma)
            val_mu_scaler = StandardScaler()

            val_sigma_scaler = Pipeline([
                ('log_transform', FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False)),
                ('scaler', StandardScaler()),
                ])
            
            #
            lat, lon, mjd, val_mu, val_sigma, = [], [], [], [], []
            val_mu_noise_variance = []
            val_sigma_noise_variance = []
            for source in self.boundarySources:
                
                bound, noise_variance = self._rescale_2DBoundary(
                    self.boundaryDistributions[source],
                    target_reduction = kwargs.get('target_reduction'),
                    target_size = kwargs.get('target_size')
                    )
                
                val_mu_noise_variance.append(noise_variance[target_var+'_mu_grid'])
                val_sigma_noise_variance.append(noise_variance[target_var+'_sigma_grid'])
                
                lon_1d = bound['lon_grid']
                mjd_1d = bound['t_grid']
                lat_1d = np.interp(mjd_1d, 
                                   self.ephemeris[source].time.mjd, 
                                   self.ephemeris[source].lat_c.to(u.deg).value)
                
                mjd_2d, lon_2d, = np.meshgrid(mjd_1d, lon_1d)
                lat_2d, lon_2d, = np.meshgrid(lat_1d, lon_1d)
                
                lon_2d, mjd_2d = np.meshgrid(lon_1d, mjd_1d)
                lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
                
                val_mu_2d = bound[target_var+'_mu_grid'].T
                val_sigma_2d = bound[target_var+'_sigma_grid'].T
                
                # We're going to transpose all of these 2D matrices
                # So, when flattened, lon is the second (faster changing) dim
                # And mjd/lat is the first (slower changing) dim
                lat.extend(lat_2d.flatten())
                lon.extend(lon_2d.flatten())
                mjd.extend(mjd_2d.flatten())
                
                val_mu.extend(val_mu_2d.flatten())
                val_sigma.extend(val_sigma_2d.flatten())

            # Recast as arrays
            lon = np.array(lon)
            lat = np.array(lat)
            mjd = np.array(mjd)
            
            val_mu = np.array(val_mu)
            val_sigma = np.array(val_sigma)
            # log_val_sigma = np.log10(val_sigma)
            
            # Normalizations & NaN removal
            xlat = lat_scaler.transform(lat[~np.isnan(val_mu),None])
            xlon = lon_scaler.transform(lon[~np.isnan(val_mu),None])
            xmjd = mjd_scaler.transform(mjd[~np.isnan(val_mu),None])
            
            val_mu_scaler.fit(val_mu[:,None])
            yval_mu = val_mu_scaler.transform(val_mu[~np.isnan(val_mu),None])
            
            # val_sigma_scaler.fit(val_sigma[:,None])
            # yval_sigma = val_sigma_scaler.transform(val_sigma[~np.isnan(val_mu),None])
            
            yval_sigma_TEST = val_sigma[~np.isnan(val_mu),None] / val_mu_scaler.scale_
            # Avoid zeros in the sigma values
            sigma_badindx = yval_sigma_TEST <= 0
            yval_sigma_TEST[sigma_badindx] = yval_sigma_TEST[~sigma_badindx].min()
            
            all_scalers.update({target_var: val_mu_scaler})
            
            # %% ==================================================================
            # GP Kernel Definitions
            # =====================================================================
            
            # lat_scale_min = 0 / lat_scaler.scale_
            # lat_scale_mid = 1 / lat_scaler.scale_
            # lat_scale_max = 3 / lat_scaler.scale_
            # lat_lengthscale = gpflow.Parameter(lat_scale_mid, 
            #    transform = tfp.bijectors.SoftClip(lat_scale_min, lat_scale_max))
            # # lat_lengthscale = gpflow.Parameter(lat_scale_mid)
            
            # mjd_scale_min = np.float64(0.0)
            # mjd_scale_mid = 0.5 * 25.38 / mjd_scaler.scale_
            # mjd_scale_max = 1 * 25.38 / mjd_scaler.scale_
            # # if mjd_scale_mid > 0.9: mjd_scale_mid[0] = 0.9
            # # if mjd_scale_max > 1.0: mjd_scale_max[0] = 1.0
            # mjd_lengthscale = gpflow.Parameter(mjd_scale_mid, 
            #    transform = tfp.bijectors.SoftClip(mjd_scale_min, mjd_scale_max))
            # # mjd_lengthscale = gpflow.Parameter(mjd_scale_mid)
            
            # lon_scale_min = np.float64(0.0)
            # lon_scale_mid = 180 / lon_scaler.scale_
            # lon_scale_max = 360 / lon_scaler.scale_
            # # lon_lengthscale = gpflow.Parameter(lon_scale_mid, 
            # #    transform = tfp.bijectors.SoftClip(lon_scale_min, lon_scale_max))
            # lon_lengthscale = gpflow.Parameter(lon_scale_mid,
            #    transform = tfp.bijectors.SoftClip(lon_scale_min, lon_scale_max))
            
            # lat_kernel = gpflow.kernels.RationalQuadratic(active_dims=[0], lengthscales=lat_lengthscale)
            
            # period_gp = gpflow.Parameter(lon_scale_max, trainable=False)
            # base_kernel = gpflow.kernels.RationalQuadratic(active_dims=[1], lengthscales=lon_lengthscale)
            # amplitude_kernel = gpflow.kernels.RationalQuadratic(active_dims=[1], lengthscales=lon_lengthscale)
            # period_kernel = gpflow.kernels.Periodic(
            #     # gpflow.kernels.SquaredExponential(active_dims=[1], lengthscales=period_gp), 
            #     gpflow.kernels.SquaredExponential(active_dims=[1]), 
            #     period=period_gp)
            # lon_kernel = base_kernel + amplitude_kernel * period_kernel
                         
            # mjd_kernel = gpflow.kernels.RationalQuadratic(active_dims=[2], lengthscales=mjd_lengthscale)
            
            # factor_kernel = gpflow.kernels.RationalQuadratic(active_dims=[0,1,2])
            
            # all_kernel = gpflow.kernels.RationalQuadratic()
            # kernel_mu = (lat_kernel + lon_kernel + mjd_kernel + 
            #              # lat_kernel*lon_kernel + lat_kernel*mjd_kernel + lon_kernel*mjd_kernel +
            #              factor_kernel*lat_kernel*lon_kernel*mjd_kernel + 
            #              all_kernel)
            kernel_mu = gpflow.kernels.RationalQuadratic()
            
            kernel_sigma = copy.deepcopy(kernel_mu)
            # %% ==================================================================
            # Optimize Clustering & Cluster
            # =====================================================================
            X = np.column_stack([xlat, xlon, xmjd])
            Y_mu = yval_mu
            # Y_sigma = yval_sigma
            Y_sigma_TEST = yval_sigma_TEST
            
            # Xc_mu, Yc_mu, opt_noise_mu = self._optimize_clustering(X, Y_mu, 0.05)
            # # Xc_sigma, Yc_sigma, opt_noise_sigma = self._optimize_clustering(X, Y_sigma, 
            # #     target_reduction=target_reduction, target_noise=target_noise, inX=True)
            # Xc_sigma, Yc_sigma, opt_noise_sigma = self._optimize_clustering(X, Y_sigma, 0.05)
            # # XYc_mu = np.column_stack([Xc_mu, Yc_mu])
            # # XYc_sigma = np.column_stack([Xc_sigma, Yc_sigma])
            # # Generous estimate; in general, the downsampling does not introduce substantial noise
            # opt_noise_mu = 0.005
            # opt_noise_sigma = 0.005
            # # 3D Plot for testing
            # fig, ax = plt.subplots(figsize=[10,5], subplot_kw={'projection': '3d'})
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # ax.scatter(Xc_mu[:,2], Xc_mu[:,1], Xc_mu[:,0], c=Yc_mu[:,0], 
            #            alpha=0.5, marker='.', s=16, vmin=-2, vmax=2)
            # ax.set(xlabel = 'Time [arb.]', ylabel='Longitude [arb.]', zlabel = 'Latitude [arb.]')
            # ax.set_box_aspect([4, 1, 1])
            # ax.view_init(elev=30, azim=80)
            # plt.show()
            # fig, ax = plt.subplots(figsize=[10,5], subplot_kw={'projection': '3d'})
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # ax.scatter(X[:,2], X[:,1], X[:,0], c=Y_mu[:,0], 
            #            alpha=0.5, marker='.', s=16, vmin=-2, vmax=2)
            # ax.set(xlabel = 'Time [arb.]', ylabel='Longitude [arb.]', zlabel = 'Latitude [arb.]')
            # ax.set_box_aspect([4, 1, 1])
            # ax.view_init(elev=30, azim=80)
            # plt.show()
    
    
    
            # %% ==================================================================
            # Chunk Data for Processing
            # =====================================================================
            # Check the total duration of the input
            total_span = mjd.max() - mjd.min()
            n_chunks = np.ceil(total_span / maximum_span.to(u.day).value).astype(int)
            
            # Shuffle the two datasets such that time is monotonic
            XY = np.column_stack([X, Y_mu, Y_sigma_TEST])
            
            sorted_indx = XY[:,2].argsort(kind='stable')
            XY = XY[sorted_indx]
            
            XY_chunks = np.array_split(XY, n_chunks)
            
            X_chunks = [c[:,:X.shape[1]] for c in XY_chunks]
            Y_mu_chunks = [c[:,X.shape[1]:Y_mu.shape[1]] for c in XY_chunks]
            Y_sigma_chunks_TEST = [c[:,-1] for c in XY_chunks]

            # kwargs['max_chunk_length'] = 2000
            # Xc_mu_chunks, Yc_mu_chunks = self._optimize_chunking(Xc_mu, Yc_mu, **kwargs)
                  
            # Xc_sigma_chunks, Yc_sigma_chunks = self._optimize_chunking(Xc_sigma, Yc_sigma, **kwargs)
            
            
            # fig, axs = plt.subplots(ncols=len(Xc_mu_chunks), figsize=[10,5], subplot_kw={'projection': '3d'})
            # for ax, Xchunk, Ychunk in zip(axs, Xc_mu_chunks, Yc_mu_chunks):
            #     ax.scatter(Xchunk[:,2], Xchunk[:,1], Xchunk[:,0], c=Ychunk[:,0], 
            #                alpha=0.5, marker='.', s=36, vmin=-2, vmax=2)
            #     ax.set(xlabel='Time', ylabel='Longitude', zlabel='Latitude')
            #     ax.view_init(elev=30., azim=80)
            # plt.show()
            # breakpoint()
            
            # # !!!!! Try random sampling?
            # rng = np.random.default_rng()
            # rand_indx = rng.choice(Xc_mu.shape[0], size=4000, replace=False)
            # Xc_mu_rand = [Xc_mu[rand_indx, :]]
            # Yc_mu_rand = [Yc_mu[rand_indx, :]]
            # model_mu = GPFlowEnsemble(kernel_mu, Xc_mu_rand, Yc_mu_rand, opt_noise_mu, SGPR=1)
            
            
            # =================================================================
            # Random Sampling: VAST performance increase over other methods
            # =================================================================
            XY_samples = []
            for XY_chunk in XY_chunks:
                XY_sample = self._random_clustering(XY_chunk, size=2000, number=1)
                XY_samples.append(XY_sample[0])
                
                
            
            # Xc_sigma_chunks, Yc_sigma_chunks = self._random_clustering(X, Y_sigma, size=1000, number=2)
            
            opt_noise_mu = 0.005
            opt_noise_sigma = 0.005
            
            def plot_randomclusters():
                # Visualize the randomly selected points
                fig, ax = plt.subplots(ncols=1, figsize=[5,5], subplot_kw={'projection': '3d', 'computed_zorder': False})
                axs = [ax]
                # Means
                axs[0].scatter(X[:,2], X[:,1], X[:,0], 
                               c=Y_mu[:,0], cmap='magma', vmin=-2, vmax=2,
                               alpha=1, marker='.', s=1)
                for i, XY_sample in enumerate(XY_samples):
                    axs[0].scatter(XY_sample[:,2], XY_sample[:,1], XY_sample[:,0], 
                                   c=[i]*len(XY_sample), cmap='GnBu', vmin=0, vmax=len(XY_samples),
                                   alpha=1, marker='x', s=2, zorder=1)
                axs[0].set(title = 'Mean Model Sampling Points')
                    
                # # Standard Deviations
                # axs[1].scatter(X[:,2], X[:,1], X[:,0], 
                #                c=Y_sigma[:,0], cmap='magma', vmin=-2, vmax=2,
                #                alpha=1, marker='.', s=1)
                # for i, (Xchunk, Ychunk) in enumerate(zip(Xc_sigma_chunks, Yc_sigma_chunks)):
                #     axs[1].scatter(Xchunk[:,2], Xchunk[:,1], Xchunk[:,0], 
                #                    c=[i]*len(Xchunk), cmap='GnBu', vmin=0, vmax=len(Xc_mu_chunks),
                #                    alpha=1, marker='x', s=2, zorder=1)
                # axs[1].set(title = 'Standard Dev. Model Sampling Points')
                
                for ax in axs:
                    ax.set(xlabel='Time', ylabel='Longitude', zlabel='Latitude')
                    ax.view_init(elev=30., azim=80)
                    
                plt.show()
                return
            plot_randomclusters()
            
            # =================================================================
            # Run the GP Regression
            # =====================================================================
            # SGPR = kwargs.get('SGPR', 0.1)
            
            
            # TRY LIKELIHOOD
            
            X_samples = [s[:,:X.shape[1]] for s in XY_samples]
            Y_mu_samples = [s[:,X.shape[1]:X.shape[1]+Y_mu.shape[1]] for s in XY_samples]
            Y_sigma_samples = [s[:,-1][:,None] for s in XY_samples]
            
            # breakpoint()
            
            model = GPFlowEnsemble(kernel_mu, X_samples, Y_mu_samples, Y_sigma_samples,
                                   SGPR=1, 
                                   interpolate_mean=(X,Y_mu))
                                   # interpolate_mean=None)
            
            # breakpoint()
            
            # model_mu = GPFlowEnsemble(kernel_mu, Xc_mu_chunks, Yc_mu_chunks, noise_variance=opt_noise_mu, SGPR=1)
            # model_sigma = GPFlowEnsemble(kernel_sigma, Xc_sigma_chunks, Yc_sigma_chunks, noise_variance=opt_noise_sigma, SGPR=1)
            
            # all_models.update({target_var+'_mu': model_mu,
            #                    target_var+'_sigma': model_sigma})
            all_models.update({target_var: model})
            
            # breakpoint()
            # TWEAK WEIGHTING IN ENSEMBLE
            # =============================================================================
            # Verify performance against input data    
            # =============================================================================
            # model_mu_results = model_mu.predict_f(X, chunk_size=4096, cpu_fraction=0.75)
            # model_sigma_results = model_sigma.predict_f(X, chunk_size=4096, cpu_fraction=0.75)
            # diff_mu = model_mu_results[0] - Y_mu
            # diff_sigma = model_sigma_results[0] - Y_sigma
            
            # fig, axs = plt.subplots(ncols=2, sharey=True, sharex=True)
            # axs[0].hist(diff_mu, np.linspace(-3, 3, 100), density=True)
            # axs[1].hist(diff_sigma, np.linspace(-3, 3, 100), density=True)
            # axs[0].set(xlabel = "Model - Data Mean Difference\n(Normalized)",
            #            ylabel = "Density")
            # axs[1].set(xlabel = "Model - Data Standard Dev. Difference\n(Normalized)")
            # plt.show()
            
            # if (diff_mu.std() > 1) | (diff_sigma.std() > 1):
            #     breakpoint()
    
            # fig, axs = plt.subplots(ncols=2, figsize=[10,5], subplot_kw={'projection': '3d'})
            # temp_result_mu, temp_result_var = model_mu.predict_f(Xc_mu_chunks[1])
            # for ax, Ychunk in zip(axs, [Yc_mu_chunks[1], temp_result_mu]):
            #     ax.scatter(Xc_mu_chunks[1][:,2], Xc_mu_chunks[1][:,1], Xc_mu_chunks[1][:,0], c=Ychunk[:,0], 
            #                alpha=0.5, marker='.', s=36, vmin=-2, vmax=2)
            #     ax.set(xlabel='Time', ylabel='Longitude', zlabel='Latitude')
            #     ax.view_init(elev=30., azim=130)
            # plt.show()
            
            # fig, ax = plt.subplots(figsize=[5,5], subplot_kw={'projection': '3d'})
    
            # ax.scatter(Xc_mu_chunks[1][:,2], Xc_mu_chunks[1][:,1], Xc_mu_chunks[1][:,0], c=temp_result_mu[:,0] - Yc_mu_chunks[1][:,0], 
            #                alpha=0.5, marker='.', s=36, vmin=-1, vmax=1)
            
            # try:
            #     iv0 = model_mu.model_list[1].inducing_variable.Z
            #     ax.scatter(iv0[:,2], iv0[:,1], iv0[:,0], color='black', marker='x', s=36)
            # except:
            #     pass
    
            # ax.set(xlabel='Time', ylabel='Longitude', zlabel='Latitude')
            # ax.view_init(elev=30., azim=70)
            # plt.show()
            
            
            # # Extract at STB position
            # stb_lon = lon_scaler.transform(self.ephemeris['stereo b'].lon_c.to(u.deg).value[:,None])
            # stb_lat = lat_scaler.transform(self.ephemeris['stereo a'].lat_c.to(u.deg).value[:,None])
            # stb_mjd = mjd_scaler.transform(self.ephemeris['stereo b'].time.mjd[:, None])
            # stb_X = np.column_stack([stb_lat, stb_lon, stb_mjd])
            
            # stb_Y_mu, _ = model_mu.predict_f(stb_X)
            # stb_val_mu = val_mu_scaler.inverse_transform(stb_Y_mu)
            
            # stb_Y_sigma, _ = model_sigma.predict_f(stb_X)
            # stb_val_sigma =  val_sigma_scaler.inverse_transform(stb_Y_sigma)
            
            # breakpoint()
            # %% ==================================================================
            # Predict values for the full grid...     
            # =====================================================================
            Xlat, Xlon, Xmjd = np.meshgrid(lat_scaler.transform(lat_for3d[:,None]),
                                           lon_scaler.transform(lon_for3d[:,None]), 
                                           mjd_scaler.transform(mjd_for3d[:,None]),
                                           indexing='ij')
            X3d = np.column_stack([Xlat.flatten()[:,None],
                                   Xlon.flatten()[:,None],
                                   Xmjd.flatten()[:,None]])
            
            # Parallel chunk processing 
            # fmu3d_mu, fmu3d_var = model_mu.predict_f(X3d, chunk_size=4096, cpu_fraction=0.75)
            # fmu_samples = model_mu.predict_f_samples(X3d, num_samples, chunk_size=4096, cpu_fraction=0.75)
            f3d_mu, f3d_var = model.predict_f(X3d, chunk_size=4096, cpu_fraction=0.75)
            
            val_mu = val_mu_scaler.inverse_transform(f3d_mu).reshape(nLat, nLon, nMjd)
            val_sig = val_mu_scaler.scale_ * tf.sqrt(f3d_var).numpy().reshape(nLat, nLon, nMjd)
            
            # For the standard deviation
            # fsig3d_mu, fsig3d_var = model_sigma.predict_f(X3d, chunk_size=4096, cpu_fraction=0.75)
            # fsig_samples = model_sigma.predict_f_samples(X3d, num_samples, chunk_size=4096, cpu_fraction=0.75)
            
            # val_sig_mu = val_sigma_scaler.inverse_transform(fsig3d_mu).reshape(nLat, nLon, nMjd)
            # val_sig_sig = val_sigma_scaler.scale_ * tf.sqrt(fsig3d_var).numpy().reshape(nLat, nLon, nMjd)
            # The uncertainty on the uncertainty is difficult to quantify
            
            # Add to dictionaries
            # all_summaries.update({target_var: {'mu': val_mu_mu, 'sigma': np.sqrt(val_mu_sig**2 + val_sig_mu**2)}})
            all_summaries.update({target_var+'_mu': val_mu,
                                  target_var+'_sigma': val_sig})
            
            # breakpoint()
            # val_sig_sig = val_sigma_scaler.scale_ * tf.sqrt(fsig3d_var).reshape(nLat, nLon, nMjd)
            # test0 = val_sigma_scaler.inverse_transform(fsig3d_mu + tf.sqrt(fsig3d_var)).reshape(nLat, nLon, nMjd) - val_sig_mu
            # test1 = val_sig_mu - val_sigma_scaler.inverse_transform(fsig3d_mu - tf.sqrt(fsig3d_var)).reshape(nLat, nLon, nMjd)
    
            # !!!! Eventually, val will apply to both U and B...
            # U_mu_3d = val_mu_mu
            # U_sigma_3d = np.sqrt(val_mu_sig**2 + val_sig_mu**2)
        
        
        # Generate an OBVIOUSLY WRONG B
        # B_3d = np.tile(self.boundaryDistributions['omni']['B_grid'], (64, 1, 1))
        
        # # %% ==================================================================
        # # TESTING PLOTS
        # # =====================================================================
        # print("Check for prediction quality!")
        # self._assign_boundaryDistributions3D(mjd_for3d, lon_for3d, lat_for3d, U_mu_3d, U_sigma_3d, B_3d)
        # test_atOMNI = self.sample_boundaryDistribution3D(at='omni')
        # test_atSTA = self.sample_boundaryDistribution3D(at='stereo a')
        
        # fig, axs = plt.subplots(nrows=2)
        # img = axs[0].pcolormesh(test_atOMNI['t_grid'], test_atOMNI['lon_grid'], 
        #                         test_atOMNI['U_mu_grid'] - self.boundaryDistributions['omni']['U_mu_grid'], 
        #                         vmin=-100, vmax=100)
        
        
        # img = axs[1].pcolormesh(test_atSTA['t_grid'], test_atSTA['lon_grid'], 
        #                         test_atSTA['U_mu_grid'] - self.boundaryDistributions['stereo a']['U_mu_grid'], 
        #                         vmin=-100, vmax=100)
        
        # fig.colorbar(img, ax=axs)
        # plt.show()
        
        # %% Return
        
        # Assign to self
        # Sample at OMNI/STA
        
        return all_summaries, all_scalers, all_models
        
        # # =============================================================================
        # # Visualization     
        # # =============================================================================
        # self._assign_boundaryDistributions3D(mjd_for3d, lon_for3d, lat_for3d, U_mu_3d, U_sigma_3d, B_3d)
        
        # for source in self.boundarySources:
            
        #     # Reconstruct the backmapped solar wind view at each source
        #     fig, axs = plt.subplots(nrows=2)
            
        #     axs[0].imshow(self.boundaryDistributions[source]['U_mu_grid'],
        #                   vmin=200, vmax=600)
            
        #     boundary = self.sample_boundaryDistribution3D(source)
        #     _ = axs[1].imshow(boundary['U_mu_grid'],
        #                       vmin=200, vmax=600)
            
        #     fig.suptitle(source)
        #     # plt.colorbar(_, cax = ax)
            
        #     plt.show()
                
            
        # breakpoint()
    
        return
    
    def sample_boundaryDistribution3D(self, at=None, num_samples=100):
        from scipy.interpolate import RegularGridInterpolator
        
        # Handle GP and extend differently
        if len(self._boundaryModels) > 0:
            # !!!! Catch exceptions better...
            if at not in self.availableSources:
                breakpoint()
            
            # Rescale all coordinates
            lat = np.interp(self.boundaryDistributions3D['t_grid'],
                            self.solar_wind['mjd'],
                            self.ephemeris[at].lat_c.to(u.deg).value)
            x_lat = self._boundaryScalers['lat_grid'].transform(lat[:, None])
            
            x_lon = self._boundaryScalers['lon_grid'].transform(self.boundaryDistributions3D['lon_grid'][:, None])
            
            x_mjd = self._boundaryScalers['t_grid'].transform(self.boundaryDistributions3D['t_grid'][:, None])
            
            # Construct 2, 2D grid
            x_lon2d, x_t2d = np.meshgrid(x_lon, x_mjd,indexing='ij')
            x_lon2d, x_lat2d = np.meshgrid(x_lon, x_lat, indexing='ij')
            
            # Finally construct 1D list of coordinates
            X = np.column_stack([x_lat2d.flatten()[:, None], 
                                 x_lon2d.flatten()[:, None],
                                 x_t2d.flatten()[:, None]])
            
            # Plug these into the model for samples
            
            # U_mu_samples = self._boundaryModels['U_mu'].predict_f_samples(X, num_samples, chunk_size=5000, cpu_fraction=0.75)
            # U_sigma_samples = self._boundaryModels['U_sigma'].predict_f_samples(X, num_samples, chunk_size=5000, cpu_fraction=0.75)
            # Br_mu_samples = self._boundaryModels['Br_mu'].predict_f_samples(X, num_samples, chunk_size=5000, cpu_fraction=0.75)
            # Br_sigma_samples = self._boundaryModels['Br_sigma'].predict_f_samples(X, num_samples, chunk_size=5000, cpu_fraction=0.75)
            U_samples = self._boundaryModels['U'].predict_f_samples(X, num_samples, chunk_size=5000, cpu_fraction=0.75)
            Br_samples = self._boundaryModels['Br'].predict_f_samples(X, num_samples, chunk_size=5000, cpu_fraction=0.75)

            samples = []
            # Convert back to real units
            for U_sample, Br_sample in zip(U_samples, Br_samples):
                U = self._boundaryScalers['U'].inverse_transform(U_sample).reshape(x_lon2d.shape)
                # U_sigma = self._boundaryScalers['U'].inverse_transform(U_sigma_sample).reshape(x_lon2d.shape)
                
                Br = self._boundaryScalers['Br'].inverse_transform(Br_sample).reshape(x_lon2d.shape)
                # Br_sigma =  self._boundaryScalers['Br_sigma'].inverse_transform(Br_sigma_sample).reshape(x_lon2d.shape)
                
                d = self.boundaryDistributions3D.copy()
                _ = d.pop('lat_grid')
                _ = d.pop('U_mu_grid')
                _ = d.pop('U_sigma_grid')
                _ = d.pop('Br_mu_grid')
                _ = d.pop('Br_sigma_grid')
                d['U_grid'] = U
                # d['U_sigma_grid'] = U_sigma
                d['Br_grid'] = Br
                # d['Br_sigma_grid'] = Br_sigma
                
                samples.append(d)
            
            # U_mu_mu, U_mu_var = self._boundaryModels['U_mu'].predict_f(X, chunk_size=5000, cpu_fraction=0.75)
            # U_sigma_mu, U_sigma_var = self._boundaryModels['U_sigma'].predict_f(X, chunk_size=5000, cpu_fraction=0.75)
            
            # Br_mu_mu, Br_mu_var = self._boundaryModels['Br_mu'].predict_f(X, chunk_size=5000, cpu_fraction=0.75)
            # Br_sigma_mu, Br_sigma_var = self._boundaryModels['Br_sigma'].predict_f(X, chunk_size=5000, cpu_fraction=0.75)
            
            U_mu, U_var = self._boundaryModels['U'].predict_f(X, chunk_size=5000, cpu_fraction=0.75)
            
            Br_mu, Br_var = self._boundaryModels['Br'].predict_f(X, chunk_size=5000, cpu_fraction=0.75)
            
            summary = self.boundaryDistributions3D.copy()
            _ = summary.pop('lat_grid')
            summary['U_mu_grid'] = self._boundaryScalers['U'].inverse_transform(U_mu).reshape(x_lon2d.shape)
            summary['U_sigma_grid'] = self._boundaryScalers['U'].scale_ * tf.sqrt(U_var).numpy().reshape(x_lon2d.shape)
            summary['Br_mu_grid'] = self._boundaryScalers['Br'].inverse_transform(Br_mu).reshape(x_lon2d.shape)
            summary['Br_sigma_grid'] = self._boundaryScalers['Br'].scale_ * tf.sqrt(Br_var).numpy().reshape(x_lon2d.shape)
            
            
            # TESTING
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            
            # Visualize the training data
            test_model = self._boundaryModels['U'].model_list[2]
            ax.scatter(test_model.data[0][:,2], test_model.data[0][:,1], test_model.data[0][:,0],
                       c=test_model.data[1], vmin=-2, vmax=2, cmap='inferno')
            plt.show()
            
            # Visualize the output of the GPR
            test_indx = (X[:,2] > np.min(test_model.data[0][:,2])) & (X[:,2] < np.max(test_model.data[0][:,2]))
            U_mu_test, U_var_test = self._boundaryModels['U'].model_list[1].predict_f(X[test_indx,:])
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.scatter(X[test_indx, 2], X[test_indx, 1], X[test_indx, 0],
                       c=U_mu_test, vmin=-2, vmax=2, cmap='inferno')
            plt.show()
            
            # # Attempt to rerun model
            # X_ = test_model.data[0]
            # Y_ = test_model.data[1]
            # likelihood_ = test_model.likelihood
            # kernel_ = test_model.kernel
            
            # model_ = gpflow.models.GPR((X_, Y_), kernel=gpflow.kernels.RationalQuadratic())
            # opt = gpflow.optimizers.Scipy()
            # opt.minimize(model_.training_loss, model_.trainable_variables)
            
            # U_mu_, U_var_, = model_.predict_f(X[test_indx,:])
            
            # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            # ax.scatter(X[test_indx, 2], X[test_indx, 1], X[test_indx, 0],
            #            c=U_mu_, vmin=-2, vmax=2, cmap='inferno')
            # plt.show()
            
            
            # breakpoint()
        else:
            
            # Rescale all coordinates
            lat = np.interp(self.boundaryDistributions3D['t_grid'],
                            self.solar_wind['mjd'],
                            self.ephemeris[at].lat_c.to(u.deg).value)
            
            x_lat = lat[:, None]
            x_lon = self.boundaryDistributions3D['lon_grid'][:, None]
            x_mjd = self.boundaryDistributions3D['t_grid'][:, None]
            
            # Construct 2, 2D grid
            x_lon2d, x_t2d = np.meshgrid(x_lon, x_mjd,indexing='ij')
            x_lon2d, x_lat2d = np.meshgrid(x_lon, x_lat, indexing='ij')
            
            interp_mu = RegularGridInterpolator((self.boundaryDistributions3D['lat_grid'], 
                                                 self.boundaryDistributions3D['lon_grid'], 
                                                 self.boundaryDistributions3D['t_grid']), 
                                                 self.boundaryDistributions3D['U_mu_grid'])
            
            U_mu_2d = interp_mu(np.column_stack((x_lat2d.flatten(), x_lon2d.flatten(), x_t2d.flatten()))).reshape(x_lon2d.shape)
            
            interp_sigma = RegularGridInterpolator((self.boundaryDistributions3D['lat_grid'], 
                                                    self.boundaryDistributions3D['lon_grid'], 
                                                    self.boundaryDistributions3D['t_grid']), 
                                                    self.boundaryDistributions3D['U_sigma_grid'])
            
            U_sigma_2d = interp_sigma(np.column_stack((x_lat2d.flatten(), x_lon2d.flatten(), x_t2d.flatten()))).reshape(x_lon2d.shape)
            
            interp_mu = RegularGridInterpolator((self.boundaryDistributions3D['lat_grid'], 
                                                 self.boundaryDistributions3D['lon_grid'], 
                                                 self.boundaryDistributions3D['t_grid']), 
                                                 self.boundaryDistributions3D['Br_mu_grid'])
            
            Br_mu_2d = interp_mu(np.column_stack((x_lat2d.flatten(), x_lon2d.flatten(), x_t2d.flatten()))).reshape(x_lon2d.shape)
            
            interp_sigma = RegularGridInterpolator((self.boundaryDistributions3D['lat_grid'], 
                                                    self.boundaryDistributions3D['lon_grid'], 
                                                    self.boundaryDistributions3D['t_grid']), 
                                                    self.boundaryDistributions3D['Br_sigma_grid'])
            
            Br_sigma_2d = interp_sigma(np.column_stack((x_lat2d.flatten(), x_lon2d.flatten(), x_t2d.flatten()))).reshape(x_lon2d.shape)
            
            samples = []
            for _ in range(num_samples):
                d = self.boundaryDistributions3D.copy()
                _ = d.pop('lat_grid')
                d['U_grid'] = U_mu_2d
                # d['U_sigma_grid'] = U_sigma_2d
                d['Br_grid'] = Br_mu_2d
                # d['Br_sigma_grid'] = Br_sigma_2d
                
                samples.append(d)
            
            summary = self.boundaryDistributions3D.copy()
            _ = summary.pop('lat_grid')
            summary['U_mu_grid'] = U_mu_2d
            summary['U_sigma_grid'] = U_sigma_2d
            summary['Br_mu_grid'] = Br_mu_2d
            summary['Br_sigma_grid'] = Br_sigma_2d
        
        return summary, samples
    
    def generate_cmeDistribution(self, search=True):
        
        # 
        t_sig_init = 3*3600 # seconds
        lon_sig_init = 10 # degrees
        lat_sig_init = 10 # degrees
        width_sig_init = 10 # degrees
        thick_mu_init = 4 # solar radii
        thick_sig_init = 1 # solar radii
        speed_sig_init = 200 # km/s
        
        # Get the CMEs
        if search == True:
            cmes = queryDONKI.CME(self.simstart, self.simstop)
        else:
            return
        
        for index, row in cmes.iterrows():
            # Extract CME Analysis info
            info = row['cmeAnalyses']
            
            # Setup a dict to hold CME params
            cmeDistribution_dict = {}
            
            t = (datetime.datetime.strptime(info['time21_5'], "%Y-%m-%dT%H:%MZ") - self.simstart).total_seconds()
            cmeDistribution_dict['t_mu'] = t
            cmeDistribution_dict['t_sig'] = t_sig_init
            
            cmeDistribution_dict['lon_mu'] = info['longitude']
            cmeDistribution_dict['lon_sig'] = lon_sig_init
            
            cmeDistribution_dict['lat_mu'] = info['latitude']
            cmeDistribution_dict['lat_sig'] = lat_sig_init
            
            cmeDistribution_dict['width_mu'] = 2*info['halfAngle']
            cmeDistribution_dict['width_sig'] = width_sig_init
            
            cmeDistribution_dict['speed_mu'] = info['speed']
            cmeDistribution_dict['speed_sig'] = speed_sig_init
            
            cmeDistribution_dict['thickness_mu'] = thick_mu_init
            cmeDistribution_dict['thickness_sig'] = thick_sig_init
            
            cmeDistribution_dict['innerbound'] = 21.5
            
            self.cmeDistribution.loc[index, :] = cmeDistribution_dict
         
        # cmeDistribution = pd.DataFrame(cmeDistribution_dict)
        
        # Drop CMEs at high lat
        lat_cutoff = np.abs(self.cmeDistribution['lat_mu']) > 2.0*self.latmax
        self.cmeDistribution.loc[lat_cutoff, 'lat_mu'] = np.nan
        
        # Drop NaNs
        self.cmeDistribution.dropna(how='any', axis='index', inplace = True)
        
        # self.cmeDistribution = cmeDistribution
        
        return
    

    
    
    def sample(self, weights):
        
        n_samples = len(weights)
        
        rng = np.random.default_rng()
        
        # Plain normal samples
        # backgroundSamples = rng.normal(loc=self.backgroundDistribution['u_mu'],
        #                                scale=self.backgroundDistribution['u_sig'])
        
        # Offset normal samples
        boundarySamples_U = []
        offsets = rng.normal(loc=0, scale=1, size=n_samples)
        offsets_ratio = 0.1
        for offset in offsets:
            boundarySamples_U.append(rng.normal(loc=self.boundaryDistribution['U_mu_grid'] + offsets_ratio*offset*self.boundaryDistribution['U_sig_grid'],
                                              scale=(1-offsets_ratio)*self.boundaryDistribution['U_sig_grid'],
                                              )) 
        
        # To sample the CMEs
        cmeSamples = []
        n_cmes = len(self.cmeDistribution)
        for i in range(n_samples):
            
            cmeSample = {}
            cmeSample['t'] = rng.normal(self.cmeDistribution['t_mu'], 
                                        self.cmeDistribution['t_sig'])
            
            cmeSample['lon'] = rng.normal(self.cmeDistribution['lon_mu'],
                                          self.cmeDistribution['lon_sig'])
            
            cmeSample['lat'] = rng.normal(self.cmeDistribution['lat_mu'],
                                          self.cmeDistribution['lat_sig'])
            
            cmeSample['width'] = rng.lognormal(self.cmeDistribution['width_mu'],
                                               self.cmeDistribution['width_sig'])
            
            cmeSample['thickness'] = rng.lognormal(self.cmeDistribution['thickness_mu'],
                                                   self.cmeDistribution['thickness_sig'])
            
            cmeSample['speed'] = rng.normal(loc=self.cmeDistribution['speed_mu'],
                                            scale=self.cmeDistribution['speed_sig'])
            
            cmeSample['innerbound'] = self.cmeDistribution['innerbound']
            
            cmeSamples.append(pd.DataFrame(data=cmeSample))
        
        # self.nSamples = n_samples
        # # self.boundarySamples = boundarySamples
        # # self.cmeSamples = cmeSamples
        
        return boundarySamples_U, cmeSamples
    
    def sample3D(self, weights, at='omni'):
        
        n_samples = len(weights)
        
        rng = np.random.default_rng()
        
        # Plain normal samples
        # backgroundSamples = rng.normal(loc=self.backgroundDistribution['u_mu'],
        #                                scale=self.backgroundDistribution['u_sig'])
        
        # Offset normal samples
        boundaryDist = self.sample_boundaryDistribution3D(at)
        # boundaryDist = self.boundaryDistributions[at]
        boundarySamples_U = []
        offsets = rng.normal(loc=0, scale=1, size=n_samples)
        offsets_ratio = 0.1
        for offset in offsets:
            boundarySamples_U.append(rng.normal(loc=boundaryDist['U_mu_grid'] + offsets_ratio*offset*boundaryDist['U_sig_grid'],
                                              scale=(1-offsets_ratio)*boundaryDist['U_sig_grid'],
                                              )) 
        
        # To sample the CMEs
        cmeSamples = []
        n_cmes = len(self.cmeDistribution)
        for i in range(n_samples):
            
            cmeSample = {}
            cmeSample['t'] = rng.normal(self.cmeDistribution['t_mu'], 
                                        self.cmeDistribution['t_sig'])
            
            cmeSample['lon'] = rng.normal(self.cmeDistribution['lon_mu'],
                                          self.cmeDistribution['lon_sig'])
            
            cmeSample['lat'] = rng.normal(self.cmeDistribution['lat_mu'],
                                          self.cmeDistribution['lat_sig'])
            
            cmeSample['width'] = rng.lognormal(self.cmeDistribution['width_mu'],
                                               self.cmeDistribution['width_sig'])
            
            cmeSample['thickness'] = rng.lognormal(self.cmeDistribution['thickness_mu'],
                                                   self.cmeDistribution['thickness_sig'])
            
            cmeSample['speed'] = rng.normal(loc=self.cmeDistribution['speed_mu'],
                                            scale=self.cmeDistribution['speed_sig'])
            
            cmeSample['innerbound'] = self.cmeDistribution['innerbound']
            
            cmeSamples.append(pd.DataFrame(data=cmeSample))
        
        # self.nSamples = n_samples
        # # self.boundarySamples = boundarySamples
        # # self.cmeSamples = cmeSamples
        
        return boundarySamples_U, cmeSamples
    
    # def predict_withDask(self, boundarySamples_U, cmeSamples, observer_name, dpadding=0.03):
    #     import multiprocessing as mp
    #     from tqdm import tqdm
    #     from dask.distributed import Client, wait, progress, as_completed
    #     import logging
    #     logging.disable(logging.INFO)
    #     # dask.config.set({'logging.distributed': 'error'})
    #     # dask.config.set({'logging.futures': 'error'})
        
    #     # DO NOT loop over this bit
    #     observer = H.Observer(observer_name, Time(self.boundaryDistribution['t_grid'], format='mjd'))
        
    #     n_cores = int(0.75 * mp.cpu_count()) 
    #     client = Client(n_workers = n_cores,
    #                     threads_per_worker = 1,
    #                     silence_logs = 40)
        
    #     futures = []
    #     for boundarySample_U, cmeSample in zip(boundarySamples_U, cmeSamples):
    #     # for i in range(self.nSamples):
    #         # DO loop over these bits
    #         cme_list = []
    #         for index, row in cmeSample.iterrows():
                
    #             cme = H.ConeCME(t_launch=row['t']*u.s, 
    #                             longitude=row['lon']*u.deg, 
    #                             latitude=row['lat']*u.deg, 
    #                             width=row['width']*u.deg, 
    #                             v=row['speed']*(u.km/u.s), 
    #                             thickness=row['thickness']*u.solRad, 
    #                             initial_height=row['innerbound']*u.solRad,
    #                             cme_expansion=False,
    #                             cme_fixed_duration=True)
                
    #             cme_list.append(cme)
            
    #         future = client.submit(hao.huxt_atObserver, self.simstart, self.simstop,
    #                                self.boundaryDistribution['t_grid'], 
    #                                boundarySample_U,
    #                                self.boundaryDistribution['B_grid'], 
    #                                observer_name, observer,
    #                                dpadding = dpadding, 
    #                                cme_list = cme_list,
    #                                r_min=self.innerbound)
            
    #         futures.append(future)
            
    #     t0 = time.time()
        
    #     # Append the results, after interpolating to internal data index
    #     ordered_dict = {}
    #     for future, result in tqdm(as_completed(futures, with_results=True), total=len(futures)):
    #         interp_result = pd.DataFrame(index=self.solar_wind.index,
    #                                      columns=result.columns)
    #         for col in interp_result.columns:
    #             interp_result[col] = np.interp(self.solar_wind['mjd'], result['mjd'], result[col])
                
    #         ordered_dict[future.key] = interp_result
        
    #     # Now reorder them based on the original futures order
    #     ensemble = [ordered_dict[future.key] for future in futures]
    #     del futures
        
    #     print("{} HUXt forecasts completed in {}s".format(len(ensemble), time.time()-t0))
        
    #     # =============================================================================
    #     # Visualize    
    #     # =============================================================================
    #     fig, ax = plt.subplots(figsize=(6,4.5))
        
    #     for member in ensemble:
    #         ax.plot(member['mjd'], member['U'], color='C3', lw=1, alpha=0.2)
    #     ax.plot(member['mjd'][0:1], member['U'][0:1], lw=1, color='C3', alpha=1, 
    #             label = 'Ensemble Members')
        
        
    #     ax.legend(scatterpoints=3, loc='upper right')
        
    #     ax.set(xlim=[self.starttime.mjd, self.stoptime.mjd])
    #     ax.set(xlabel='Date [MJD], from {}'.format(datetime.datetime.strftime(self.start, '%Y-%m-%d %H:%M')), 
    #            ylabel='Solar Wind Speed [km/s]', 
    #            title='HUXt Ensemble @ {}'.format(observer_name))
        
    #     plt.show()
            
            
    #     return ensemble
    
    # def predict(self, boundarySamples_U, cmeSamples, observer_name, dpadding=0.03):
    #     import multiprocessing as mp
    #     from tqdm import tqdm
    #     from joblib import Parallel, delayed
        
    #     t0 = time.time()
    #     nSamples = len(boundarySamples_U)
        
    #     # DO NOT loop over this bit
    #     observer = H.Observer(observer_name, Time(self.boundaryDistributions3D['t_grid'], format='mjd'))
        
    #     nCores = int(0.75 * mp.cpu_count()) 
        
    #     # Calculate boundary distributions by backmapping each sample
    #     def runHUXt(boundarySample_U, cmeSample):
            
    #         cme_list = []
    #         for index, row in cmeSample.iterrows():
                
    #             cme = H.ConeCME(t_launch=row['t']*u.s, 
    #                             longitude=row['lon']*u.deg, 
    #                             latitude=row['lat']*u.deg, 
    #                             width=row['width']*u.deg, 
    #                             v=row['speed']*(u.km/u.s), 
    #                             thickness=row['thickness']*u.solRad, 
    #                             initial_height=row['innerbound']*u.solRad,
    #                             cme_expansion=False,
    #                             cme_fixed_duration=True)
                
    #             cme_list.append(cme)
                
    #         future = hao.huxt_atObserver(self.simstart, self.simstop,
    #                                      self.boundaryDistributions3D['t_grid'], 
    #                                      boundarySample_U,
    #                                      self.boundaryDistributions3D['B_grid'][0,:,:], 
    #                                      observer_name, observer,
    #                                      dpadding = dpadding, 
    #                                      cme_list = cme_list,
    #                                      r_min=self.innerbound)
            
    #         # Do a bit of reformatting
    #         future.drop(columns=['r', 'lon'], inplace=True)
    #         future.rename(columns={'U': 'U', 'BX': 'Br'}, inplace=True)
            
    #         futureInterpolated = pd.DataFrame(index=self.solar_wind.index,
    #                                           columns=future.columns)
    #         for col in futureInterpolated.columns:
    #             futureInterpolated[col] = np.interp(self.solar_wind['mjd'], future['mjd'], future[col])
            
    #         return futureInterpolated
        
    #     futureGenerator = Parallel(return_as='generator', n_jobs=nCores)(
    #         delayed(runHUXt)(boundarySample_U, cmeSample) 
    #         for boundarySample_U, cmeSample in zip(boundarySamples_U, cmeSamples)
    #         )
        
    #     ensemble = list(tqdm(futureGenerator, total=nSamples))
    #     # !!!! ditch ephemeris info in these files
        
    #     print("{} HUXt forecasts completed in {}s".format(len(ensemble), time.time()-t0))
        
    #     # =============================================================================
    #     # Visualize    
    #     # =============================================================================
    #     # fig, ax = plt.subplots(figsize=(6,4.5))
        
    #     # for member in ensemble:
    #     #     ax.plot(member['mjd'], member['U'], color='C3', lw=1, alpha=0.2)
    #     # ax.plot(member['mjd'][0:1], member['U'][0:1], lw=1, color='C3', alpha=1, 
    #     #         label = 'Ensemble Members')
        
        
    #     # ax.legend(scatterpoints=3, loc='upper right')
        
    #     # ax.set(xlim=[self.starttime.mjd, self.stoptime.mjd])
    #     # ax.set(xlabel='Date [MJD], from {}'.format(datetime.datetime.strftime(self.start, '%Y-%m-%d %H:%M')), 
    #     #        ylabel='Solar Wind Speed [km/s]', 
    #     #        title='HUXt Ensemble @ {}'.format(observer_name))
        
    #     # plt.show()
        
    #     # Save ensemble
    #     self.current_ensemble = ensemble
        
    #     return ensemble
    
    def estimate(self, ensemble, weights, columns=None): # in loop
        """
        Return a weighted median metamodel
    
        Parameters
        ----------
        ensemble : TYPE
            DESCRIPTION.
        weights : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        metamodel = pd.DataFrame(index = ensemble[0].index)
        ensemble_columns = ensemble[0].columns
        
        if columns is None:
            columns = ['U', 'Br']
        
        for col in ensemble_columns:
            for index in metamodel.index:
                vals = [m.loc[index, col] for m in ensemble]
                valsort_indx = np.argsort(vals)
                cumsum_weights = np.cumsum(np.array(weights)[valsort_indx])
                
                weighted_median = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.5 * cumsum_weights[-1])]]
                weighted_upper95 = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.975 * cumsum_weights[-1])]]
                weighted_lower95 = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.025 * cumsum_weights[-1])]]

                if col in columns:
                    metamodel.loc[index, col+"_median"] = weighted_median
                    metamodel.loc[index, col+"_upper95"] = weighted_upper95
                    metamodel.loc[index, col+"_lower95"] = weighted_lower95
                else:
                    metamodel.loc[index, col] = weighted_median
                    
                # breakpoint()
        
        return metamodel
    
    
    def _rescale_2DBoundary(self, bound, target_reduction=None, target_size=None):
        from scipy import ndimage
        from skimage.transform import rescale
        from skimage.measure import block_reduce
        from scipy.interpolate import RegularGridInterpolator
        
        data_shape = bound['U_mu_grid'].shape
        
        if target_reduction is None and target_size is None:
            target_reduction = 0.25
        elif target_reduction is not None:
            zoom_scale = np.sqrt(target_reduction)
        else:
            zoom_scale = np.sqrt(target_size/np.product(data_shape))
        
        new_bound = {}
        for key, val in bound.items():
            
            # Create a mask for valid (non-NaN) pixels
            mask = ~np.isnan(val)
            val_clean = np.where(mask, val, 0.0)
            
            # Resize both image and mask
            val_rescaled = rescale(val_clean, zoom_scale, 
                                  anti_aliasing=True, preserve_range=True)
            mask_rescaled = rescale(mask.astype(float), zoom_scale, 
                                  anti_aliasing=True, preserve_range=True)
            
            new_val = val_rescaled/mask_rescaled
            new_val[~mask_rescaled.astype(bool)] = np.nan
                
            new_bound[key] = new_val
        
        
        # Estimate noise 
        noise_variance = {}
        for key, val in new_bound.items():
            if len(val.shape) == 2:
                interp = RegularGridInterpolator(
                    (new_bound['lon_grid'], new_bound['t_grid']), 
                    val,
                    bounds_error=False)
            
                lon2d, t2d = np.meshgrid(bound['lon_grid'], bound['t_grid'], indexing='ij')
                upscaled = interp(np.column_stack([lon2d.flatten(), t2d.flatten()])).reshape(lon2d.shape)
                difference = upscaled - bound[key]
                
                noise_variance[key] = np.nanpercentile(difference, 95)
        
        return new_bound, noise_variance
                                         
    # =========================================================================
    # Utility Functions 
    # (that could be separated from this file with no loss of generalization 
    # or context)
    # =========================================================================
    def _getChunksInTime(self, df, delta=90 * u.day):
        
        # We want each chunk to be as close to delta in length as possible
        # And to overlap on each side by overlap
        total_span = self.simstoptime - self.simstarttime
        overlap = 10 * u.day
        core_length = delta - 2 * overlap
        approx_chunks = (total_span - overlap) / (core_length + overlap)
        
        n_chunks = int(np.floor(approx_chunks))
        eff_core_length = ((total_span - overlap) - n_chunks*overlap)/n_chunks
        eff_delta = eff_core_length + 2 * overlap
        
        dfs = []
        for i in range(int(np.ceil(n_chunks))):
            
            # subsimstart = (self.simstarttime + i * eff_delta - (0 if i == 0 else 1) * overlap)
            subsimstart = (self.simstarttime + i * eff_delta - i * overlap)
            subsimstop = (subsimstart + eff_delta)
            
            dfs.append(df.query("@subsimstart.mjd <= mjd < @subsimstop.mjd"))    
            
        # Make sure the dfs cover the full range
        if not (pd.concat(dfs).drop_duplicates().index == df.index).all():
            print("Missing dataframe coverage!")
            breakpoint()
        
        return dfs
    
    
    def _optimize_clustering(self, X, Y, target_noise_variance=0.01):
                             #target_reduction=None, target_noise=None, inX=None, inXY=None):
        from sklearn.cluster import MiniBatchKMeans
        from scipy.optimize import curve_fit
        from sklearn.cluster import HDBSCAN
        
        # target_reduction = kwargs.get('target_reduction')
        # target_noise = kwargs.get('target_noise')
        # inX = kwargs.get('inX')
        # inXY = kwargs.get('inXY')
        
        hdb = HDBSCAN(min_cluster_size=2, 
                      max_cluster_size=6, 
                      cluster_selection_epsilon=target_noise_variance)
        
        # Naturally, independent variables will be more closely spaced than independent variables
        # Here we adjust for this for better clustering
        X_adjustment_factor = np.abs(np.diff(Y, axis=0)).mean(axis=0) / np.abs(np.diff(X, axis=0)).mean(axis=0)
        hdb.fit(np.hstack([X_adjustment_factor * X, Y]))

        # Separate true labels from "noise" (-1) label
        true_labels = set(hdb.labels_) - {-1}

        # Loop over true labels to assign each to a centroid
        Xc_mu, Yc_mu = [], []
        Xc_sigma, Yc_sigma = [], []
        for l in true_labels:
            
            X_label = X[hdb.labels_ == l,:]
            Y_label = Y[hdb.labels_ == l,:]
            
            Xc_mu.append(X_label.mean(axis=0))
            Yc_mu.append(Y_label.mean(axis=0))
            
            Xc_sigma.append(X_label.std(axis=0))
            Yc_sigma.append(Y_label.std(axis=0))
            
        # Add the noise points back in    
        Xc_mu.extend(X[hdb.labels_ == -1,:])
        Yc_mu.extend(Y[hdb.labels_ == -1,:])
        
        Xc_sigma.extend(X[hdb.labels_ == -1,:] * 0)
        Yc_sigma.extend(Y[hdb.labels_ == -1,:] * 0)
        
        # Convert back to an array
        Xc_mu = np.array(Xc_mu)
        Yc_mu = np.array(Yc_mu)
        
        Xc_sigma = np.array(Xc_sigma)
        Yc_sigma = np.array(Yc_sigma)
        
        # Finally, reorder to match input
        cluster_sort_index = None
        for i_col, col in enumerate(X.T):
            # Below is true if monotonic along this column
            if (col[1:] >= col[:-1]).all():
                cluster_sort_index = np.argsort(Xc_mu[:, i_col])
        if cluster_sort_index is None:
            cluster_sort_index = np.argsort(Xc_mu[:, -1])
                
        Xc_mu = Xc_mu[cluster_sort_index]
        Yc_mu = Yc_mu[cluster_sort_index]
        
        Xc_sigma = Xc_sigma[cluster_sort_index]
        Yc_sigma = Yc_sigma[cluster_sort_index]
        
        return Xc_mu, Yc_mu, target_noise_variance
    
    def _optimize_chunking(self, X, Y, **kwargs):
        # Keywords
        max_chunk_length    = kwargs.get('max_chunk_length', 2048)
        byDimension         = kwargs.get('byDimension')
        byCluster           = kwargs.get('byCluster')
            
        #    
        XY = np.column_stack([X, Y])
        
        if (byDimension is None) & (byCluster is None):
            print("By default, chunking linearly in current order.")
            sort_indx = np.arange(0, XY.shape[0])
        elif byDimension is not None:
            sort_indx = np.argsort(XY[:,byDimension])
        elif byCluster is not None:
            sort_indx = np.arange(0, XY.shape[0])
        
        # Sort XY
        XY_sorted = XY[sort_indx,:]
        
        # Number of chunks
        nChunks = np.ceil(XY_sorted.shape[0] / max_chunk_length).astype(int)
        
        if byCluster is None:
            XY_chunks = np.array_split(XY_sorted, nChunks)
        else:
            if byCluster == 'X':
                kmeans = KMeans(n_clusters=nChunks).fit(X)
            else: # byCluster == 'XY':
                kmeans = KMeans(n_clusters=nChunks).fit(XY)
            XY_chunks = [XY[kmeans.labels_ == i, :] for i in range(kmeans.n_clusters)]
        
        X_chunks = [XY_chunk[:,:X.shape[1]] for XY_chunk in XY_chunks]
        Y_chunks = [XY_chunk[:,X.shape[1]:] for XY_chunk in XY_chunks]
        
        return X_chunks, Y_chunks
        
    def _random_clustering(self, XY, size=10, number=10):
        # Randomly choose indices for n(umber) samples into X, Y, of length s(ize)
        
        rng = np.random.default_rng()
        
        XYcs = []
        for _ in range(number):
            indx = np.sort(rng.choice(np.arange(XY.shape[0], dtype=int), size, replace=False))
            XYc = XY[indx, :]
            XYcs.append(XYc)
        
        # CHECK THAT DISTRIBUTION IS REPRESENTATIVE ???
            
        return XYcs
    

# Define an inner function to be run in parallel
def _map_vBoundaryInwards(simstart, simstop, source, insitu_df, corot_type, ephemeris, innerbound):
    
    # Reformat for HUXt inputs expectation
    insitu_df['BX_GSE'] =  -insitu_df['Br']
    insitu_df['V'] = insitu_df['U']
    insitu_df['datetime'] = insitu_df.index
    insitu_df = insitu_df.reset_index()
    
    # Generate the Carrington grids
    t, vcarr, bcarr = Hin.generate_vCarr_from_insitu(simstart, simstop, 
                                                     insitu_source=source, insitu_input=insitu_df, 
                                                     corot_type=corot_type)
    
    # Map to 210 solar radii, then to the inner boundary for the model
    vcarr_inner = vcarr.copy()
    bcarr_inner = bcarr.copy()
    for i, _ in enumerate(t):
        current_r = np.interp(t[i], ephemeris.time.mjd, ephemeris.r)
        results = Hin.map_v_boundary_inwards(
            vcarr[:,i]*u.km/u.s, 
            current_r.to(u.solRad),
            innerbound,
            b_orig = bcarr[:,i]
            )
        
        vcarr_inner[:,i] = results[0]
        bcarr_inner[:,i] = results[1]
        
    return vcarr_inner, bcarr_inner


# %% Define custom mean function for use in GPFlow models   
import gpflow
 
from check_shapes import inherit_check_shapes
# @wrap_non_picklable_objects
class CustomMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self, X, Y):
        
        self.X = X
        self.Y = Y
        
        self.bins_d2 = np.linspace(-3, 3, 100)
        self.bins_d2_indx = np.digitize(X[:,2], self.bins_d2)
        
        self.bins_d1 = np.linspace(-3, 3, 200)
        self.bins_d1_indx = np.digitize(X[:,1], self.bins_d1)       
        
        self.mean = np.zeros([200,100])
        for i in range(200):
            for j in range(100):
                indx = (self.bins_d1_indx == i) & (self.bins_d2_indx == j)
                if indx.any():
                    self.mean[i,j] = np.mean(Y[indx])
                    
        from scipy.interpolate import RegularGridInterpolator
        self.interp = RegularGridInterpolator((self.bins_d1, self.bins_d2), self.mean)

    @inherit_check_shapes
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        result = tf.numpy_function(self.interp, [X[:,1:]], tf.float64)[:,None]
        return result

