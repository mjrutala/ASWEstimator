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
# import time
import matplotlib.pyplot as plt
import pandas as pd
# import tqdm
import copy
import tensorflow as tf
# import pickle
import dill as pickle
import tensorflow_probability  as     tfp
SoftClip = tfp.bijectors.SoftClip
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sunpy.coordinates.sun import carrington_rotation_number as sunpy_crn

import sys
path = '/Users/mrutala/projects/ASWEstimator/'
sys.path.append(path + '/code/')
import ASWReaders
import ASWEphemeris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import gpflow
import GPFlowEnsemble

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

feature_range = (0,10)
feature_quantiles = [0.1, 0.5, 0.9]
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
            import tensorflow as tf
            pickle.dump(self, f)
    
    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as f:
            import tensorflow as tf
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
            data_df = ASWReaders.SolarWindData(source, self.simstart, self.simstop).data
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
            
            # Avoid importing HUXt; instead, do a similar linear interpolation
            source_icme_df = icme_df.query('affiliated_source == @source')
            source_icme_series = pd.Series(index=self.solar_wind.index, data=False)
            if len(source_icme_df) > 0:
                for _, entry in source_icme_df.iterrows():
                    start_mjd = entry['mjd'].mjd - self._icme_duration_buffer.to(u.day).value
                    stop_mjd = start_mjd + entry['duration'] + 2*self._icme_duration_buffer.to(u.day).value
                    
                    icme_index = (self.solar_wind['mjd'] >= start_mjd) & (self.solar_wind['mjd'] < stop_mjd)
                    source_icme_series.loc[icme_index] = True
            
            # # Format insitu data for HUXt's remove_ICMEs function
            # insitu = self.solar_wind[source].copy()
            # insitu.loc[:, 'mjd'] = self.solar_wind.loc[:, 'mjd']
            
            # # Format ICME data for HUXt's remove_ICMEs function
            # icmes = icme_df.query('affiliated_source == @source')
            # icmes.reset_index(inplace=True, drop=True)
            # if 'eventTime' in icmes.columns: 
            #     icmes = icmes.rename(columns = {'eventTime': 'Shock_time'})
            #     icmes['ICME_end'] = [row['Shock_time'] + datetime.timedelta(days=(row['duration'])) 
            #                          for _, row in icmes.iterrows()]
            
            # # Interpolate over existing data gaps (NaNs), so they aren't caught as ICMEs
            # insitu.interpolate(method='linear', axis='columns', limit_direction='both', inplace=True)
            
            # # Extract the timesteps during which there is an ICME
            # if len(icmes) > 0:
            #     breakpoint()
            #     insitu_noicme = Hin.remove_ICMEs(insitu, icmes, 
            #                                      params=['U'], 
            #                                      interpolate = False, 
            #                                      icme_buffer = self._icme_duration_buffer, 
            #                                      interp_buffer = self._icme_interp_buffer, 
            #                                      fill_vals = np.nan)
                
            #     icme_series = insitu_noicme['U'].isna().to_numpy()
                
            # else:
                # insitu_noicme = insitu
                
                # icme_series = [None] * len(insitu)
                
            # Add ICME indices to background data
            idx = self.solar_wind.columns.get_loc((source, self.solar_wind[source].columns[-1]))
            self.solar_wind.insert(idx+1, (source, 'ICME'), source_icme_series)
                          
        return # insitu_noicme['U'].isna()
    
    @property
    def ephemeris(self):
        from astropy.time import Time
        # If this hasn't been run before, run for all 
        if len(self._ephemeris) == 0:
            print("No ephemeris loaded. Now generating...")
            for source in self.availableSources:
                eph = ASWEphemeris.ephemeris(source, Time(self.solar_wind.index), ephemeris_dir=path+'/ephemeris/')
                # eph = H.Observer(source, Time(self.solar_wind.index))
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
                                   GP = False, interpolate = False,
                                   n_samples = 1, 
                                   target_variables = ['U']):
        
        
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
                
                carrington_period = self.get_carringtonPeriod(self.ephemeris[source]['r'].mean())
                
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
        df_chunks = self._getChunksInTime(df, delta=90*u.day, overlap=10*u.day)
        
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
                time_scaler = MinMaxScaler(feature_range=feature_range)
                time_scaler.fit(df_chunk['mjd'].to_numpy()[:,None])
                
                X_all = time_scaler.transform(df_chunk['mjd'].to_numpy()[:,None])
                
                bgScalers[target_var]['mjd'].append([time_scaler]) # append as list to allow for multiple X scalers later
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
                
                X_list.append(X_train)
                Y_list.append(Y_train)
                
                # =================================================================
                # Define kernel for each dimension separately, then altogether
                # =================================================================
                period_rescaled = np.float64(carrington_period.to(u.day).value * time_scaler.scale_[0])
                period_gp = gpflow.Parameter(period_rescaled, trainable=False)
                
                # Only predict 3 Carrington Rotation forward
                min_x = np.float64(0)
                mid_x = np.float64(period_rescaled)
                max_x = np.float64(3*period_rescaled) # 4*period_rescaled
                
                lengthscale_gp = gpflow.Parameter(mid_x, 
                    transform = tfp.bijectors.SoftClip(min_x, max_x))
                ls_amplitude = gpflow.Parameter(np.float64(1.1*period_rescaled), 
                    transform = tfp.bijectors.SoftClip(np.float64(0.99*period_rescaled), max_x))
                
                base_kernel = gpflow.kernels.RationalQuadratic(lengthscales = lengthscale_gp)
                amplitude_kernel = gpflow.kernels.SquaredExponential(lengthscales = ls_amplitude)
                period_kernel = gpflow.kernels.Periodic(
                    gpflow.kernels.SquaredExponential(lengthscales=period_gp),
                    period=period_gp)
                
                noise_kernel = gpflow.kernels.White(gpflow.Parameter(0.05**2, trainable=False))
                # noise_kernel = gpflow.kernels.White(0.05**2)
                # noise_kernel = gpflow.kernels.White(0.1**2)
                
                kernel = base_kernel + (amplitude_kernel * period_kernel) + noise_kernel
                # kernel = base_kernel 
                kernel_backup = gpflow.kernels.RationalQuadratic() + noise_kernel
                
                k_list.append([kernel, kernel_backup])
              
            # =============================================================================
            # Plug into the ensemble GP model
            # =============================================================================
            model = GPFlowEnsemble.EnsembleGPR(X_list, Y_list, k_list, 
                                               bgScalers[target_var]['mjd'], 
                                               bgScalers[target_var]['val'])
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
    
    def makeBoundaryDistributions(self, target_variables=['U'], constant_percent_error=0.0):
        
        ref_r = 1 * u.AU
        
        # Longitudinal x Time Grids
        lon_step = 3
        lon_grid = np.arange(0, 360+lon_step/2, lon_step) * u.deg
        mjd_grid = (np.arange(self.simstarttime.mjd, self.simstoptime.mjd+0.5, 1)) * u.day
        
        # Sidereal period from quasi-infinite distance
        period_Carr_sidereal = self.get_carringtonPeriod(ref_r*1e9)
        # Ballistically project background samples onto a sphere at ref_r
        def ballistic_projection(df, eph, target_vars=['U']):
            
            speed = df['U'].to_numpy() * u.km / u.s
            delta_r = ref_r - eph['r']
            delta_t = delta_r / speed
            mjd_ref = df['mjd'].to_numpy() * u.day + delta_t
            
            delta_lon = delta_t * (2 * np.pi * u.rad / period_Carr_sidereal)
            lon_ref = np.unwrap(eph['lon_c']) + delta_lon
            
            # Estimate how many complete rotations are observed
            rotation_number = np.ceil((lon_ref[0] - lon_ref[-1]) / (2*np.pi*u.rad))
            
            var_grid_dict = {}
            for target_var in target_vars:
                var_ref = df[target_var].to_numpy()
                
                var_grid = np.full((*mjd_grid.shape, *lon_grid.shape), np.nan)
                
                for l, lon in enumerate(lon_grid):
                    
                    # Find all times this carrington longitude is sampled
                    lon_sample = lon + np.arange(-rotation_number, 1)*360*u.deg
                    mjd_sample = np.interp(lon_sample, np.flip(lon_ref.to(u.deg)), np.flip(mjd_ref), left=np.nan, right=np.nan)
                    var_sample = np.interp(lon_sample, np.flip(lon_ref.to(u.deg)), np.flip(var_ref), left=np.nan, right=np.nan)
                    
                    # Drop nans from overestimating the sampled lons
                    mjd_sample = mjd_sample[~np.isnan(mjd_sample)]
                    var_sample = var_sample[~np.isnan(var_sample)]
                    
                    # 
                    res = np.interp(mjd_grid, np.flip(mjd_sample), np.flip(var_sample))
                    var_grid[:, l] = res
                
            var_grid_dict.update({target_var: var_grid})
            
            return var_grid_dict
        
        boundaryDistributions_d = {}
        boundarySamples_d = {}
        for source in self.boundarySources:
            
            dfSamples = [df[source] for df in self.backgroundSamples]
            
            bpResult = [ballistic_projection(dfSample, self.ephemeris[source], target_variables) for dfSample in dfSamples]
            
            # Go from list of dicts to dict of lists
            varResult = {v:[d[v] for d in bpResult] for v in target_variables}
            
            # Get the mean and stadard deviation for each variable
            # var_mu = {v: np.mean(varResult[v], axis=0) for v in target_variables}
            # var_sigma = {v: np.std(varResult[v], axis=0) for v in target_variables}
            
            source_dict = {'t_grid': mjd_grid.value, 'lon_grid': lon_grid.value}
            for v in target_variables:
                source_dict[v+'_mu_grid'] = np.mean(varResult[v], axis=0)
                source_dict[v+'_sigma_grid'] = np.std(varResult[v], axis=0)
                
            boundaryDistributions_d[source] = source_dict
            
            # For completeness, add boundarySamples here
            boundarySamples_d[source] = []
            for item in bpResult:
                source_dict = {'t_grid': mjd_grid.value, 'lon_grid': lon_grid.value}
                for v in target_variables:
                    source_dict[v] = item[v]
                boundarySamples_d[source].append(source_dict)
        
        self.boundaryDistributions = boundaryDistributions_d
        self.boundarySamples = boundarySamples_d
        
        return
    
    def generate_boundaryDistribution3D(self, nLat=32, extend=None, GP=True, 
                                        target_variables = ['U'], num_samples=0,
                                        **kwargs):
        
        # Get dimensions from OMNI boundary distribution, which *must* exist
        nLon, nTime = self.boundaryDistributions['omni']['U_mu_grid'].shape
        
        # Coordinates = (lat, lon, time)
        # Values = boundary speed, magnetic field* (*not implemented fully)
        mjd_for3d = self.boundaryDistributions['omni']['t_grid']
        lon_for3d = self.boundaryDistributions['omni']['lon_grid']
        lat_for3d = np.linspace(-self.latmax.value, self.latmax.value, nLat)
        # lon_for3d = np.linspace(0, 360, nLon+1)[:-1]
        
        if (type(extend) == str) & (GP == True):
            print("Cannot have extend=str and GP=True!")
            return
        
        #
        if type(extend) == str:
            summary = self._extendBoundaryDistributions(
                lat_for3d, lon_for3d, mjd_for3d, extend, num_samples=num_samples, target_variables=target_variables)
            model_d = {key: None for key in target_variables}
        #
        elif GP is True:
            summary, model_d = self._imputeBoundaryDistributions(
                lat_for3d, lon_for3d, mjd_for3d, num_samples=num_samples, target_variables=target_variables, **kwargs)
        
        # Assign to dict
        self.boundaryDistributions3D = {}
        self.boundaryModels = {}
        self.boundaryDistributions3D.update({'t_grid': mjd_for3d,
                                             'lon_grid': lon_for3d,
                                             'lat_grid': lat_for3d,
                                             })
        for target_var in target_variables:
            self.boundaryDistributions3D.update({target_var+'_mu_grid': summary[target_var+'_mu_grid']})
            self.boundaryDistributions3D.update({target_var+'_sigma_grid': summary[target_var+'_sigma_grid']})
            self.boundaryModels[target_var] = model_d[target_var]
        
        return
        
    def _extendBoundaryDistributions(self, lat_for3d, lon_for3d, mjd_for3d,
                                     name,
                                     target_variables = ['U'],
                                     **kwargs):
        
        summary = {}
        for target_var in target_variables:
            summary[target_var+'_mu_grid'] = np.repeat(
                self.boundaryDistributions[name][target_var+'_mu_grid'][:,:,None], lat_for3d.shape, axis=2
                )
            summary[target_var+'_sigma_grid'] = np.repeat(
                self.boundaryDistributions[name][target_var+'_sigma_grid'][:,:,None], lat_for3d.shape, axis=2
                )
            
        return summary
        
    def _imputeBoundaryDistributions(self, lat_for3d, lon_for3d, mjd_for3d,
                                     # maximum_span = 90*u.day, 
                                     target_variables = ['U'],
                                     chunk_duration = 20*u.day,
                                     chunk_overlap = 5*u.day,
                                     samples_per_chunk = 2000,
                                     sample_grid = False,
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
        
        # Incredibly naive chunking -- to be added
        chunk_step = (chunk_duration - chunk_overlap).to(u.day).value
        chunk_mjd_starts = np.arange(self.simstarttime.mjd, self.simstoptime.mjd, chunk_step)
        chunk_mjd_stops = chunk_mjd_starts + chunk_duration.to(u.day).value
        
        # Get dimensions from supplied grid parameters
        nLat = len(lat_for3d)
        nLon = len(lon_for3d)
        nMjd = len(mjd_for3d)
        
        all_summaries = {}
        all_models = {}
        
        boScalers = {}
        boGPModels = {}
        
        # Repeat everything for each target variable
        for target_var in target_variables:
            
            # Set up dicts to hold scalers and models
            boScalers[target_var] = {'mjd': [], 'lon': [], 'lat': [], 'val': []}
            boGPModels[target_var] = {}
            
            # Extract the positions of each input spacecraft
            mjds, lons, lats = [], [], []
            μvals, σvals = [], []
            for source, boDist in self.boundaryDistributions.items():
                
                # lat is degenerate with time, lon from corotation assumption
                mjd_1d = boDist['t_grid']
                lon_1d = boDist['lon_grid']
                lat_1d = np.interp(mjd_1d, 
                                   self.ephemeris[source]['time'].mjd, 
                                   self.ephemeris[source]['lat_c'].to(u.deg).value)
                
                mjd_2d, lon_2d, = np.meshgrid(mjd_1d, lon_1d, indexing='ij')
                lat_2d, lon_2d, = np.meshgrid(lat_1d, lon_1d, indexing='ij')
                
                # # Longitude needs to be non-circular for periodic trends to vary
                # # Tie it to self.starttime and mjd to keep it consistent
                # lon_2d += (mjd_1d - self.starttime.mjd)[:,None] * 360
                
                mjds.extend(mjd_2d.flatten())
                lons.extend(lon_2d.flatten())
                lats.extend(lat_2d.flatten())
                
                μvals.extend(boDist[target_var+'_mu_grid'].flatten())
                σvals.extend(boDist[target_var+'_sigma_grid'].flatten())
            
            # Cast all coordinates as arrays
            mjds = np.array(mjds)[:,None]
            lons = np.array(lons)[:,None]
            lats = np.array(lats)[:,None]
            μvals = np.array(μvals)[:,None]
            σvals = np.array(σvals)[:,None]
            
            
            # # breakpoint()
            # # SCRATCH WORK - GET DECAYING PERIODIC KERNEL TO WORK
            # t_ = mjd_2d.flatten()[:3600,None]
            # l_ = lon_2d.flatten()[:3600,None]
            # X_ = np.hstack([t_, l_])
            # t_kernel = gpflow.kernels.RationalQuadratic(active_dims=[0], lengthscales=40, variance=10, alpha=1e+2)
            # l_kernel = gpflow.kernels.Periodic(gpflow.kernels.RationalQuadratic(active_dims=[1], lengthscales=1, variance=1, alpha=1e-2), period = 360)
            # kernel = t_kernel * l_kernel
            # K = kernel.K(X_).numpy() + np.eye(len(X_)) * 1e-4
            # L = np.linalg.cholesky(K)
            # num_samples = 6
            # random_z = np.random.randn(len(X_), num_samples)
            # samples = L @ random_z

            # fig, ax = plt.subplots()
            # for sample in samples.T:
            #     ax.plot(l_.flatten(), sample, lw=1)
            # plt.show()
            
            
            # Chunk all the coordinates
            df_chunks = []
            for cmjd_start, cmjd_stop in zip(chunk_mjd_starts, chunk_mjd_stops):
                
                cindx = (mjds >= cmjd_start) & (mjds < cmjd_stop)
                chunk = pd.DataFrame({'mjds': mjds[cindx],
                                      'lons': lons[cindx],
                                      'lats': lats[cindx],
                                      'μvals': μvals[cindx], 
                                      'σvals': σvals[cindx]})
                df_chunks.append(chunk)
            
            # Loop through each chunk to scale the coordinates
            X_list_full,  X_scaler_list     = [], []
            Y_list_full,  Y_scaler_list     = [], []
            Σ2_list_full, Σ2_scaler_list    = [], []
            
            kernel_list = []
            mean_function_list = []
            var_mean_function_list = []
            
            for df_chunk in df_chunks:
                
                # Scale MJD, lon, lat with MinMax
                mjd_scaler = MinMaxScaler(feature_range=feature_range)
                mjd_scaler.fit(df_chunk['mjds'].to_numpy()[:,None])
                
                lon_scaler = MinMaxScaler(feature_range=feature_range)
                lon_scaler.fit(df_chunk['lons'].to_numpy()[:,None])
                
                lat_scaler = MinMaxScaler(feature_range=feature_range)
                lat_scaler.fit(df_chunk['lats'].to_numpy()[:,None])
                
                X = np.hstack([mjd_scaler.transform(df_chunk['mjds'].to_numpy()[:,None]), 
                               lon_scaler.transform(df_chunk['lons'].to_numpy()[:,None]),
                               lat_scaler.transform(df_chunk['lats'].to_numpy()[:,None])])
                
                X_list_full.append(X)
                X_scaler_list.append([mjd_scaler, lon_scaler, lat_scaler])
                
                # StandardScale the mean value
                μval_scaler = StandardScaler()
                μval_scaler.fit(df_chunk['μvals'].to_numpy()[:,None])
                
                Y = μval_scaler.transform(df_chunk['μvals'].to_numpy()[:,None])
                
                Y_list_full.append(Y)
                Y_scaler_list.append(μval_scaler)
                
                σ2val_scaler = Pipeline(
                    steps=[('log', FunctionTransformer(np.log, np.exp)),
                           ('scale', StandardScaler())])
                σ2val_scaler.fit(df_chunk['σvals'].to_numpy()[:,None]**2)
                
                Σ2 = σ2val_scaler.transform(df_chunk['σvals'].to_numpy()[:,None]**2)
                
                Σ2_list_full.append(Σ2)
                Σ2_scaler_list.append(σ2val_scaler)
                
                # σ2val_scaler = StandardScaler()
                # σ2val_scaler.fit(df_chunk['σvals'].to_numpy()[:,None]**2)
                
                # Σ2 = σ2val_scaler.transform(df_chunk['σvals'].to_numpy()[:,None]**2)
                
                # Σ2_list_full.append(Σ2)
                # Σ2_scaler_list.append(σ2val_scaler)
                
                # 
                # mean_function_list.append(LinearAverage(X, Y))
                mean_function_list.append(SpatialAverage(X, Y))
                
                # =================================================================
                # Define kernel for each dimension separately, then altogether
                # =================================================================
                
                # MJD 
                mjd_min, mjd_mid, mjd_max = np.float64([10, 15, 100]) * mjd_scaler.scale_ # From Milosic 2025/6 (?)
                # mjd_min, mjd_mid, mjd_max = np.float64([0.1, 1, 30])
                
                mjd_kernel = gpflow.kernels.RationalQuadratic(active_dims=[0])
                # mjd_kernel = gpflow.kernels.SquaredExponential(active_dims=[0])
                mjd_kernel.lengthscales = gpflow.Parameter(
                    mjd_mid, transform = SoftClip(mjd_min, mjd_max))
                
                
                # LONGITUDE
                lon_min, lon_mid, lon_max = np.float64([0, 10, 90]) * lon_scaler.scale_
                # lon_min, lon_mid, lon_max = np.float64([0.1, 1, 30]) 
                
                # lon_kernel_trend = gpflow.kernels.RationalQuadratic(active_dims=[1])
                # # lon_kernel_trend = gpflow.kernels.SquaredExponential(active_dims=[1])
                # lon_kernel_trend.lengthscales = gpflow.Parameter(
                #     lon_mid, transform = SoftClip(lon_min, lon_max))
                
                # lon_kernel_amplitude = gpflow.kernels.RationalQuadratic(active_dims=[1])
                # lon_kernel_amplitude.lengthscales = gpflow.Parameter(
                #     lon_mid, transform = SoftClip(lon_min, lon_max))
                
                lon_kernel_period = gpflow.kernels.Periodic(
                    gpflow.kernels.RationalQuadratic(active_dims=[1]), 
                    period = gpflow.Parameter(np.float64(lon_scaler.scale_[0] * 360), trainable=False))
                lon_kernel_period.base_kernel.lengthscales = gpflow.Parameter(
                    lon_mid, transform = SoftClip(lon_min, lon_max))
                
                # LATITUDE
                lat_min, lat_mid, lat_max = np.float64([0, 1, 3]) * lat_scaler.scale_
                # lat_min, lat_mid, lat_max = np.float64([0.1, 1, 20]) 
                
                lat_kernel = gpflow.kernels.RationalQuadratic(active_dims=[2])
                # lat_kernel = gpflow.kernels.SquaredExponential(active_dims=[2])
                lat_kernel.lengthscales = gpflow.Parameter(
                    lat_mid, transform = SoftClip(lat_min, lat_max))
                
                # noise_kernel = gpflow.kernels.White(gpflow.Parameter(0.05**2, trainable=False), active_dims=[0,1,2])
                
                # Kernel option #1
                # kernel = (mjd_kernel * (lon_kernel_trend + lon_kernel_amplitude * lon_kernel_period) * lat_kernel) # + noise_kernel
                # kernel = mjd_kernel + (lon_kernel_trend + lon_kernel_amplitude * lon_kernel_period) + lat_kernel
                kernel = mjd_kernel * lon_kernel_period * lat_kernel #+ noise_kernel
                
                # Kernel option #2
                # mjd_kernel = gpflow.kernels.RationalQuadratic(active_dims=[0])
                # lon_kernel = gpflow.kernels.RationalQuadratic(active_dims=[1])
                # lat_kernel = gpflow.kernels.RationalQuadratic(active_dims=[2])
                kernel_backup =  gpflow.kernels.RationalQuadratic(active_dims=[0,1,2]) # + noise_kernel
                
                kernel_list.append( [kernel, kernel_backup] )
            
            # =================================================================
            # Implement random sampling?
            # =================================================================
            import scipy
            random_sample_stats = []
            
            rng = np.random.default_rng()
            X_list, Y_list, Σ2_list = [], [], []
            typical_len = len(X_list_full[0])
            for i in range(len(X_list_full)):
                
                this_len = len(X_list_full[i])
                
                # Scale the length of each sample by the length of the chunk
                n_rng = int((samples_per_chunk/typical_len) * this_len)
                
                # Draw non-repeating indices
                random_indx = np.sort(rng.choice(np.arange(this_len), n_rng, replace=False))
                
                # Subsample
                X_list.append(X_list_full[i][random_indx, :])
                Y_list.append(Y_list_full[i][random_indx, :])
                Σ2_list.append(Σ2_list_full[i][random_indx, :])
                
                # Get KS statistics to confirm sampling is sufficient
                stat_row = []
                for _x_full, _x in zip(X_list_full[i].T, X_list[i].T):
                    stat_row.append(scipy.stats.kstest(_x_full, _x).pvalue)
                stat_row.append(scipy.stats.kstest(Y_list_full[i][:,0], Y_list[i][:,0]).pvalue)
                stat_row.append(scipy.stats.kstest(Σ2_list_full[i][:,0], Σ2_list[i][:,0]).pvalue)
                random_sample_stats.append(stat_row)
            
            
            if np.min(random_sample_stats) < 0.05:
                print("Random sampling failed to draw a representative distribution!")
                print("Run again with a larger sample size.")
                break
           
            # =================================================================
            # Run the GP Regression
            # =================================================================
            model = GPFlowEnsemble.EnsembleGPR(X_list, Y_list, kernel_list, 
                                               X_scaler_list=X_scaler_list, 
                                               Y_scaler_list=Y_scaler_list, 
                                               Y_variance_list = Σ2_list,
                                               Y_variance_scaler_list = Σ2_scaler_list, 
                                               variance_kernels=[gpflow.kernels.RationalQuadratic(active_dims=[0,1,2])],
                                               mean_function_list = mean_function_list,
                                               jitter=1e-4)
            model.optimize()
            
            
           
            
            
            # =================================================================
            # Predict values for the full grid in parallel
            # =================================================================
            
            # # TESTING
            # mjd2d, lon2d = np.meshgrid(mjd_for3d, lon_for3d, indexing='ij')
            # lat2d, lon2d = np.meshgrid(np.interp(mjd_for3d, self.ephemeris['omni']['time'].mjd, self.ephemeris['omni']['lat_c'].to(u.deg).value), lon_for3d, indexing='ij')
            # X_test = np.hstack([mjd2d.flatten()[:,None], 
            #                     lon2d.flatten()[:,None],
            #                     lat2d.flatten()[:,None]])
            # f_mu, f_sigma2 = model.predict_f(unscaled_X = X_test)
            
            
            # fig, axs = plt.subplots(ncols=2, figsize=(8,5))
            
            # im0 = axs[0].pcolormesh(mjd_for3d, lon_for3d, 
            #                         self.boundaryDistributions['omni']['U_sigma_grid'].T)
            # c0 = plt.colorbar(im0, ax=axs[0])
            # im1 = axs[1].pcolormesh(mjd_for3d, lon_for3d, 
            #                         np.sqrt(f_sigma2).reshape(mjd2d.shape).T)
            # c1 = plt.colorbar(im1, ax=axs[1])
               
            # plt.show()
            
            # f_samples = model.predict_f_samples(unscaled_X = X_test, num_samples=10)
            
            # Make sure lon changes most rapidly
            mjd3d, lat3d, lon3d = np.meshgrid(mjd_for3d, 
                                              lat_for3d, 
                                              lon_for3d, 
                                              indexing='ij')
            lon3d +=  (mjd_for3d - self.starttime.mjd)[:,None,None] * 360
            X_flat = np.hstack([mjd3d.flatten()[:,None], 
                                lon3d.flatten()[:,None] ,
                                lat3d.flatten()[:,None]])
            
            if sample_grid is True:
                
                f_mu, f_sigma2 = model.predict_f(unscaled_X=X_flat, cpu_fraction=0.75, chunk_size=2000)
                f3d_mu = f_mu.reshape(mjd3d.shape)
                f3d_sigma = np.sqrt(f_sigma2).reshape(mjd3d.shape)
            else:
                f3d_mu = np.full(mjd3d.shape, -1)
                f3d_sigma = np.full(mjd3d.shape, -1)
            
            # Add to dictionaries
            # all_summaries.update({target_var: {'mu': val_mu_mu, 'sigma': np.sqrt(val_mu_sig**2 + val_sig_mu**2)}})
            all_summaries.update({target_var+'_mu_grid': f3d_mu,
                                  target_var+'_sigma_grid': f3d_sigma})
            
            all_models.update({target_var: model})
            
        return all_summaries, all_models

    
    def sample_boundaryDistribution3D(self, at=None, target_variables=['U'], num_samples=100, chunk_size=5000, cpu_fraction=0.8):
        """
        This function samples the 3D boundary distribution in 2D, returning a 
        huxt-input-like array. This could maybe have a better name.
        """
        
        from scipy.interpolate import RegularGridInterpolator
        
        summary = self.boundaryDistributions3D.copy()
        samples = [self.boundaryDistributions3D.copy() for _ in range(num_samples)]
        
        # Rescale all coordinates
        mjd = self.boundaryDistributions3D['t_grid']
        lon = self.boundaryDistributions3D['lon_grid']
        lat = np.interp(mjd, self.solar_wind['mjd'],
                        self.ephemeris[at]['lat_c'].to(u.deg).value)
        
        # Construct 2, 2D grid
        mjd2d, lon2d, = np.meshgrid(mjd, lon, indexing='ij')
        lat2d, lon2d, = np.meshgrid(lat, lon, indexing='ij')
        
        # # Longitude needs to be non-circular for periodic trends to vary
        # # Tie it to self.starttime and mjd to keep it consistent
        # lon2d += (mjd - self.starttime.mjd)[:,None] * 360
        
        # Construct 1D list of coordinates
        X = np.hstack([mjd2d.flatten()[:, None], 
                       lon2d.flatten()[:, None],
                       lat2d.flatten()[:, None]])
        
        # Handle GP and extend differently
        for target_var in target_variables:
            if self.boundaryModels[target_var] is not None :
                # !!!! Catch exceptions better...
                if at not in self.availableSources:
                    breakpoint()
                
                # Plug these into the model for samples
                var_mu, var_sigma2 = self.boundaryModels[target_var].predict_f(
                    unscaled_X=X, chunk_size=chunk_size, cpu_fraction=cpu_fraction)
                var2d_mu = var_mu[:,0].reshape(lon2d.shape)
                var2d_sigma2 = var_sigma2[:,0].reshape(lon2d.shape)
                _ = summary.pop('lat_grid')
                summary[target_var+'_mu_grid'] = var2d_mu
                summary[target_var+'_sigma_grid'] = np.sqrt(var2d_sigma2)
                
                if num_samples > 0:
                    var_samples = self.boundaryModels[target_var].predict_f_samples(
                        unscaled_X=X, num_samples=num_samples, chunk_size=chunk_size, cpu_fraction=cpu_fraction)
                    for i, var_sample in enumerate(var_samples):
                        _ = samples[i].pop('lat_grid')
                        _ = samples[i].pop(target_var+'_mu_grid')
                        _ = samples[i].pop(target_var+'_sigma_grid')
                        samples[i][target_var] = var_sample.reshape(lon2d.shape)
                
            else:
                
                # Rescale all coordinates                
                # X[:,1] -= np.tile((mjd - self.starttime.mjd)[:,None] * 360, 121).flatten()
                interp_mu = RegularGridInterpolator((self.boundaryDistributions3D['t_grid'],
                                                        self.boundaryDistributions3D['lon_grid'], 
                                                        self.boundaryDistributions3D['lat_grid']), 
                                                        self.boundaryDistributions3D[target_var+'_mu_grid'])
                var2d_mu = interp_mu(X).reshape(lon2d.shape)
                
                interp_sigma = RegularGridInterpolator((self.boundaryDistributions3D['t_grid'],
                                                        self.boundaryDistributions3D['lon_grid'], 
                                                        self.boundaryDistributions3D['lat_grid']), 
                                                        self.boundaryDistributions3D[target_var+'_sigma_grid'])
                
                var2d_sigma = interp_sigma(X).reshape(lon2d.shape)
                
                summary[target_var+'_mu_grid'] = var2d_mu
                summary[target_var+'_sigma_grid'] = var2d_sigma
                
                for i in range(num_samples):
                    _ = samples[i].pop('lat_grid')
                    _ = samples[i].pop(target_var+'_mu_grid')
                    _ = samples[i].pop(target_var+'_sigma_grid')
                    samples[i][target_var] = var2d_mu
                    
                
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
    
    
    # def _rescale_2DBoundary(self, bound, target_reduction=None, target_size=None):
    #     from scipy import ndimage
    #     from skimage.transform import rescale
    #     from skimage.measure import block_reduce
    #     from scipy.interpolate import RegularGridInterpolator
        
    #     data_shape = bound['U_mu_grid'].shape
        
    #     if target_reduction is None and target_size is None:
    #         target_reduction = 0.25
    #     elif target_reduction is not None:
    #         zoom_scale = np.sqrt(target_reduction)
    #     else:
    #         zoom_scale = np.sqrt(target_size/np.product(data_shape))
        
    #     new_bound = {}
    #     for key, val in bound.items():
            
    #         # Create a mask for valid (non-NaN) pixels
    #         mask = ~np.isnan(val)
    #         val_clean = np.where(mask, val, 0.0)
            
    #         # Resize both image and mask
    #         val_rescaled = rescale(val_clean, zoom_scale, 
    #                               anti_aliasing=True, preserve_range=True)
    #         mask_rescaled = rescale(mask.astype(float), zoom_scale, 
    #                               anti_aliasing=True, preserve_range=True)
            
    #         new_val = val_rescaled/mask_rescaled
    #         new_val[~mask_rescaled.astype(bool)] = np.nan
                
    #         new_bound[key] = new_val
        
        
    #     # Estimate noise 
    #     noise_variance = {}
    #     for key, val in new_bound.items():
    #         if len(val.shape) == 2:
    #             interp = RegularGridInterpolator(
    #                 (new_bound['lon_grid'], new_bound['t_grid']), 
    #                 val,
    #                 bounds_error=False)
            
    #             lon2d, t2d = np.meshgrid(bound['lon_grid'], bound['t_grid'], indexing='ij')
    #             upscaled = interp(np.column_stack([lon2d.flatten(), t2d.flatten()])).reshape(lon2d.shape)
    #             difference = upscaled - bound[key]
                
    #             noise_variance[key] = np.nanpercentile(difference, 95)
        
    #     return new_bound, noise_variance
                                         
    # =========================================================================
    # Utility Functions 
    # (that could be separated from this file with no loss of generalization 
    # or context)
    # =========================================================================
    def _getChunksInTime(self, df, delta=90 * u.day, overlap=10*u.day):
        
        # We want each chunk to be as close to delta in length as possible
        # And to overlap on each side by overlap
        total_span = self.simstoptime - self.simstarttime
        # overlap = 10 * u.day
        core_length = delta - 2 * overlap
        approx_chunks = (total_span - overlap) / (core_length + overlap)
        
        # Round the number of chunks down, unless the result would be <1
        n_chunks = int(np.floor(approx_chunks))
        if n_chunks < 1:
            n_chunks = 1
            
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
# # @wrap_non_picklable_objects
# class CustomMeanFunction(gpflow.functions.MeanFunction):
#     def __init__(self, X, Y):
        
#         self.X = X
#         self.Y = Y
        
#         self.bins_d2 = np.linspace(-3, 3, 100)
#         self.bins_d2_indx = np.digitize(X[:,2], self.bins_d2)
        
#         self.bins_d1 = np.linspace(-3, 3, 200)
#         self.bins_d1_indx = np.digitize(X[:,1], self.bins_d1)       
        
#         self.mean = np.zeros([200,100])
#         for i in range(200):
#             for j in range(100):
#                 indx = (self.bins_d1_indx == i) & (self.bins_d2_indx == j)
#                 if indx.any():
#                     self.mean[i,j] = np.mean(Y[indx])
                    
#         from scipy.interpolate import RegularGridInterpolator
#         self.interp = RegularGridInterpolator((self.bins_d1, self.bins_d2), self.mean)

#     @inherit_check_shapes
#     def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
#         result = tf.numpy_function(self.interp, [X[:,1:]], tf.float64)[:,None]
#         return result
# from scipy.interpolate import RegularGridInterpolator

class LinearAverage(gpflow.functions.MeanFunction):
    def __init__(self, X, Y):
        
        # In this case, longitude is a hybrid longitude-temporal measure
        # So the longitude alone is sufficient to return a mean
        X1 = np.round(X[:,1], decimals=9)
        uX1 = np.unique(np.sort(X1))
        nuX1 = uX1.shape[0]

        #
        mean = np.full(nuX1, np.nan)
        for j in range(len(uX1)):
            indx = (X1 == uX1[j])
            mean[j] = np.mean(Y[indx])
        
        self.linear_abcissa = uX1
        self.linear_mean = mean
        return
    
    @inherit_check_shapes
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        X1 = X[:,1]
        
        result = tf.numpy_function(np.interp, [X1, self.linear_abcissa, self.linear_mean], tf.float64)[:,None]
        return result
        
class SpatialAverage(gpflow.functions.MeanFunction):
    def __init__(self, X, Y):
        
        # This X parsing is not particularly robust...
        X0 = np.round(X[:,0], decimals=9)
        # Longitude spans [0,360], so catch all 360s missed by mod
        X1 = np.round(X[:,1], decimals=9)
        
        # Get the unique mjds (0) and lons (1), ignoring lats
        uX0 = np.unique(np.sort(X0))
        uX1 = np.unique(np.sort(X1))
        
        nuX0 = uX0.shape[0]
        nuX1 = uX1.shape[0]
        
        # Generate a mean value for each of point on the grid
        mean = np.full([nuX0,nuX1], np.nan)
        for i in range(len(uX0)):
            for j in range(len(uX1)):
                indx = (X0 == uX0[i]) & (X1 == uX1[j])
                mean[i,j] = np.mean(Y[indx])
        
        # Store the interpolator locally
        self.interp = RegularGridInterpolator(
            (uX0, uX1), mean, 
            bounds_error=False, fill_value=None, method='nearest')
        return
    @inherit_check_shapes
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        
        result = tf.numpy_function(self.interp, [X[:,0:2]], tf.float64)[:,None]
        return result 
        