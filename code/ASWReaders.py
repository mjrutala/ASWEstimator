#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 12:29:25 2025

@author: mrutala
"""
import astropy.units as u
from astropy.time import Time
import datetime
import numpy as np
from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
import scipy
from pathlib import Path
import pandas as pd

def lookup_omni(starttime, stoptime):
    """
    A function to grab and process the most up-to-date OMNI data from 
    COHOWeb directly
    
    Args:
        starttime : datetime for start of requested interval
        endtime : datetime for start of requested interval

    Returns:
        omni: Dataframe of the OMNI timeseries

    """
    # trange = attrs.Time(starttime, endtime)
    # dataset = attrs.cdaweb.Dataset('OMNI_COHO1HR_MERGED_MAG_PLASMA')
    # result = Fido.search(trange, dataset)
    # downloaded_files = Fido.fetch(result, path='../Data/OMNI/')
    
    # omni = TimeSeries(downloaded_files, concatenate=True).to_dataframe()
    # omni = omni.rename(columns = {'ABS_B': 'B',
    #                               'V': 'U'})
    # # Set invalid data points to NaN
    # id_bad = omni['U'] == 9999.0
    # omni.loc[id_bad, 'U'] = np.NaN

    
    # Get the relevant URLs
    skeleton_url = 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni_m{YYYY:4.0f}.dat'
    years_covered = np.arange(starttime.year, stoptime.year+1, 1)
    all_urls = [skeleton_url.format(YYYY=year) for year in years_covered]
    
    # Read each OMNI file, concatenate, and set NaNs 
    null_values = {'year': None, 'doy': None, 'hour': None,
                   'lat_hgi': 9999.9, 'lon_hgi': 9999.9, 
                   'BR': 999.9, 'BT': 999.9, 'BN': 999.9, 'B': 999.9, 
                   'U': 9999., 'U_theta': 999.9, 'U_phi': 999.9,
                   'n': 999.9, 'T': 9999999.}
    df_list = []
    for url in all_urls:
        df = pd.read_csv(url, header=0, sep='\s+', names=null_values.keys())
        df_list.append(df)
    df = pd.concat(df_list, axis='index') 
    df = df.reset_index(drop=True).sort_index()
    for col, null_val in null_values.items():
        null_index = df[col] == null_val
        df.loc[null_index, col] = np.nan
    
    # Add datetime, MJD
    parse_dt = datetime.datetime.strptime
    date_cols = ['year', 'doy', 'hour']
    str_fmt   = '{:04.0f}-{:03.0f} {:02.0f}'
    dt_fmt    = '%Y-%j %H'
    df['dt'] = [parse_dt(str_fmt.format(*row[date_cols]), dt_fmt) 
                      for _, row in df.iterrows()]
    df['mjd'] = Time(df['dt']).mjd

    # create a BX_GSE field that is expected by some HUXt fucntions
    df['BX_GSE'] = -df['BR']
    
    # Limit the dataframe to the requested dates
    omni = df.query('@starttime <= dt < @stoptime')
    # omni = omni.rename(columns={'dt': 'datetime'})
    omni = omni.reset_index(drop=True)
    
    return omni

class SolarWindData:
    def __init__(self, source, start, stop):
        self.input_source = source
        self.source = self.identify_source(source)
        self.start = start
        self.stop = stop
        self.resolution = '1h'
        self.variables = ['U', 'n', 'B', 'Br']
        
        self.path = Path('/Users/mrutala/projects/OHTransients/data/')
        filename = (source + ' 1hr.csv.zip').replace(' ', '_')
        self.filepath = self.path / (filename)
        
        # Generate an empty dataframe to fill with data
        index = pd.date_range(start=start, end=stop, freq=self.resolution)[:-1]
        self.data = pd.DataFrame(index = pd.DatetimeIndex(index, name='Epoch'), 
                                 columns = self.variables,
                                 dtype=np.float64)
        
        
        self.search()
        
    @property
    def starttime(self):
        return Time(self.start)
    
    @property
    def stoptime(self):
        return Time(self.stop)
            
    def search(self):
        
        # Get the correct reader function
        fn = self.identifyDatasetLookupFunction()
        
        # if filepath does not exist, download all data
        # if filepath does exist, check the coverage
        
        if self.filepath.exists():
            # Load the existing data
            data_df = pd.read_csv(self.filepath, index_col='Epoch')
            data_df.index = pd.DatetimeIndex(data_df.index)
            data_df.query("@self.start <= index < @self.stop", inplace=True)
            
            # Find datetimes missing from dt_index
            missing_set = set(self.data.index) - set(data_df.index)
            if len(missing_set) > 0:
                
                missing_index = pd.DatetimeIndex(np.sort(list(missing_set)))
                
                # Split missing_index into continuous (1 hour res) chunks
                breaks = np.where(np.diff(missing_index).astype('float64')/1e9 > 3600)
                chunks = np.split(missing_index, breaks[0]+1)
                
                # Download the missing data
                partial_dfs = [data_df]
                for chunk in chunks:
                    partial_df = fn(chunk[0], chunk[-1]+datetime.timedelta(hours=1))
                    partial_dfs.append(partial_df)
                    
                data_df = pd.concat(partial_dfs, axis='index')
                data_df.sort_index(inplace=True)
                
            # if missing_set has 0 len, we already have what we need
            else:
                pass
            
        else:
            data_df = fn(self.start, self.stop)
        
        # very minor post-processing to decrease NaNs
        data_df = data_df.interpolate('time', limit=6, 
                                      limit_direction='both', limit_area='inside')
        
        # Assign this to self.data
        for col in self.data.columns:
            if col in data_df.columns:
                self.data.loc[:,col] = pd.to_numeric(data_df.loc[:,col])

        # Wrap up by updating/creating the csv
        self.update_csv(self.data)
        
        return data_df
    
    def update_csv(self, df):
        if self.filepath.exists():
            # Read the existing df
            existing_df = pd.read_csv(self.filepath, index_col='Epoch')
            existing_df .index = pd.DatetimeIndex(existing_df.index)
            
            # Concatenate with the new one
            combined_df = pd.concat([existing_df, df])
            
            # Ensure monotonic index
            combined_df.sort_index(inplace=True)
            combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first'), :]
            
            # Save
            combined_df.to_csv(self.filepath, index=True, compression='zip')
            
        else:
            df.to_csv(self.filepath, index=True, compression='zip')


    def identify_source(self, input_source):
        
        source_aliases = {'omni': ['omni2'],
                          'parker solar probe': ['parkersolarprobe', 'psp'],
                          'stereo a': ['stereoa', 'sta'],
                          'stereo b': ['stereob', 'stb'],
                          # 'helios1': ['helios1', 'helios 1'],
                          # 'helios2': ['helios2', 'helios 2'],
                          'ulysses': ['uy'],
                          # 'maven': ['maven'],
                          'voyager 1': ['voyager1'],
                          'voyager 2': ['voyager2']} 
        
        source = None
        for possible_source, aliases in source_aliases.items():
            if input_source in [*aliases, possible_source]:
                source = possible_source
        
        return source
   
    def identifyDatasetLookupFunction(self):
        LookupFunctions = {'omni': self.omni,
                           'parker solar probe': self.parkersolarprobe,
                           'stereo a': self.stereoa,
                           'stereo b': self.stereob,
                           # 'helios1': ['helios1', 'helios 1'],
                           # 'helios2': ['helios2', 'helios 2'],
                           'ulysses': self.ulysses,
                           # 'maven': ['maven'],
                           'voyager 1': self.voyager1,
                           'voyager 2': self.voyager2,}
        return LookupFunctions[self.source]
    
    def _fetch(self, source, datasetID, start, stop):
        # Mute warnings
        import warnings
        warnings.filterwarnings("ignore")
           
        # Simulation time range in Fido format
        # timerange = attrs.Time(self.starttime, self.stoptime)
        timerange = attrs.Time(Time(start), Time(stop))
        
        expected_index = pd.date_range(start, stop, freq=self.resolution)
        expected_df = pd.DataFrame(index = expected_index[:-1])

        # 
        dataset = attrs.cdaweb.Dataset(datasetID)
        result = Fido.search(timerange, dataset)
        downloaded_files = Fido.fetch(result, path = str(self.path / source.upper()) + '/{file}')
        
        # Check for errors and retry if needed
        retry_count = 0
        while len(downloaded_files.errors) > 0:
            downloaded_files = Fido.fetch(downloaded_files)
            retry_count += 1
            if retry_count > 10:
                breakpoint()
        
        # If there's any files to download, download them and get the dataframe
        if len(downloaded_files) > 0:
            # Combine the data for this source into one df
            try:
                df = TimeSeries(downloaded_files, concatenate=True).to_dataframe()
            
            except ValueError:
                # Some time series have an extra, errant 0 in a data column, which breaks the TimeSeries assignment
                # In these cases, we can do this manually...
                # (e.g. 'STA_L2_PLA_1DMAX_1MIN')
                import cdflib
                partial_dfs = []
                for dfile in downloaded_files:
                    temp1 = cdflib.CDF(dfile)
                    temp1vars = temp1.cdf_info().zVariables
                    
                    # Find the typical length of variables in this cdf file
                    var_lengths = []
                    for v in temp1vars:
                        try:    var_lengths.append(len(temp1.varget(v)))
                        except: var_lengths.append(0)
                    typical_length, _ = scipy.stats.mode(var_lengths)
                    
                    # Remove all elements with atypical length
                    vars_to_remove = np.array(temp1vars)[var_lengths != typical_length]
                    for vr in vars_to_remove:
                        temp1vars.remove(vr)

                    # Add columns with appropriate lengths to df
                    temp1_df = pd.DataFrame(columns = temp1.cdf_info().zVariables, index = np.arange(typical_length))
                    for v in temp1vars:
                        if np.shape(temp1.varget(v)) == (typical_length,):
                            # Get the fill value to replace w/ NaN
                            fill_val = temp1.varattsget(v)['FILLVAL']
                            nan_indx = temp1.varget(v) == fill_val
                            temp1_df.loc[~nan_indx,v] = temp1.varget(v)[~nan_indx]
                            
                    if temp1.varattsget('epoch')['UNITS'] == 'ms':
                        t = temp1_df['epoch'] / (1000 * 60 * 60 * 24) * u.day
                        
                    if temp1.varattsget('epoch')['FIELDNAM'] == 'Time since 0 AD':
                        t0 = Time("0000-01-01", format='isot', scale='utc')
                        
                    temp1_df.index = pd.DatetimeIndex((t0 + t).to_datetime())
                    
                    temp2_df = temp1_df.resample(self.resolution).mean()
                        
                    partial_dfs.append(temp2_df)
                
                df = pd.concat(partial_dfs)

            # Truncate to the appropriate start/stop time
            df = df.resample(self.resolution).mean()
            df = df.query("@start <= index < @stop")
            
            # Merge onto existing df index
            df = expected_df.join(df)
           
        else:
            df = expected_df
        
        return df
    
    def omni(self, start, stop):
        
        # Source and Dataset names
        source = 'omni'
        fetch_datasetID = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'
        
        
        # Which columns to keep, and what to call them
        column_map = {'V': 'U',
                      'N': 'n',
                      'ABS_B': 'B',
                      'BR': 'Br'}
        # Get the data
        data_df = self._fetch(source, fetch_datasetID, start, stop)
        
        # If data exists, rename columns & add distance
        if len(data_df.columns) > 0:
            data_df.rename(columns = column_map, inplace=True)
            # Add the radial distance
            data_df = data_df[column_map.values()]
            data_df.index.name = 'Epoch'
            
        return data_df
    
    def stereoa(self, start, stop):
        # Source and Dataset names
        source = 'stereo a'
        
        ordered_ID_list = [['STA_COHO1HR_MERGED_MAG_PLASMA'],
                           ['STA_L2_PLA_1DMAX_1MIN', 'STA_L1_MAG_RTN']]
        
        columns_maps = {
            'STA_COHO1HR_MERGED_MAG_PLASMA':    {'B': 'B',
                                                 'BR': 'Br',
                                                 'plasmaSpeed': 'U',
                                                 'plasmaDensity': 'n'},
            'STA_L2_PLA_1DMAX_1MIN':            {'proton_bulk_speed': 'U',
                                                 'proton_number_density': 'n'},
            'STA_L1_MAG_RTN':                   {'BFIELD_3': 'B',
                                                 'BFIELD_0': 'Br'}
                        }
        # Descend the data options until we have all the data
        # As there are some gaps which persist across data products,
        # we don't want to simply check for 'full' coverage
        # Instead: if either start or stop have data, consider this sufficient
        for datasetIDs in ordered_ID_list:
            partial_dfs = []
            for dID in datasetIDs:
                test_df = self._fetch(source, dID, start, stop)
                if len(test_df.columns) > 0:
                    test_df = test_df.rename(columns = columns_maps[dID])
                    test_df = test_df[columns_maps[dID].values()]
                    
                partial_dfs.append(test_df)
            
            
            combined_df = pd.concat(partial_dfs, axis='columns')
            
            if ~combined_df.iloc[0].isna().any() or ~combined_df.iloc[-1].isna().any():
                break
        
        # If data exists, rename columns
        if combined_df is not None:
            combined_df.index.name = 'Epoch'

        return combined_df
        
    def stereob(self, start, stop):
        # Source and Dataset names
        source = 'stereo b'
        fetch_datasetID = 'STB_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'plasmaSpeed': 'U',
                      'plasmaDensity': 'n',
                      'B': 'B',
                      'BR': 'Br',}
        
        # Get the data
        data_df = self._fetch(source, fetch_datasetID, start, stop)
        
        # If data exists, rename columns
        if len(data_df.columns) > 0:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
            data_df.index.name = 'Epoch'
        
        return data_df    
    
    def voyager1(self, start, stop):
        
        # Source and Dataset names
        source = 'voyager 1'
        fetch_datasetID = 'VOYAGER1_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'V': 'U',
                      'ABS_B': 'B',
                      'BR': 'Br'}
        
        # Get the data
        data_df = self._fetch(source, fetch_datasetID, start, stop)
        
        # If data exists, rename columns
        if len(data_df.columns) > 0:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
            data_df.index.name = 'Epoch'
        
        return data_df
    
    def voyager2(self, start, stop):
        
        # Source and Dataset names
        source = 'voyager 2'
        fetch_datasetID = 'VOYAGER2_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'V': 'U',
                      'ABS_B': 'B',
                      'BR': 'Br',}
        
        # Get the data
        data_df = self._fetch(source, fetch_datasetID, start, stop)
        
        # If data exists, rename columns
        if len(data_df.columns) > 0:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
            data_df.index.name = 'Epoch'
        
        return data_df
    
    def ulysses(self, start, stop):
        # Source and Dataset names
        source = 'ulysses'
        fetch_datasetID = 'UY_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'plasmaFlowSpeed': 'U',
                      'ABS_B': 'B',
                      'BR': 'Br',}
        
        # Get the data
        data_df = self._fetch(source, fetch_datasetID, start, stop)
        
        # If data exists, rename columns
        if len(data_df.columns) > 0:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
            data_df.index.name = 'Epoch'
        
        return data_df    
    
    def parkersolarprobe(self, start, stop):
        # Source and Dataset names
        source = 'parker solar probe'
        fetch_datasetID = 'PSP_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'ProtonSpeed': 'U',
                      'B': 'B',
                      'BR': 'Br'}
        
        # Get the data
        data_df = self._fetch(source, fetch_datasetID, start, stop)
        
        # If data exists, rename columns
        if len(data_df.columns) > 0:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
            data_df.index.name = 'Epoch'
            
        return data_df    
    
