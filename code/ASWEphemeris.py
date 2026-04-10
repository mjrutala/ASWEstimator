#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:12:46 2026

@author: mrutala
"""
# Import here to avoid conflict with HDF5-only usage requirements
from astroquery.jplhorizons import Horizons
from astropy.coordinates import SkyCoord
from astropy.coordinates import HeliocentricMeanEcliptic
from sunpy.coordinates import frames
import pandas as pd
import astropy.units as u
from astropy.time import Time, TimeDelta
import os
import numpy as np

def ephemeris(target, times, ephemeris_dir=''):
    
    # Standardize inputs
    target = target.upper()
    
    # Swapping the keys and values would make more intuitive sense, but
    # this allows better multiple-pattern-matching for spacecraft
    body_id_dict = {"199": ["MERCURY"],
                    "299": ["VENUS"],
                    "399": ["EARTH"],
                    "499": ["MARS"],
                    "599": ["JUPITER"],
                    "699": ["SATURN"],
                    "799": ["URANUS"],
                    "899": ["NEPTUNE"],
                    "-55": ["ULYSSES"],
                    "-61": ["JUNO"],
                    "-96": ["PARKER SOLAR PROBE", "PARKER", "PSP"],
                    "-144": ["SOLAR ORBITER", "SOLO"],
                    "-202": ["MAVEN"],
                    "-234": ["STEREO A", "STA"],
                    "-235": ["STEREO B", "STB"]
                    }
    body_id = ''
    for key, names in body_id_dict.items():
        if target in names:
            body_id = key
    if body_id == '':
        # Default to Earth
        target = 'EARTH'
        body_id = 399
    
    # 4 hour steps, rounded to nearest day to ensure the same hours are sampled
    epoch_dict = {'start': times[0].iso[:-12] + '00:00:00', 
                  'stop': (times[-1] + 1*u.day).iso[:-12] + '00:00:00', 
                  'step': '4h'}
    
    # Check if there's a file locally
    filename = 'body{}_ephemeris_fromHorizons.zip'.format(body_id)
    filepath = '/'.join(ephemeris_dir.split('/')[0:-1]) + '/' + filename
    
    if os.path.isfile(filepath):
        # If a file exists, read it and check overlap
        existing_ephemeris = pd.read_csv(filepath)
        epoch_start = Time(epoch_dict['start']).mjd
        epoch_stop = Time(epoch_dict['stop']).mjd
        
        # If the file contains all ephemeris, we can skip the Horizons query
        overlap = existing_ephemeris.query("@epoch_start <= mjd < @epoch_stop")
        if len(overlap) == ((epoch_stop-epoch_start) * 24 / 4):
            need_ephemeris = False
        else:
            need_ephemeris = True
    else:
       existing_ephemeris = False
       need_ephemeris = True
       
    if need_ephemeris:
        print("Downloading ephemeris from JPL Horizons...")
        # Horizons will grab ICRF coords given location @0
        pos = Horizons(id = body_id, location = '@0', epochs = epoch_dict)
        vec = pos.vectors(refplane='earth').to_pandas()
        epoch_time = Time(vec['datetime_jd'], format='jd')
        
        icrf_coords = SkyCoord(vec['x'].to_numpy() * u.AU,
                               vec['y'].to_numpy() * u.AU,
                               vec['z'].to_numpy() * u.AU,
                               obstime = epoch_time,
                               frame='icrs', representation_type='cartesian')
        
        # Transform to HEEQ frame
        heeq_coords = icrf_coords.transform_to(frames.HeliographicStonyhurst)
        
        # Transform to HAE frame
        hae_coords = heeq_coords.transform_to(HeliocentricMeanEcliptic)
        
        # Transform to Heliographic Carrington (CARR)
        carr_coords = heeq_coords.transform_to(frames.HeliographicCarrington(observer='self'))
        
        new_ephemeris = pd.DataFrame(data = {'time': epoch_time.datetime,
                                             'mjd': epoch_time.mjd,
                                             'r_heeq': heeq_coords.radius.to(u.solRad).value,
                                             'lon_heeq': heeq_coords.lon.to(u.rad).value,
                                             'lat_heeq': heeq_coords.lat.to(u.rad).value,
                                             'r_hae': hae_coords.distance.to(u.solRad).value,
                                             'lon_hae': hae_coords.lon.to(u.rad).value,
                                             'lat_hae': hae_coords.lat.to(u.rad).value,
                                             'r_carr': carr_coords.radius.to(u.solRad).value,
                                             'lon_carr': carr_coords.lon.to(u.rad).value,
                                             'lat_carr': carr_coords.lat.to(u.rad).value})
        
        # Merge body_df with existing_df, if it exists
        if existing_ephemeris is not False:
            # Merge existing_ephemeris and new_ephemeris
            existing_ephemeris = pd.concat([existing_ephemeris, new_ephemeris], axis='index')
            existing_ephemeris.drop_duplicates(['mjd'], inplace=True, ignore_index=True)
            existing_ephemeris.sort_values(by=['mjd'], axis='index', inplace=True)
            existing_ephemeris.reset_index(inplace=True, drop=True)
            
        else:
            existing_ephemeris = new_ephemeris
        
        # Save the new results
        existing_ephemeris.to_csv(filepath, compression='zip', index=False)
    
    # Now get observers coordinates
    all_time = Time(existing_ephemeris['mjd'], format='mjd')
    # Pad out the window to account for single values being passed. 
    dt = TimeDelta(2 * 60 * 60, format='sec')
    id_epoch = (all_time >= (times.min() - dt)) & (all_time <= (times.max() + dt))
    epoch_time = all_time[id_epoch]
    
    eph = {}
    eph['time']     = times
    if len(epoch_time.jd) == 0:
        eph['r']        = np.full(len(times), np.nan)
        eph['lon']      = np.full(len(times), np.nan)
        eph['lat']      = np.full(len(times), np.nan)

        eph['r_hae']    = np.full(len(times), np.nan)
        eph['lon_hae']  = np.full(len(times), np.nan)
        eph['lat_hae']  = np.full(len(times), np.nan)

        eph['r_c']      = np.full(len(times), np.nan)
        eph['lon_c']    = np.full(len(times), np.nan)
        eph['lat_c']    = np.full(len(times), np.nan)

    else:
        r = existing_ephemeris['r_heeq'][id_epoch]
        eph['r'] = np.interp(times.jd, epoch_time.jd, r) 
        eph['r'] *= u.solRad

        lon = np.unwrap(existing_ephemeris['lon_heeq'])[id_epoch]
        eph['lon'] = np.interp(times.jd, epoch_time.jd, lon) % (2*np.pi)
        eph['lon'] *= u.rad

        lat = existing_ephemeris['lat_heeq'][id_epoch]
        eph['lat'] = np.interp(times.jd, epoch_time.jd, lat)
        eph['lat'] *= u.rad

        r = existing_ephemeris['r_hae'][id_epoch]
        eph['r_hae'] = np.interp(times.jd, epoch_time.jd, r)
        eph['r_hae'] *= u.solRad

        lon = np.unwrap(existing_ephemeris['lon_hae'])[id_epoch]
        eph['lon_hae'] = np.interp(times.jd, epoch_time.jd, lon) % (2*np.pi)
        eph['lon_hae'] *= u.rad

        lat = existing_ephemeris['lat_hae'][id_epoch]
        eph['lat_hae'] = np.interp(times.jd, epoch_time.jd, lat)
        eph['lat_hae'] *= u.rad

        r = existing_ephemeris['r_carr'][id_epoch]
        eph['r_c'] = np.interp(times.jd, epoch_time.jd, r)
        eph['r_c'] *= u.solRad
        
        lon = np.unwrap(existing_ephemeris['lon_carr'])[id_epoch]
        eph['lon_c'] = np.interp(times.jd, epoch_time.jd, lon) % (2*np.pi)
        eph['lon_c'] *= u.rad

        lat = existing_ephemeris['lat_carr'][id_epoch]
        eph['lat_c'] = np.interp(times.jd, epoch_time.jd, lat)
        eph['lat_c'] *= u.rad
    
    return eph