#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:02:32 2026

@author: mrutala
"""

from astropy.time import Time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time 
import pandas as pd
import statsmodels as sm
import astropy.units as u
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
import dill as pickle
import matplotlib.dates as mdates

from pathlib import Path
import sys
localpath = Path('/Users/mrutala/projects/ASWEstimator/code/')
sys.path.append(localpath.as_posix())

import ASWEstimator as ASWE
import performance

experiment_dir = Path('/Users/mrutala/projects/ASWEstimator/code/experiments/data/')


start = datetime.datetime(2010, 4, 1)
stop = datetime.datetime(2014, 9, 27)
# stop = datetime.datetime(2010, 5, 1)

# Setup some common params
rmax = 1.1
latmax = 15
latnum = 8

icme_duration   = 3.75 * u.day # conservative duration (Richardson & Cane 2010)
icme_buffer     = 0.25 * u.day # onservative duration (Richardson & Cane 2010)
interp_buffer   = 1.0 * u.day # how much to use in the interpolation

# bg_target_noise = 0.05
# bg_max_chunk_length = 1000
bg_num_samples = 200

# bo_num_samples = 20
# bo_target_reduction = 0.25
# bo_max_chunk_length = 1000
# bo_SGPR = 1.0

# %%
# =============================================================================
# Step 2) Get background (ambient) solar wind data + transients
# =============================================================================
test_background_file = experiment_dir / 'test_background.pkl'
if not test_background_file.exists():
    
    inputs = ASWE.ASWEstimator(start, stop, rmax=rmax, latmax=latmax)
    
    # Specify typically-hard-coded parameters for clarity
    inputs.boundarySources              = ['omni', 'stereo a', 'stereo b']
    inputs._icme_duration               = icme_duration
    inputs._icme_duration_buffer        = icme_buffer 
    inputs._icme_interp_buffer          = interp_buffer
    
    # 
    inputs.getSolarWind()
    inputs.getTransients()

    # inputs.makeBackgroundDistribution(
    #     # inducing_variable   = False,
    #     GP                  = True,
    #     # target_noise        = bg_target_noise,
    #     # max_chunk_length    = bg_max_chunk_length,
    #     n_samples           = bg_num_samples
    #     )
    inputs.save(test_background_file)
else:
    inputs = ASWE.ASWEstimator.load(test_background_file)

# %%
# Figure for paper

# Set subplot sizes carefully
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 8

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

fig, axd = plt.subplot_mosaic("""
                              a
                              b
                              c
                              d
                              e
                              """,
                              height_ratios=[1, 1, 1, 1, 1],
                              figsize=[20/3, 8/3])
plt.subplots_adjust(hspace=0.15, wspace=0.1,
                    left=0.075, bottom=0.15, 
                    right=1-0.01, top=1-0.01)
c_omni  = '#90be6d'
c_sta   = '#43aa8b'
c_stb   = '#577590'

solar_wind_daily = inputs.solar_wind.groupby(inputs.solar_wind['mjd'].round()).mean()

axd['a'].plot(solar_wind_daily['mjd'], solar_wind_daily['omni']['U'],
              color=c_omni, lw=0.75, label='OMNI/Earth')
axd['a'].fill_between(inputs.solar_wind['mjd'], 1250, 
                      where=inputs.solar_wind['omni']['ICME'], 
                      color='black', alpha=0.33, lw=0)
axd['a'].annotate('OMNI', (0,1), (60,-10), 
                  'axes fraction', 'offset pixels', ha='left', va='top', fontsize=SMALL_SIZE, 
                  bbox={'boxstyle':"round,pad=0.05", 'fc':(c_omni, 0.33), 'ec':c_omni, 'lw':0.75})

axd['b'].plot(solar_wind_daily['mjd'], solar_wind_daily['stereo a']['U'],
              color=c_sta, lw=0.75, label='STEREO-A')
axd['b'].fill_between(inputs.solar_wind['mjd'], 1250, 
                      where=inputs.solar_wind['stereo a']['ICME'], 
                      color='black', alpha=0.33, lw=0)
axd['b'].annotate('STEREO-A', (0,1), (60,-10), 
                  'axes fraction', 'offset pixels', ha='left', va='top', fontsize=SMALL_SIZE, 
                  bbox={'boxstyle':"round,pad=0.05", 'fc':(c_sta, 0.33), 'ec':c_sta, 'lw':0.75})

axd['c'].plot(solar_wind_daily['mjd'], solar_wind_daily['stereo b']['U'],
              color=c_stb, lw=0.75, label='STEREO-B')
axd['c'].fill_between(inputs.solar_wind['mjd'], 1250, 
                      where=inputs.solar_wind['stereo b']['ICME'], 
                      color='black', alpha=0.33, lw=0)
axd['c'].annotate('STEREO-B', (0,1), (60,-10), 
                  'axes fraction', 'offset pixels', ha='left', va='top', fontsize=SMALL_SIZE, 
                  bbox={'boxstyle':"round,pad=0.05", 'fc':(c_stb, 0.33), 'ec':c_stb, 'lw':0.75})

# Dummy lines for legend"
axd['a'].plot([], [], color=c_sta, label='STEREO-A')
axd['a'].plot([], [], color=c_stb, label='STEREO-B')
axd['a'].fill_between([], 0, color='black', alpha=0.33, lw=0, label='ICMEs')
# l = axd['a'].legend(bbox_to_anchor=(0.075, 0.92, 0.915, 0.06), loc='lower left',
#                     mode="expand", borderaxespad=0., edgecolor='black', ncol=4, 
#                     bbox_transform=fig.transFigure)
# for line in l.get_lines():
#     line.set_linewidth(2.0)

# omni_lon_unwrapped = np.unwrap(inputs.ephemeris['omni']['lon_c'].value, discont=np.pi, period=2*np.pi)
# sta_lon_unwrapped = np.unwrap(inputs.ephemeris['stereo a']['lon_c'].value, discont=np.pi, period=2*np.pi)
# stb_lon_unwrapped = np.unwrap(inputs.ephemeris['stereo b']['lon_c'].value, discont=np.pi, period=2*np.pi)
# axd['d'].plot(inputs.solar_wind['mjd'], np.rad2deg(omni_lon_unwrapped - omni_lon_unwrapped),
#               color=c_omni, lw=1)
# axd['d'].plot(inputs.solar_wind['mjd'], np.rad2deg(sta_lon_unwrapped - omni_lon_unwrapped),
#               color=c_sta, lw=1)
# axd['d'].plot(inputs.solar_wind['mjd'], np.rad2deg(stb_lon_unwrapped - omni_lon_unwrapped),
#               color=c_stb, lw=1)
# axd['d'].set(ylim=[-180, 180], yticks=[-120, 0, 120])
omni_lon_unwrapped = np.unwrap(inputs.ephemeris['omni']['lon_c'].value, discont=np.pi, period=2*np.pi)
sta_lon_unwrapped = np.unwrap(inputs.ephemeris['stereo a']['lon_c'].value, discont=np.pi, period=2*np.pi)
stb_lon_unwrapped = np.unwrap(inputs.ephemeris['stereo b']['lon_c'].value, discont=np.pi, period=2*np.pi)
axd['d'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['omni']['lon_c'].to(u.deg).value,
              color=c_omni, lw=0.75)
axd['d'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo a']['lon_c'].to(u.deg).value,
              color=c_sta, lw=0.75)
axd['d'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo b']['lon_c'].to(u.deg).value,
              color=c_stb, lw=0.75)
axd['d'].set(ylim=[0, 360], yticks=[0, 180, 360], yticklabels=[0, 180, 360])

axd['e'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['omni']['lat_c'].to(u.deg),
              color=c_omni, lw=1)
axd['e'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo a']['lat_c'].to(u.deg),
              color=c_sta, lw=1)
axd['e'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo b']['lat_c'].to(u.deg),
              color=c_stb, lw=1)
axd['e'].set(ylim=[-12, 12], yticks=[-8, 0, 8])

# axd['f'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['omni']['r'].to(u.au).value, 
#               color=c_omni, lw=1)
# axd['f'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo a']['r'].to(u.au).value,
#               color=c_sta, lw=1)
# axd['f'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo b']['r'].to(u.au).value, 
#               color=c_stb, lw=1)
# axd['f'].set(ylim=[0.9, 1.15], yticks=[0.9, 1.0, 1.1])


axd['b'].set(ylabel = 'Flow Speed $U$ [km/s]')
axd['d'].set(ylabel = '$\lambda$ [$^{\circ}$]')
axd['e'].set(ylabel = '$\phi$ [$^{\circ}$]')
# axd['f'].set(ylabel = 'r [AU]')
fig.align_ylabels()

mjd2dt = lambda x: Time(x, format='mjd').datetime64
dt2mjd = lambda x: Time(mdates.num2date(x), format='datetime').mjd
x2 = axd['e'].secondary_xaxis(0.07, functions=(mjd2dt, dt2mjd), transform=fig.transFigure)
x2.xaxis.set_minor_locator(mdates.MonthLocator())
x2.xaxis.set_major_locator(mdates.YearLocator())

# Get y pos from 
# b = x2.get_xticklabels()[0].get_window_extent() 
# fig.transFigure.inverted().transform((b.x0, b.y0+0.5*b.height))
fig.text(0.0, 0.01479167, 'Date [Year]', transform=fig.transFigure, ha='left', va='center')
fig.text(0.0, 0.09479167, 'Date [MJD]', transform=fig.transFigure, ha='left', va='center')

for ax in [axd[key] for key in ['a', 'b', 'c']]:
    ax.set(ylim = [250, 900], yticks=[300, 700])
    
for ax in list(axd.values())[:-1]:
    ax.tick_params('x', which='major', bottom=True)
    
for key, ax in axd.items():
    ax.annotate('({})'.format(key), (0,1), (10,-10), 
                'axes fraction', 'offset pixels', ha='left', va='top', fontsize=BIGGER_SIZE)
    ax.set(xlim=[inputs.starttime.mjd, inputs.stoptime.mjd])
    ax.grid(True, which='major', axis='x', color='black', alpha=0.75, zorder=-99, lw=0.5)
    
plt.savefig(experiment_dir / '1_data_exploration.png', dpi=300)
plt.show()

# %%
# Figure for paper
fig, axd = plt.subplot_mosaic("""
                              abc
                              ddd
                              eee
                              fff
                              """, 
                              height_ratios=[1, 1, 1, 1], 
                              width_ratios=[1, 1, 1], 
                              figsize=[20/3, 10/3])
plt.subplots_adjust(hspace=0, wspace=0.1,
                    left=0.1, bottom=0.125, 
                    right=1-0.05, top=1-0.1)
c_omni  = '#90be6d'
c_sta   = '#43aa8b'
c_stb   = '#577590'

# axd['a'].plot(inputs.solar_wind['mjd'], inputs.solar_wind['omni']['U'],
#               color=c_omni, lw=1, label='OMNI/Earth')
# axd['a'].fill_between(inputs.solar_wind['mjd'], 1250, 
#                       where=inputs.solar_wind['omni']['ICME'], 
#                       color='black', alpha=0.1, lw=0)

# axd['c'].plot(inputs.solar_wind['mjd'], inputs.solar_wind['stereo a']['U'],
#               color=c_sta, lw=1, label='STEREO-A')
# axd['c'].fill_between(inputs.solar_wind['mjd'], 1250, 
#                       where=inputs.solar_wind['stereo a']['ICME'], 
#                       color='black', alpha=0.1, lw=0)

# axd['e'].plot(inputs.solar_wind['mjd'], inputs.solar_wind['stereo b']['U'],
#               color=c_stb, lw=1, label='STEREO-B')
# axd['e'].fill_between(inputs.solar_wind['mjd'], 1250, 
#                       where=inputs.solar_wind['stereo b']['ICME'], 
#                       color='black', alpha=0.1, lw=0)

# for ax in [axd[key] for key in ['a', 'c', 'e']]:
#     ax.set(ylim = [200, 1000], xlim=inputs.solar_wind['mjd'].to_numpy()[[0, -1]])

# # Dummy lines for legend
# axd['a'].plot([], [], color=c_sta, label='STEREO-A', lw=1)
# axd['a'].plot([], [], color=c_stb, label='STEREO-B', lw=1)
# axd['a'].fill_between([], 0, color='black', alpha=0.1, lw=0, label='ICMEs')
# axd['a'].legend(bbox_to_anchor=(0.1, 0.925, 0.85*(3/4), 0.05), loc='lower left',
#                 mode="expand", borderaxespad=0., edgecolor='black', ncol=4, 
#                 bbox_transform=fig.transFigure)

hbins = np.arange(200, 1000, 50)
axd['a'].hist(inputs.solar_wind['omni']['U'], hbins,
              color=c_omni,)
axd['b'].hist(inputs.solar_wind['stereo a']['U'], hbins,
              color=c_sta,)
axd['c'].hist(inputs.solar_wind['stereo b']['U'], hbins,
              color=c_stb)


omni_lon_unwrapped = np.unwrap(inputs.ephemeris['omni']['lon_c'].value, discont=np.pi, period=2*np.pi)
sta_lon_unwrapped = np.unwrap(inputs.ephemeris['stereo a']['lon_c'].value, discont=np.pi, period=2*np.pi)
stb_lon_unwrapped = np.unwrap(inputs.ephemeris['stereo b']['lon_c'].value, discont=np.pi, period=2*np.pi)
axd['d'].plot(inputs.solar_wind['mjd'], np.rad2deg(omni_lon_unwrapped - omni_lon_unwrapped),
              color=c_omni, lw=1)
axd['d'].plot(inputs.solar_wind['mjd'], np.rad2deg(sta_lon_unwrapped - omni_lon_unwrapped),
              color=c_sta, lw=1)
axd['d'].plot(inputs.solar_wind['mjd'], np.rad2deg(stb_lon_unwrapped - omni_lon_unwrapped),
              color=c_stb, lw=1)

axd['e'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['omni']['lat_c'].to(u.deg),
              color=c_omni, lw=1)
axd['e'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo a']['lat_c'].to(u.deg),
              color=c_sta, lw=1)
axd['e'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo b']['lat_c'].to(u.deg),
              color=c_stb, lw=1)

axd['f'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['omni']['r'].to(u.au).value, 
              color=c_omni, lw=1)
axd['f'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo a']['r'].to(u.au).value,
              color=c_sta, lw=1)
axd['f'].plot(inputs.solar_wind['mjd'], inputs.ephemeris['stereo b']['r'].to(u.au).value, 
              color=c_stb, lw=1)


# for ax in axs[0:3]:
#     ax.set(ylim = [250, 1250])
# axs[1].set(ylabel = '$U$ [km s$^{-1}$]')
# axs[3].set(ylabel = '$\lambda$ [$^{\circ}$]', ylim=[-180, 180])
# axs[4].set(ylabel = '$\phi$ [$^{\circ}$]', ylim=[-8, 8])
# axs[5].set(ylabel = 'r [AU]', ylim=[0.9, 1.1])

# for ax, letter in zip(axs, ['a', 'b', 'c', 'd', 'e', 'f']):
#     ax.grid(which='major', axis='x')
#     ax.annotate('({})'.format(letter), (0,1), (1,-1), 
#                 'axes fraction', 'offset fontsize', ha='left', va='top')
    
# axs[5].set(xlabel='MJD [days]', xlim=[inputs.starttime.mjd, inputs.stoptime.mjd])
# fig.align_ylabels()
# # plt.savefig('/Users/mrutala/projects/OHTransients/code/figures/data.png', dpi=300)
plt.show()