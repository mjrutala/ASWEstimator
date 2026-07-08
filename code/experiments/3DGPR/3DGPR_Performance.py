#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:44:00 2025

@author: mrutala
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
# import scipy
# import time 
import pandas as pd
# import statsmodels as sm
import astropy.units as u
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn import metrics
import dill as pickle
from astropy.time import Time
import matplotlib.dates as mdates


from pathlib import Path
import sys
localpath = Path('/Users/mrutala/projects/ASWEstimator/code/')
sys.path.append(localpath.as_posix())

import ASWEstimator as ASWE
import performance

experiment_dir = Path('/Users/mrutala/projects/ASWEstimator/code/experiments/3DGPR/')


start = datetime.datetime(2010, 4, 1)
stop = datetime.datetime(2014, 9, 27)
# stop = datetime.datetime(2011, 4, 1)

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

# %% ==========================================================================
# Step 1) Get the boundary distribution defined from STEREO B data
# =============================================================================
control_background_file = experiment_dir / 'control_background.pkl'
if not control_background_file.exists():
    inputs_1DSTB = ASWE.ASWEstimator(start, stop, rmax=rmax, latmax=latmax)
    
    # Specify typically-hard-coded parameters for clarity
    inputs_1DSTB.boundarySources        = ['stereo b']
    inputs_1DSTB._icme_duration         = icme_duration
    inputs_1DSTB._icme_duration_buffer  = icme_buffer 
    inputs_1DSTB._icme_interp_buffer    = interp_buffer
    
    # 
    inputs_1DSTB.getSolarWind()
    inputs_1DSTB.getTransients()
    inputs_1DSTB.makeBackgroundDistribution(
        GP                  = True,
        n_samples           = bg_num_samples
        )
    
    inputs_1DSTB.save(control_background_file)
else:
    inputs_1DSTB = ASWE.ASWEstimator.load(control_background_file)

# %% =============================================================================
# 
# =============================================================================
control_boundary_file = experiment_dir / 'control_boundary.pkl'
if not control_boundary_file.exists():
    inputs_1DSTB.makeBoundaryDistributions(constant_percent_error=0)
    inputs_1DSTB.save(control_boundary_file)
else:
    inputs_1DSTB = ASWE.ASWEstimator.load(control_boundary_file)

# 
bound1DSTB_fromSTB = inputs_1DSTB.boundaryDistributions['stereo b']

# %%
# =============================================================================
# Step 2) Get background (ambient) solar wind data + transients
# =============================================================================
test_background_file = experiment_dir / 'test_background.pkl'
if not test_background_file.exists():
    
    inputs = ASWE.ASWEstimator(start, stop, rmax=rmax, latmax=latmax)

    # Specify typically-hard-coded parameters for clarity
    inputs.boundarySources              = ['omni', 'stereo a']
    inputs._icme_duration               = icme_duration
    inputs._icme_duration_buffer        = icme_buffer 
    inputs._icme_interp_buffer          = interp_buffer

    # 
    inputs.getSolarWind()
    inputs.getTransients()

    inputs.makeBackgroundDistribution(
        # inducing_variable   = False,
        GP                  = True,
        # target_noise        = bg_target_noise,
        # max_chunk_length    = bg_max_chunk_length,
        n_samples           = bg_num_samples
        )
    inputs.save(test_background_file)
else:
    inputs = ASWE.ASWEstimator.load(test_background_file)

# %% ==========================================================================
# Step 4) Backmap to 21.5 solar radii to generate a boundary distribution
#         Then split into 3 for 3D boundary distribution generation:
#           - one for for 3D Gaussian Processes (inputs)
#           - one for 1D extrapolation from OMNI (omni1d_inputs)
#           - one for 1D extrapolation from STEREO A (sta_inputs)
# =============================================================================
inputs.makeBoundaryDistributions(constant_percent_error=0)

inputs_3DGP = inputs.copy()
inputs_1DOMNI = inputs.copy()
inputs_1DSTA = inputs.copy()

# %% =============================================================================
# Step 5) Get the boundary distributions
# =============================================================================
test_boundary_file = experiment_dir / 'test_boundary.pkl'
if not test_boundary_file.exists():
    inputs_3DGP.generate_boundaryDistribution3D(
        nLat                = latnum,
        GP                  = True,
        )
    inputs_3DGP.save(test_boundary_file)
else:
    inputs_3DGP = ASWE.ASWEstimator.load(test_boundary_file)

# %%
bound_1DOMNI_file = experiment_dir / 'bound_1DOMNI.pkl'
if not bound_1DOMNI_file.exists():
    bound_1DOMNI, bound_samples_1DOMNI = inputs_3DGP.sample_boundaryDistribution3D('omni', num_samples=0, chunk_size=5000, cpu_fraction=0.75)
    with open(bound_1DOMNI_file, 'wb') as f:
        pickle.dump(bound_1DOMNI, f)
else:
    with open(bound_1DOMNI_file, 'rb') as f:
        bound_1DOMNI = pickle.load(f)
        
bound_1DSTA_file = experiment_dir / 'bound_1DSTA.pkl'
if not bound_1DSTA_file.exists():
    bound_1DSTA, bound_samples_1DSTA = inputs_3DGP.sample_boundaryDistribution3D('stereo a', num_samples=0, chunk_size=5000, cpu_fraction=0.75)
    with open(bound_1DSTA_file, 'wb') as f:
        pickle.dump(bound_1DSTA, f)
else:
    with open(bound_1DSTA_file, 'rb') as f:
        bound_1DSTA = pickle.load(f)

# CHECK PERFORMANCE
inputs_1DOMNI.generate_boundaryDistribution3D(nLat=latnum, extend='omni', GP=False)
inputs_1DSTA.generate_boundaryDistribution3D(nLat=latnum, extend='stereo a', GP=False)

# Ground-truth performance -- better be high!
perfOMNI = performance.measure(pd.Series(inputs_3DGP.boundaryDistributions['omni']['U_mu_grid'].flatten()), 
                               pd.Series(bound_1DOMNI['U_mu_grid'].flatten()), 
                               pd.Series(bound_1DOMNI['U_sigma_grid'].flatten()), 
                               )

perfSTA = performance.measure(pd.Series(inputs_3DGP.boundaryDistributions['stereo a']['U_mu_grid'].flatten()), 
                              pd.Series(bound_1DSTA['U_mu_grid'].flatten()), 
                              pd.Series(bound_1DSTA['U_sigma_grid'].flatten()), 
                              )


# Check performance
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,4))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0)

sample_slices = np.linspace(0, bound_1DOMNI['t_grid'].shape[0], 5).astype(int)[1:4]

for ax_pair, indx in zip(axs.T, sample_slices):
    ax_pair[0].plot(inputs_3DGP.boundaryDistributions['omni']['lon_grid'], 
                    inputs_3DGP.boundaryDistributions['omni']['U_mu_grid'][indx,:], 
                    color='black')
    ax_pair[0].fill_between(inputs_3DGP.boundaryDistributions['omni']['lon_grid'], 
                            inputs_3DGP.boundaryDistributions['omni']['U_mu_grid'][indx,:] - inputs_3DGP.boundaryDistributions['omni']['U_sigma_grid'][indx,:], 
                            inputs_3DGP.boundaryDistributions['omni']['U_mu_grid'][indx,:] + inputs_3DGP.boundaryDistributions['omni']['U_sigma_grid'][indx,:], 
                            color='black', alpha=0.33)

    ax_pair[0].plot(bound_1DOMNI['lon_grid'], 
                    bound_1DOMNI['U_mu_grid'][indx,:], 
                    color='red')
    ax_pair[0].fill_between(bound_1DOMNI['lon_grid'], 
                            bound_1DOMNI['U_mu_grid'][indx,:] - bound_1DOMNI['U_sigma_grid'][indx,:], 
                            bound_1DOMNI['U_mu_grid'][indx,:] + bound_1DOMNI['U_sigma_grid'][indx,:], 
                            color='red', alpha=0.33)

    ax_pair[1].plot(inputs_3DGP.boundaryDistributions['stereo a']['lon_grid'], 
                    inputs_3DGP.boundaryDistributions['stereo a']['U_mu_grid'][indx,:], 
                    color='black')
    ax_pair[1].fill_between(inputs_3DGP.boundaryDistributions['stereo a']['lon_grid'], 
                            inputs_3DGP.boundaryDistributions['stereo a']['U_mu_grid'][indx,:] - inputs_3DGP.boundaryDistributions['stereo a']['U_sigma_grid'][indx,:], 
                            inputs_3DGP.boundaryDistributions['stereo a']['U_mu_grid'][indx,:] + inputs_3DGP.boundaryDistributions['stereo a']['U_sigma_grid'][indx,:], 
                            color='black', alpha=0.33)
    
    ax_pair[1].plot(bound_1DSTA['lon_grid'], 
                    bound_1DSTA['U_mu_grid'][indx,:], 
                    color='red')
    ax_pair[1].fill_between(bound_1DSTA['lon_grid'], 
                            bound_1DSTA['U_mu_grid'][indx,:] - bound_1DSTA['U_sigma_grid'][indx,:], 
                            bound_1DSTA['U_mu_grid'][indx,:] + bound_1DSTA['U_sigma_grid'][indx,:], 
                            color='red', alpha=0.33)
    
    mjd_indx = inputs_3DGP.boundaryDistributions['omni']['t_grid'][indx]
    omni_lat_indx = np.interp( mjd_indx, inputs_3DGP.ephemeris['omni']['time'].mjd, inputs_3DGP.ephemeris['omni']['lat_c'].to(u.deg))
    sta_lat_indx = np.interp( mjd_indx, inputs_3DGP.ephemeris['stereo a']['time'].mjd, inputs_3DGP.ephemeris['stereo a']['lat_c'].to(u.deg))
    
    for ax in ax_pair:
        ax.set(xlabel = 'Longitude [deg.]', ylabel = r'$U_{SW}$ [km/s]', ylim=[200,800])
        ax.annotate('MJD: {}'.format(mjd_indx), 
                    (0,1), (1,-1), xycoords='axes fraction', textcoords='offset fontsize')
        ax.annotate('~{:.1f} degrees separation in latitude'.format(np.abs(omni_lat_indx - sta_lat_indx)), 
                    (0,1), (1,-3), xycoords='axes fraction', textcoords='offset fontsize')

#  ==========================================================================
# Step 6) Sample all three distributions at STEREO B
# =============================================================================
bound1DSTB_from3DGP_file = experiment_dir / 'bound1DSTB_from3DGP.pkl'
if not bound1DSTB_from3DGP_file.exists():
    bound1DSTB_from3DGP, sample1DSTB_from3DGP = inputs_3DGP.sample_boundaryDistribution3D('stereo b', num_samples=0, chunk_size=1000, cpu_fraction=0.75)
    with open(bound1DSTB_from3DGP_file, 'wb') as f:
        pickle.dump(bound1DSTB_from3DGP, f)
else:
    with open(bound1DSTB_from3DGP_file, 'rb') as f:
        bound1DSTB_from3DGP = pickle.load(f)
    
bound1DSTB_fromOMNI, sample1DSTB_fromOMNI = inputs_1DOMNI.sample_boundaryDistribution3D('stereo b')
bound1DSTB_fromSTA, sample1DSTB_fromSTA = inputs_1DSTA.sample_boundaryDistribution3D('stereo b')



# ==========================================================================
# Step 7) Compare the three models to the actual values at STEREO B
# =============================================================================

bounds_d = {'omni': bound1DSTB_fromOMNI, 
          'stereo a': bound1DSTB_fromSTA,
          '3D GP': bound1DSTB_from3DGP}

# statstimes = {}
# for key, bound in bounds_d.items():
#     statstime_df = pd.DataFrame(columns = ('mjd', 'r', 'σd', 'σm', 'E', 'PE'))
#     for i, t in enumerate(bound['t_grid']):
#         s = performance.measure(bound1DSTB_fromSTB['U_mu_grid'].T[i], bound['U_mu_grid'].T[i], bound['U_sig_grid'].T[i], )
#         s['mjd'] = t
#         statstime_df.loc[i,:] = s
        
#     statstimes[key] = statstime_df
    
mjd2D, lon2D = np.meshgrid(bound1DSTB_from3DGP['t_grid'], bound1DSTB_from3DGP['lon_grid'], indexing='ij')
trange_indx = (mjd2D >= inputs_3DGP.starttime.mjd) & (mjd2D < inputs_3DGP.stoptime.mjd)
    
omni = performance.measure(pd.Series(bound1DSTB_fromSTB['U_mu_grid'][trange_indx]), 
                           pd.Series(bound1DSTB_fromOMNI['U_mu_grid'][trange_indx]), 
                           pd.Series(bound1DSTB_fromOMNI['U_sigma_grid'][trange_indx]), 
                           )
sta = performance.measure(pd.Series(bound1DSTB_fromSTB['U_mu_grid'][trange_indx]), 
                          pd.Series(bound1DSTB_fromSTA['U_mu_grid'][trange_indx]), 
                          pd.Series(bound1DSTB_fromSTA['U_sigma_grid'][trange_indx]), 
                          )
gp = performance.measure(pd.Series(bound1DSTB_fromSTB['U_mu_grid'][trange_indx]), 
                         pd.Series(bound1DSTB_from3DGP['U_mu_grid'][trange_indx]), 
                         pd.Series(bound1DSTB_from3DGP['U_sigma_grid'][trange_indx]), 
                         )
# # gp_samples = []
# for i in range(len(sample1DSTB_from3DGP)):
#     gp_sample = performance.measure(pd.Series(bound1DSTB_fromSTB['U_mu_grid'][trange_indx]),
#                                     pd.Series(sample1DSTB_from3DGP[i]['U'][trange_indx]))
#     gp_samples.append(gp_sample)
# gp_samples = pd.concat(gp_samples)

# =============================================================================
# The real figure
# =============================================================================
fig, axd = plt.subplot_mosaic("""
                              a_
                              be
                              cf
                              dg
                              """, width_ratios=[2, 1])
bounds_d = {"STEREO B": bound1DSTB_fromSTB,
            "OMNI": bound1DSTB_fromOMNI,
            "STEREO A": bound1DSTB_fromSTA,
            "3D GP": bound1DSTB_from3DGP}   
target_mjd = 55400
for (label, bound), ax_key in zip(bounds_d.items(), ['a', 'b', 'c', 'd']):      
    img = axd[ax_key].pcolormesh(bound['t_grid'], bound['lon_grid'], 
                                 bound['U_mu_grid'].T, 
                                 vmin=300, vmax=500, cmap='plasma')
    
    for i, j in zip(img.get_facecolors(), bound['U_sigma_grid'].flatten()):
        norm_j = (j - np.nanmin(j))/(np.nanmax(j) - np.nanmin(j))
        i[3] = norm_j # Set the alpha value of the RGBA tuple using m2
    
    axd[ax_key].annotate("({})".format(ax_key), (0,1), (1,-1), 
                         'axes fraction', 'offset fontsize', 
                         ha='left', va='top', color='white')
    axd[ax_key].annotate("{}".format(label), (1,1), (-1,-1), 
                         'axes fraction', 'offset fontsize',
                         ha='right', va='top', color='white')
    axd[ax_key].set(xlim = [inputs_1DSTB.starttime.mjd, inputs_1DSTB.stoptime.mjd], xlabel = 'MJD [days]', ylabel = 'Longitude [deg.]')
    
    axd[ax_key].axvline(target_mjd, color='white', linestyle='--')

# Plot the estimated conditons under STEREO B for each boundary
bounds_d.pop("STEREO B")
for (label, bound), ax_key in zip(bounds_d.items(), ['e', 'f', 'g']):
    axd[ax_key].annotate("({})".format(ax_key), (0,1), (1,-1), 
                         'axes fraction', 'offset fontsize', 
                         ha='left', va='top', color='black')
    
    t_indx = np.argmin(np.abs(bound1DSTB_fromSTB['t_grid'] - target_mjd))
    axd[ax_key].plot(bound1DSTB_fromSTB['lon_grid'], bound1DSTB_fromSTB['U_mu_grid'][t_indx, :],
                     color='black', label = 'STEREO B')
    
    t_indx = np.argmin(np.abs(bound['t_grid'] - target_mjd))
    axd[ax_key].plot(bound['lon_grid'], bound['U_mu_grid'][t_indx, :],
                     color='C0', label=label)
    axd[ax_key].fill_between(bound['lon_grid'], 
                             bound['U_mu_grid'][t_indx, :] + 1.96*bound['U_sigma_grid'][t_indx, :],
                             bound['U_mu_grid'][t_indx, :] - 1.96*bound['U_sigma_grid'][t_indx, :],
                             color='C0', alpha=0.33)
    axd[ax_key].set(xlim=[0,360], ylim=[250, 550], ylabel='SW Speed [km/s]')
    
    # axd[ax_key].legend(loc='upper right')
    
    
# for (label, bound), ax_key in zip(bounds_d.items(), ['f', 'h', 'k']):
#     axd[ax_key].annotate("({})".format(ax_key), (0,1), (1,-1), 
#                          'axes fraction', 'offset fontsize', 
#                          ha='left', va='top', color='black')
    
#     t_indx = np.argmin(np.abs(bound['t_grid'] - target_mjd))
#     z_score = np.sqrt((bound['U_mu_grid'][:, t_indx] - bound1DSTB_fromSTB['U_mu_grid'][:, t_indx])**2 / bound['U_sig_grid'][:, t_indx]**2)
#     axd[ax_key].plot(bound['lon_grid'], 
#                      z_score,
#                      color='black', alpha=0.66)
    
#     axd[ax_key].set(xlim=[0,360], ylim=[0, 10], xlabel = 'Longitude [deg.]', ylabel = 'Res.')

plt.show()
    

#

def init_TaylorDiagram(σlim, rlim):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.9)

    # Remove right, top spines & enclose with arc
    ax.set(xlim=σlim, ylim=σlim, aspect=1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.plot(σlim[1]*np.cos(np.linspace(0, np.pi/2, 100)),
            σlim[1]*np.sin(np.linspace(0, np.pi/2, 100)),
            color = ax.spines['bottom'].get_edgecolor(),
            lw = ax.spines['bottom'].get_linewidth(),
            clip_on=False,
            zorder=-997)
    ax.fill_between(σlim[1]*np.cos(np.linspace(0, np.pi/2, 100)),
                    σlim[1]*np.sin(np.linspace(0, np.pi/2, 100)),
                    np.zeros(100) + σlim[1],
                    color=ax.get_facecolor(),
                    zorder=-998)
    
    
    # Plot radial grid
    for xtick in ax.get_xticks()[ax.get_xticks() < np.max(σlim)]:
        ax.plot(xtick*np.cos(np.linspace(0, np.pi/2, 100)),
                xtick*np.sin(np.linspace(0, np.pi/2, 100)),
                color = ax.spines['bottom'].get_edgecolor(),
                lw = ax.spines['bottom'].get_linewidth(),
                alpha=0.33,
                zorder=-999)
        
    # Plot azimuthal grid
    for rtick in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        ax.plot([0, σlim[1]*np.cos(np.arccos(rtick))],
                [0, σlim[1]*np.sin(np.arccos(rtick))],
                color = ax.spines['bottom'].get_edgecolor(),
                lw = ax.spines['bottom'].get_linewidth(),
                alpha=0.33,
                zorder=-999)
        
        ax.annotate(rtick, 
                    [σlim[1]*np.cos(np.arccos(rtick)), σlim[1]*np.sin(np.arccos(rtick))],
                    xycoords='data', 
                    ha='left', va='bottom',
                    clip_on=False)
        
    # Plot RMS rings
    for xtick in ax.get_xticks()[ax.get_xticks() < np.max(σlim)]:
        ax.plot(xtick*np.cos(np.linspace(0, np.pi, 100)) + 1,
                xtick*np.sin(np.linspace(0, np.pi, 100)),
                color = ax.spines['bottom'].get_edgecolor(),
                lw = ax.spines['bottom'].get_linewidth(),
                alpha=0.33,
                zorder=-999)
    
    return fig, ax

def TaylorCoords(σ, r):
    x = σ * np.cos(np.arccos(r))
    y = σ * np.sin(np.arccos(r))
    return (x,y)

fig, ax = init_TaylorDiagram([0,1.2], [0,1])



x, y = TaylorCoords(omni['σm']/omni['σd'], omni['r'])
c = omni['PE'][0]
ax.scatter(x, y, c=c,
           vmin=0, vmax=0.5, s=128, alpha=1.0,
           marker = 'o', label = 'LI overall')
ax.annotate('OMNI\nPE = {:.2f}'.format(c), (x, y), (-6,-3), 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=1),
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
            xycoords = 'data', textcoords = 'offset fontsize', 
            ha = 'left', va = 'center')

x, y = TaylorCoords(sta['σm']/sta['σd'], sta['r'])
c = sta['PE'][0]
ax.scatter(x, y, c=c,
           vmin=0, vmax=0.5, s=128, alpha=1.0,
           marker = 'o', label = 'LI overall')
ax.annotate('STEREO A\nPE = {:.2f}'.format(c), (x, y), (-6,+3), 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=1),
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
            xycoords = 'data', textcoords = 'offset fontsize', 
            ha = 'left', va = 'center')

x, y = TaylorCoords(gp['σm']/gp['σd'], gp['r'])
c = gp['PE'][0]
ax.scatter(x, y, c=c,
           vmin=0, vmax=0.5, s=128, alpha=1.0,
           marker = 'o', label = 'LI overall')
ax.annotate('3D GP\nPE = {:.2f}'.format(c), (x, y), (+3,-6), 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=1),
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
            xycoords = 'data', textcoords = 'offset fontsize', 
            ha = 'left', va = 'center')

ax.yaxis.set_visible(False)

ax.text(0.5, -0.1, "Model Standard Deviation, Data-Normalized",
        ha='center', va='center', fontsize='large')
ax.text(0.9, 0.9, "Model-Data Correlation Coefficient",
        ha='center', va='center', rotation=-45, fontsize='large')

import matplotlib as mpl
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=0.5)

cax = fig.add_axes([0.125, 0.15, 0.05, 0.75]) 
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=cax, orientation='vertical', location='left',
                  label='Prediction Efficiency (PE)')
cb.set_label('Prediction Efficiency (PE)', size='large')


plt.show()


# %%

# Define function to plot the Taylor Diagram axes
def init_TaylorDiagram(σlim, rlim, ax=None):
    
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        return_fig = False

    ticklen = 0.02
    
    # Remove right, top spines & enclose with arc
    ax.set(xlim=σlim, ylim=σlim, aspect=1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#000000')
    ax.tick_params(color=ax.spines['bottom'].get_edgecolor(), which='both')
    
    inner_spine_color = '#c4c4c4'
    
    ax.plot(σlim[1]*np.cos(np.linspace(0, np.pi/2, 100)),
            σlim[1]*np.sin(np.linspace(0, np.pi/2, 100)),
            color = ax.spines['bottom'].get_edgecolor(),
            lw = ax.spines['bottom'].get_linewidth(),
            clip_on=False,
            zorder=-997)
    ax.fill_between(σlim[1]*np.cos(np.linspace(0, np.pi/2, 100)),
                    σlim[1]*np.sin(np.linspace(0, np.pi/2, 100)),
                    np.zeros(100) + σlim[1],
                    color=ax.get_facecolor(),
                    zorder=-998)
    
    
    # Plot radial grid
    for xtick in ax.get_xticks()[ax.get_xticks() < np.max(σlim)]:
        ax.plot(xtick*np.cos(np.linspace(0, np.pi/2, 100)),
                xtick*np.sin(np.linspace(0, np.pi/2, 100)),
                color = inner_spine_color,
                lw = ax.spines['bottom'].get_linewidth(),
                alpha=1.0,
                zorder=-999)
        
    # Plot azimuthal grid
    for rtick in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        ax.plot([0, (σlim[1])*np.cos(np.arccos(rtick))],
                [0, (σlim[1])*np.sin(np.arccos(rtick))],
                color = inner_spine_color,
                lw = ax.spines['bottom'].get_linewidth(),
                alpha=1.0,
                zorder=-996)
        ax.plot([σlim[1]*np.cos(np.arccos(rtick)), (σlim[1]+ticklen)*np.cos(np.arccos(rtick))],
                [σlim[1]*np.sin(np.arccos(rtick)), (σlim[1]+ticklen)*np.sin(np.arccos(rtick))],
                color = ax.spines['bottom'].get_edgecolor(),
                lw = ax.spines['bottom'].get_linewidth(),
                alpha=1.0,
                zorder=-996)
        
        ax.annotate(rtick, 
                    [(σlim[1]+ticklen)*np.cos(np.arccos(rtick)), 
                     (σlim[1]+ticklen)*np.sin(np.arccos(rtick))],
                    xycoords='data', 
                    ha='left', va='bottom',
                    clip_on=False, zorder=-996, annotation_clip=False)
        
    # Plot RMS rings
    for xtick in ax.get_xticks()[ax.get_xticks() < np.max(σlim)]:
        ax.plot(xtick*np.cos(np.linspace(0, np.pi, 100)) + 1,
                xtick*np.sin(np.linspace(0, np.pi, 100)),
                color = inner_spine_color,
                lw = ax.spines['bottom'].get_linewidth(),
                alpha=1.0,
                zorder=-999)
    if return_fig:
        return fig, ax
    else:
        return ax

# And a function to convert coordinates to TD format
def TaylorCoords(σ, r):
    x = σ * np.cos(np.arccos(r))
    y = σ * np.sin(np.arccos(r))
    return (x,y)

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

gp_c = '#F37735'
li_c = '#FFC425'
cmap = plt.get_cmap('winter')
norm = mpl.colors.Normalize(vmin=0, vmax=0.6)

fig = plt.figure(figsize=[20/3, 15/3])

ax1 = fig.add_subplot(6, 1, 1)
ax1.set_position([0.1, 0.850, 0.875, 0.125])

ax2 = fig.add_subplot(6, 1, 2)
ax2.set_position([0.1, 0.7125, 0.875, 0.125])

ax3 = fig.add_subplot(6, 1, 3)
ax3.set_position([0.1, 0.575, 0.875, 0.125])

ax4 = fig.add_subplot(6, 1, 4)
ax4.set_position([0.1, 0.4375, 0.875, 0.125])

ax5 = fig.add_subplot(6, 1, 5)
ax5.set_position([0.1, 0.075, 0.40, 0.3])

ax6 = fig.add_subplot(6, 1, 6)
ax6.set_position([0.6625, 0.075, 0.3, 0.3 * (fig.get_size_inches()[0]/fig.get_size_inches()[1])])

ax6 = init_TaylorDiagram([0, 1.2], [0, 1], ax=ax6)

ax3_pos = ax3.get_position().extents
cax = fig.add_axes([0.6125, 0.075, 0.025, 0.3]) 

plt.savefig(experiment_dir / "3_3DGPR_Performance_option1.png", dpi=300)
plt.savefig(experiment_dir / "3_3DGPR_Performance_option1.eps")

# %%
fig = plt.figure(figsize=[10/3, 23/3])

ax1 = fig.add_subplot(6, 1, 1)
ax1.set_position([0.15, 1 - 0.075*(10/23) - 1*0.075 - 0, 0.675, 0.075])

ax2 = fig.add_subplot(6, 1, 2)
ax2.set_position([0.15, 1 - 0.075*(10/23) - 2*0.075 - 0.0125, 0.675, 0.075])

ax3 = fig.add_subplot(6, 1, 3)
ax3.set_position([0.15, 1 - 0.075*(10/23) - 3*0.075 - 0.025, 0.675, 00.075])

ax4 = fig.add_subplot(6, 1, 4)
ax4.set_position([0.15, 1 - 0.075*(10/23) - 4*0.075 - 0.0375, 0.675, 0.075])

cax4 = fig.add_axes([0.85, 1 - 0.075*(10/23) - 4*0.075 - 0.0375, 0.05, (1 - 0.075*(10/23)) - (1 - 0.075*(10/23) - 4*0.075 - 0.0375)])

ax5 = fig.add_subplot(6, 1, 5)
ax5.set_position([0.15, 0.425, 0.775, 0.15])

ax6 = fig.add_subplot(6, 1, 6)
ax6.set_position([0.25, 0.05, 0.675, 0.675 * (10/23)])

ax6 = init_TaylorDiagram([0, 1.2], [0, 1], ax=ax6)

# ax6_pos = ax6.get_position().extents
cax6 = fig.add_axes([0.15, 0.05, 0.05, 0.675 * (10/23)]) 

c_omni  = '#90be6d'
c_sta   = '#43aa8b'
c_stb   = '#577590'
c_gp    = '#F37735'

cmap_2D = plt.get_cmap('plasma')
norm_2D = mpl.colors.Normalize(vmin=300, vmax=600)

cmap_2D_err = cmap_2D(np.arange(cmap_2D.N))
cmap_2D_err[:,0:3] = 1
cmap_2D_err[:,-1] = np.linspace(0, 1, cmap_2D.N)
cmap_2D_err = mpl.colors.ListedColormap(cmap_2D_err)
norm_2D_err = mpl.colors.Normalize(vmin=0, vmax=0.25)

cmap_TD = plt.get_cmap('winter')
norm_TD = mpl.colors.Normalize(vmin=0, vmax=0.5)

bounds_d = {"STEREO B": bound1DSTB_fromSTB,
            "OMNI": bound1DSTB_fromOMNI,
            "STEREO A": bound1DSTB_fromSTA,
            "3D GP": bound1DSTB_from3DGP}   

target_mjd = 56279

# Plot the extracted 2D boundaries
for (label, bound), ax in zip(bounds_d.items(), [ax1, ax2, ax3, ax4]):
    img = ax.pcolormesh(bound['t_grid'], 
                        bound['lon_grid'], 
                        bound['U_mu_grid'].T, 
                        cmap=cmap_2D, norm=norm_2D)
    
    test = ax.pcolormesh(bound['t_grid'], 
                         bound['lon_grid'], 
                         (bound['U_sigma_grid'] / bound['U_mu_grid']).T, 
                         cmap=cmap_2D_err, norm=norm_2D_err)
    # for rgba, 
    # #     for i, j in zip(img.get_facecolors(), bound['U_sigma_grid'].flatten()):
    # #         norm_j = (j - np.nanmin(j))/(np.nanmax(j) - np.nanmin(j))
    # #         i[3] = norm_j # Set the alpha value of the RGBA tuple using m2
    
    ax.set(xlim=[inputs_1DSTB.starttime.mjd, inputs_1DSTB.stoptime.mjd], 
           ylim=[0,360], yticks=[0,120,240,360])
    
    ax.axvline(target_mjd, color='xkcd:lime green', linestyle='--', lw=2)

plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap_2D, norm=norm_2D), cax=cax4)

# Plot the extracted 1D boundaries
for (label, bound), color in zip(bounds_d.items(), [c_stb, c_omni, c_sta, c_gp]):

     t_indx = np.argmin(np.abs(bound['t_grid'] - target_mjd))
     ax5.plot(bound['lon_grid'], bound['U_mu_grid'][t_indx, :],
              color=color, lw=1)
     ax5.fill_between(bound['lon_grid'], 
                      bound['U_mu_grid'][t_indx, :] - bound['U_sigma_grid'][t_indx, :], 
                      bound['U_mu_grid'][t_indx, :] + bound['U_sigma_grid'][t_indx, :], 
                      color=color, alpha=0.2)
     
# Plot the Taylor Diagram
for perf, color, marker in zip([omni, sta, gp], [c_omni, c_sta, c_gp], ['D', 'D', 'X']):
    
    x, y = TaylorCoords(perf['σm']/perf['σd'], perf['r'])
    c = perf['PE'][0]
    ax6.scatter(x, y, c=c,
                cmap=cmap_TD, norm=norm_TD, s=128, alpha=1.0,
                marker = marker, ec=color, label = 'LI overall')
    
    # ax.annotate('OMNI\nPE = {:.2f}'.format(c), (x, y), (-6,-3), 
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=1),
    #             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
    #             xycoords = 'data', textcoords = 'offset fontsize', 
    #             ha = 'left', va = 'center')

plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap_TD, norm=norm_TD), cax=cax6)

plt.savefig(experiment_dir / "3_3DGPR_Performance_option1.png", dpi=300)
plt.savefig(experiment_dir / "3_3DGPR_Performance_option1.eps")

# %%
fig = plt.figure(figsize=[(244 + 244 + 20)/72, (244 + 10)/72])

ax1 = fig.add_subplot(6, 1, 1)
ax1.set_position([0.075, 0.6675, 0.45, 0.16])

ax2 = fig.add_subplot(6, 1, 2)
ax2.set_position([0.075, 0.4825, 0.45, 0.16])

ax3 = fig.add_subplot(6, 1, 3)
ax3.set_position([0.075, 0.2975, 0.45, 0.16])

ax4 = fig.add_subplot(6, 1, 4)
ax4.set_position([0.075, 0.1125, 0.45, 0.16])

cax4 = fig.add_axes([0.5375, 0.1125, 0.025, 0.715])
cax4_err = fig.add_axes([0.075, 0.8525, 0.45, 0.05])

ax5 = fig.add_subplot(6, 1, 5)
ax5.set_position([0.700, 0.6675, 0.275, 0.235])

ax6 = fig.add_subplot(6, 1, 6)
ax6.set_position([0.7375, 0.1125, 0.225, 0.45])

ax6 = init_TaylorDiagram([0, 1.2], [0, 1], ax=ax6)

cax6 = fig.add_axes([0.700, 0.1125, 0.025, 0.45]) 

c_omni  = '#90be6d'
c_sta   = '#43aa8b'
c_stb   = '#577590'
c_gp    = '#F37735'

cmap_2D = plt.get_cmap('plasma')
norm_2D = mpl.colors.Normalize(vmin=300, vmax=600)

cmap_2D_err = cmap_2D(np.arange(cmap_2D.N))
cmap_2D_err[:,0:3] = 1
cmap_2D_err[:,-1] = np.linspace(0, 1, cmap_2D.N)
cmap_2D_err = mpl.colors.ListedColormap(cmap_2D_err)
norm_2D_err = mpl.colors.Normalize(vmin=0, vmax=0.25)

cmap_TD = plt.get_cmap('winter')
norm_TD = mpl.colors.Normalize(vmin=0, vmax=0.5)

bounds_d = {"STEREO B": bound1DSTB_fromSTB,
            "OMNI": bound1DSTB_fromOMNI,
            "STEREO A": bound1DSTB_fromSTA,
            "3D GP": bound1DSTB_from3DGP}   

target_mjd = 56279

# Plot the extracted 2D boundaries
for (label, bound), ax in zip(bounds_d.items(), [ax1, ax2, ax3, ax4]):
    img = ax.pcolormesh(bound['t_grid'], 
                        bound['lon_grid'], 
                        bound['U_mu_grid'].T, 
                        cmap=cmap_2D, norm=norm_2D)
    
    test = ax.pcolormesh(bound['t_grid'], 
                         bound['lon_grid'], 
                         (bound['U_sigma_grid'] / bound['U_mu_grid']).T, 
                         cmap=cmap_2D_err, norm=norm_2D_err)
    # for rgba, 
    # #     for i, j in zip(img.get_facecolors(), bound['U_sigma_grid'].flatten()):
    # #         norm_j = (j - np.nanmin(j))/(np.nanmax(j) - np.nanmin(j))
    # #         i[3] = norm_j # Set the alpha value of the RGBA tuple using m2
    
    ax.set(xlim=[inputs_1DSTB.starttime.mjd, inputs_1DSTB.stoptime.mjd], 
           ylim=[0,360], yticks=[0,120,240,360])
    
    ax.axvline(target_mjd, color='xkcd:lime green', linestyle='--', lw=2)

plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap_2D, norm=norm_2D), cax=cax4, label='Flow Speed U [km/s]')

def percent_error_colorbar(err_cmap, err_norm, base_cmap, base_norm, n=11, cax=None):
    import matplotlib.ticker as ticker
    
    # x = [base_norm.vmin, (base_norm.vmin + base_norm.vmax)/2, base_norm.vmax]
    x = np.linspace(base_norm.vmin, base_norm.vmax, 5)[1:4]
    # base_vals = np.tile(np.linspace(base_norm.vmin, base_norm.vmax, n)[:,None], len(x))
    y =  np.linspace(err_norm.vmin, err_norm.vmax, n)
    err_vals = np.tile(y[:,None], len(x))
    
    cax.pcolormesh(y, x, np.tile(x, (n,1)).T, norm=base_norm, cmap=base_cmap)
    cax.pcolormesh(y, x, np.tile(y[:,None], len(x)).T, norm=err_norm, cmap=err_cmap)
    
    # Massage the axes
    cax.yaxis.set_visible(False)
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top') 
    cax.set(xticks = y[::2], 
            xticklabels = ['{:.0f}%'.format(_*100) for _ in y[::2]], 
            xlabel='Percent Uncertainty')
    cax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    return
percent_error_colorbar(cmap_2D_err, norm_2D_err, cmap_2D, norm_2D, n=11, cax=cax4_err)

# Plot the extracted 1D boundaries
for (label, bound), color, zorder, name in zip(bounds_d.items(), [c_stb, c_omni, c_sta, c_gp], [1, -1, -2, 2], ['STEREO-B', 'STEREO-A', 'OMNI', 'GP']):

     t_indx = np.argmin(np.abs(bound['t_grid'] - target_mjd))
     ax5.plot(bound['lon_grid'], bound['U_mu_grid'][t_indx, :],
              color=color, lw=1, zorder=zorder, label=name)
     ax5.fill_between(bound['lon_grid'], 
                      bound['U_mu_grid'][t_indx, :] - bound['U_sigma_grid'][t_indx, :], 
                      bound['U_mu_grid'][t_indx, :] + bound['U_sigma_grid'][t_indx, :], 
                      color=color, alpha=0.2, zorder=zorder)
ax5.legend(ncols=4, bbox_to_anchor=(0.7, 0.9275, 0.275, 0.05), loc='lower left',
           mode="expand", borderaxespad=0., edgecolor='black', bbox_transform=fig.transFigure, 
           labelcolor='linecolor', handlelength=0, prop={'weight': 'bold'})
ax5.set(xlim=[0,360], xticks=[0, 90, 180, 270, 360], 
        ylim=[norm_2D.vmin-50, 550], ylabel='U [km/s]')
fig.text(0.66318898, 0.62576772, r'$\lambda [^\circ]$', transform=fig.transFigure, ha='left', va='center')

# Plot the TD
x, y = TaylorCoords(omni['σm']/omni['σd'], omni['r'])
c = (omni['PE'][0] + sta['PE'][0])/2
ax6.scatter(x, y, c=c,
            cmap=cmap_TD, norm=norm_TD, s=64, alpha=1.0,
            marker = 'P', lw=1, ec=c_omni)

x, y = TaylorCoords(sta['σm']/sta['σd'], sta['r'])
c = (omni['PE'][0] + sta['PE'][0])/2
ax6.scatter(x, y, c=c,
            s=64, alpha=1.0,
            marker = 'P', lw=1, ec=c_sta)

x, y = TaylorCoords(gp['σm']/gp['σd'], gp['r'])
c = gp['PE'][0]
ax6.scatter(x, y, c=c,
            cmap=cmap_TD, norm=norm_TD, s=64, alpha=1.0,
            marker = 'X', lw=1, ec=c_gp)
    
x, y = TaylorCoords(1, 1)
ax6.scatter(x, y, c=1, cmap=cmap_TD, norm=norm_TD, s=128, alpha=1.0,
           marker='o', ec='black', lw=1)

plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap_TD, norm=norm_TD), 
             cax=cax6, orientation='vertical', location='left',
             label='Prediction Efficiency (PE)')

ax6.annotate(r"$P_{\sigma} = \sigma_M / \sigma_D$",
             (0.5, -0.2), (0, 0), 'axes fraction', 'offset fontsize',
             ha='center', va='center', fontsize=ax3.xaxis.get_label().get_fontsize())
ax6.annotate(r"$R$",
             (0.85, 0.85), (0, 0), 'axes fraction', 'offset fontsize',
             ha='center', va='center', rotation=-45, fontsize=ax3.xaxis.get_label().get_fontsize())


# Add years to MJD axes
mjd2dt = lambda x: Time(x, format='mjd').datetime64
dt2mjd = lambda x: Time(mdates.num2date(x), format='datetime').mjd
x2 = ax4.secondary_xaxis(0.05, functions=(mjd2dt, dt2mjd), transform=fig.transFigure)
x2.xaxis.set_minor_locator(mdates.MonthLocator())
x2.xaxis.set_major_locator(mdates.YearLocator())

# Get y pos from 
# b = x2.get_xticklabels()[0].get_window_extent() 
# fig.transFigure.inverted().transform((b.x0, b.y0+0.5*b.height))
fig.text(0.0, 0, 'Date [Year]', transform=fig.transFigure, ha='left', va='bottom')
fig.text(0.0, 0.07076772, 'Date [MJD]', transform=fig.transFigure, ha='left', va='center')
fig.text(0.0, 0.47, r'Heliolongitude ($\lambda$) [$^\circ$]', 
         transform=fig.transFigure, ha='left', va='center', rotation='vertical')

for ax, letter in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd']):
    ax.annotate("({})".format(letter), (0,1), (10,-10), xycoords='axes fraction', textcoords='offset pixels', 
                ha='left', va='top', color='white')
    
for ax, letter in zip([ax5, ax6], ['e', 'f']):
    ax.annotate("({})".format(letter), (0,1), (10,-10), xycoords='axes fraction', textcoords='offset pixels', 
                ha='left', va='top', color='black')

plt.savefig(experiment_dir / "3_3DGPR_Performance_option1.png", dpi=600)
plt.savefig(experiment_dir / "3_3DGPR_Performance_option1.pdf")

# Print Improvement metrics
gp['abs_1-P'] = np.abs(1 - gp['σm']/gp['σd'])
omni['abs_1-P'] = np.abs(1 - omni['σm']/omni['σd'])
sta['abs_1-P'] = np.abs(1 - sta['σm']/sta['σd'])

omni_comparison = []
sta_comparison = []
for col in ['P', 'r', 'RMSE', 'PE']:
    omni_comparison.append(((gp[col] - omni[col])/omni[col]).to_numpy()[0])
    sta_comparison.append(((gp[col] - sta[col])/sta[col]).to_numpy()[0])
    
# omni_stat = (omni_comparison[0] - omni_comparison[1] + omni_comparison[2] - omni_comparison[3])/4
# sta_stat = (sta_comparison[0] - sta_comparison[1] + sta_comparison[2] - sta_comparison[3])/4
# print("The GP performs {:.1f}% better than OMNI".format(omni_stat * 100))
# print("The GP performs {:.1f}% better than STEREO-A".format(sta_stat * 100))

# %%

# # %%
# # Just for fun: animation
# import matplotlib.animation as animation
# from astropy.time import Time
# def video():
    
#     frames = 1
    
#     step = 1 # degrees
#     lon = np.arange(0, 360, step)+0.5*step
#     lat = np.arange(-10, 10, step)+0.5*step
#     lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')
    
#     mjds = np.arange(55400, 55400+frames, 1)
    
#     # Initial initialization
#     fig, axs = plt.subplots(ncols=2, figsize=(6,4), width_ratios=[2,1])
#     axs[0].set(xlim=[0,360], xticks=np.arange(0,360+30,30), xlabel='Heliolongitude [deg.]',
#                ylim=[-10,10], yticks=np.arange(-10,10+2,2), ylabel='Heliolatitude [deg.]')
#     date = axs[0].annotate("Date: ", #"Date: {}".format(Time(mjd, format='mjd').datetime().strftime("%Y-%m-%d")), 
#                            (0,1), (1,-1), xycoords='axes fraction', textcoords='offset fontsize', 
#                            ha='left', va='top')
    
#     img = axs[0].pcolormesh(lon2d, lat2d, np.full(lon2d.shape, np.nan), 
#                             vmin=200, vmax=600, cmap='plasma')
    
#     line_omni = axs[0].axhline(np.nan, color='black', ls=':', lw=1)
#     point_omni = axs[0].scatter([np.nan], [np.nan], marker='x', s=32, color='black')
    
#     line_sta = axs[0].axhline(np.nan, color='black', ls=':', lw=1)
#     point_sta = axs[0].scatter([np.nan], [np.nan], marker='x', s=32, color='xkcd:green')
    
#     line_stb = axs[0].axhline(np.nan, color='black', ls=':', lw=1)
#     point_stb = axs[0].scatter([np.nan], [np.nan], marker='x', s=32, color='xkcd:blue')
    
#     curve_stb = axs[1].plot(lon, lon*np.nan, color='black')
#     fill_stb = axs[1].fill_between(lon, lon*np.nan, lon*np.nan, color='black', alpha=0.33)
#     axs[1].set(xlim=[0,360], xticks=np.arange(0,360+30,30), xlabel='Heliolongitude [deg.]',
#                ylim=[100, 800], ylabel='Solar Wind Speed [km/s]')
    
#     fig.colorbar(img, ax=axs[0])
    
#     def animate(i):
#         mjd = mjds[i]
#         mjd2d = np.full(lon2d.shape, mjd)
        
#         X = np.hstack([mjd2d.flatten()[:,None], 
#                        lon2d.flatten()[:,None], 
#                        lat2d.flatten()[:,None]])
#         mu, sigma = inputs_3DGP.boundaryModels['U'].predict_f(unscaled_X = X, chunk_size=2000, cpu_fraction=0.75)
        
#         mu2d = mu.reshape(lon2d.shape)
#         sigma2d = sigma.reshape(lon2d.shape)
        
#         img.set_array(mu2d.ravel())
        
#         # Get spacecraft positions
#         omni_lon = np.interp(mjd, inputs_3DGP.ephemeris['omni']['time'].mjd, inputs_3DGP.ephemeris['omni']['lon_c'].to(u.deg).value)
#         omni_lat = np.interp(mjd, inputs_3DGP.ephemeris['omni']['time'].mjd, inputs_3DGP.ephemeris['omni']['lat_c'].to(u.deg).value)
#         line_omni.set_ydata([omni_lat])
#         point_omni.set_offsets([[omni_lon, omni_lat]])
        
#         sta_lon = np.interp(mjd, inputs_3DGP.ephemeris['stereo a']['time'].mjd, inputs_3DGP.ephemeris['stereo a']['lon_c'].to(u.deg).value)
#         sta_lat = np.interp(mjd, inputs_3DGP.ephemeris['stereo a']['time'].mjd, inputs_3DGP.ephemeris['stereo a']['lat_c'].to(u.deg).value)
#         line_sta.set_ydata([sta_lat])
#         point_sta.set_offsets([[sta_lon, sta_lat]])
        
#         stb_lon = np.interp(mjd, inputs_3DGP.ephemeris['stereo b']['time'].mjd, inputs_3DGP.ephemeris['stereo b']['lon_c'].to(u.deg).value)
#         stb_lat = np.interp(mjd, inputs_3DGP.ephemeris['stereo b']['time'].mjd, inputs_3DGP.ephemeris['stereo b']['lat_c'].to(u.deg).value)
#         line_stb.set_ydata([stb_lat])
#         point_stb.set_offsets([[stb_lon, stb_lat]])
        
#         date.set_text("Date: {}".format(Time(mjd, format='mjd').datetime().strftime("%Y-%m-%d")))
        
        
#         interp = scipy.interpolate.RegularGridInterpolator((lon, lat), mu2d)
#         stb_sample = interp((lon, np.full(lon.shape, stb_lat)))
#         curve_stb.set_data(lon, stb_sample)
        
#         interp_sigma = scipy.interpolate.RegularGridInterpolator((lon, lat), sigma2d)
#         stb_error = interp((lon, np.full(lon.shape, stb_lat)))
#         fill_stb.remove()
#         fill_stb = axs[1].fill_between(lon, stb_sample-stb_error, stb_sample+stb_error, color='black', alpha=0.33)
        
#         return date, img, line_omni, point_omni, line_sta, point_sta, line_stb, point_stb, curve_stb, fill_stb
    
#     anim = animation.FuncAnimation(fig,animate,frames=frames,interval=1000,blit=False,repeat=False)
#     anim.save('test.mp4')
#     plt.show()
# video()