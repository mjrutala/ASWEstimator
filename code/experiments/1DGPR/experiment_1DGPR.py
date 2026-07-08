#!/usr/bin/env python
# coding: utf-8

import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time 
import pandas as pd
import tqdm
import astropy.units as u
from pathlib import Path

import sys
localpath = Path('/Users/mrutala/projects/ASWEstimator/code/')
sys.path.append(localpath.as_posix())
import ASWEstimator as ASWE
import performance 

experiment_dir = localpath / 'experiments/1DGPR/'


# First we'll set up some basic parameters for the experiment overall: the start and stop dates; the duration and buffer (padding on each end) of the ICMEs; the interpolation buffer (how much data is averaged on each end for the linear interpolation); the number of splits; the number of samples to draw; and a directory for all figures and files to be saved in.

#%%
ref_input_filepath = experiment_dir / 'reference_input.pkl'

# Check if any (or all) of the splits have already been run
GP_filename = 'split{0:02d}_GP_input.pkl'
LI_filename = 'split{0:02d}_LI_input.pkl'
def check_completion():
    complete_GP_tests, incomplete_GP_tests = [], []
    complete_LI_tests, incomplete_LI_tests = [], []
    for i in range(n_splits):
        if (experiment_dir / GP_filename.format(i)).exists():
            complete_GP_tests.append(experiment_dir / GP_filename.format(i))
        else:
            incomplete_GP_tests.append(i)
        if (experiment_dir / LI_filename.format(i)).exists():
            complete_LI_tests.append(experiment_dir / LI_filename.format(i))
        else:
            incomplete_LI_tests.append(i)
    return complete_GP_tests, incomplete_GP_tests, complete_LI_tests, incomplete_LI_tests

# %% Setup some initial parameters

start = datetime.datetime(2010, 4, 1) # 2010/04/01 -> First DONKI Data
stop = datetime.datetime(2014, 9, 27) # 2014/09/27 -> Last STEREO-B Data

icme_duration   = 3.75 * u.day # conservative duration (Richardson & Cane 2010)
icme_buffer     = 0.25 * u.day # onservative duration (Richardson & Cane 2010)
interp_buffer   = 1.0 * u.day # how much to use in the interpolation

n_splits = 10

num_samples = 10

# %% Get in-situ solar wind and transient data

background_file =  experiment_dir / 'background.pkl'
if not background_file.exists():
    ref_input = ASWE.ASWEstimator(start, stop, rmax=1.1, latmax=20)
    ref_input._icme_duration = icme_duration # conservative duration (Richardson & Cane 2010)
    ref_input._icme_duration_buffer = icme_buffer # conservative buffer (Richardson & Cane 2010)
    ref_input._icme_interp_buffer = interp_buffer

    ref_input.getSolarWind()
    ref_input.getTransients()
    
    ref_input.save(background_file)
else:
    ref_input = ASWE.ASWEstimator.load(background_file)


# %%
# =============================================================================
# Add fake ICMEs
# =============================================================================

split_bool_columns = pd.MultiIndex.from_product((ref_input.boundarySources, 
                                                     ['test', 'train', 'icme']))
split_bool_df = pd.DataFrame(data = False, 
                             index = ref_input.solar_wind.index,
                             columns = split_bool_columns)

split_dfs = [split_bool_df.copy() for _ in range(n_splits)] 

for source in ref_input.boundarySources:

    df = ref_input.solar_wind[source].copy()
    df['mjd'] = ref_input.solar_wind['mjd']

    # Define ICME-sized groups
    group_length = (icme_duration + 2*icme_buffer).to(u.day).value # days
    groups = pd.Series(
        index   = df.index, 
        data    = np.floor((df['mjd'] - df['mjd'].iloc[0])/group_length)
        )

    # Exclude ICMEs from the train/test split
    groups[df.query("ICME == True").index] = np.nan

    # Perform Group-KFold split, leaving ICMES out
    from sklearn.model_selection import GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits, shuffle=True)
    nonICME_index = df.query("ICME == False").index

    for i, (train, test) in enumerate(group_kfold.split(df.loc[nonICME_index,:], groups=groups[nonICME_index])):

        split_dfs[i].loc[nonICME_index[train], (source, 'train')] = True
        split_dfs[i].loc[nonICME_index[test], (source, 'test')] = True
        split_dfs[i].loc[df.query("ICME == True").index, (source, 'icme')] = True

# In[7]:


# Visualize training and test sets
fig, axs = plt.subplots(nrows=len(ref_input.boundarySources), figsize=(6,4), sharex=True)
for j, source in enumerate(ref_input.boundarySources):
    axs[j].set(ylabel = source, xlim=[ref_input.solar_wind['mjd'].min(), ref_input.solar_wind['mjd'].max()])
    for i, split_df  in enumerate(split_dfs):

        axs[j].fill_between(ref_input.solar_wind['mjd'], 
                            i+1-0.33, i+1+0.33, where=split_df[(source, 'train')],
                            color='xkcd:cornflower', lw=0)
        axs[j].fill_between(ref_input.solar_wind['mjd'], 
                            i+1-0.33, i+1+0.33, where=split_df[(source, 'test')],
                            color='xkcd:purple', lw=0)
        axs[j].fill_between(ref_input.solar_wind['mjd'], 
                            i+1-0.33, i+1+0.33, where=split_df[(source, 'icme')],
                            color='xkcd:red', lw=0)


fig.supylabel('Split Number')
fig.supxlabel('MJD')


plt.savefig(experiment_dir / 'Train_Test_split.png')


# In[8]:


# Compare our train/test split to the real background/ICME split
elapsed_time_ICME = []
elapsed_time_test = []
for s_df in split_dfs:
    for source in ref_input.boundarySources:
        # Gaps caused by ICMEs
        elapsed_time_ICME.extend(np.diff(ref_input.solar_wind['mjd'][~s_df[(source, 'icme')]]))
        elapsed_time_test.extend(np.diff(ref_input.solar_wind['mjd'][s_df[(source, 'train')]]))


fig, ax = plt.subplots()
h_ICME, bins = np.histogram(elapsed_time_ICME, bins=np.arange(0,15,1))
h_test, bins = np.histogram(elapsed_time_test, bins=bins)

ax.stairs(h_ICME, bins, label='True', fill=False, lw=1)
ax.stairs(h_test, bins, label='Simulated', fill=False, lw=1)

KS_results = scipy.stats.ks_2samp(elapsed_time_ICME, elapsed_time_test)

ax.set(xlabel='Time Since Background Solar Wind [days]', 
       ylabel='#', yscale='log')
ax.annotate("K-S p-value: {:.3f}".format(KS_results.pvalue), (0,1), (1,-1), 
            'axes fraction', 'offset fontsize')

plt.savefig(experiment_dir / 'Train_Test_split_stats.png')


# Now we'll train a GP and LI model for each train/test split. Prior to running the models, we can check whether they ahve already been run to save time.

# In[13]:
ref_input.save(ref_input_filepath)

# Save the split dfs for future reference
for i, s_df in enumerate(split_dfs):
    s_df.to_csv(experiment_dir / 'split{0:02d}.csv'.format(i))

# In[ ]:

# 
ref_input_copy = ASWE.ASWEstimator.load(ref_input_filepath)

# Determine which models still need to be run
GP_done, GP_todo, LI_done, LI_todo = check_completion()

# Perform modeling
_GP_timing = []
_LI_timing = []
for i in [GP_todo, LI_todo][np.argmax([len(GP_todo), len(LI_todo)])]:
    
    print("Processing split {0:02d}".format(i))
    s_df = pd.read_csv(experiment_dir / 'split{0:02d}.csv'.format(i), 
                       header=[0,1], index_col=0, parse_dates=True)

    # Copy the original inputs for the Gaussian process, and add test ICMEs
    GP_input = ref_input_copy.copy()
    LI_input = ref_input_copy.copy()
    for source in GP_input.boundarySources:
        GP_input.solar_wind.loc[:, (source, 'ICME')] = s_df.loc[:, (source, 'icme')] | s_df.loc[:, (source, 'test')]
        LI_input.solar_wind.loc[:, (source, 'ICME')] = s_df.loc[:, (source, 'icme')] | s_df.loc[:, (source, 'test')]

    t0 = time.time()
    GP_input.makeBackgroundDistribution(GP = True, n_samples = num_samples)
    GP_input.save(experiment_dir / 'split{0:02d}_GP_input.pkl'.format(i))
    _GP_timing.append(time.time() - t0)

    # Generate the background distribution, removing the fake ICMEs
    t0 = time.time()
    LI_input.makeBackgroundDistribution(interpolate = True, n_samples = num_samples)
    LI_input.save(experiment_dir / 'split{0:02d}_LI_input.pkl'.format(i))
    _LI_timing.append(time.time() - t0)


# %% Read all generated files back in for testing
ref_input = ASWE.ASWEstimator.load(ref_input_filepath)
GP_done, GP_todo, LI_done, LI_todo = check_completion()

split_dfs = [pd.read_csv(experiment_dir / 'split{0:02d}.csv'.format(i), header=[0,1], index_col=0, parse_dates=True) for i in range(n_splits)]
GP_inputs = [ASWE.ASWEstimator.load(file) for file in GP_done]
LI_inputs = [ASWE.ASWEstimator.load(file) for file in LI_done]    

GP_performance_by_split = []
GP_sample_performance_by_split = []
LI_performance_by_split = []
for s_df, GP_input, LI_input in zip(split_dfs, GP_inputs, LI_inputs):
    
    GP_performance_by_source = {}
    GP_sample_performance_by_source = {}
    LI_performance_by_source = {}
    
    for source in tqdm.tqdm(GP_input.boundarySources):
        
        # Get the fake ICME (i.e. test) indices
        test_index = s_df.loc[:, (source, 'test')]
        
        # Resample the GP, if you want
        # Though, for the 1D case, the samples tend to underperform the mean
        # So this does not really enhance our statistics
        # GP_input.sampleBackgroundDistributions(n_samples = 100)
        
        # Calculate the performance stats for the mean model
        performance_stats_df = performance.measure(
             ref_input.solar_wind.loc[test_index, (source, 'U')], 
             GP_input.backgroundDistributions.loc[test_index, (source, 'U_mu')],
             GP_input.backgroundDistributions.loc[test_index, (source, 'U_sigma')])
        GP_performance_by_source.update({source: performance_stats_df})
        
        # Calculate the performance stats for each sample
        sample_performance_stats_df = pd.DataFrame()
        for sample in GP_input.backgroundSamples:
            perf_df = performance.measure(
                ref_input.solar_wind.loc[test_index, (source, 'U')], 
                sample.loc[test_index, (source, 'U')])
            sample_performance_stats_df = pd.concat([sample_performance_stats_df, perf_df], ignore_index=True)
        GP_sample_performance_by_source.update({source: sample_performance_stats_df})
        
        # Calculate the performance stats for the LI model
        performance_stats_df = performance.measure(
            ref_input.solar_wind.loc[test_index, (source, 'U')], 
            LI_input.backgroundDistributions.loc[test_index, (source, 'U_mu')])
        LI_performance_by_source.update({source: performance_stats_df})

    GP_performance_by_source_df = pd.concat(
        GP_performance_by_source, keys=GP_performance_by_source.keys(), axis=1)
    GP_performance_by_split.append(GP_performance_by_source_df)
    
    GP_sample_performance_by_source_df = pd.concat(
        GP_sample_performance_by_source, keys=GP_sample_performance_by_source.keys(), axis=1)
    GP_sample_performance_by_split.append(GP_sample_performance_by_source_df)
    
    LI_performance_by_source_df = pd.concat(
        LI_performance_by_source, keys=LI_performance_by_source.keys(), axis=1)
    LI_performance_by_split.append(LI_performance_by_source_df)

# Finally, summarize everything into useful statistics
summarize_stats = lambda l: pd.concat(l, axis=0, ignore_index=True)
GP_performance_overall = summarize_stats(GP_performance_by_split)
GS_performance_overall = summarize_stats(GP_sample_performance_by_split)
LI_performance_overall = summarize_stats(LI_performance_by_split)

# Also flatten these, ignoring which source they characterize
flatten_stats = lambda df: pd.concat([df[source] for source in df.columns.levels[0]], axis=0, ignore_index=True)
GP_performance_flat = flatten_stats(GP_performance_overall)
GS_performance_flat = flatten_stats(GS_performance_overall)
LI_performance_flat = flatten_stats(LI_performance_overall)
# %%

# Test: What if we plot the time where GP performs best, and where LI performs best?

GP_perf_by_span = []
LI_perf_by_span = []
# source = 'omni'
for i in range(len(split_dfs)):
    for source in GP_input.boundarySources:
        test_indx = split_dfs[i][(source, 'test')]
        test_ids = (ref_input.solar_wind.loc[test_indx, :].index.diff() > datetime.timedelta(hours=1)).cumsum()
        
        ref_test_spans = [g for _, g in ref_input.solar_wind.loc[test_indx, :].groupby(test_ids) if len(g) >= 72]
        GP_test_spans = [g for _, g in GP_inputs[i].backgroundDistributions.loc[test_indx, :].groupby(test_ids) if len(g) >= 72]
        LI_test_spans = [g for _, g in LI_inputs[i].backgroundDistributions.loc[test_indx, :].groupby(test_ids) if len(g) >= 72]
        
        for _ref, _GP, _LI in zip(ref_test_spans, GP_test_spans, LI_test_spans):
            
            GP_perf = performance.measure(_ref[(source, 'U')], _GP[(source, 'U_mu')])
            GP_perf['split'] = i
            GP_perf['mjd'] = _ref['mjd'].mean()
            GP_perf['source'] = source
            
            LI_perf = performance.measure(_ref[(source, 'U')], _LI[(source, 'U_mu')])
            LI_perf['split'] = i
            LI_perf['mjd'] = _ref['mjd'].mean()
            LI_perf['source'] = source
            
            GP_perf_by_span.append(GP_perf)
            LI_perf_by_span.append(LI_perf)
        
GP_perf_by_span = pd.concat(GP_perf_by_span, ignore_index=True)
LI_perf_by_span = pd.concat(LI_perf_by_span, ignore_index=True)

opt = GP_perf_by_span['E']/GP_perf_by_span['σd'] - LI_perf_by_span['E']/LI_perf_by_span['σd']
opt_sort, opt_sort_indx = np.sort(opt.to_numpy()), np.argsort(opt.to_numpy())

# %%
ex1_indx    = 21 # 8
ex1         = GP_perf_by_span.loc[ex1_indx, :]

# ex2_indx    = opt_sort_indx[398]
# ex2         = GP_perf_by_span.loc[ex2_indx, :]

# %% =============================================================================
# A simple Taylor Diagram
# =============================================================================
from astropy.time import Time
import matplotlib as mpl

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

gp_c = '#f23524'
li_c = '#FFC425'
cmap = plt.get_cmap('winter')
norm = mpl.colors.Normalize(vmin=0, vmax=0.6)

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

# Initialize axes
fig = plt.figure(figsize=[10/3, 15/3])
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_position([0.15, 0.666, 0.75, 0.266])

ax3 = fig.add_subplot(2, 1, 2)
ax3.set_position([0.25, 0.075, 0.65, 0.4333])

ax3 = init_TaylorDiagram([0,1.25], [0,1], ax=ax3)

ax3_pos = ax3.get_position().extents
cax = fig.add_axes([0.15, 0.075, 0.05, 0.4333]) 

# Plot the data, GP, and Linear Interpolation models
def plot_ax1(ax, ex):
    icme_indx = split_dfs[ex['split']][(ex['source'], 'icme')]
    test_indx = split_dfs[ex['split']][(ex['source'], 'test')]
    source = ex['source']
    split = ex['split']
    center_mjd = ex['mjd']
    
    ax.plot(ref_input.solar_wind['mjd'].mask(icme_indx, np.nan), 
            ref_input.solar_wind.loc[:, (source, 'U')],
            color = 'black', lw = 1,
            label = 'Data')
    
    ax.plot(GP_inputs[split].backgroundDistributions['mjd'].mask(icme_indx, np.nan), 
            GP_inputs[split].backgroundDistributions.loc[:, (source, 'U_mu')],
            color = gp_c, lw = 1,
            label = 'GP')
    
    ax.fill_between(GP_inputs[split].backgroundDistributions['mjd'].mask(icme_indx, np.nan), 
                    GP_inputs[split].backgroundDistributions.loc[:, (source, 'U_mu')] + GP_inputs[split].backgroundDistributions.loc[:, (source, 'U_sigma')], 
                    GP_inputs[split].backgroundDistributions.loc[:, (source, 'U_mu')] - GP_inputs[split].backgroundDistributions.loc[:, (source, 'U_sigma')], 
                    color=gp_c, lw=0, alpha=0.66)
    
    ax.plot(LI_inputs[split].backgroundDistributions['mjd'].mask(icme_indx, np.nan), 
            LI_inputs[split].backgroundDistributions.loc[:, (source, 'U_mu')],
            color = li_c, lw = 1, ls=':',
            label = 'LI')
    
    # Setup the axes
    delta1, delta2 = 20, 30
    ax.set(xlim=[center_mjd-delta1, center_mjd+delta2],
           xticks=np.arange(center_mjd-delta1, center_mjd+delta2+10, 10),
           xticklabels=np.arange(0,delta1+delta2+10,10),
           ylim = [250, 850],
           xlabel = 'Days Since {}'.format(Time(center_mjd-delta1, format='mjd').datetime.strftime('%Y %b %d %H:00')), 
           ylabel = r'Flow Speed $U$ [km s$^{-1}$]')
    
    # Finally, shade the ICME & test locations
    ax.fill_between(ref_input.solar_wind['mjd'], ax1.get_ylim()[1],
                    where=split_dfs[split][(source, 'icme')], 
                    color='black', alpha=0.66, lw=0, 
                    label = 'True ICMEs')
    ax.fill_between(ref_input.solar_wind['mjd'], ax1.get_ylim()[1],
                    where=split_dfs[split][(source, 'test')], 
                    color='black', alpha=0.11, lw=0, label = 'Artificial ICME-length Gaps')
    
plot_ax1(ax1, ex1)

# Manual legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
legenda_handles = [Line2D([], [], color='black', lw=2), 
                   Line2D([], [], color=gp_c, lw=2), 
                   Line2D([], [], color=li_c, lw=2, ls=':')]
legenda = ax1.legend(legenda_handles, ['Data', 'GP', 'LI'], 
                     ncols=3, bbox_to_anchor=(0.15, 0.96, 0.75, 0.022), loc='lower left',
                     mode="expand", borderaxespad=0., frameon=False, bbox_transform=fig.transFigure)

legendb_handles = [Patch(color='xkcd:black', alpha=0.44, lw=0), 
                   Patch(color='xkcd:black', alpha=0.11, lw=0)]
legendb = ax1.legend(legendb_handles, ['True ICMEs', 'Artificial ICME-length Gaps'], 
                     ncols=2, bbox_to_anchor=(0.15, 0.933, 0.75, 0.033), loc='lower left',
                     mode="expand", borderaxespad=0., frameon=False, bbox_transform=fig.transFigure)

rect = FancyBboxPatch((0.155, 0.945), 0.74, 0.045, transform=fig.transFigure, 
                      boxstyle="Round, pad=0.005", clip_on=False, edgecolor='black', facecolor='none')
ax1.add_patch(rect)

ax1.add_artist(legenda)
ax1.add_artist(legendb)

# Load a colormap
from scipy.stats import gaussian_kde



# Linear interpolation: mean and spread
def plot_LI():
    σ = LI_performance_flat.loc[:, 'σm']/LI_performance_flat.loc[:, 'σd']
    r = LI_performance_flat.loc[:, 'r']
    c = LI_performance_flat.loc[:, 'PE']

    x_mean, y_mean = TaylorCoords(σ.mean(), r.mean())
    ax3.scatter(x_mean, y_mean, c=c.mean(),
                cmap=cmap, norm=norm, s=128, alpha=1.0,
                marker = '^', ec=li_c, lw=1.0, ls='-', zorder=10)
    
    x, y = TaylorCoords(σ, r)
    ax3.scatter(x, y, c=c,
                cmap=cmap, norm=norm, s=32, alpha=0.50,
                marker = '^', lw=0, zorder=0)

    kde = gaussian_kde(np.vstack([x, y]))
    x_mesh, y_mesh = np.meshgrid(np.arange(*ax3.get_xlim(),0.01), np.arange(*ax3.get_ylim(),0.01))
    density = kde(np.vstack([x_mesh.flatten(), y_mesh.flatten()]))
    scaled_density = (density - density.min())/(density.max() - density.min())
    ax3.contour(x_mesh, y_mesh, scaled_density.reshape(x_mesh.shape), 
                linewidths=1.0, linestyles='-', colors=li_c, levels=[1-0.68],
                zorder=1)
    
    return
plot_LI()

# ax3.annotate('Linear Interpolation\nPE = {:.2f}'.format(c), (x, y), (-2, -6), 
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=1),
#              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
#              xycoords = 'data', textcoords = 'offset fontsize', 
#              ha = 'left', va = 'center')

# GP Performance: mean and spread
def plot_GP():
    σ = GP_performance_flat.loc[:, 'σm']/GP_performance_flat.loc[:, 'σd']
    r = GP_performance_flat.loc[:, 'r']
    c = GP_performance_flat.loc[:, 'PE']

    x_mean, y_mean = TaylorCoords(σ.mean(), r.mean())
    ax3.scatter(x_mean, y_mean, c=c.mean(),
                cmap=cmap, norm=norm, s=128, alpha=1.0,
                marker = 'X', ec = gp_c, lw=1, ls='-', zorder=10)
    
    x, y = TaylorCoords(σ, r)
    ax3.scatter(x, y, c=c,
                cmap=cmap, norm=norm, s=32, alpha=0.50,
                marker = 'X', lw=0, zorder=0)

    kde = gaussian_kde(np.vstack([x, y]))
    x_mesh, y_mesh = np.meshgrid(np.arange(*ax3.get_xlim(),0.01), np.arange(*ax3.get_ylim(),0.01))
    density = kde(np.vstack([x_mesh.flatten(), y_mesh.flatten()]))
    scaled_density = (density - density.min())/(density.max() - density.min())
    ax3.contour(x_mesh, y_mesh, scaled_density.reshape(x_mesh.shape), 
                linewidths=1.0, linestyles='-', colors=gp_c, levels=[1-0.68],
                zorder=1)
    
    return
plot_GP()

# plt.show()
# For reference, plot where a 'perfect' model would lie
x, y, c = 1, 0, 1
ax3.scatter(1, 0, c=c,
           s=128, alpha=1.0, lw=1, cmap=cmap, norm=norm,
           marker='o', ec='black')
# ax3.annotate('Perfect Model\nPE = {:.2f}'.format(c), (x, y), (-4, +2), 
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=1),
#             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
#             xycoords = 'data', textcoords = 'offset fontsize', 
#             ha = 'left', va = 'bottom')   

# Setup the axes
ax3.yaxis.set_visible(False)

ax3.annotate(r"$P_{\sigma} = \sigma_M / \sigma_D$",
             (0.5, -0.13), (0, 0), 'axes fraction', 'offset fontsize',
             ha='center', va='center', fontsize=ax3.xaxis.get_label().get_fontsize())
ax3.annotate(r"$R$",
             (0.82, 0.82), (0, 0), 'axes fraction', 'offset fontsize',
             ha='center', va='center', rotation=-45, fontsize=ax3.xaxis.get_label().get_fontsize())

ax3.scatter([], [], color=li_c, facecolor='white', marker='^', s=128, label='LI')
ax3.scatter([], [], color=gp_c, facecolor='white', marker='X', s=128, label='GP')
ax3.scatter([], [], color='black', facecolor='white', marker='o', s=128, label='Perfect Model')
ax3.legend(ncols=3, bbox_to_anchor=(0.15, 0.533, 0.75, 0.033), loc='lower left',
           mode="expand", borderaxespad=0., edgecolor='black', markerscale=0.75, bbox_transform=fig.transFigure)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='vertical', location='left',
             label='Prediction Efficiency (PE)',)


ax1.annotate('(a)', (0,1), (10,-10), 'axes fraction', 'offset pixels', ha='left', va='top', fontsize=BIGGER_SIZE)
# ax2.annotate('(b)', (0,1), (1,-1), 'axes fraction', 'offset fontsize', ha='center', va='center')
ax3.annotate('(b)', (0,1), (10,-10), 'axes fraction', 'offset pixels', ha='left', va='top', fontsize=BIGGER_SIZE)

plt.savefig(experiment_dir / 'measure_1DGPRPerformance.png', dpi=300)
plt.savefig(experiment_dir / 'measure_1DGPRPerformance.eps', dpi=300)
plt.show()