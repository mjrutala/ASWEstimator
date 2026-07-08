# <ins>A</ins>mbient <ins>S</ins>olar <ins>W</ins>ind Estimator
## A Gaussian-process-based tool for estimating unobserved solar wind conditions at 1 AU
The idea behind this project is to use Gaussian Process regression---a probablistic, physics-aware, non-parametric machine learning technique---to estimate unobserved ambient solar wind conditions at 1 AU (i.e., near the Earth's orbit). The two main use cases currently explored are:
- Estimating the ambient solar wind at an individual data source (OMNI, *Wind*, *ACE*, *STEREO-A*, *STEREO-B*, etc.) during an ICME
- Estimating the ambient solar wind structure in longitude, latitude, and time from a collection of individual data sources
This project builds on the GPs-for-solar-wind approach of the [Virtual Solar Wind Monitor at Mars](https://github.com/abbyazari/vswim), and is designed for interoperability with the [HUXt Solar Wind propagation model](https://github.com/University-of-Reading-Space-Science/HUXt). 
This project relies on the [GPFlow package](https://gpflow.github.io/) to perform Gaussian process regression, and uses [astroquery](https://astroquery.readthedocs.io/) and [SunPy](https://sunpy.org/) to automatically search for, download, and use data from multiple spacecraft.


## Contents
1. [**Conda Environment File**](https://github.com/mjrutala/ASWEstimator/blob/main/ASWEstimator-env.yml): this file describes a minimal conda environment for using this repo. With Anaconda or miniconda running on your machine, you can create a conda environment to run this project using
```
conda env create -f ASWEstimator-env.yml
```
Alternatively, you can use another function to parse `ASWEstimator-env.yml` or open it with a text editor and manually install the listed packages.
2. [**Experiments:**](https://github.com/mjrutala/ASWEstimator/tree/main/code/experiments) contains scripts which demonstrate the basic usage of the code in this repo by generating the three figures contained in the paper [under review]. Specifically, scripts are included for:
    - [Data Exploration](https://github.com/mjrutala/ASWEstimator/blob/main/code/experiments/data/data_exploration.py) introduces the data searching, downloading, and reading interface used by this project, and produces the following figure:

    **Note** `localpath` and `experiment_dir` should be changed to reflect the installation directory of this project on your local machine.
    - [1D GP Model](https://github.com/mjrutala/ASWEstimator/blob/main/code/experiments/1DGPR/experiment_1DGPR.py) introduces the one-dimensional GP model, which fills in gaps in an ambient solar wind time series caused by ICMEs, and produces the following figure:

    **Note** `localpath` and `experiment_dir` should be changed to reflect the installation directory of this project on your local machine. By default, this script will run the 1D GP model on the entire, 4+ year dataset 10 times, which will take a substantial amount of time; you may want to tweak the `start` and `stop` times before running.
    - [3D GP Model](https://github.com/mjrutala/ASWEstimator/blob/main/code/experiments/3DGPR/3DGPR_Performance.py) introduces the three-dimensional GP model, which fills in gaps in the ambient solar wind in longitude, latitude, and time using data from multiple spacecraft, and prdocues the following figure:

    **Note** `localpath` and `experiment_dir` should be changed to reflect the installation directory of this project on your local machine. By default, this script will run the 3D GP model on the entire, 4+ year dataset 10 times, which will take a substantial amount of time; you may want to tweak the `start` and `stop` times before running.
3. [**Source Code:**](https://github.com/mjrutala/ASWEstimator/tree/main/code) contains the classes, methods, and functions used to: locate, download, and read spacecraft data; identify and label ICMEs; and perform the 1D and 3D GP regressions. The source code is divided into 5 files:
    - [ASWEphemeris](https://github.com/mjrutala/ASWEstimator/blob/main/code/ASWEphemeris.py) - fetches spacecraft ephemeris (i.e., positioning) data in various useful coordinate systems.
    - [ASWReaders](https://github.com/mjrutala/ASWEstimator/blob/main/code/ASWReaders.py) - locates, downloads, and reads measured solar wind data from various spacecraft.
    - [queryDONKI](https://github.com/mjrutala/ASWEstimator/blob/main/code/queryDONKI.py) - searches for labelled ICMEs in the [DONKI](https://ccmc.gsfc.nasa.gov/tools/DONKI/) database, and labels the spacecraft data accordingly.
    - [GPFlowEnsemble](https://github.com/mjrutala/ASWEstimator/blob/main/code/GPFlowEnsemble.py) - a convenience wrapper for GPFlow which performs GP regression in series on lists of subsetted data, and performs GP prediction in parallel. This mainly serves to improve performance.
    - [ASWEstimator](https://github.com/mjrutala/ASWEstimator/blob/main/code/ASWEstimator.py) - contains a class, `ASWEstimator`, which coordinates functions from the other source code files to generate 1D and 3D GP estimates of the solar wind.