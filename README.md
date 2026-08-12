## Description
This repository contains tools relevant for training and evaluating anomaly detection algorithms on CMS DQM data, with updates made to allow for per-Lumisection data fetching, training, and model assessing. Additionally, data fetching has been updated with automatic metadata dictated by [OMS](https://cmsoms.cern.ch/cms/runs/lumisection?cms_run=397209&cms_run_sequence=GLOBAL-RUN).
Core code is contained in `autodqm_ml`, core scripts are contained in `scripts` and some helpful examples are in `examples`.
The following instructions have been adapted from the [AutoDQM-ML Readme](https://github.com/AutoDQM/AutoDQM_ML/blob/main/README.md) and [AutoDQM ML Introduction](https://autodqm.github.io/autodqm_ml.github.io/).

## Required Certificates
**1. VOMS Proxy**

**2. OMS API Access**
A registered CERN OpenID application is required to access data from OMS API, which the data fetching pipeline uses for filtering and assigning metadata to the relevant histograms.

For instructions on setting up access to OMS API, see [the linked CMS OMS gitlab](https://gitlab.cern.ch/cmsoms/oms-api-client/-/blob/master/README.md?ref_type=heads).

Once you have registered, make not of your key and secret, as this will be necessary data fetching.

## Installation
**1. Clone repository**
```
git clone https://github.com/avafaubus/AutoDQM_ML 
cd AutoDQM_ML
```
**2. Install dependencies**

Dependencies are listed in ```environment.yml``` and installed using `conda`. If you do not already have `conda` set up on your system, you can install (for linux) with:
```
curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
```
You can then set `conda` to be available upon login with
```
~/miniconda3/bin/conda init # adds conda setup to your ~/.bashrc, so relogin after executing this line
```

Once `conda` is installed and set up, install dependencies with (warning: this step may take a while)
```
conda env create -f environment.yml -p <path to install conda env>
```

Some packages cannot be installed via `conda` or take too long and need to be installed with `pip` (after activating your `conda` env above):
```
pip install yahist
pip install tensorflow==2.11.0
pip install omsapi
```
Note: if you are running on `lxplus`, you may run into permissions errors, which may be fixed with:
```
chmod 755 -R /afs/cern.ch/user/s/<your_user_name>/.conda
```
and then rerunning the command to create the `conda` env. The resulting `conda env` can also be several GB in size, so it may also be advisable to specify the installation location in your work area if running on `lxplus`, i.e. running the `conda env create` command with `-p /afs/cern.ch/work/...`.

Note: I recommend specify your EOS, not AFS, working directory because it has more available space.

**3. Install autodqm-ml**

Install with:
```
pip install -e .
```

Once your setup is installed, you can activate your python environment with
```
conda activate autodqm-ml
```

**Note**: `CMSSW` environments can interfere with `conda` environments. Recommended to unset your CMSSW environment (if any) by running
```
eval `scram unsetenv -sh`
```
before attempting installation and each time before activating the `conda` environment.

## Using the Tool
### 1. Data Fetching
Per-LS data fetching can be split into three steps: 

i. Collection of the urls of relevant histograms.
From the AutoDQM directory, run:

```
python write_file_list.py
```
Note: In the code, update input_directory to the file path to the directory you are interested in.

This code will output batches of txt files that will be saved in ./AutoDQM_ML/autodqm_ml/data_prep/batches/

ii. Extraction of specified histograms into parquet batches with training metadata.

Before beginning this step, make sure you have a registered CERN OpenID application with access to OMS API.

In the /data_prep/condor_make_training directory,

```
vim api_key.txt 
```
and update the txt file with your information: 
```
my_app_id=<your_app_id>
my_app_secret=<yourappsecret> 
```
Now, you can safely run:
```
condor_submit submit.sub 
```
This will follow the URLs indicated in batches in step i and process them into parquet shards. You can check the progress of the Condor job with: 
```
condor_q
```
iii. Merging of parquet shards into a complete training file.
Once the parquet shards have been produced, run

```
python merge_parquets.py
```
This will produce a complete training set in /AutoDQM_ML/training_sets/.

### 2. Training
First, navigate:
```
cd /AutoDQM_ML/scripts
```
Then, run the training script with the desired algorithm and training set:
```
python train.py --input_file "/AutoDQM_ML/training_sets/your-training-set-name.parquet" 
                --output_dir "/AutoDQM_ML/training_sets/labelled_addMLAlgos" 
                --algorithm "pca" 
                --tag "default_pca" 
                --histograms "path/to/histogram1, path/to/histogram2, path/to/histogram3" 
                --reco_assess_plots False
                --debug
```

Note: For CSCs, the histograms of interest are "CSC/CSCOfflineMonitor/recHits/hRHGlobalm1,CSC/CSCOfflineMonitor/recHits/hRHGlobalm2,CSC/CSCOfflineMonitor/recHits/hRHGlobalm3,CSC/CSCOfflineMonitor/recHits/hRHGlobalm4,CSC/CSCOfflineMonitor/recHits/hRHGlobalp1,CSC/CSCOfflineMonitor/recHits/hRHGlobalp2,CSC/CSCOfflineMonitor/recHits/hRHGlobalp3,CSC/CSCOfflineMonitor/recHits/hRHGlobalp4"

Replace "pca" with "autoencoder" or "ae" and "default_pca" with "default_ae" to train an autoencoder model with the training set.
### 3. Model Assessment

After training, you can use the assessing script (assess.py) to create useful plots and metrics for evaluating the model(s). 
```
python assess.py --input_file "/AutoDQM_ML/training_sets/labelled_addMLAlgos/your-training-set-name.parquet"
                 --output_dir "/AutoDQM_ML/training_sets/labelled_addMLAlgos/plots"
                 --histograms "CSC/CSCOfflineMonitor/recHits/hRHGlobalm1,CSC/CSCOfflineMonitor/recHits/hRHGlobalm2,CSC/CSCOfflineMonitor/recHits/hRHGlobalm3,CSC/CSCOfflineMonitor/recHits/hRHGlobalm4,CSC/CSCOfflineMonitor/recHits/hRHGlobalp1,CSC/CSCOfflineMonitor/recHits/hRHGlobalp2,CSC/CSCOfflineMonitor/recHits/hRHGlobalp3,CSC/CSCOfflineMonitor/recHits/hRHGlobalp4"
                 --hist_layout 2d
                 --algorithms "default_pca"
                 --debug
```
Optionally, you can add:
```
                 --samples "run-number1:LS1,run-number2:LS2,run-number3:LS3"
                 --plots_only
```
Samples allows you to specify run, lumisectoin pairs you wish to plot. Plots_only will produce only plots, rather than all of the training metrics.

For each histogram type provided, the assessing script provides a comprehensive list of anomaly scores for each run and lumisection, a summary plot providing Fraction of runs vs. Anomaly score, and plots of the reconstructed  

How is anomaly score determined?

### 4. Main Changes

