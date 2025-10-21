Tristan Pipeline
================
fMRI preprocessing and analysis pipeline for ultra-high-field data

Overview
--------
Tristan Pipeline is a modular Python workflow for preprocessing and analyzing ultra-high-field functional MRI (fMRI) data.
The dataset used for development and validation was acquired at 11.7 T using a well-validated functional localizer paradigm (P. Pinel, BMC Neuroscience, 2007), enabling precise mapping of cortical activations for different stimuli types (motor, visual, math, language).

Key Features
------------
- Modular structure separating preprocessing, analysis, plotting, I/O, and utility components.
- Reproducible configuration files for consistent data handling and parameter tracking.
- Support for both single-subject and group-level analyses (in progress).
- Open-source, flexible, and research-oriented.

Intended Audience
-----------------
- Neuroscientists working with ultra-high-field MRI (≥ 7 T).
- Researchers studying fine-grained cortical organization using validated localizer paradigms.
- Labs needing a customizable starting point for fMRI pipeline automation.
- Students learning how to build and document neuroimaging workflows.

Data Context
------------
The pipeline was tested on fMRI data acquired at 11.7 T using a functional localizer paradigm designed to elicit robust activations in well-defined cortical regions.
This dataset provides a benchmark for evaluating preprocessing accuracy, motion correction, and spatial normalization at ultra-high resolution.

Project Structure
-----------------
tristan_pipeline/
│
├── preproc/       ← preprocessing scripts (motion correction, normalization, etc)
├── analysis/      ← model fitting, statistics, group-level analysis
├── plotting/      ← visualization scripts & notebooks
├── io/            ← data loading and saving utilities
├── utils/        
├── notebooks/    
├── scripts/     
├── LICENSE
├── README.md

Getting Started
---------------
1. Clone the repository
   git clone https://github.com/Zaineb18/tristan_pipeline.git
   cd tristan_pipeline

2. Install dependencies
    Typical dependencies include nibabel, nilearn, numpy, scipy, matplotlib, pandas.

3. Prepare your data
   - Organize your fMRI data following a BIDS-like structure (recommended).
   - Edit configuration files to match your acquisition parameters (e.g., TR, voxel size, smoothing kernel).

License
-------
This project is licensed under the MIT License — see the LICENSE file for details.