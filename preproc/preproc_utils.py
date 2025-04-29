import re
import os, glob, h5py
from nilearn.image import mean_img, load_img, clean_img,math_img,new_img_like,resample_to_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiSpheresMasker,NiftiMasker
import numpy as np
import nibabel as nib
from scipy.signal import detrend
from scipy.stats import pearsonr
from nibabel.affines import apply_affine

from tristan_pipeline.io.params import *

def extract_runs(file_list):
    runs = set()
    for f in file_list:
        match = re.search(r'run-(\d+)', f)
        if match:
            runs.add(f"run-{match.group(1)}")
    return sorted(runs)

def sort_by_run(files):
    return sorted(files, key=lambda x: int(re.search(r'run-(\d+)', x).group(1)))

def add_ignore_suffix(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    new_path = file_path + '.ignore'
    os.rename(file_path, new_path)
    print(f"Renamed:\n  {file_path}\n→ {new_path}")

def clean_bold(fpath,tr=1.09):
    confounds, sample_mask = load_confounds(fpath, strategy=('motion','wm_csf','global_signal'), motion='derivatives', scrub=0)
    clean_func = clean_img(imgs=fpath, confounds=confounds, standardize=False, detrend=False, low_pass=None, high_pass=None, t_r=tr)
    mean_func = mean_img(clean_func)
    return(clean_func, mean_func, sample_mask, confounds)  