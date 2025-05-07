import re
import os, glob, h5py
import gzip
import shutil
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

def clean_bold(fpath,tr=2.12):
    confounds, _ = load_confounds(fpath, strategy=('motion','wm_csf','global_signal'), motion='full',
                                  wm_csf='full', global_signal='full',scrub=0)
    clean_func = clean_img(imgs=fpath, confounds=confounds, standardize=False, detrend=True, low_pass=0.1, high_pass=0.01, t_r=tr)
    mean_func = mean_img(clean_func)
    return(clean_func, mean_func, confounds)

def compress_nii_to_niigz(nii_path):
    if nii_path.endswith('.nii') and not nii_path.endswith('.nii.gz'):
        with open(nii_path, 'rb') as f_in:
            with gzip.open(nii_path + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(nii_path)  