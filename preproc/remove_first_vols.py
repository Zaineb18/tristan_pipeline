from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *

import glob
import nibabel as nib
from nilearn.image import mean_img,load_img, index_img
import os
import shutil

n_vols = 3
for subj in subjects: 
    for ses in sessions: 
        RFUNC_PATH, RFMAP_PATH = load_rawdata(GLOB_DIR, subj, ses)           
        for path in RFUNC_PATH:
            img = load_img(path) 
            trimmed_img = index_img(img, slice(n_vols, None)) 
            trimmed_img.to_filename(path)
