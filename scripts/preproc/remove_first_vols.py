from tristan_pipeline.io.params import *
from tristan_pipeline.utils.loading_utils import *
from tristan_pipeline.utils.preproc_utils import *
from nilearn.image import load_img, index_img
import nibabel as nib
import json
for subj in subjects: 
    for ses in sessions: 
        RFUNC_PATH, RFMAP_PATH = load_rawdata(RAW_PATH, subj, ses, moco=None) 
        for path in RFMAP_PATH:
            print(path)
            img = load_img(path) 
            print(nib.load(path).shape)
            trimmed_img = index_img(img, 0) 
            trimmed_img.to_filename(path)
