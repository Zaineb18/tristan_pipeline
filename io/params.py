import numpy as np 
import os, glob

GLOB_DIR = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/bids"
space = "MNI152NLin2009cAsym"
subjects = [1]
sessions = [1]
RAW_PATH = os.path.join(GLOB_DIR, 'rawdata')


DATA_DIR = "/home/zamor/Documents/TRISTAN/data/bids"
FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
TRANSFORM_PATH = os.path.join(DATA_DIR, 'derivatives', 'h5_transforms')
CHARM_PATH =os.path.join(DATA_DIR, 'derivatives', 'charmtms')
SIMNIBS_PATH =os.path.join(DATA_DIR, 'derivatives', 'simnibs')
change_from_LPS_to_RAS = np.array([[1, -1, 1, 1], [-1, 1, 1, 1], [1, 1, 1, -1], [1, 1, 1, 1]])
