import numpy as np 
import os, glob

GLOB_DIR = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/bids"
space = "MNI152NLin2009cAsym"
subjects = [3]
sessions = [1]
RAW_PATH = os.path.join(GLOB_DIR, 'rawdata')


DATA_DIR = "/home/zamor/Documents/TRISTAN/sub-04/data_onavON"
FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
