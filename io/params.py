import numpy as np 
import os, glob

GLOB_DIR = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/bids"
RAW_PATH = os.path.join(GLOB_DIR, 'rawdata')

stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
#stimfile = "/home/zamor/Documents/TRISTAN/data_Caro/session1_localizer_standard.csv"
contrasts_names = ['calculations','checkerboard vs the others','clic right vs clic left']
mocos = ["ONAVonPEERSon"]
spaces = ["MNI152NLin2009cAsym", "T1w"]#
onav = True
subjects = [1,2,3]
sessions = [1]
onav_files = {1:"Y_B0_sent_2025-04-2311_47_37.753099.npy",
             2:"Y_B0_sent_2025-05-2714_30_55.782043.npy", 
             3:"Y_B0_sent_2025-09-0311-17-46.993207.npy", 
             4:"Y_B0_sent_2025-06-1111_13_47.267598.npy"}
d_vols= {1:0, 2:2, 3:2, 4:2}
n_vols = {1:153, 2:155, 3:155, 4:155}
trs= {1:2.12, 2:2.12, 3:2.16, 4:2.12}
min_onsets= {1:0, 2:0, 3:0, 4:0}

#d_vols= {1:0, 2:0, 3:0, 4:0}
#n_vols = {1:263, 2:263, 3:263, 4:263}
#trs= {1:1.2, 2:1.2, 3:1.2, 4:1.2}
#min_onsets= {1:0, 2:0, 3:0, 4:0} 
