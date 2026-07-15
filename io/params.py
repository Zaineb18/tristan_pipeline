import numpy as np 
import os, glob

GLOB_DIR = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/bids"
RAW_PATH = os.path.join(GLOB_DIR, 'rawdata')
DATA_DIR = f"/home/zamor/Documents/TRISTAN/imag_dataset"
grp_dir = os.path.join(DATA_DIR,"grp_output")

stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
contrasts_names = ['calculations','clic right vs clic left', 'checkerboard', 'phrases']
spaces = [
          "MNI152NLin2009cAsym",
          #"T1w"
           ]
subjects = [1,2,3,4]
sessions = [1]

mocos = {
    "SNAVoffPEERSoff":False,
        "SNAVonPEERSon":True}

mocos_ = {
    "Servo off":False,
        "Servo on + PEERS ":True}
onav_files = {1:"Y_B0_sent_2025-04-2311_47_37.753099.npy",
             2:"Y_B0_sent_2025-05-2714_30_55.782043.npy", 
             3:"Y_B0_sent_2025-06-1111_13_47.267598.npy", 
             4:"Y_B0_sent_2025-09-0311-17-46.993207.npy"}

d_vols= {1:0, 2:2, 3:2, 4:2}
n_vols = {1:153, 2:155, 3:155, 4:155}
trs= {1:2.12, 2:2.12, 3:2.12, 4:2.16}
min_onsets= {1:2, 2:2, 3:2, 4:2}


#GLOB_DIR = "/home/zamor/Documents/TRISTAN/dataset_Caroline"
#RAW_PATH = os.path.join(GLOB_DIR, 'rawdata')
#DATA_DIR = f"/home/zamor/Documents/TRISTAN/dataset_Caroline"
#grp_dir = os.path.join(DATA_DIR,"grp_output")

#stimfile = "/home/zamor/Documents/TRISTAN/dataset_Caroline/session1_localizer_standard.csv"
#contrasts_names = ['checkerboard vs calculations']
#spaces = [
#          "MNI152NLin2009cAsym",
#          "T1w"
#           ]
#subjects = [1,2,3,4,5,6,7,8,9]
#sessions = [1]
#d_vols= {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
#n_vols = {1:263, 2:263, 3:263, 4:263, 5:263, 6:263, 7:263, 8:263, 9:263}
#trs= {1:1.2, 2:1.2, 3:1.2, 4:1.2, 5:1.2, 6:1.2, 7:1.2, 8:1.2, 9:1.2}
#min_onsets= {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0} 
