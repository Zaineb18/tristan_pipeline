
#####DO NOT REVIEW, CHANGES IN PATHS#####

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import math_img
from nilearn.masking import apply_mask
from nilearn.glm import threshold_stats_img
from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
from tristan_pipeline.analysis.glm_utils import *
from tristan_pipeline.analysis.analysis_utils import *
from tristan_pipeline.plotting.plotting_utils import *


#####SET UP PARAMS#####
subjects = [5]
sessions = [1]
spaces = ["MNI152NLin2009cAsym"]
contrasts= ['clic right vs clic left']
datasets = [
    ("/home/zamor/Documents/TRISTAN/sub-05/data_onavOFFPEERSOFF", False, "ONAVoffPEERSoff"),
    ("/home/zamor/Documents/TRISTAN/sub-05/data_onavOFFPEERSON",  False, "ONAVoffPEERSon"),
    ("/home/zamor/Documents/TRISTAN/sub-05/data_onavONPEERSOFF",  True,  "ONAVonPEERSoff"),
    ("/home/zamor/Documents/TRISTAN/sub-05/data_onavONPEERSON",   True,  "ONAVonPEERSon")
]
n_scans, delay_volumes, tr = 155, 2, 2.12
stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
events, task_vector_right, task_vector_left, task_vector_calc = events_task_vectors(stimfile,n_scans=155,delay_volumes=2,tr=2.12)
all_means_right = []
all_means_left = []
all_means_calc = []
all_means_visu = []
linestyles = ['-', '--', 'dashdot', ':']
colors = {'Right': 'red', 'Left': 'blue', 'Calculation': 'green'
          ,'Checkerboard': 'orange'}

for idx, (data_dir, onav, moco_label) in enumerate(datasets):
    FMRIPREP_PATH = os.path.join(data_dir, 'derivatives', 'fmriprep_reconall')
    for subj in subjects:
        for ses in sessions:
            for space in spaces:
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH, xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(FMRIPREP_PATH, subj, ses, space)
                bold_file,mask_file = FUNC_PATH[0],MASK_PATH[0]
                for contrast in contrasts:
                    z_map = nib.load(os.path.join(FMRIPREP_PATH, 'stat',
                    f'sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco_label}.nii'))
                    thresholded_map, threshold = threshold_stats_img(z_map,alpha=0.001,
                    height_control='fpr',two_sided=True)
                    if contrast == 'clic right vs clic left':
                        pos_mask = math_img("img > 3.1", img=z_map)
                        neg_mask = math_img("img < -3.1", img=z_map)
                        mean_right = apply_mask(bold_file, pos_mask).mean(axis=1)
                        mean_left  = apply_mask(bold_file, neg_mask).mean(axis=1)
                        all_means_right.append({'label': moco_label, 'mean': mean_right, 'linestyle': linestyles[idx]})
                        all_means_left.append({'label': moco_label, 'mean': mean_left, 'linestyle': linestyles[idx]})
                    elif contrast == 'calculations':            
                        pos_mask = math_img("img > 3.1", img=z_map)
                        mean_calc = apply_mask(bold_file, pos_mask).mean(axis=1)
                        all_means_calc.append({'label': moco_label, 'mean': mean_calc, 'linestyle': linestyles[idx]})
                    elif contrast == 'checkerboard vs the others':            
                        pos_mask = math_img("img > 3.1", img=z_map)
                        mean_visu = apply_mask(bold_file, pos_mask).mean(axis=1)
                        all_means_visu.append({'label': moco_label, 'mean': mean_calc, 'linestyle': linestyles[idx]})

plt.figure(figsize=(20, 12))
x = np.arange(n_scans)
click_lines = [
    plt.Line2D([0], [0], color='red', label='Right Click'),
    plt.Line2D([0], [0], color='blue', label='Left Click')
]

# Motion correction lines in black for the legend
linestyles = ['-', '--', 'dashdot', ':']
strategy_lines = [
    plt.Line2D([0], [0], color='black', linestyle=item['linestyle'], label=item['label'])
    for item in all_means_right
]

## Plot Right clicks (percentage change)
for item in all_means_right:
    rest_idx = np.where(~task_vector_right)[0]
    rest_idx = rest_idx[12:30]   
    baseline = item['mean'][rest_idx].mean()
    percent_change = ((item['mean'][3:] - baseline) / baseline) * 100
    plt.plot(x[3:], percent_change, color='red', linestyle=item['linestyle'])

## Plot Left clicks (percentage change)
for item in all_means_left:
    rest_idx = np.where(~task_vector_right)[0]
    rest_idx = rest_idx[15:27]
    baseline = item['mean'][rest_idx].mean()
    percent_change = ((item['mean'][3:] - baseline) / baseline) * 100
    plt.plot(x[3:], percent_change + 10, color='blue', linestyle=item['linestyle'])  # small offset if needed

# Shaded blocks
ymin, ymax = plt.ylim()
for start, end in consecutive_blocks(task_vector_right[3:]):
    plt.fill_between(range(start, end+1), ymin, ymax, color='red', alpha=0.5)
for start, end in consecutive_blocks(task_vector_left[3:]):
    plt.fill_between(range(start, end+1), ymin, ymax, color='blue', alpha=0.5)

plt.xlabel('Volumes')
plt.ylabel('% BOLD Signal Change')
plt.title('% BOLD Change - Click Right vs Click Left Across Motion Correction Strategies')

# Motion correction legend
plt.gca().add_artist(plt.legend(handles=strategy_lines, title='Motion Correction Strategies', loc='upper right'))

# Click type legend
plt.gca().add_artist(plt.legend(handles=click_lines, title='Click Type', loc='upper left'))

plt.grid(True)
plt.tight_layout()
plt.show()



