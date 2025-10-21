import os
import numpy as np
import nibabel as nib
from nilearn import plotting, image
import matplotlib.pyplot as plt
from tristan_pipeline.plotting.plotting_utils import *

##########CONFIG###########
subjects = [1,2,3,4]
sessions = [1]
space = "MNI152NLin2009cAsym"
datasets = [
    #("/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/data_onavoffPEERSoff", "ONAVoffPEERSoff"),
    #("/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/data_ONAVoffPEERSon", "ONAVoffPEERSon"),
    #("/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/data_ONAVonPEERSoff", "ONAVonPEERSoff"),
    ("/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/data_ONAVonPEERSon", "ONAVonPEERSon")]
output_dir = "/home/zamor/Documents/TRISTAN/ismrm_dataset/grptSNR_maps"
os.makedirs(output_dir, exist_ok=True)
##########CONFIG###########

for data_dir_template, moco_label in datasets:
    subject_maps = []
    #########LOOP OVER SUBJECTS AND SESSIONS#########
    for subj in subjects:
        for ses in sessions:
            data_dir = data_dir_template.format(subj=subj)
            fmriprep_path = os.path.join(data_dir, 'derivatives', 'fmriprep')            
            tsnr_file = os.path.join(fmriprep_path, 'stat',f"sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco_label}.nii")
            if not os.path.exists(tsnr_file):
                print(f"Missing file: {tsnr_file}")
                continue
            img = nib.load(tsnr_file)
            subject_maps.append(img)
    if not subject_maps:
        print(f"No maps found for {moco_label}")
        continue
    #########COMPUTE GROUP AVERAGE#########
    group_img = image.mean_img(subject_maps)
    group_img = image.math_img("img * (img > 0)", img=group_img)
    #########SAVE NII#########
    group_file = os.path.join(output_dir, f"group_tSNR_space-{space}_{moco_label}.nii.gz")
    nib.save(group_img, group_file)
    print(f"Saved group tSNR map → {group_file}")
    #########DISPLAY#########
    disp = plotting.plot_stat_map(
        group_img,
        title=f"Group tSNR ({moco_label}) - {space}",
        threshold=0.0,
        #vmin=0,
        vmax=130,
        display_mode='ortho',
        cut_coords=(0, 0, 0),
        colorbar=True,draw_cross=False, 
        symmetric_cbar=False,
    )
    plt.savefig(os.path.join(output_dir, f"group_tSNR_space-{space}_{moco_label}"))
    plotting.show()        
