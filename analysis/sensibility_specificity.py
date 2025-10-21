from tristan_pipeline.analysis.analysis_utils import *
from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
from tristan_pipeline.analysis.glm_utils import *
from nilearn.glm import threshold_stats_img
from nilearn import image
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import os
# ------------------------- SETTINGS -------------------------
subjects = [1, 2, 3, 4]
sessions = [1]
mocos = ["ONAVonPEERSon"]
spaces = ["T1w"]
contrasts_names = ['calculations', 'clic right vs clic left']

# Unique styles per subject
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
markers = ['o', 's', '^', '*']
alphas = [1.0, 1.0, 0.7, 0.5]

activation_fraction_df = pd.DataFrame()

# ------------------------- LOOP OVER DATA -------------------------
for subj in subjects:
    for ses in sessions:
        for moco in mocos:
            DATA_DIR = f"/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/data_{moco}"
            FMRIPREP_PATH = os.path.join(DATA_DIR, 'derivatives', 'fmriprep')

            for space in spaces:
                # Load FMRIPREP paths
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH, WM_PATH, CSF_PATH, xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(FMRIPREP_PATH, subj, ses, space)

                # Load anatomical/tissue images
                anat_file,mask_file = ANAT_PATH[0], MASK_PATH[0]
                gm_file, wm_file, csf_file = GM_PATH[0], WM_PATH[0], CSF_PATH[0]

                # Process each contrast
                for contrast in contrasts_names:
                    z_map_file = os.path.join(FMRIPREP_PATH, 'stat', f'sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco}.nii')
                    z_map = nib.load(z_map_file)
                    thresholded_map, threshold= threshold_stats_img(z_map, alpha=0.001, height_control='fpr', two_sided=True)

                    gm,wm,csf,zmap_data, brain_mask = prep_stats_anats_tissues(mask_file,
                                                gm_file, wm_file, csf_file, z_map_file)
                    wm_core, gm_core, csf_core, gm_wm_interface, gm_csf_interface, ambiguous = make_tissues(wm, 
                                                                         gm,
                                                                         csf,
                                                                         min_thresh=0.85)
                    gm_core = gm_core.astype(bool)
                    wm_core = wm_core.astype(bool)
                    csf_core = csf_core.astype(bool)
                    gm_wm_interface = gm_wm_interface.astype(bool)
                    gm_csf_interface = gm_csf_interface.astype(bool)

                    roi_masks = {
                    "GM": gm_core,
                    "WM": wm_core,
                    "CSF": csf_core,
                    "GM/WM interface": gm_wm_interface,
                    "Pial surface": gm_csf_interface,
                }

                    # Apply brain mask
                    if contrast == "calculations":
                        activated_mask = zmap_data > threshold
                        total_activated_voxels = np.sum(activated_mask)
                        #print("total from zdata", total_activated_voxels)
                        if total_activated_voxels == 0:
                            continue
                        for tissue, mask in roi_masks.items():  # <-- use resampled masks!
                            tissue_activated = np.sum(activated_mask & mask)
                            #print(f"{tissue}", tissue_activated)
                            fraction = tissue_activated / total_activated_voxels * 100
                            activation_fraction_df = pd.concat([
                            activation_fraction_df,
                            pd.DataFrame({
                            'Subject': [f"sub-{subj:02}"],
                            'Contrast': [contrast],
                            'Tissue': [tissue],
                            'Fraction': [fraction],
                            'Sign': ['positive']
                            })
                            ], ignore_index=True)

                    elif contrast == "clic right vs clic left":
                        activated_mask = (zmap_data > threshold) | (zmap_data < -threshold)
                        total_voxels = np.sum(activated_mask)
                        #print("total from zdata", total_voxels)
                        if total_voxels == 0:
                            continue
                        for tissue, mask in roi_masks.items():  # <-- use resampled masks!
                            tissue_activated = np.sum(activated_mask & mask)
                            #print(f"{tissue}", tissue_activated)
                            fraction = tissue_activated / total_voxels *100
                            activation_fraction_df = pd.concat([
                            activation_fraction_df,
                            pd.DataFrame({
                            'Subject': [f"sub-{subj:02}"],
                            'Contrast': [contrast],
                            'Tissue': [tissue],
                            'Fraction': [fraction],
                            'Sign': ['positive']
                            })
                            ], ignore_index=True)


# Plot style
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

tissues = ["GM","Pial surface","GM/WM interface","WM","CSF",]
contrasts = ["clic right vs clic left", "calculations"]
subject_list = [f"sub-{s:02}" for s in subjects]

# Colors
subject_colors = colors  # for subjects
tissue_colors = ['blue', 'lightblue', 'lightgreen','yellow','orange']  # stacked colors for tissues
bar_width = 0.3

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle("Percentage of activated voxels by tissue (all subjects)")

for ax, contrast in zip(axes, contrasts):
    x = np.arange(len(subject_list))
    
    for subj_idx, subj in enumerate(subject_list):
        bottom_val = 0  # starting point for stacking
        for tissue_idx, tissue in enumerate(tissues):
            val = activation_fraction_df.loc[
                (activation_fraction_df['Contrast'] == contrast) &
                (activation_fraction_df['Tissue'] == tissue) &
                (activation_fraction_df['Subject'] == subj),
                'Fraction'
            ]
            frac = val.values[0] if not val.empty else 0
            
            ax.bar(
                x[subj_idx], 
                frac,  # convert fraction to percentage
                width=bar_width, 
                bottom=bottom_val, 
                color=tissue_colors[tissue_idx], 
                edgecolor='k'
            )
            bottom_val += frac  # increment bottom for next tissue
        
        # Outline the full bar for each subject
        ax.bar(
            x[subj_idx], 
            0, 
            width=bar_width, 
            color='none', 
            edgecolor='k'
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(subject_list)
    ax.set_title(contrast.replace("_", " ").title())
    ax.set_ylabel("Percentage of activated voxels" if contrast == contrasts[0] else "")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.yaxis.set_major_formatter(lambda x, _: f'{int(x)}%')  # format y-axis as percentages

# Legend for tissues
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='k', label=tissue) for tissue, color in zip(tissues, tissue_colors)]
axes[1].legend(handles=legend_elements, title="Tissues", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()