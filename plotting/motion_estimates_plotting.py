import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
from tristan_pipeline.plotting.plotting_utils import *
import pandas as pd 
#################CONFIG#####################
subjects = [1, 2, 3, 4]
sessions = [1]
space = "MNI152NLin2009cAsym"
onav_files = {
    1: "Y_B0_sent_2025-04-2311_47_37.753099.npy",
    2: "Y_B0_sent_2025-05-2714_30_55.782043.npy", 
    3: "Y_B0_sent_2025-09-0311-17-46.993207.npy", 
    4: "Y_B0_sent_2025-06-1111_13_47.267598.npy"
}
d_vols = {1: 0, 2: 2, 3: 2, 4: 2}
n_vols = {1: 153, 2: 155, 3: 155, 4: 155}
# Unique styles per subject
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
markers = ['o', 's', '^', '*']   # circle, square, triangle, star
line_widths = [1.5, 1.5, 2.5, 2.5]
alphas = [1.0, 1.0, 0.7, 0.5]
results = {'Subject': [],'ONAV Mean Abs Disp Y (mm)': [],'ONAV RMS Disp Y (mm)': [],'fMRIPrep Mean Abs Disp Y (mm)': [],'fMRIPrep RMS Disp Y (mm)': []}
#################CONFIG#####################
plt.figure(figsize=(12, 6))

for idx, subj in enumerate(subjects):
    for ses in sessions:
        print(f"\nProcessing subject {subj}, session {ses}")
        # fMRIPREP path
        FMRIPREP_PATH = f"/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02d}/data_ONAVonPEERSon/derivatives/fmriprep"
        FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,_,_, xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(
            FMRIPREP_PATH, subj, ses, space
        )
        bold_file = FUNC_PATH[0]
        # Load confounds
        confounds, _ = load_confounds(
            bold_file,
            strategy=('motion', 'global_signal', 'compcor', 'high_pass'),
            motion='power2',
            global_signal='power2',
            compcor="temporal_anat_combined",
            n_compcor=4,
            scrub=0
        )
        # fMRIPrep motion
        trans_y = confounds['trans_y'].values
        fmriprep_rms_disp_y = np.sqrt(np.mean(trans_y**2))
        fmriprep_mean_abs_disp_y = np.max(np.abs(trans_y))
        trans_norm = np.sqrt(
                confounds['trans_x']**2 +
                confounds['trans_y']**2 +
                confounds['trans_z']**2
            )
        plt.plot(
            trans_y[d_vols[subj]:],
            color=colors[idx],
            linestyle='-',
            marker=markers[idx],
            markevery=20,       # place markers every 20 timepoints
            linewidth=line_widths[idx], 
            alpha=alphas[idx]
        )
        # ONAV motion
        motion_file = os.path.join(
            f"/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02d}/onav_data",
            onav_files[subj]
        )
        onav_mean_abs_disp_y, onav_rms_disp_y = None, None
        if os.path.exists(motion_file):
            motion_reg, motion_labels = load_onav_reg(
                filepath=motion_file,
                labels=["Rx ", "Ry ", "Rz ", "x ", "y ", "z ", r"$\phi$ ", "f0 ", "G$_x$", "G$_y$", "G$_z$"],
                y_labels=[" / °", " / °", " / °", " / mm", " / mm", " / mm", " / rad", " / Hz", " / µT/m", " / µT/m", "/ µT/m"]
            )
            onav_trans_y = motion_reg[:, 4]
            onav_mean_abs_disp_y = np.max(np.abs(onav_trans_y))
            onav_rms_disp_y = np.sqrt(np.mean(onav_trans_y**2))
            onav_trans_norm = np.sqrt(
                    motion_reg[:, 3]**2 +
                    motion_reg[:, 4]**2 +
                    motion_reg[:, 5]**2
                )
            plt.plot(
                onav_trans_y[d_vols[subj]:],
                color=colors[idx],
                linestyle='--',
                marker=markers[idx],
                markevery=20,
                linewidth=line_widths[idx],
                alpha=alphas[idx]
            )

# Axis formatting
#plt.ylabel('Translation Y (mm)')
plt.xlabel("Timepoint (fMRI Volume #)")
#plt.title('Translation Y: fMRIPrep vs ONAV (all subjects)')
plt.ylabel('Translation Y (mm)')
plt.title('Translation Y: fMRIPrep vs ONAV (all subjects)')
plt.ylim((-0.2, 0.2))
plt.xlim((0,160))

# --- Two legends ---
# Legend 1: subjects (colors + markers)
subject_handles = [Line2D([0], [0], color=colors[i], lw=1.0, marker=markers[i]) for i in range(len(subjects))]
subject_labels = [f"sub-{s:02d}" for s in subjects]
leg1 = plt.legend(subject_handles, subject_labels, title="Subjects",loc="upper left", 
                  bbox_to_anchor=(0.15, 1.0))

# Legend 2: estimate types (solid = fMRIPrep, dashed = ONAV)
style_handles = [
    Line2D([0], [0], color="black", lw=0.8, linestyle='-'),
    Line2D([0], [0], color="black", lw=0.8, linestyle='--')
]
style_labels = ["fMRIPrep", "ONAV"]
leg2 = plt.legend(style_handles, style_labels, title="Estimates", loc="upper left")
plt.gca().add_artist(leg1)

plt.tight_layout()
plt.show()

