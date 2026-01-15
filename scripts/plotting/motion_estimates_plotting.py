import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tristan_pipeline.utils.loading_utils import *
from tristan_pipeline.utils.preproc_utils import *
from tristan_pipeline.utils.plotting_utils import *

onav_marker = 'D'
onav_linewidth = 2.0
onav_alpha = 0.3

# ---------------------------------------
for subj_idx, subj in enumerate(subjects):
    plt.figure(figsize=(8, 4))
    subj_color = subject_colors[subj]
    for ses in sessions:
        FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
        for m_idx, moco in enumerate(list(mocos.keys())):
            try:
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,_,_, xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(
                    FMRIPREP_PATH, subj, ses, "MNI152NLin2009cAsym", moco)
                
                bold_file = FUNC_PATH[0]
                confounds, _ = load_confounds(
                    bold_file,
                    strategy=('motion', 'global_signal', 'compcor', 'high_pass'),
                    motion='power2',
                    global_signal='power2',
                    compcor="temporal_anat_combined",
                    n_compcor=4,
                    scrub=0
                )

                trans_norm = np.sqrt(confounds['trans_x']**2 +
                                     confounds['trans_y']**2 +
                                     confounds['trans_z']**2)

                plt.plot(
                    trans_norm,
                    color=adjust_color_tone(subj_color, moco_brightness[m_idx]),
                    linestyle='-',
                    marker=markers[m_idx],
                    markevery=20,
                    linewidth=moco_widths[m_idx],
                    alpha=0.9,
                    label=f"{moco} (fMRIPrep)" if ses == sessions[0] else None
                )
            except Exception as e:
                print(f"[WARN] Skipped sub-{subj:02d}, {moco}: {e}")
                continue
            
        # --- SNAV motion estimates ---
        motion_file = os.path.join(DATA_DIR, 'rawdata',f'sub-{subj:02}',f'ses-{ses}', 'onav_data',onav_files[subj]) 
        if os.path.exists(motion_file):
            motion_reg, motion_labels = load_onav_reg(
                filepath=motion_file,
                labels=["Rx ", "Ry ", "Rz ", "x ", "y ", "z ", r"$\phi$ ", "f0 ", "G$_x$", "G$_y$", "G$_z$"],
                y_labels=[" / °", " / °", " / °", " / mm", " / mm", " / mm", " / rad", " / Hz", " / µT/m", " / µT/m", "/ µT/m"]
            )

            onav_trans_norm = np.sqrt(motion_reg[:, 3]**2 +
                                      motion_reg[:, 4]**2 +
                                      motion_reg[:, 5]**2)
            plt.plot(
                onav_trans_norm,
                color=adjust_color_tone(subj_color,onav_alpha),
                linestyle='--',
                marker=onav_marker,
                markevery=20,
                linewidth=onav_linewidth,
                label="SNAV" if ses == sessions[0] else None
            )
        else:
            print(f"[WARN] Missing ONAV file: {motion_file}")

    # --- Formatting ---
    plt.xlabel("Timepoint (fMRI Volume #)")
    plt.ylabel("Translation L2 norm (mm)")
    plt.title(f"fMRIPrep vs SNAV: Motion estimates - sub-{subj:02d}", color=subj_color, fontweight='bold')
    plt.ylim(0, 0.85)
    plt.xlim(-1, 160)
    plt.grid(True, linestyle='--', alpha=0.3)

    # --- Legends ---
    moco_legend = [
        Line2D([0], [0],
               color=adjust_color_tone(subj_color, moco_brightness[i]),
               lw=moco_widths[i],
               linestyle='-',
               marker=markers[i],
               markersize=6,
               label=m)
        for i, m in enumerate(list(mocos.keys()))
    ]
    onav_legend = [
        Line2D([0], [0],
               color=adjust_color_tone(subj_color,onav_alpha),
               lw=onav_linewidth,
               linestyle='--',
               marker=onav_marker,
               markersize=6,
               label='SNAV')
    ]

    first_legend = plt.legend(handles=moco_legend, title="fMRIPrep estimates", loc="upper center")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=onav_legend, title="SNAV estimates", loc="upper left"#,bbox_to_anchor=(0.45, 1.0)
               )
    plt.tight_layout()
    os.makedirs(os.path.join(grp_dir,'figures') ,exist_ok=True)
    plt.savefig(os.path.join(grp_dir,'figures',f"sub-{subj:02}_ses-{ses}_MotionEst.pdf"))

    plt.show()