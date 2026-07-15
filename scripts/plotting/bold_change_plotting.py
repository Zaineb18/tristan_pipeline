import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.image import math_img
from nilearn.masking import apply_mask
from nilearn.glm import threshold_stats_img

from tristan_pipeline.io.params import *
from tristan_pipeline.utils.loading_utils import *
from tristan_pipeline.utils.preproc_utils import *
from tristan_pipeline.utils.glm_utils import *
from tristan_pipeline.utils.analysis_utils import *
from tristan_pipeline.utils.plotting_utils import *

contrasts = ["clic right vs clic left"]
spaces = ["T1w"]

figtype = "tog"

# ----------------
# LOOP OVER SUBJECTS
# ----------------
for subj in subjects:
    for ses in sessions:
        n_scans = n_vols[subj]
        delay_volumes = min_onsets[subj]
        tr = trs[subj]
        subj_color = subject_colors.get(subj, 'black')

        stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
        events, task_vector_right, task_vector_left, task_vector_calc, task_vector_lang, task_vector_visu = events_task_vectors(
            stimfile, n_scans=n_scans, delay_volumes=delay_volumes, tr=tr
        )

        all_means_right, all_means_left, all_means = [], [], []
        print(subj)
        # ----------------
        # LOAD DATA
        # ----------------
        for idx_moco, moco_label in enumerate(list(mocos.keys())):
            FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
            print(moco_label)
            for idx_space, space in enumerate(spaces):
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,WM_PATH,CSF_PATH,xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(
                    FMRIPREP_PATH, subj, ses, space,moco_label
                )
                bold_file, mask_file = FUNC_PATH[0], MASK_PATH[0]
                print(space)
                for contrast in contrasts:
                    z_map_path = os.path.join(
                        FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}','stats',
                        f"sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco_label}.nii"
                    )
                    if not os.path.exists(z_map_path):
                        print(f"Missing z-map: {z_map_path}")
                        continue

                    z_map = nib.load(z_map_path)
                    thresholded_map, threshold = threshold_stats_img(
                        z_map, alpha=0.001, height_control="fpr", two_sided=True
                    )

                    # ----------------
                    # Extract mean timecourse for clicks
                    # ----------------
                    if contrast == "clic right vs clic left":
                        pos_mask = math_img(f"img > {threshold:.4f}", img=z_map)
                        neg_mask = math_img(f"img < {-threshold:.4f}", img=z_map)
                        mean_right = apply_mask(bold_file, pos_mask).mean(axis=1)
                        mean_left = apply_mask(bold_file, neg_mask).mean(axis=1)

                        all_means_right.append({
                            "moco": moco_label,
                            "space": space,
                            "mean": mean_right,
                            "color": adjust_color_tone("red", moco_brightness[idx_moco]),
                            "marker": markers[idx_moco % len(markers)],
                            "linewidth": moco_widths[idx_moco]
                        })
                        all_means_left.append({
                            "moco": moco_label,
                            "space": space,
                            "mean": mean_left,
                            "color": adjust_color_tone("blue", moco_brightness[idx_moco]*0.8),
                            "marker": markers[idx_moco % len(markers)],
                            "linewidth": moco_widths[idx_moco]
                        })
                             
        #plt.figure(figsize=(20, 8))
        fig, axes = plt.subplots(1,2, figsize=(22, 8))
        x = np.arange(n_scans)

        # Click type legend
        click_lines = [
            plt.Line2D([0], [0], color="red", lw=3, label="Right Click"),
            plt.Line2D([0], [0], color="blue", lw=3, label="Left Click"),
        ]

        # Motion correction legend (markers)
        seen_moco = set()
        moco_lines = []

        for idx, item in enumerate(all_means_right):
            if item["moco"] not in seen_moco:
                moco_lines.append(
                    plt.Line2D([0], [0], color="black", marker=markers[idx % len(markers)],
                               lw=moco_widths[idx % len(moco_widths)], label=item["moco"], linestyle='-')
                )
                seen_moco.add(item["moco"])

        if figtype=="sep":
            # ----------------
            # Plot Right clicks
            # ----------------
            for item in all_means_right:
                rest_idx = np.where(~task_vector_right)[0][:7]#[12:30]
                baseline = item["mean"][rest_idx].mean()
                percent_change_right = ((item["mean"][delay_volumes:] - baseline) / baseline) * 100
                axes[0].plot(
                    x[delay_volumes:], percent_change_right,
                    color=item["color"], marker=item["marker"],
                    lw=item["linewidth"], alpha=0.9, label= item["moco"]
                )
                axes[0].grid(True)
                ymin, ymax = plt.ylim()
                for start, end in consecutive_blocks(task_vector_right[delay_volumes:]):
                    axes[0].fill_between(range(start, end+1), 6, 9, color="red", alpha=1.0, edgecolor="red", linewidth=2)
                axes[0].set_xlabel("Timepoint (fMRI Volume #)")
                axes[0].set_ylabel("%ΔBOLD")
                axes[0].set_xlim(-1, 160)
                axes[0].set_ylim(-5, 10)
                axes[0].figure.tight_layout()
                axes[0].set_title(
                    f"%ΔBOLD: Click Right \n Sub-{subj:02}",
                    color=subj_color,
                    fontweight='bold',
                    fontsize=40,
                )
                axes[0].legend(loc='lower right')
            # ----------------
            # Plot Left clicks
            # ----------------
            for item in all_means_left:
                rest_idx = np.where(~task_vector_left)[0][:7]#[15:27]
                baseline = item["mean"][rest_idx].mean()
                percent_change_left = ((item["mean"][delay_volumes:] - baseline) / baseline) * 100
                axes[1].plot(
                    x[delay_volumes:], percent_change_left,
                    color=item["color"], marker=item["marker"],
                    lw=item["linewidth"], alpha=0.9, label= item["moco"]
                )
                axes[1].grid(True)
                ymin, ymax = plt.ylim()
                for start, end in consecutive_blocks(task_vector_left[delay_volumes:]):
                    axes[1].fill_between(range(start, end+1), 6, 9, color="blue", alpha=1.0, edgecolor="blue", linewidth=2)
                axes[1].set_xlabel("Timepoint (fMRI Volume #)")
                axes[1].set_ylabel("%ΔBOLD")
                axes[1].set_xlim(-1, 160)
                axes[1].set_ylim(-5, 10)
                axes[1].set_title(
                    f"% BOLD Change: Click Right vs Click Left \n Sub-{subj:02}",
                    color=subj_color,
                    fontweight='bold',
                    fontsize=20,
                )
                axes[1].legend(loc='lower right')                
        if figtype=="tog":
            # Plot Right clicks - Left Clicks
            i=0
            for ax, item_r, item_l in zip(axes, all_means_right,all_means_left):                
                rest_idx_r = np.where(~task_vector_right)[0][:7]#[12:30]
                baseline_r = item_r["mean"][rest_idx_r][delay_volumes:].mean()
                percent_change_right = ((item_r["mean"][delay_volumes:] - baseline_r) / baseline_r) * 100
                
                rest_idx_l = np.where(~task_vector_left)[0][:7]#[12:30]
                baseline_l = item_l["mean"][rest_idx_l][delay_volumes:].mean()
                percent_change_left = ((item_l["mean"][delay_volumes:] - baseline_l) / baseline_l) * 100
                ax.plot(
                    x[delay_volumes:], percent_change_right-percent_change_left,
                    color=adjust_color_tone(subj_color,moco_brightness[i]), marker=item_r["marker"], lw=item_r["linewidth"], alpha=0.9,
                    #label= item_r["moco"]
                )
                i+=1
                ax.grid(True)
                ymin, ymax = plt.ylim()
                for start, end in consecutive_blocks(task_vector_right[delay_volumes:]):
                    ax.fill_between(range(start, end+1), 6, 9, color="red", alpha=1.0, edgecolor="red", linewidth=2)
                for start, end in consecutive_blocks(task_vector_left[delay_volumes:]):
                    ax.fill_between(range(start, end+1), -9, -6, color="blue", alpha=1.0, edgecolor="blue", linewidth=2)

            # Add legends
            #plt.gca().add_artist(plt.legend(handles=click_lines, title="Click Type", loc="center right"))
            #plt.gca().add_artist(plt.legend(handles=moco_lines, title="Motion Correction", loc="upper right"))
                ax.legend(loc='lower right')
                ax.set_xlabel("Timepoint (fMRI Volume #)")
                ax.set_ylabel("%ΔBOLD: Click Right - Click Left")
                ax.set_xlim(-1, 160)
                ax.set_ylim(-12, 12)
                ax.figure.tight_layout()
                fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at top for suptitle
                fig.suptitle(
                    f"%ΔBOLD: Click Right vs Click Left \n Sub-{subj:02}",
                    color=subj_color,
                    fontweight='bold',
                    fontsize=40,
                    x=0.5,     # center horizontally
                    y=0.92     # slightly below top edge
                )
        # Save figure
        out_dir = os.path.join(FMRIPREP_PATH, 'figures')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(os.path.join(grp_dir,'figures'),
            f'sub-{subj:02}_ses-{ses}_DeltaBOLDTest_contrast-{contrast}_ISMRM.png'))
            
        plt.show()

