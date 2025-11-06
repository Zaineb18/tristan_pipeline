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
#spaces = ["MNI152NLin2009cAsym"]
spaces = ["T1w"]

# ----------------
# LOOP OVER SUBJECTS
# ----------------
for subj in subjects:
    for ses in sessions:
        n_scans = n_vols[subj]
        delay_volumes = d_vols[subj]
        tr = trs[subj]
        subj_color = subject_colors.get(subj, 'black')

        stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
        events, task_vector_right, task_vector_left, task_vector_calc = events_task_vectors(
            stimfile, n_scans=n_scans, delay_volumes=delay_volumes, tr=tr
        )

        all_means_right, all_means_left = [], []
        print(subj)
        # ----------------
        # LOAD DATA
        # ----------------
        for idx_moco, moco_label in enumerate(mocos):
            data_dir = f"{base_dir}/sub-{subj:02}/data_{moco_label}"
            FMRIPREP_PATH = os.path.join(data_dir, "derivatives", "fmriprep")
            print(moco_label)
            for idx_space, space in enumerate(spaces):
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,WM_PATH,CSF_PATH,xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(
                    FMRIPREP_PATH, subj, ses, space
                )
                bold_file, mask_file = FUNC_PATH[0], MASK_PATH[0]
                print(space)
                for contrast in contrasts:
                    z_map_path = os.path.join(
                        FMRIPREP_PATH, "stat",
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

        # ----------------
        # PLOT - RIGHT & LEFT ON SAME FIGURE
        # ----------------
        plt.figure(figsize=(25, 14))
        x = np.arange(n_scans)

        # Click type legend
        click_lines = [
            plt.Line2D([0], [0], color="red", lw=3, label="Right Click"),
            plt.Line2D([0], [0], color="blue", lw=3, label="Left Click")
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

        # Space legend (line style)
        seen_space = set()
        space_lines = []
        for idx, item in enumerate(all_means_right):
            if item["space"] not in seen_space:
                space_lines.append(
                    plt.Line2D([0], [0], color="black", lw=2,
                               linestyle=line_styles[idx % len(line_styles)], label=item["space"])
                )
                seen_space.add(item["space"])

        # ----------------
        # Plot Right clicks
        # ----------------
        for item in all_means_right:
            rest_idx = np.where(~task_vector_right)[0][:7]#[12:30]
            baseline = item["mean"][rest_idx].mean()
            percent_change = ((item["mean"][delay_volumes:] - baseline) / baseline) * 100
            plt.plot(
                x[delay_volumes:], percent_change,
                color=item["color"], marker=item["marker"], lw=item["linewidth"], alpha=0.9
            )

        # ----------------
        # Plot Left clicks (offset +10)
        # ----------------
        for item in all_means_left:
            rest_idx = np.where(~task_vector_left)[0][:7]#[15:27]
            baseline = item["mean"][rest_idx].mean()
            percent_change = ((item["mean"][delay_volumes:] - baseline) / baseline) * 100
            plt.plot(
                x[delay_volumes:], percent_change + 10,
                color=item["color"], marker=item["marker"], lw=item["linewidth"], alpha=0.9
            )

        # Shaded task blocks
        ymin, ymax = plt.ylim()
        for start, end in consecutive_blocks(task_vector_right[delay_volumes:]):
            plt.fill_between(range(start, end+1), ymin, ymax, color="red", alpha=0.9)
        for start, end in consecutive_blocks(task_vector_left[delay_volumes:]):
            plt.fill_between(range(start, end+1), ymin, ymax, color="blue", alpha=0.9)

        # Add legends
        plt.gca().add_artist(plt.legend(handles=space_lines, title="Space", loc="upper center"))
        plt.gca().add_artist(plt.legend(handles=click_lines, title="Click Type", loc="upper left"))
        plt.gca().add_artist(plt.legend(handles=moco_lines, title="Motion Correction", loc="upper right"))

        plt.xlabel("Timepoint (fMRI Volume #)")
        plt.ylabel("% BOLD Signal Change")
        plt.title(f"% BOLD Change - Click Right vs Click Left - sub-{subj:02}", color=subj_color, fontweight='bold', )
        plt.grid(True)
        plt.xlim(-1,160)
        plt.ylim(-7,20)
        plt.tight_layout()

        # Save figure
        out_dir = os.path.join(FMRIPREP_PATH, 'figures')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(os.path.join(grp_dir,'figures'),
        f'sub-{subj:02}_ses-{ses}_tCNR_contrast-{contrast}.png'))
        plt.show()
