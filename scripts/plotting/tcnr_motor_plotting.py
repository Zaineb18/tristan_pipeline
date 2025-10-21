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

# ----------------
# CONFIG
# ----------------
subjects = [1,2,3,4]#,2,3,4]
sessions = [1]
mocos = [
    # ("ONAVoffPEERSoff", 'tab:blue'),
    # ("ONAVoffPEERSon", 'tab:orange'),
    # ("ONAVonPEERSoff", 'tab:green'),
    ("ONAVonPEERSon", 'tab:red')
]
spaces = ["MNI152NLin2009cAsym", "T1w"]
space_styles = ['-', '--', ':']  # assign one per space
contrasts = ["clic right vs clic left"]
# Subject-specific acquisition params
d_vols = {1:0, 2:2, 3:2, 4:2}
n_vols = {1:153, 2:155, 3:155, 4:155}
trs = {1:2.12, 2:2.12, 3:2.16, 4:2.12}
linestyles = ['-', '--', 'dashdot', ':']  # differentiate mocos if multiple
base_dir = "/home/zamor/Documents/TRISTAN/ismrm_dataset"
#stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"



base_dir = "/home/zamor/Documents/TRISTAN/data_Caro/"
linestyles = ['-', '--', 'dashdot', ':']  # differentiate mocos if multiple
stimfile = "/home/zamor/Documents/TRISTAN/data_Caro/session1_localizer_standard.csv"
contrasts_names = ['calculations','checkerboard vs the others','clic right vs clic left']
mocos = [
    ("NA", 'tab:red')
]
spaces = ["MNI152NLin2009cAsym", "T1w"]
space_styles = ['-', '--', ':']  # assign one per space
contrasts = ["clic right vs clic left"]
subjects = [1,2,3]
sessions = [1]
d_vols= {1:0, 2:0, 3:0, 4:0}
n_vols = {1:263, 2:263, 3:263, 4:263}
trs= {1:1.2, 2:1.2, 3:1.2, 4:1.2}

# ----------------
# LOOP OVER SUBJECTS
# ----------------
for subj in subjects:
    for ses in sessions:
        n_scans = n_vols[subj]
        delay_volumes = d_vols[subj]
        tr = trs[subj]

        #stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
        events, task_vector_right, task_vector_left, task_vector_calc = events_task_vectors(
            stimfile, n_scans=n_scans, delay_volumes=delay_volumes, tr=tr
        )

        all_means_right, all_means_left = [], []

        # ----------------
        # LOAD DATA
        # ----------------
        for idx_moco, (moco_label, _) in enumerate(mocos):
            #data_dir = f"{base_dir}/sub-{subj:02}/data_{moco_label}"
            #FMRIPREP_PATH = os.path.join(data_dir, "derivatives", "fmriprep")
            FMRIPREP_PATH = os.path.join(base_dir, "derivatives", "fmriprep")

            for idx_space, space in enumerate(spaces):
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,WM_PATH,CSF_PATH,xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(
                    FMRIPREP_PATH, subj, ses, space
                )
                bold_file, mask_file = FUNC_PATH[0], MASK_PATH[0]

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

                    if contrast == "clic right vs clic left":
                        pos_mask = math_img("img > 3.2905", img=z_map)
                        neg_mask = math_img("img < -3.2905", img=z_map)
                        mean_right = apply_mask(bold_file, pos_mask).mean(axis=1)
                        mean_left = apply_mask(bold_file, neg_mask).mean(axis=1)
                        std_right = apply_mask(bold_file, pos_mask).std(axis=1)
                        std_left = apply_mask(bold_file, neg_mask).std(axis=1)
                        all_means_right.append({
                            "moco": moco_label,
                            "space": space,
                            "mean": mean_right,
                            "std": std_right,
                            "linestyle": linestyles[idx_moco % len(linestyles)],
                            "space_style": space_styles[idx_space % len(space_styles)]
                        })
                        all_means_left.append({
                            "moco": moco_label,
                            "space": space,
                            "mean": mean_left,
                            "std": std_left, 
                            "linestyle": linestyles[idx_moco % len(linestyles)],
                            "space_style": space_styles[idx_space % len(space_styles)]
                        })

        # ----------------
        # PLOT
        # ----------------
        plt.figure(figsize=(20, 12))
        x = np.arange(n_scans)

        # Click type legend (red/blue)
        click_lines = [
            plt.Line2D([0], [0], color="red", label="Right Click"),
            plt.Line2D([0], [0], color="blue", label="Left Click")
        ]

        # Motion correction legend (linestyle)
        seen_moco = set()
        strategy_lines = []
        for item in all_means_right:
            if item["moco"] not in seen_moco:
                strategy_lines.append(
                    plt.Line2D([0], [0], color="black", linestyle=item["linestyle"], label=item["moco"])
                )
                seen_moco.add(item["moco"])

        # Space legend (line style)
        seen_space = set()
        space_lines = []
        for item in all_means_right:
            if item["space"] not in seen_space:
                space_lines.append(
                    plt.Line2D([0], [0], color="black", lw=2, linestyle=item["space_style"], label=item["space"])
                )
                seen_space.add(item["space"])

        # ----------------
        # Plot Right clicks (red)
        # ----------------
        for item in all_means_right:
            rest_idx = np.where(~task_vector_right)[0]
            rest_idx = rest_idx[:7]
            baseline = item["mean"][rest_idx].mean()
            percent_change = ((item["mean"][delay_volumes:] - baseline) / baseline) * 100
            #plt.plot(x[delay_volumes:], percent_change,
            #         color="red", linestyle=item["space_style"])
            dispersion = (item["std"][delay_volumes:] / baseline) * 100  # <-- use std
            plt.plot(x[delay_volumes:], percent_change,
             color="red", linestyle=item["space_style"])
            #plt.fill_between(x[delay_volumes:], percent_change - dispersion, percent_change + dispersion,
            #         color="red", alpha=0.2)
        # ----------------
        # Plot Left clicks (blue)
        # ----------------
        for item in all_means_left:
            rest_idx = np.where(~task_vector_left)[0]
            rest_idx = rest_idx[:7]
            baseline = item["mean"][rest_idx].mean()
            percent_change = ((item["mean"][delay_volumes:] - baseline) / baseline) * 100
            #plt.plot(x[delay_volumes:], percent_change + 10,
            #         color="blue", linestyle=item["space_style"])
            dispersion = (item["std"][delay_volumes:] / baseline) * 100  # <-- use std
            plt.plot(x[delay_volumes:], percent_change + 10,
             color="blue", linestyle=item["space_style"])
            #plt.fill_between(x[delay_volumes:], percent_change + 10 - dispersion, percent_change + 10 + dispersion,
            #         color="blue", alpha=0.2)

        # Shaded task blocks
        ymin, ymax = plt.ylim()
        for start, end in consecutive_blocks(task_vector_right[delay_volumes:]):
            plt.fill_between(range(start, end+1), ymin, ymax, color="red", alpha=0.5)
        for start, end in consecutive_blocks(task_vector_left[delay_volumes:]):
            plt.fill_between(range(start, end+1), ymin, ymax, color="blue", alpha=0.5)

        # Add legends
        plt.gca().add_artist(plt.legend(handles=strategy_lines, title="Motion Correction", loc="upper right"))
        plt.gca().add_artist(plt.legend(handles=space_lines, title="Space", loc="upper center"))
        plt.gca().add_artist(plt.legend(handles=click_lines, title="Click Type", loc="upper left"))

        plt.xlabel("Volumes")
        plt.ylabel("% BOLD Signal Change")
        plt.title(f"% BOLD Change - Click Right vs Click Left\nsub-{subj:02}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FMRIPREP_PATH, 'figures',
        f'sub-{subj:02}_ses-{ses}_tCNR_contrast-{contrast}.png'))
        plt.show()