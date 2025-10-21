import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.image import math_img
from nilearn.masking import apply_mask
from nilearn.glm import threshold_stats_img

from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
from tristan_pipeline.analysis.glm_utils import *
from tristan_pipeline.analysis.analysis_utils import *
from tristan_pipeline.plotting.plotting_utils import *

# ----------------
# CONFIG
# ----------------
subjects = [1,2,3,4]
sessions = [1]
mocos = [
    # ("ONAVoffPEERSoff", 'tab:blue'),
    # ("ONAVoffPEERSon", 'tab:orange'),
    # ("ONAVonPEERSoff", 'tab:green'),
    ("ONAVonPEERSon", 'tab:red')
]
spaces = ["MNI152NLin2009cAsym", "T1w"]
space_styles = ['-', '--', ':']  # assign one per space
contrasts = ["calculations"]

# Subject-specific acquisition params
d_vols = {1:0, 2:2, 3:2, 4:2}
n_vols = {1:153, 2:155, 3:155, 4:155}
trs = {1:2.12, 2:2.12, 3:2.16, 4:2.12}

linestyles = ['-', '--', 'dashdot', ':']  # differentiate mocos if multiple
base_dir = "/home/zamor/Documents/TRISTAN/ismrm_dataset"

# ----------------
# LOOP OVER SUBJECTS
# ----------------
for subj in subjects:
    for ses in sessions:
        n_scans = n_vols[subj]
        delay_volumes = d_vols[subj]
        tr = trs[subj]

        stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
        events, task_vector_right, task_vector_left, task_vector_calc = events_task_vectors(
            stimfile, n_scans=n_scans, delay_volumes=delay_volumes, tr=tr
        )

        all_means = []

        # ----------------
        # LOAD DATA
        # ----------------
        for idx_moco, (moco_label, _) in enumerate(mocos):
            data_dir = f"{base_dir}/sub-{subj:02}/data_{moco_label}"
            FMRIPREP_PATH = os.path.join(data_dir, "derivatives", "fmriprep")

            for idx_space, space in enumerate(spaces):
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH, xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(
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

                    if contrast == "calculations":
                        mask = math_img("img > 3.1", img=z_map)
                        mean = apply_mask(bold_file, mask).mean(axis=1)
                        std = apply_mask(bold_file, mask).std(axis=1)
                        all_means.append({
                            "moco": moco_label,
                            "space": space,
                            "mean": mean,
                            "std": std,
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
            plt.Line2D([0], [0], color="green", label="Calculations"),
            #plt.Line2D([0], [0], color="blue", label="Left Click")
        ]

        # Motion correction legend (linestyle)
        seen_moco = set()
        strategy_lines = []
        for item in all_means:
            if item["moco"] not in seen_moco:
                strategy_lines.append(
                    plt.Line2D([0], [0], color="black", linestyle=item["linestyle"], label=item["moco"])
                )
                seen_moco.add(item["moco"])

        # Space legend (line style)
        seen_space = set()
        space_lines = []
        for item in all_means:
            if item["space"] not in seen_space:
                space_lines.append(
                    plt.Line2D([0], [0], color="black", lw=2, linestyle=item["space_style"], label=item["space"])
                )
                seen_space.add(item["space"])

        for item in all_means:
            rest_idx = np.where(~task_vector_calc)[0]
            rest_idx = rest_idx[12:30]
            baseline = item["mean"][rest_idx].mean()
            percent_change = ((item["mean"][delay_volumes:] - baseline) / baseline) * 100
            plt.plot(x[delay_volumes:], percent_change,
             color="red", linestyle=item["space_style"])

        # Shaded task blocks
        ymin, ymax = plt.ylim()
        for start, end in consecutive_blocks(task_vector_calc[delay_volumes:]):
            plt.fill_between(range(start, end+1), ymin, ymax, color="green", alpha=0.5)

        # Add legends
        plt.gca().add_artist(plt.legend(handles=strategy_lines, title="Motion Correction", loc="upper right"))
        plt.gca().add_artist(plt.legend(handles=space_lines, title="Space", loc="upper center"))
        plt.gca().add_artist(plt.legend(handles=click_lines, title="Calculations", loc="upper left"))

        plt.xlabel("Volumes")
        plt.ylabel("% BOLD Signal Change")
        plt.title(f"% BOLD Change - Calculations \nsub-{subj:02}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FMRIPREP_PATH, 'figures',
        f'sub-{subj:02}_ses-{ses}_tCNR_contrast-{contrast}.png'))
        plt.show()