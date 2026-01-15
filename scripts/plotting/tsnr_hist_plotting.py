import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.io.params import *
spaces = ["MNI152NLin2009cAsym", "T1w", "native bold"]

for subj in subjects:
    for ses in sessions:
        plt.figure(figsize=(10, 6))
        subj_color = subject_colors.get(subj, 'black')

        # Plot each moco × space combination
        for moco_label, marker, m_lw, bright_factor in zip(list(mocos.keys()), markers, moco_widths, moco_brightness):
            moco_color = adjust_color_tone(subj_color, bright_factor)

            for space, ls in zip(spaces, line_styles):
                FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
                tsnr_file = os.path.join(
                    FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}', 'stats',
                    f"sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco_label}.npy"
                )

                if not os.path.exists(tsnr_file):
                    print(f"Missing file: {tsnr_file}")
                    continue

                tsnr_data = np.load(tsnr_file)
                tsnr_data = tsnr_data[(tsnr_data > 0) & np.isfinite(tsnr_data)]
                if tsnr_data.size == 0:
                    print(f"No valid tSNR values in {tsnr_file}")
                    continue

                counts, bins = np.histogram(tsnr_data, bins=100, density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])

                plt.plot(
                    bin_centers, counts,
                    color=moco_color,
                    linestyle=ls,
                    linewidth=m_lw,
                    alpha=0.9,
                    marker=marker,
                    markevery=10,
                    markersize=4
                )

        # Legends
        moco_legend = [
            Line2D([0], [0],
                   color=adjust_color_tone(subj_color, bright),
                   marker=m,
                   linestyle='-',
                   lw=lw,
                   markersize=6,
                   label=ml)
            for ml, m, bright, lw in zip(mocos, markers, moco_brightness, moco_widths)
        ]
        space_legend = [
            Line2D([0], [0], color='black', lw=2, ls=ls, label=sp)
            for sp, ls in zip(spaces, line_styles)
        ]

        # Plot formatting
        plt.xlim(0, 120)
        plt.xlabel("tSNR")
        plt.ylabel("Density")
        plt.title(f"Temporal SNR Histograms - sub-{subj:02}", color=subj_color, fontweight='bold')

        # Add legends
        first_legend = plt.legend(handles=moco_legend, title="Motion correction", loc="center right")
        plt.gca().add_artist(first_legend)
        plt.legend(handles=space_legend, title="Space", loc="upper right")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.join(grp_dir,'figures') ,exist_ok=True)
        plt.savefig(os.path.join(grp_dir,'figures',f"sub-{subj:02}_ses-{ses}_tSNRHist.png"))
        plt.show()


import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.io.params import *

spaces = ["MNI152NLin2009cAsym", "T1w"]

# Create one single figure
plt.figure(figsize=(8, 5))

for subj in subjects:
    subj_color = subject_colors.get(subj, 'black')

    for ses in sessions:

        for moco_label, marker, m_lw, bright_factor in zip(
            list(mocos.keys()), markers, moco_widths, moco_brightness
        ):

            moco_color = adjust_color_tone(subj_color, bright_factor)

            for space, ls in zip(spaces, line_styles):

                FMRIPREP_PATH = os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
                tsnr_file = os.path.join(
                    FMRIPREP_PATH,
                    f"sub-{subj:02}",
                    f"ses-{ses}",
                    "stats",
                    f"sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco_label}.npy"
                )

                if not os.path.exists(tsnr_file):
                    print(f"Missing file: {tsnr_file}")
                    continue

                tsnr_data = np.load(tsnr_file)
                tsnr_data = tsnr_data[(tsnr_data > 0) & np.isfinite(tsnr_data)]
                if tsnr_data.size == 0:
                    print(f"No valid tSNR values in {tsnr_file}")
                    continue

                counts, bins = np.histogram(tsnr_data, bins=100, density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])

                plt.plot(
                    bin_centers, counts,
                    color=moco_color,
                    linestyle=ls,
                    linewidth=m_lw,
                    alpha=0.7,
                    marker=marker,
                    markevery=10,
                    markersize=3,
                )

# Motion-correction legend
moco_legend = [
    Line2D([0], [0],
           color='black',
           marker=m,
           linestyle='-',
           lw=lw,
           markersize=6,
           label=ml)
    for ml, m, lw in zip(mocos.keys(), markers, moco_widths)
]

# Space legend
space_legend = [
    Line2D([0], [0], color='black', lw=2, ls=ls, label=sp)
    for sp, ls in zip(spaces, line_styles)
]

# Subject legend
subject_legend = [
    Line2D([0], [0],
           color=subject_colors.get(subj, 'black'),
           lw=4,
           label=f"sub-{subj:02}")
    for subj in subjects
]

# Formatting
plt.xlim(0, 120)
plt.xlabel("tSNR")
plt.ylabel("Density")
plt.title("Temporal SNR Histograms Across All Subjects")

# Add legends
subj_legend_obj = plt.legend(handles=subject_legend, title="Subjects",
                             loc="center right")
plt.gca().add_artist(subj_legend_obj)

plt.legend(handles=space_legend, title="Space", loc="upper right")

plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(os.path.join(grp_dir, 'figures'), exist_ok=True)
plt.savefig(os.path.join(grp_dir, 'figures', "AllSubjects_tSNRHist.png"))
plt.show()



