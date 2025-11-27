import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.io.params import *
spaces = ["MNI152NLin2009cAsym", "T1w", "native bold"]
mocos_ = ["SNAVoffPEERSoff", 
         "SNAVonPEERSon"]
for subj in subjects:
    for ses in sessions:
        plt.figure(figsize=(10, 6))
        subj_color = subject_colors.get(subj, 'black')

        # Plot each moco × space combination
        for moco_label, moco_, marker, m_lw, bright_factor in zip(mocos, mocos_, markers, moco_widths, moco_brightness):
            moco_color = adjust_color_tone(subj_color, bright_factor)

            for space, ls in zip(spaces, line_styles):
                data_dir = f"{base_dir}/sub-{subj:02}/data_{moco_label}"
                FMRIPREP_PATH = os.path.join(data_dir, 'derivatives', 'fmriprep')
                tsnr_file = os.path.join(
                    FMRIPREP_PATH, 'stat',
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
            for ml, m, bright, lw in zip(mocos_, markers, moco_brightness, moco_widths)
        ]
        space_legend = [
            Line2D([0], [0], color='black', lw=2, ls=ls, label=sp)
            for sp, ls in zip(spaces, line_styles)
        ]

        # Plot formatting
        plt.xlim(0, 200)
        plt.xlabel("tSNR")
        plt.ylabel("Density")
        plt.title(f"tSNR Histograms - sub-{subj:02}", color=subj_color, fontweight='bold')

        # Add legends
        first_legend = plt.legend(handles=moco_legend, title="Motion correction", loc="upper right")
        plt.gca().add_artist(first_legend)
        plt.legend(handles=space_legend, title="Space", loc="center right")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.join(grp_dir,'figures') ,exist_ok=True)
        plt.savefig(os.path.join(grp_dir,'figures',f"sub-{subj:02}_ses-{ses}_tSNRHist.png"))
        plt.show()


