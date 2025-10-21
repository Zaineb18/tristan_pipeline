import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tristan_pipeline.utils.plotting_utils import *

##########CONFIG###########
subjects = [1,2,3,4]   # loop over multiple subjects
sessions = [1]
mocos = [
#    ("ONAVoffPEERSoff", 'tab:blue'),
#    ("ONAVoffPEERSon", 'tab:orange'),
#    ("ONAVonPEERSoff", 'tab:green'),
    ("ONAVonPEERSon", 'tab:red')]
spaces = ["MNI152NLin2009cAsym", "T1w", "native bold"]   # anatomical spaces
line_styles = ['-', '--', ':']                 # match number of spaces
alphas = [0.7, 0.9, 1]
widths = [1.5, 2.0, 2.5]
base_dir = "/home/zamor/Documents/TRISTAN/ismrm_dataset"
##########CONFIG###########

for subj in subjects:
    for ses in sessions:
        plt.figure(figsize=(10, 6))        
        for space, ls, alpha, lw in zip(spaces, line_styles, alphas, widths):
            for moco_label, color in mocos:                
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
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha
                )
        # Legend for motion/peer conditions
        color_legend = [
            Line2D([0], [0], color=color, lw=2, label=moco_label)
            for moco_label, color in mocos
        ]
        # Legend for spaces
        style_legend = [
            Line2D([0], [0], color='black', lw=lw, ls=ls, alpha=alpha, label=space)
            for space, ls, alpha, lw in zip(spaces, line_styles, alphas, widths)
        ]
        plt.xlim(0, 200)
        plt.xlabel("tSNR")
        plt.ylabel("Density")
        plt.title(f"tSNR Histograms - sub-{subj:02}")
        first_legend = plt.legend(handles=color_legend, title="Motion correction", loc="upper right")
        plt.gca().add_artist(first_legend)  # Keep first legend
        plt.legend(handles=style_legend, title="Space", loc="upper center")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FMRIPREP_PATH, 'figures',
        f'sub-{subj:02}_ses-{ses}_tSNRHist_{space}'))
        plt.show()
