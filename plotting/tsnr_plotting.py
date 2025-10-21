import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ----------------
# CONFIG
# ----------------
space = "MNI152NLin2009cAsym"
group_dir = "/home/zamor/Documents/TRISTAN/group_tSNR_maps"

datasets = [
    "ONAVoffPEERSoff",  # baseline
    "ONAVoffPEERSon",
    "ONAVonPEERSoff",
    "ONAVonPEERSon"
]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# ----------------
# LOAD GROUP MAPS
# ----------------
group_tsnr_values = []
labels = []

for moco_label in datasets:
    group_file = os.path.join(group_dir, f"group_tSNR_space-{space}_{moco_label}.nii.gz")
    if not os.path.exists(group_file):
        print(f"Missing file: {group_file}")
        continue
    
    img = nib.load(group_file)
    data = img.get_fdata()
    data = data[(data > 0) & np.isfinite(data)]  # filter invalid values
    group_tsnr_values.append(data)
    labels.append(moco_label)

# ----------------
# STATISTICAL TESTS (vs. baseline)
# ----------------
baseline_data = group_tsnr_values[0]
p_values = [np.nan]  # no test for baseline

for data in group_tsnr_values[1:]:
    stat, p = mannwhitneyu(baseline_data, data, alternative='two-sided')
    p_values.append(p)

# Bonferroni correction
_, p_corrected, _, _ = multipletests(p_values[1:], method='bonferroni')
p_corrected = [np.nan] + list(p_corrected)

# Assign stars based on corrected p-values
stars = []
for p in p_corrected:
    if np.isnan(p):
        stars.append("")
    elif p < 0.001:
        stars.append("***")
    elif p < 0.01:
        stars.append("**")
    elif p < 0.05:
        stars.append("*")
    else:
        stars.append("")

# Determine direction and add arrow
baseline_median = np.median(baseline_data)
stars_with_dir = []
for vals, star in zip(group_tsnr_values, stars):
    if star == "":
        stars_with_dir.append("")
        continue
    current_median = np.median(vals)
    if current_median > baseline_median:
        stars_with_dir.append(star + "↑")
    elif current_median < baseline_median:
        stars_with_dir.append(star + "↓")
    else:
        stars_with_dir.append(star)

# ----------------
# PLOT
# ----------------
plt.figure(figsize=(10, 6))
box = plt.boxplot(
    group_tsnr_values,
    labels=labels,
    patch_artist=True,
    showfliers=False
)

# Color boxes
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add stars & arrows INSIDE boxes
for i, (star, median) in enumerate(zip(stars_with_dir, box['medians'])):
    if star:
        median_y = median.get_ydata()[0]
        plt.text(
            i+1,
            median_y + (median_y * 0.05),  # small offset above median
            star,
            ha='center',
            va='bottom',
            fontsize=14,
            color='black'
        )

plt.ylabel("tSNR")
plt.title(f"Group-level tSNR Distributions - {space}\n(Significance vs. {labels[0]})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

























































import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
plt.rcParams.update({
    'font.size': 16,         # main font size
    'axes.titlesize': 18,    # title
    'axes.labelsize': 18,    # x/y labels
    'legend.fontsize': 14,   # legend
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

subjects = [5]
sessions = [1]
datasets = [
    ("/home/zamor/Documents/TRISTAN/sub-05/data_onavOFFPEERSOFF",
      "ONAVoffPEERSoff", 'tab:blue'),
    ("/home/zamor/Documents/TRISTAN/sub-05/data_onavOFFPEERSON", 
      "ONAVoffPEERSon", 'tab:orange'),
    ("/home/zamor/Documents/TRISTAN/sub-05/data_onavONPEERSOFF",
       "ONAVonPEERSoff", 'tab:green'),
    ("/home/zamor/Documents/TRISTAN/sub-05/data_onavONPEERSON", 
       "ONAVonPEERSon", 'tab:red')
]
spaces = ["MNI152NLin2009cAsym"]#, ["T1w", "native bold"]
line_styles = ['-', '--', ':']   # one per space
alphas = [0.7, 0.9, 1.0]         # transparency per space
widths = [1.5, 2.0, 2.5]         # line width per space

for subj in subjects:
    for ses in sessions:
        plt.figure(figsize=(10, 6))
        
        for space, ls, alpha, lw in zip(spaces, line_styles, alphas, widths):
            for data_dir, moco_label, color in datasets:
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

        # Legend for colors → motion/peers
        color_legend = [
            Line2D([0], [0], color=color, lw=2, label=moco_label)
            for _, moco_label, color in datasets
        ]
        
        # Legend for line style/width → space
        style_legend = [
            Line2D([0], [0], color='black', lw=lw, ls=ls, alpha=alpha, label=space)
            for space, ls, alpha, lw in zip(spaces, line_styles, alphas, widths)
        ]
        
        plt.xlim(0, 200)
        plt.xlabel("tSNR")
        plt.ylabel("Density")
        plt.title("tSNR Histograms")
        first_legend = plt.legend(handles=color_legend, title="Motion correction", loc="upper right")
        plt.gca().add_artist(first_legend)  # Keep first legend
        #plt.legend(handles=style_legend, title="Space", loc="upper center")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()






