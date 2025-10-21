#####DO NOT REVIEW, CHANGES IN PATHS#####

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import matplotlib.patches as mpatches

space = "MNI152NLin2009cAsym"
contrasts_names = ['calculations', 'clic right vs clic left']
subjects = [1]
sessions = [1]

conditions = ["ONAVoffPEERSoff", "ONAVoffPEERSon", "ONAVonPEERSoff", "ONAVonPEERSon"]
color_map = {
    "ONAVoffPEERSoff": "#1f77b4",  # blue
    "ONAVoffPEERSon": "#ffbf00",   # yellow
    "ONAVonPEERSoff": "#2ca02c",   # green
    "ONAVonPEERSon": "#d62728"     # red
}

# Collect data
all_data = {contrast: {cond: [] for cond in conditions} for contrast in contrasts_names}

for subj in subjects:
    for ses in sessions:
        for contrast in contrasts_names:
            data_paths = [
                "/home/zamor/Documents/TRISTAN/sub-01/data_onavOFFPEERSOFF",
                "/home/zamor/Documents/TRISTAN/sub-01/data_onavOFFPEERSON",
                "/home/zamor/Documents/TRISTAN/sub-01/data_onavONPEERSOFF",
                "/home/zamor/Documents/TRISTAN/sub-01/data_onavONPEERSON"
            ]

            file_suffixes = [
                "nomoconopeers",
                "nomocopeers",
                "moconopeers",
                "mocopeers"
            ]

            for cond, data_dir, suffix in zip(conditions, data_paths, file_suffixes):
                FMRIPREP_PATH = os.path.join(data_dir, 'derivatives', 'fmriprep')
                arr = np.load(os.path.join(FMRIPREP_PATH, 'stat',
                    f'sub-{subj:02}_ses-{ses}_zarray_{contrast}_{space}_{suffix}.npy'))
                arr = arr[arr > 0]
                all_data[contrast][cond].extend(arr)

# Prepare boxplot positions
n_conditions = len(conditions)
gap = 1  # gap between contrasts
positions = []
group_values = []
group_colors = []
for i, contrast in enumerate(contrasts_names):
    base_pos = i * (n_conditions + gap)
    for j, cond in enumerate(conditions):
        positions.append(base_pos + j + 1)
        group_values.append(all_data[contrast][cond])
        group_colors.append(color_map[cond])

# Statistical tests vs. baseline (blue) for each contrast
stars_with_dir = []
for contrast in contrasts_names:
    baseline_data = all_data[contrast]["ONAVoffPEERSoff"]
    for cond in conditions:
        vals = all_data[contrast][cond]
        if cond == "ONAVoffPEERSoff":
            stars_with_dir.append("")
            continue
        stat, p = mannwhitneyu(baseline_data, vals, alternative='two-sided')
        p_corr = min(p*3, 1.0)  # Bonferroni correction per contrast
        if p_corr < 0.001:
            star = "***"
        elif p_corr < 0.01:
            star = "**"
        elif p_corr < 0.05:
            star = "*"
        else:
            star = ""
        if star:
            direction = "↑" if np.median(vals) > np.median(baseline_data) else "↓"
            star += direction
        stars_with_dir.append(star)

# Plot
plt.figure(figsize=(12, 6))
box = plt.boxplot(group_values, positions=positions, patch_artist=True, widths=0.6, showfliers=False)

# Color boxes
for patch, color in zip(box['boxes'], group_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add stars
for i, (star, median) in enumerate(zip(stars_with_dir, box['medians'])):
    if star:
        median_y = median.get_ydata()[0]
        plt.text(positions[i], median_y + (median_y * 0.05), star, ha='center', va='bottom', fontsize=12)

# Add contrast names centered above each group of 4 boxes
for i, contrast in enumerate(contrasts_names):
    group_center = i*(n_conditions+gap) + (n_conditions+1)/2
    plt.text(group_center, plt.ylim()[1]*0.95, contrast, ha='center', va='top', fontsize=14, fontweight='bold')

# Add threshold line
plt.axhline(3.0, color='gray', linestyle='--', linewidth=1, label='Z=3 threshold')
plt.ylabel("Z-score")
plt.xticks([])  # hide x-axis ticks
plt.title(f"Z-score Distributions by Condition ({space})", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.3)

# Create legend for conditions
patches = [mpatches.Patch(color=color_map[cond], label=cond) for cond in conditions]
plt.legend(handles=patches, title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()