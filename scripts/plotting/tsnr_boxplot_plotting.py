import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.utils.analysis_utils import *
from tristan_pipeline.io.params import *

spaces = ["MNI152NLin2009cAsym"]

###########LOAD SUBJECT-WISE tSNR DATA###########
subject_tsnr_values = {}
for subj in subjects:
    subj_values = []
    for moco_label in mocos:
        tsnr_all_spaces = []
        for space in spaces:
            data_dir = f"{base_dir}/sub-{subj:02}/data_{moco_label}"
            FMRIPREP_PATH = os.path.join(data_dir, 'derivatives', 'fmriprep')
            tsnr_file = os.path.join(FMRIPREP_PATH, 'stat',
                                     f"sub-{subj:02}_ses-{sessions[0]}_tSNRmap_space-{space}_{moco_label}.npy")
            if os.path.exists(tsnr_file):
                tsnr_data = np.load(tsnr_file)
                tsnr_data = tsnr_data[(tsnr_data > 0) & np.isfinite(tsnr_data)]
                tsnr_all_spaces.append(tsnr_data)
        if tsnr_all_spaces:
            subj_values.append(np.concatenate(tsnr_all_spaces))
        else:
            subj_values.append(np.array([]))
    subject_tsnr_values[subj] = subj_values

###########LOAD GROUP tSNR DATA###########
group_tsnr_values = []
for moco_label in mocos:
    group_file = os.path.join(grp_dir,'stats', f"group_tSNR_space-{spaces[0]}_{moco_label}.nii.gz")
    if os.path.exists(group_file):
        img = nib.load(group_file)
        data = img.get_fdata()
        data = data[(data > 0) & np.isfinite(data)]
        group_tsnr_values.append(data)
    else:
        print(f"Missing group file: {group_file}")
        group_tsnr_values.append(np.array([]))



plt.figure(figsize=(12,6))
n_mocos = len(mocos)
cluster_spacing = 3       # large gap between mocos
subject_offset = 0.5      # small offset inside cluster
group_offset = 0 * subject_offset  # slightly larger offset for group
box_positions = []
for moco_idx in range(n_mocos):
    cluster_start = moco_idx * cluster_spacing
    for subj_idx in range(len(subjects)):
        box_positions.append(cluster_start + subj_idx * subject_offset)
    box_positions.append(cluster_start + len(subjects) * subject_offset + group_offset)  # group
# prepare values, colors, linewidths
all_values = []
all_colors = []
all_lw = []
for moco_idx in range(n_mocos):
    for subj_idx, subj in enumerate(subjects):
        vals = subject_tsnr_values[subj][moco_idx]
        all_values.append(vals)
        base_color = subject_colors[subj]
        all_colors.append(adjust_color_tone(base_color, moco_brightness[moco_idx]))
        all_lw.append(moco_widths[moco_idx])
    # group
    all_values.append(group_tsnr_values[moco_idx])
    all_colors.append(adjust_color_tone('lightsalmon', moco_brightness[moco_idx]))
    all_lw.append(moco_widths[moco_idx])
# plot boxplots
box = plt.boxplot(all_values, positions=box_positions, widths=0.25, patch_artist=True, showfliers=False,
                  medianprops=dict(color='black', linewidth=2))
for patch, color, lw in zip(box['boxes'], all_colors, all_lw):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_linewidth(lw)
# add stars/arrows for subjects
for subj_idx, subj in enumerate(subjects):
    stars = compute_stars(subject_tsnr_values[subj])
    for moco_idx, star in enumerate(stars):
        pos = moco_idx * cluster_spacing + subj_idx * subject_offset
        median_y = np.median(subject_tsnr_values[subj][moco_idx])
        if star:
            plt.text(pos, median_y + median_y*0.05, star,
                     ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
# add stars for group
group_stars = compute_stars(group_tsnr_values)
for moco_idx, star in enumerate(group_stars):
    pos = moco_idx * cluster_spacing + len(subjects) * subject_offset + group_offset
    median_y = np.median(group_tsnr_values[moco_idx])
    if star:
        plt.text(pos, median_y + median_y*0.05, star,
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
# x-ticks at center of each cluster
tick_positions = [moco_idx * cluster_spacing + (len(subjects) * subject_offset + group_offset)/2 for moco_idx in range(n_mocos)]
plt.xticks(tick_positions, mocos)
plt.ylabel("tSNR")
plt.grid(True, alpha=0.3, linestyle='--')
plt.title(f"Subject and Group-level tSNR Distributions - {space}\n(Significance vs. ONAVoffPEERSoff)", fontweight='bold')
# legend
legend_elements = [Line2D([0],[0], color=c, lw=3, label=f"sub-{subj:02}") for subj,c in subject_colors.items()]
legend_elements.append(Line2D([0],[0], color='lightsalmon', lw=3, label='Group'))
plt.legend(handles=legend_elements, title="Subjects", loc='upper right')
plt.tight_layout()

os.makedirs(os.path.join(grp_dir,'figures') ,exist_ok=True)
plt.savefig(os.path.join(os.path.join(grp_dir,'figures'), f"boxplot_tSNR_space-{space}"))

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






