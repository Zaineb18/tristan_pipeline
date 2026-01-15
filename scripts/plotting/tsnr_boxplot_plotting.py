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
    for moco_label in list(mocos.keys()):
        tsnr_all_spaces = []
        for space in spaces:
            FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
            tsnr_file = os.path.join(
                    FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{sessions[0]}', 'stats',
                    f"sub-{subj:02}_ses-{sessions[0]}_tSNRmap_space-{space}_{moco_label}.npy"
                )
            if os.path.exists(tsnr_file):
                tsnr_data = np.load(tsnr_file)
                tsnr_data = tsnr_data[(tsnr_data > 0) & np.isfinite(tsnr_data)]
                tsnr_all_spaces.append(tsnr_data)
        if tsnr_all_spaces:
            subj_values.append(np.concatenate(tsnr_all_spaces))
        else:
            subj_values.append(np.array([]))
    subject_tsnr_values[subj] = subj_values

if space=="MNI152NLin2009cAsym":
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


plt.figure(figsize=(9,3))
n_mocos = len(mocos)
cluster_spacing = 3.5       # large gap between mocos
subject_offset = 0.5      # small offset inside cluster
group_offset = 0 * subject_offset  # slightly larger offset for group
box_positions = []
for moco_idx in range(n_mocos):
    cluster_start = moco_idx * cluster_spacing
    for subj_idx in range(len(subjects)):
        box_positions.append(cluster_start + subj_idx * subject_offset)
    if space=="MNI152NLin2009cAsym":
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
    if space=="MNI152NLin2009cAsym":
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
    stars,pvalues, pvalues_corr, uvalues, esvalues = compute_stars(subject_tsnr_values[subj])
    print('subject', subj)
    print('stars', stars)
    print('pvalues', pvalues)
    print('pvalues_corr', pvalues_corr)
    print('uvalues', uvalues)
    print('effect size', esvalues)
    for moco_idx, star in enumerate(stars):
        pos = moco_idx * cluster_spacing + subj_idx * subject_offset
        median_y = np.median(subject_tsnr_values[subj][moco_idx])
        if star:
            plt.text(pos, median_y + median_y*0.05, star,
                     ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')        
# add stars for group
if space=="MNI152NLin2009cAsym":
    group_stars,group_p, group_pbon, group_u, group_es = compute_stars(group_tsnr_values)
    print('stars', group_stars)
    print('pvalues', group_p)
    print('pvalues_corr', group_pbon)
    print('uvalues', group_pbon)
    print('effect size', group_es)
    for moco_idx, star in enumerate(group_stars):
        pos = moco_idx * cluster_spacing + len(subjects) * subject_offset + group_offset
        median_y = np.median(group_tsnr_values[moco_idx])
        if star:
            plt.text(pos, median_y + median_y*0.05, star,
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
# x-ticks at center of each cluster
tick_positions = [moco_idx * cluster_spacing + (len(subjects) * subject_offset + group_offset)/2 for moco_idx in range(n_mocos)]
plt.xticks(tick_positions, list(mocos.keys()))
plt.ylabel("tSNR")
plt.grid(True, alpha=0.3, linestyle='--')
if space=="MNI152NLin2009cAsym":
    plt.title(f"Subject and Group-level tSNR Distributions - {space}\n(Significance vs. SNAVoffPEERSoff)", fontweight='bold', fontsize=12)
else: 
    plt.title(f"Subject-wise tSNR Distributions - {space}\n(Significance vs. SNAVoffPEERSoff)", fontweight='bold', fontsize=12)

# legend
legend_elements = [Line2D([0],[0], color=c, lw=3, label=f"sub-{subj:02}") for subj,c in subject_colors.items()]
if space=="MNI152NLin2009cAsym":
    legend_elements.append(Line2D([0],[0], color='lightsalmon', lw=3, label='Group'))

plt.legend(handles=legend_elements, 
           #loc='upper right'
           loc='center',
           fontsize=12,
           )
plt.tight_layout()

os.makedirs(os.path.join(grp_dir,'figures') ,exist_ok=True)
plt.savefig(os.path.join(os.path.join(grp_dir,'figures'), f"boxplot_tSNR_space-{space}.pdf"))

plt.show()

