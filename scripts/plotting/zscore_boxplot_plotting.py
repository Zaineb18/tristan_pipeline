import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.utils.analysis_utils import *
from tristan_pipeline.io.params import *
from nilearn.glm import threshold_stats_img


space = "T1w"
mocos_ = ["SNAVoffPEERSoff", 
         "SNAVonPEERSon"]
#contrasts_names = [
                #'clic right vs clic left' 
                #'checkerboard',
                #'calculations',
#                'phrases'
#                    ]

subject_z_values = {contrast: {subj: [] for subj in subjects} for contrast in contrasts_names}

#################LOAD DATA PER SUBJECT / PER MOCO#################
for subj in subjects:
    #print('!!!SUBJECT!!!', subj)
    for ses in sessions:
        FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
        for contrast in contrasts_names:
            #print('!!!CONTRAST!!!',contrast)
            subj_values = []
            for moco in list(mocos.keys()):
                #print(moco)
                zmap_path = os.path.join(
                    FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}','stats',
                    f"sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco}.nii"                )
                if not os.path.exists(zmap_path):
                    print(f"Missing file: {zmap_path}")
                    subj_values.append(np.array([]))
                    continue
                img = nib.load(zmap_path)
                img, threshold = threshold_stats_img(img,alpha=0.001,
                    height_control='fpr',two_sided=True)
                arr = img.get_fdata()
                if contrast=="clic right vs clic left":
                    arr_pos = arr[(arr > threshold) & np.isfinite(arr)]
                    arr_neg = arr[(arr < -threshold) & np.isfinite(arr)]
                    subj_values.append({"pos": arr_pos, "neg": arr_neg})
                    #print('Right - MAX ', arr_pos.max(),
                    #      'MEAN', arr_pos.mean(),
                    #      'MEADIAN',np.median(arr_pos))
                    #print('Left - MAX ', arr_neg.min(),
                    #      'MEAN', arr_neg.mean(),
                    #      'MEADIAN',np.median(arr_neg))                    
                else: 
                    arr = arr[(arr > threshold) & np.isfinite(arr)]
                    subj_values.append({"pos": arr, "neg": np.array([])})
                    #print('MAX ', arr.max(),
                    #      'MEAN', arr.mean(),
                    #      'MEADIAN',np.median(arr))
                                        
            subject_z_values[contrast][subj] = subj_values

for contrast in contrasts_names:
    dual_polarity = (contrast == "clic right vs clic left")

    # Larger figure if positive/negative
    fig_size = (12, 6) if dual_polarity else (12, 3)
    plt.figure(figsize=fig_size)

    n_mocos = len(mocos)
    cluster_spacing = 3.0
    subject_offset = 0.6

    # Build box positions and collect data
    box_positions, all_values, all_colors, all_lw = [], [], [], []
    for m_idx, moco_label in enumerate(mocos):
        cluster_start = m_idx * cluster_spacing
        for subj_idx, subj in enumerate(subjects):
            subj_data = subject_z_values[contrast][subj][m_idx]
            base_color = subject_colors[subj]
            color_mod = adjust_color_tone(base_color, moco_brightness[m_idx])

            # Positive side
            pos_pos = cluster_start + subj_idx * subject_offset
            box_positions.append(pos_pos)
            all_values.append(subj_data["pos"])
            all_colors.append(color_mod)
            all_lw.append(moco_widths[m_idx])

            if dual_polarity:
                # Negative side
                neg_pos = cluster_start + subj_idx * subject_offset + subject_offset - 0.6
                box_positions.append(neg_pos)
                all_values.append(subj_data["neg"])
                all_colors.append(adjust_color_tone(base_color, moco_brightness[m_idx] * 0.7))
                all_lw.append(moco_widths[m_idx])

    # Plot boxplots
    box = plt.boxplot(
        all_values,
        positions=box_positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2)
    )
    for patch, color, lw in zip(box['boxes'], all_colors, all_lw):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(lw)

    # Significance stars
    for subj_idx, subj in enumerate(subjects):
        subj_data = subject_z_values[contrast][subj]
        if dual_polarity:
            pos_data = [d["pos"] for d in subj_data]
            neg_data = [d["neg"] for d in subj_data]
            pos_stars, pos_pvalues, pos_pvalues_corr, pos_uvalues = compute_stars_z(pos_data)
            neg_data_abs = [np.abs(arr) for arr in neg_data]
            neg_stars, neg_pvalues, neg_pvalues_corr, neg_uvalues = compute_stars_z(neg_data_abs)
            
            print('CONTRAST', contrast , ' - SUBJECT', subj )
            print('stars_pos', pos_stars)
            print('pvalues_pos', pos_pvalues)
            print('pvalues_corr_pos', pos_pvalues_corr)
            print('uvalues_pos', pos_uvalues)

            print('stars_neg', neg_stars)
            print('pvalues_neg', neg_pvalues)
            print('pvalues_corr_neg', neg_pvalues_corr)
            print('uvalues_neg', neg_uvalues)

            for m_idx, (p_star, n_star, p_pos, p_neg) in enumerate(zip(pos_stars, neg_stars, pos_pvalues_corr, neg_pvalues_corr)):
                if p_star:
                    pos = m_idx * cluster_spacing + subj_idx * subject_offset
                    median_y = np.median(pos_data[m_idx]) if len(pos_data[m_idx]) > 0 else 0
                    label = f"{p_star}\np={p_pos:.3g}"
                    plt.text(pos, median_y + abs(median_y) * 0.05, label,
                             ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
                if n_star:
                    pos = m_idx * cluster_spacing + subj_idx * subject_offset + subject_offset - 0.6
                    median_y = np.median(neg_data[m_idx]) if len(neg_data[m_idx]) > 0 else 0
                    label = f"{n_star}\np={p_neg:.3g}"
                    plt.text(pos, median_y - abs(median_y) * 0.05, label,
                             ha='center', va='top', fontsize=12, fontweight='bold', color='black')
        else:
            pos_data = [d["pos"] for d in subj_data]
            stars, pvalues, pvalues_corr, uvalues = compute_stars_z(pos_data)
            
            print('CONTRAST', contrast , ' - SUBJECT', subj )
            print('stars', stars)
            print('pvalues', pvalues)
            print('pvalues_corr', pvalues_corr)
            print('uvalues', uvalues)

            for m_idx, (star, p) in enumerate(zip(stars,pvalues_corr)):
                if not star:
                    continue
                pos = m_idx * cluster_spacing + subj_idx * subject_offset
                median_y = np.median(pos_data[m_idx]) if len(pos_data[m_idx]) > 0 else 0
                label = f"{star}\np={p:.3g}"
                plt.text(pos, median_y + abs(median_y) * 0.05, label,
                         ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    # Reference lines
    plt.axhline(3.0, color='gray', linestyle='--', linewidth=1, label=f'Z = +{threshold:.2f}')
    if dual_polarity:
        plt.axhline(-3.0, color='gray', linestyle='--', linewidth=1, label=f'Z = –{threshold:.2f}')

    # Axis and layout
    tick_positions = [
        m_idx * cluster_spacing + (len(subjects) * subject_offset) / 2  for m_idx in range(n_mocos)
    ]
    plt.xticks(tick_positions, mocos_, rotation=15)
    plt.ylabel("z-score")
    plt.title(f"{contrast.capitalize()} — Subject-wise z-score distributions - {space} \n (Significance vs. SNAVoffPEERSoff)", fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Legend
    legend_elements = [
        Line2D([0], [0], color=c, lw=3, label=f"sub-{subj:02}") for subj, c in subject_colors.items()
    ]
    if dual_polarity:
        plt.legend(handles=legend_elements, loc='center', fontsize=12)
    else:     
        plt.legend(handles=legend_elements, loc='center', fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.join(grp_dir, 'figures'), exist_ok=True)
    plt.savefig(
        os.path.join(grp_dir, 'figures', f"boxplot_Z_{contrast.replace(' ', '_')}_space-{space}_permMean.pdf"),
        dpi=300
    )
    plt.show()