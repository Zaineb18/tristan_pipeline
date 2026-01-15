import os
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.io.params import *

spaces = ["MNI152NLin2009cAsym", "T1w", "native bold"]
for subj in subjects:
    for ses in sessions:
        subj_color = subject_colors.get(subj, 'black')
        ########LOAD DATA########
        lines_data = []
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
                lines_data.append((moco_label, space, bin_centers, counts, moco_color, ls, marker, m_lw))

        if not lines_data:
            print(f"No valid data for subject {subj}")
            continue

        ########DETERMINE GLOBAL Y-RANGE########
        all_y = np.concatenate([counts for _, _, _, counts, *_ in lines_data])
        ymin, ymax = 0, np.max(all_y) * 1.05

        # ---- GROUP LINES BY MOCO ----
        grouped_indices = [
            [i for i, (moco_label, *_rest) in enumerate(lines_data) if moco_label == mocos[0]],
            [i for i, (moco_label, *_rest) in enumerate(lines_data) if moco_label == mocos[1]],
            [i for i, (moco_label, *_rest) in enumerate(lines_data) if moco_label == mocos[2]]
        ]

        ########SETUP FIGURE########
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 200)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("tSNR")
        ax.set_ylabel("Density")
        ax.set_title(f"tSNR Histograms - sub-{subj:02}", color=subj_color, fontweight='bold')
        ax.grid(True, alpha=0.3)

        lines = []
        for _ in lines_data:
            line, = ax.plot([], [], alpha=0.0)  # start invisible
            lines.append(line)

        ########ANIMATION UPDATE########
        def update(frame):
            # 0 → first 3 lines; 1 → second 3; 2 → last 3
            for g in range(frame + 1):
                for idx in grouped_indices[g]:
                    moco_label, space, x, y, color, ls, marker, lw = lines_data[idx]
                    lines[idx].set_data(x, y)
                    lines[idx].set_color(color)
                    lines[idx].set_linestyle(ls)
                    lines[idx].set_linewidth(lw)
                    lines[idx].set_marker(marker)
                    lines[idx].set_markevery(10)
                    lines[idx].set_markersize(4)
                    lines[idx].set_alpha(0.9)
            return lines

        ########CREATE ANIMATION########
        ani = FuncAnimation(
            fig,
            update,
            frames=len(grouped_indices),
            interval=1000,  # 1 sec per group
            blit=False,
            repeat=False
        )

        ########LEGENDS########
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
        first_legend = ax.legend(handles=moco_legend, title="Motion correction", loc="upper right")
        ax.add_artist(first_legend)
        ax.legend(handles=space_legend, title="Space", loc="center right")

        ########SAVE########
        os.makedirs(os.path.join(grp_dir,'figures') ,exist_ok=True)
        gif_path = os.path.join(os.path.join(grp_dir,'figures'), f"sub-{subj:02}_ses-{ses}_tSNRHist.gif")
        print(f"Saving animation for sub-{subj:02}...")
        ani.save(gif_path, writer=PillowWriter(fps=1))  # 1 frame/sec for clear group appearance
        plt.close(fig)
