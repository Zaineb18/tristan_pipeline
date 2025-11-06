import os
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.io.params import *

def update(frame):
    ax.clear()
    disp = plotting.plot_stat_map(
        group_imgs[frame],
        axes=ax,
        threshold=0.0,
        vmax=150,
        display_mode='ortho',
        cut_coords=(0,0,0),
        colorbar=True,
        draw_cross=False,
        symmetric_cbar=False,
        title=f"Group tSNR ({datasets[frame][1]}) - {space}"
    )
    return disp,

space = "MNI152NLin2009cAsym"
group_files = [os.path.join(os.path.join(grp_dir, 'stats'), f"group_tSNR_space-{space}_{moco_label}.nii.gz") 
               for _, moco_label in datasets]
group_imgs = []
for f in group_files:
    if os.path.exists(f):
        group_imgs.append(nib.load(f))
    else:
        print(f"Missing file: {f}")

fig, ax = plt.subplots(figsize=(10,5))
disp_list = []
anim = FuncAnimation(fig, update, frames=len(group_imgs), interval=2000, blit=False)
gif_file = os.path.join(os.path.join(grp_dir, 'figures'), f"group_tSNR_animation_{space}.gif")
writer = PillowWriter(fps=1)  # 0.5 fps → 2 seconds per frame
anim.save(gif_file, writer=writer)
print(f"Saved animation → {gif_file}")
plt.close(fig)
