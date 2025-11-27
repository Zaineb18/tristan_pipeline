import os
import matplotlib.pyplot as plt
from nilearn import plotting
from tristan_pipeline.io.params import *
import nibabel as nib

space = "MNI152NLin2009cAsym"
mocos_ = ["SNAVoffPEERSoff", 
         "SNAVonPEERSon"]
i=0
for _, moco_label in datasets:
    group_img = nib.load(os.path.join(os.path.join(grp_dir,"stats"),
                                       f"group_tSNR_space-{space}_{moco_label}.nii.gz"))
    disp = plotting.plot_stat_map(
        group_img,
        title=f"Group tSNR ({mocos_[i]}) - {space}",
        threshold=0.0,
        #vmin=0,
        vmax=100,
        display_mode='ortho',
        cut_coords=(0, 0, 0),
        colorbar=True,draw_cross=False, 
        symmetric_cbar=False,
        cmap='rainbow'
    )
    os.makedirs(os.path.join(grp_dir,'figures') ,exist_ok=True)
    plt.savefig(os.path.join(os.path.join(grp_dir,'figures'), f"group_tSNR_space-{space}_{moco_label}"))
    plotting.show()        
    i=i+1