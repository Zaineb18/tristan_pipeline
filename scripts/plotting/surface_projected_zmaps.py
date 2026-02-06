import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.utils.analysis_utils import *
from tristan_pipeline.io.params import *
from nilearn.glm import threshold_stats_img
from nilearn import surface, plotting, datasets
from nilearn.datasets import fetch_surf_fsaverage, fetch_atlas_surf_destrieux
from matplotlib import cm
import pyvista as pv
import nibabel.freesurfer as fs

space = "T1w"
contrasts_names = ['phrases']
hemis ={'lh':'left', 'rh':'right'}

for subj in subjects:
    print(subj)
    subj_color = subject_colors.get(subj, 'black')
    for ses in sessions:
        for contrast in contrasts_names:
            print(contrast)
            for m_idx, moco in enumerate(mocos.keys()):
                print(moco)
                FMRIPREP_PATH = os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
                FREESURFER_PATH = os.path.join(DATA_DIR, 'derivatives', 'freesurfer', f'sub-{subj:02}')
                zmap_path = os.path.join(
                    FMRIPREP_PATH, f"sub-{subj:02}", f"ses-{ses}", "stats",
                    f"sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco}.nii")
                zmap_vol = nib.load(zmap_path)
                for h in hemis.keys():    
                    native_pial = os.path.join(FREESURFER_PATH,'surf',f"{h}.pial")
                    native_inflated = os.path.join(FREESURFER_PATH,'surf', f"{h}.inflated")
                    native_sulc = os.path.join(FREESURFER_PATH,'surf', f"{h}.sulc")
                    labels_native_path = os.path.join(FREESURFER_PATH, 'label', f"{h}.aparc.a2009s.annot")
                    labels_native, ctab, names = fs.read_annot(labels_native_path)
                    names = [n.decode("utf-8") for n in names]
                    valid_labels = np.unique(labels_native)
                    valid_labels = valid_labels[valid_labels >= 0]
                    #coords, faces = native_inflated
                    texture = surface.vol_to_surf(zmap_vol,native_pial)
                
                    bg_map = labels_native.astype(float)
                    bg_map[bg_map == 0] = np.nan
                    parcel_levels = np.unique(labels_native)
                    parcel_levels = parcel_levels[parcel_levels != 0]
                
                    # lh_ctab[:, :3] gives RGB in 0-255, normalize to 0-1
                    colors_rgb = ctab[:, :3] / 255.0
                    alpha = 1.0
                    colors_rgba = np.hstack([colors_rgb, np.ones((colors_rgb.shape[0], 1))])  # N x 4 RGBA
                    # pick only the colors corresponding to your levels
                    colors_for_contours = colors_rgba[parcel_levels]  # length matches levels
                    
                    fig = plotting.plot_surf_stat_map(
                    surf_mesh=native_inflated,
                    stat_map=texture,          # z-map
                    hemi=hemis[h],
                    bg_map=native_sulc,   #bg_map          # atlas as background
                    threshold=1.5,
                    alpha=0.3,                 # makes atlas semi-transparent
                    vmax=6,
                    colorbar=True,
                    cmap="seismic"
                    )
                    ax = fig.axes[0]
                    plotting.plot_surf_contours(
                    surf_mesh=native_inflated,
                    roi_map=labels_native,
                    hemi=hemis[h],
                    levels=parcel_levels,
                    colors=colors_for_contours,
                    linewidths=1,
                    figure=fig
                    )
                    fig.gca().set_title(f"Contrast: {contrast} - sub-{subj:02} \n {moco}",
                                        color=adjust_color_tone(subj_color,moco_brightness[m_idx]),
                                        x=-8, y=1.15, pad=0)
                    plt.savefig(os.path.join(grp_dir, 'figures', f'sub-{subj:02}_ses-{ses}_surf-{h}_Destrieux_{contrast}_{space}_{moco}.pdf'))                
                    #plt.show()
                """# Create a discrete colormap
                cmap = cm.get_cmap('tab20', len(np.unique(lh_labels_native)))
                bg_map_color = lh_labels_native.copy()
                bg_map_color[bg_map_color == 0] = np.nan  # background

                fig = plotting.plot_surf_stat_map(
                surf_mesh=lh_native_inflated,
                stat_map=texture,
                hemi='left',
                bg_map=bg_map_color,
                threshold=2.5,
                alpha=0.3,
                cmap=cmap,
                vmax=len(np.unique(lh_labels_native)),
                colorbar=True
                )
                """
                