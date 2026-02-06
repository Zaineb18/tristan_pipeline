import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.utils.analysis_utils import *
from tristan_pipeline.io.params import *
from nilearn import surface, plotting
import nibabel.freesurfer as fs
from mpl_toolkits.mplot3d import proj3d

space = "T1w"
contrasts_names = ['phrases']
hemis = {'lh': 'left', 'rh': 'right'}

subjects = [2]

for subj in subjects:
    subj_color = subject_colors.get(subj, 'black')

    for ses in sessions:
        for contrast in contrasts_names:
            for m_idx, moco in enumerate(mocos.keys()):

                print(moco)

                FMRIPREP_PATH = os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
                FREESURFER_PATH = os.path.join(
                    DATA_DIR, 'derivatives', 'freesurfer', f'sub-{subj:02}'
                )

                zmap_path = os.path.join(
                    FMRIPREP_PATH,
                    f"sub-{subj:02}",
                    f"ses-{ses}",
                    "stats",
                    f"sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco}.nii"
                )
                zmap_vol = nib.load(zmap_path)

                for h in hemis.keys():

                    native_pial = os.path.join(FREESURFER_PATH, 'surf', f"{h}.pial")
                    native_inflated = os.path.join(FREESURFER_PATH, 'surf', f"{h}.inflated")
                    native_sulc = os.path.join(FREESURFER_PATH, 'surf', f"{h}.sulc")

                    labels_native_path = os.path.join(
                        FREESURFER_PATH, 'label', f"{h}.aparc.a2009s.annot"
                    )
                    texture = surface.vol_to_surf(zmap_vol, native_pial)

                    labels_native, ctab, names = fs.read_annot(labels_native_path)
                    names = [n.decode("utf-8") for n in names]

                    coords, faces = fs.read_geometry(native_inflated)

                    parcel_levels = np.unique(labels_native)
                    parcel_levels = parcel_levels[parcel_levels > 0]

                    colors_rgb = ctab[:, :3] / 255.0
                    colors_rgba = np.hstack(
                        [colors_rgb, np.ones((colors_rgb.shape[0], 1))]
                    )

                    for lab in parcel_levels:

                        raw_name = names[lab]

                        if raw_name == "unknown":
                            continue

                        # Base surface
                        fig = plotting.plot_surf_stat_map(
                            surf_mesh=native_inflated,
                            stat_map=texture,
                            hemi=hemis[h],
                            bg_map=native_sulc,
                            colorbar=False,
                            threshold=6,
                            vmax=6,
                            cmap="gray",
                            alpha=1.0
                        )
                        ax = fig.axes[0]

                        # Plot SINGLE Destrieux contour
                        plotting.plot_surf_contours(
                            surf_mesh=native_inflated,
                            roi_map=labels_native,
                            hemi=hemis[h],
                            levels=[lab],
                            colors=[colors_rgba[lab]],
                            linewidths=2,
                            figure=fig
                        )

                        # Label at centroid
                        verts = np.where(labels_native == lab)[0]
                        if len(verts) > 100:
                            centroid = coords[verts].mean(axis=0)
                            ax.text(
                                centroid[0],
                                centroid[1],
                                centroid[2],
                                raw_name,
                                fontsize=7,
                                color="black",
                                ha="center",
                                va="center"
                            )

                        out_path = os.path.join(
                            grp_dir,
                            "atlas",
                            f"sub-{subj:02}_ses-{ses}_surf-{h}_parcel-{raw_name}_{space}_{moco}.png"
                        )

                        plt.savefig(out_path)
                        plt.close(fig)