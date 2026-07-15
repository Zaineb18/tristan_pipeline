import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import surface, plotting
import nibabel.freesurfer.io as fsio
from matplotlib.colors import ListedColormap

from tristan_pipeline.utils.plotting_utils import *
from tristan_pipeline.utils.analysis_utils import *
from tristan_pipeline.io.params import *

# ---------------- SETTINGS ----------------
space = "T1w"
contrasts_names = ['phrases']
hemis = {'lh': 'left'}
subjects = [2]

# ---------------- YOUR PARCELS ----------------
PARCEL_NAME_MAP = {
    "G_front_inf-Opercular": "Inferior frontal gyrus (Broca)",
    "G_front_inf-Triangul": "Inferior frontal gyrus (Broca)",
    #"S_precentral-inf-part": "Inferior precentral Sulcus",
    #"S_precentral-sup-part": "Superior Precentral Sulcus",
    "G_temp_sup-Plan_tempo": "Planum temporale",
    "S_temporal_sup": "Superior temporal sulcus",
    "G_pariet_inf-Supramar": "Supramarginal gyrus",
    "Lat_Fis-post": "Posterior sylvian fissure"
    #"S_intrapariet_and_P_trans": "Intraparietal sulcus (IPS)",
    #"G_pariet_inf-Angular": "Angular gyrus",
    #"G_parietal_sup": "Superior parietal lobule",
    #"S_central": "Central Sulcus",
    #"G_front_middle": "Middle frontal gyrus",
}

TARGET_PARCELS = set(PARCEL_NAME_MAP.keys())

# ---------------- MAIN ----------------
for subj in subjects:

    for ses in sessions:
        for contrast in contrasts_names:
            for moco in mocos.keys():

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

                    # ---------------- LOAD SURF ----------------
                    native_pial = os.path.join(FREESURFER_PATH, 'surf', f"{h}.pial")
                    native_inflated = os.path.join(FREESURFER_PATH, 'surf', f"{h}.inflated")
                    native_sulc = os.path.join(FREESURFER_PATH, 'surf', f"{h}.sulc")

                    labels_path = os.path.join(
                        FREESURFER_PATH,
                        'label',
                        f"{h}.aparc.a2009s.annot"
                    )

                    labels_native, ctab, names = fsio.read_annot(labels_path)
                    names = [n.decode("utf-8") for n in names]

                    coords, faces = fsio.read_geometry(native_inflated)

                    texture = surface.vol_to_surf(zmap_vol, native_pial)

                    # ======================================================
                    # STEP 1: SELECT ONLY YOUR PARCELS
                    # ======================================================
                    target_labs = [
                        lab for lab in np.unique(labels_native)
                        if lab > 0 and names[lab] in TARGET_PARCELS
                    ]

                    # ======================================================
                    # STEP 2: BUILD SINGLE ROI MAP (CRITICAL FIX)
                    # ======================================================
                    roi_map = np.zeros_like(labels_native, dtype=int)

                    for i, lab in enumerate(target_labs, start=1):
                        roi_map[labels_native == lab] = i

                    # build colors in SAME ORDER
                    roi_colors = np.array([
                        ctab[lab, :3] / 255.0 for lab in target_labs
                    ])

                    cmap = ListedColormap(roi_colors)

                    # ======================================================
                    # BASE SURFACE
                    # ======================================================
                    fig = plotting.plot_surf_stat_map(
                        surf_mesh=native_inflated,
                        stat_map=texture,
                        hemi=hemis[h],
                        bg_map=native_sulc,
                        threshold=6,
                        alpha=0.3,
                        vmax=6,
                        cmap="seismic",
                        colorbar=False
                    )

                    # ======================================================
                    # ADD FILLED PARCELS (CORRECT WAY)
                    # ======================================================
                    plotting.plot_surf_roi(
                        surf_mesh=native_inflated,
                        roi_map=roi_map,
                        hemi=hemis[h],
                        cmap=cmap,
                        alpha=0.35,
                        figure=fig
                    )

                    # ======================================================
                    # ADD CONTOURS (MATCH COLORS)
                    # ======================================================
                    plotting.plot_surf_contours(
                        surf_mesh=native_inflated,
                        roi_map=roi_map,
                        hemi=hemis[h],
                        levels=np.arange(1, len(target_labs) + 1),
                        colors=roi_colors,
                        linewidths=1.5,
                        figure=fig
                    )

                    # ---------------- SAVE ----------------
                    plt.gcf().set_size_inches(8.27, 11.69)

                    plt.savefig(
                        os.path.join(
                            grp_dir,
                            'figures',
                            f'sub-{subj:02}_ses-{ses}_surf-{h}_FINAL.png'
                        ),
                        dpi=300
                    )

                    #plt.show()