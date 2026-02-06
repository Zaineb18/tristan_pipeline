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
from mpl_toolkits.mplot3d import proj3d

space = "T1w"
contrasts_names = ['phrases']
hemis = {'lh': 'left',
         #'rh': 'right'
         }

PARCEL_NAME_MAP = {
    # Language
    "G_front_inf-Opercular": "Inferior frontal gyrus (opercular / Broca)",
    "G_front_inf-Triangul": "Inferior frontal gyrus (triangular / Broca)",
    "G_front_inf-Orbital": "Inferior frontal gyrus (orbital)",
    "G_temp_sup-Lateral": "Superior temporal gyrus",
    "G_temp_sup-Plan_tempo": "Planum temporale",
    "G_temp_sup-Plan_polar": "Temporal pole (superior)",
    "G_temporal_middle": "Middle temporal gyrus",
    "G_temporal_inf": "Inferior temporal gyrus",
    "Pole_temporal": "Temporal pole",
    "S_temporal_sup": "Superior temporal sulcus",
    "S_temporal_inf": "Inferior temporal sulcus",
    # Math
    "G_parietal_sup": "Superior parietal lobule",
    "G_parietal_inf-Angular": "Angular gyrus",
    "G_parietal_inf-Supramar": "Supramarginal gyrus",
    "S_parietal_inf": "Inferior parietal sulcus",
    "S_parietal_sup": "Superior parietal sulcus",
    "G_front_middle": "Middle frontal gyrus",
    "S_front_middle": "Middle frontal sulcus",
    "S_front_sup": "Superior frontal sulcus",
}
PARCEL_NAME_MAP = {
    # ---------- LANGUAGE ----------
    "G_front_inf-Opercular": "Inferior frontal gyrus (Broca)", #Speech production, articulation, grammar (core Broca’s area) (ok)
    "G_front_inf-Triangul": "Inferior frontal gyrus (Broca)", #Sentence structure, controlled language output (Broca’s area) (ok)
    "S_precentral-inf-part": "Inferior precentral Sulcus", #Motor planning for speech (mouth, tongue, lips) (ok)
    "G_temp_sup-Lateral": "Superior temporal gyrus", #Understanding spoken language, phoneme processing (ok)
    "G_temp_sup-Plan_tempo": "Planum temporale", #Auditory–language integration, phonology (Planum temporale) (ok)
    "G_temp_sup-Plan_polar": "Superior temporal pole", #High-level speech and voice processing (ok)
    "G_temporal_middle": "Middle temporal gyrus", #Word meaning, lexical–semantic processing (ok)
    "Pole_temporal": "Temporal pole", #Conceptual knowledge, semantic memory (ok)
    "S_temporal_sup": "Superior temporal sulcus", #Linking sounds to meaning, social language cues (ok)
    "S_temporal_inf": "Inferior temporal sulcus", #Visual–semantic associations (words, objects) (ok)
    "G_pariet_inf-Supramar": "Supramarginal gyrus", #Phonological working memory, reading, sound–symbol mapping (ok)
    "Lat_Fis-post": "Posterior sylvian fissure", #Anatomical hub of the perisylvian language network  
    #"G_front_inf-Orbital": "Inferior frontal gyrus",
    #"G_temp_sup-Lateral": "Superior temporal gyrus",
    #"G_temporal_inf": "Inferior temporal gyrus",
    # ---------- MATH / NUMBER ----------
    "S_intrapariet_and_P_trans": "Intraparietal sulcus (IPS)", #Core number sense, quantity comparison, calculation (IPS) (ok)
    "G_parietal_sup": "Superior parietal lobule", #Spatial attention, mental calculation, number manipulation (ok)
    #"S_parietal_inf": "Inferior parietal sulcus", #Numeric operations, visuospatial processing 
    #"S_parietal_sup": "Superior parietal sulcus", #Spatial reasoning, magnitude manipulation
    "G_pariet_inf-Angular": "Angular gyrus", #Arithmetic facts, symbolic meaning, number–word links
    "S_front_middle": "Middle frontal sulcus", #Executive control during calculations (ok)
    "S_front_sup": "Superior frontal sulcus", #Attention and monitoring during complex math (ok)
    "S_precentral-sup-part": "Superior Precentral Sulcus", #Motor planning for hand and speech actions (ok)
    "S_central": "Central Sulcus", #Boundary integrating perception and action (ok)
    # "S_precentral-sup-part": "Superior Precentral Sulcus",
    #"G_front_middle": "Middle frontal gyrus", #Working memory, multi-step problem solving
    #"G_precentral": "Primary Motor Cortex", 
    #"G_postcentral": "Primary Somatosensory Cortex",

}
TARGET_PARCELS = set(PARCEL_NAME_MAP.keys())
subjects=[2]

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
                    labels_native_path = os.path.join(FREESURFER_PATH, 'label', f"{h}.aparc.a2009s.annot")
                    labels_native, ctab, names = fs.read_annot(labels_native_path)
                    names = [n.decode("utf-8") for n in names]
                    coords, faces = fs.read_geometry(native_inflated)
                    texture = surface.vol_to_surf(zmap_vol, native_pial)
                    parcel_levels = np.unique(labels_native)
                    parcel_levels = parcel_levels[parcel_levels > 0]
                    colors_rgb = ctab[:, :3] / 255.0
                    colors_rgba = np.hstack([colors_rgb, np.ones((colors_rgb.shape[0], 1))])
                    colors_for_contours = colors_rgba[parcel_levels]
                    fig = plotting.plot_surf_stat_map(
                        surf_mesh=native_inflated,
                        stat_map=texture,
                        hemi=hemis[h],
                        bg_map=native_sulc,
                        threshold=6,
                        alpha=0.3,
                        vmax=6,
                        colorbar=False,
                        cmap="seismic")
                    ax = fig.axes[0]
                    
                    target_levels = [lab for lab in parcel_levels if names[lab] in TARGET_PARCELS]
                    colors_for_target_contours = colors_rgba[target_levels]
                    
                    plotting.plot_surf_contours(
                        surf_mesh=native_inflated,
                        roi_map=labels_native,
                        hemi=hemis[h],
                        #levels=parcel_levels,
                        #colors=colors_for_contours,
                        levels=target_levels,
                        colors=colors_for_target_contours,
                        linewidths=1,
                        figure=fig)
                    for lab in parcel_levels:
                        #parcel_name = names[lab]
                        #if parcel_name not in TARGET_PARCELS:
                        #    continue
                        raw_name = names[lab]
                        if raw_name not in TARGET_PARCELS:
                            continue
                        verts = np.where(labels_native == lab)[0]
                        if len(verts) < 200:
                            continue
                        centroid = coords[verts].mean(axis=0)
                        label_text = PARCEL_NAME_MAP[raw_name]
                        
                        #x, y, _ = proj3d.proj_transform(centroid[0], centroid[1], centroid[2],
                        #                                ax.get_proj())
                        if label_text in ["Superior parietal lobule", "Intraparietal sulcus (IPS)"]:
                            ax.text(centroid[0], centroid[1], centroid[2]+7,
                            #names[lab],
                            label_text,
                            fontsize=15,
                            color="black",
                            ha="center",
                            va="center")
                        #elif label_text in ["Superior frontal sulcus"]:
                        #    ax.text(centroid[0], centroid[1], centroid[2]+3,
                        #    #names[lab],
                        #    label_text,
                        #    fontsize=15,
                        #    color="black",
                        #    ha="center",
                        #    va="center")                            
                                                      
                        elif label_text in ["Planum temporale"]:
                            ax.text(centroid[0], centroid[1], centroid[2]-5,
                            #names[lab],
                            label_text,
                            fontsize=15,
                            color="black",
                            ha="center",
                            va="center")   
                        elif label_text in ["Superior temporal sulcus"]:
                            ax.text(centroid[0], centroid[1], centroid[2]-10,
                            #names[lab],
                            label_text,
                            fontsize=15,
                            color="black",
                            ha="center",
                            va="center")                          
                        else:     
                            ax.text(centroid[0], centroid[1], centroid[2],
                            #names[lab],
                            label_text,
                            fontsize=15,
                            color="black",
                            ha="center",
                            va="center")
                    
                        #x2d, y2d, _ = proj3d.proj_transform(centroid[0], centroid[1], centroid[2], ax.get_proj())
                        #ax.text2D(
                        #    x2d, y2d,
                        #    label_text,
                        #    fontsize=6,
                        #    color="black",
                        #    ha="center",
                        #    va="center",
                        #    transform=ax.transData
                        #)
                    plt.gcf().set_size_inches(8.27, 11.69)   # A4 portrait    
                    plt.savefig(
                        os.path.join(
                            grp_dir,
                            'figures',
                            f'sub-{subj:02}_ses-{ses}_surf-{h}_Destrieux_{space}_{moco}.pdf'),
                        
                    )
                    plt.show()
                    #plt.close(fig)
