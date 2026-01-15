from nilearn import surface, plotting
from nilearn.datasets import fetch_surf_fsaverage, fetch_atlas_surf_destrieux
from tristan_pipeline.io.params import *
from tristan_pipeline.utils.plotting_utils import *

FREESURFER_PATH =os.path.join(DATA_DIR, 'derivatives', 'freesurfer')
os.environ["SUBJECTS_DIR"] = FREESURFER_PATH
#annotid='aparc.a2009s'
#annotid='aparc'

for subj in subjects: 
    fsaverage_annot_to_native_surf(f'sub-{subj:02}', FREESURFER_PATH, annotid='aparc')

"""def map_labels_fsavg_to_native(labels_fsavg, coords_fsavg, coords_native):
    tree = cKDTree(coords_fsavg)
    _, idx = tree.query(coords_native)
    return labels_fsavg[idx]

def compute_surface_borders(labels, faces):
    border = np.zeros(len(labels), dtype=bool)
    for tri in faces:
        if len({labels[tri[0]], labels[tri[1]], labels[tri[2]]}) > 1:
            border[tri] = True
    return border

# ---------------------------
# 1) Load fsaverage surfaces and Destrieux atlas
# ---------------------------
def project_atlas(DATA_DIR, FREESURFER_PATH, subj):
    fsaverage = fetch_surf_fsaverage('fsaverage5')
    destrieux = fetch_atlas_surf_destrieux()
    lh_labels_fsavg = destrieux['map_left']   # fsaverage left GIFTI labels
    rh_labels_fsavg = destrieux['map_right']  # fsaverage right GIFTI labels

    FREESURFER_PATH = os.path.join(DATA_DIR, 'derivatives', 'freesurfer', f'sub-{subj:02}','surf')
    lh_native_pial = os.path.join(FREESURFER_PATH, "lh.pial")
    lh_native_inflated = os.path.join(FREESURFER_PATH, "lh.inflated")
    lh_native_sulc = os.path.join(FREESURFER_PATH, "lh.sulc")
    rh_native_pial = os.path.join(FREESURFER_PATH, "rh.pial")
    rh_native_inflated = os.path.join(FREESURFER_PATH, "rh.inflated")
    rh_native_sulc = os.path.join(FREESURFER_PATH, "rh.sulc")

    coords_lh_fsavg, _ = surface.load_surf_mesh(fsaverage.pial_left)
    coords_rh_fsavg, _ = surface.load_surf_mesh(fsaverage.pial_right)

    coords_lh_native, faces_lh = surface.load_surf_mesh(lh_native_pial)
    coords_rh_native, faces_rh = surface.load_surf_mesh(rh_native_pial)
    lh_labels_fsavg_data = surface.load_surf_data(lh_labels_fsavg)
    rh_labels_fsavg_data = surface.load_surf_data(rh_labels_fsavg)
    lh_labels_native = map_labels_fsavg_to_native(
        lh_labels_fsavg_data,
        coords_lh_fsavg,
        coords_lh_native
        )
    rh_labels_native = map_labels_fsavg_to_native(
        rh_labels_fsavg_data,
        coords_rh_fsavg,
        coords_rh_native
        )
    return(lh_native_inflated,lh_labels_native ,lh_native_sulc, 
           rh_native_inflated,rh_labels_native, rh_native_sulc )


fig_lh = plotting.plot_surf_roi(
    surf_mesh=lh_native_inflated,
    roi_map=lh_labels_native,
    hemi='left',
    bg_map=lh_native_sulc,
    title="Destrieux Atlas – Left Hemisphere (native)",
    colorscheme='viridis',
    alpha=0.1
)

fig_rh = plotting.plot_surf_roi(
    surf_mesh=rh_native_inflated,
    roi_map=rh_labels_native,
    hemi='right',
    bg_map=rh_native_sulc,
    title="Destrieux Atlas – Right Hemisphere (native)",
    colorscheme='viridis',
    alpha=0.1
    
)

plt.show()



"""
