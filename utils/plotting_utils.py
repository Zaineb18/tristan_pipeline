import os 
import numpy as np 
import nibabel as nib 
import os.path as op
from glob import glob
from time import time
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import nipype.interfaces.freesurfer as fs
from nipype.interfaces.ants import ApplyTransforms
from nilearn.plotting import plot_stat_map
from nilearn import datasets, surface, plotting,image 
from nilearn.datasets import fetch_surf_fsaverage, fetch_atlas_surf_destrieux
import nilearn.image as inl
import nilearn.masking as mnl

#####PLOTTING SETTINGS#####
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})
line_styles = ['-', '--', ':']   # distinguish spaces
# Subject base colors
subject_colors = {
    1: 'tab:blue',
    2: 'tab:red',
    3: 'tab:green',
    4: 'tab:orange'
}
# Visual distinction for mocos
markers = [#'o',
           's', 
           '^']        # different markers per moco
#moco_widths = [1.5, 2.0, 2.5]    # progressively thicker
moco_widths = [#0.5, 
               1.0,
               1.5]    # progressively thicker
moco_brightness = [#2,
                   1.0,
                   0.7]  # lighter → darker
dims = {1: -0.5,
        2: -0.5,
        3: -0.5,
        4: -0.5,
        5: -0.5,
        6: -0.5,
        7: -0.5,
        8: -0.5,
        9: -0.5}

###########FUNCTIONS#######################
def adjust_color_tone(base_color, factor):
    """Brighten or darken an RGB color by a factor."""
    rgb = np.array(mcolors.to_rgb(base_color))
    if factor > 1:  # lighten
        rgb = 1 - (1 - rgb) / factor
    else:  # darken
        rgb = rgb * factor
    return np.clip(rgb, 0, 1)

def coord_plot11(mopa_pred_list, Y_names=None, TR=0.04, sub=0):

    fig, ax = plt.subplots(3, 4, figsize=(15, 5), constrained_layout=True, sharex=True)
    # plt.style.use('dark_background')
    n = 0
    labels = ["Rx ", "Ry ", "Rz ", "x ", "y ", "z ", r"$\phi$ ", "f0 ", "G$_x$", "G$_y$", "G$_z$"]

    if Y_names is None or len(Y_names) < len(mopa_pred_list):
        Y_names = ["Set %i" % (x + 1) for x in range(len(mopa_pred_list))]

    y_labels = [" / °", " / °", " / °", " / mm", " / mm", " / mm", " / rad", " / Hz", " / µT/m", " / µT/m", "/ µT/m"]

    for j in range(4):
        for i in range(3):
            if i == 2 and j == 2:
                continue
            if n < 11:
                for m in range(len(mopa_pred_list)):

                    ax[i][j].plot(np.arange(mopa_pred_list[m].shape[0]) * TR, mopa_pred_list[m][:, n], zorder=5, linewidth=1, label=Y_names[m])

                ax[i][j].grid()
                ax[i][j].tick_params(labelsize=12)
                ax[i][j].set_ylabel(labels[n] + y_labels[n], fontsize=12)
                if i == 2 or i == 1 and j == 2:
                    ax[i][j].set_xlabel("Time [s]", fontsize=12)

            n += 1

    fig.delaxes(ax[-1][-2])
    lines, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='center right', bbox_to_anchor=(0.69, 0.2), ncol=1, fontsize=12)
    ax[0][0].set_title("Rotations", fontsize=15)
    ax[0][1].set_title("Translations", fontsize=15)
    ax[0][2].set_title("B0", fontsize=15)
    ax[0][3].set_title("Shims", fontsize=15)

    #plt.savefig(f'/home/sergerm/PhD/onav-pmc/Data/plots/Coord_plot11_{sub}.png')
    plt.show()
    return None

def plot_activations(z_map, anat_file, gm_file, threshold, contrast, moco, space, cut_coords, subj, ses, FMRIPREP_PATH, thresh_strag='fpr'): 
    disp = plot_stat_map(z_map, anat_file,#colorbar=True,
    title=f'Custom contrast: {contrast} \n {moco} - {space}', vmax=10, dim=dims[subj], 
    display_mode='z',threshold=threshold,cut_coords=cut_coords, interpolation='none')
    disp.add_contours(gm_file, levels=[0.5], colors='cyan',linewidths=0.5, alpha=0.5)
    plt.savefig(os.path.join(FMRIPREP_PATH,'figures',
    f'sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco}_{thresh_strag}.pdf'))
    plt.show()
    plt.close() 

def disp_surf_activations(space, z_map, FREESURFER_PATH, FMRIPREP_PATH, contrast, moco, subj, ses ):
    if space=='T1w':
        texture = surface.vol_to_surf(z_map, os.path.join(FREESURFER_PATH, 'lh.pial'))
        plotting.plot_surf_stat_map(
                        os.path.join(FREESURFER_PATH, 'lh.inflated'),
                        texture,
                        hemi='left',
                        title=f'Custom contrast: {contrast} \n {moco}',
                        threshold=2.5,
                        bg_map=os.path.join(FREESURFER_PATH, 'lh.sulc'),
                        vmax=6,colorbar=False,
                        output_file=os.path.join(FMRIPREP_PATH, 'figures',
                        f'sub-{subj:02}_ses-{ses}_surf-left_{contrast}_{space}_{moco}'))           
        
        texture = surface.vol_to_surf(z_map, os.path.join(FREESURFER_PATH, 'rh.pial'))
        plotting.plot_surf_stat_map(
                        os.path.join(FREESURFER_PATH, 'rh.inflated'),
                        texture,
                        hemi='right',
                        title=f'Custom contrast: {contrast} \n {moco}',
                        threshold=2.5,
                        bg_map=os.path.join(FREESURFER_PATH, 'rh.sulc'),
                        vmax=6,colorbar=False,
                        output_file=os.path.join(FMRIPREP_PATH, 'figures',
                        f'sub-{subj:02}_ses-{ses}_surf-right_{contrast}_{space}_{moco}'))
    else:               
        fsaverage = datasets.fetch_surf_fsaverage()
                    
        texture = surface.vol_to_surf(z_map,fsaverage.pial_left)
        plotting.plot_surf_stat_map(
                        fsaverage.infl_left,
                        texture,
                        hemi='left',
                        title=f'Custom contrast: {contrast} \n {moco}',
                        threshold=2.5,
                        bg_map=fsaverage.sulc_left,
                        vmax=6,colorbar=False,
                        output_file=os.path.join(FMRIPREP_PATH, 'figures',
                        f'sub-{subj:02}_ses-{ses}_surf-left_{contrast}_{space}_{moco}'))

        texture = surface.vol_to_surf(z_map,fsaverage.pial_right)
        plotting.plot_surf_stat_map(
                        fsaverage.infl_right,
                        texture,
                        hemi='right',
                        title=f'Custom contrast: {contrast} \n {moco}',
                        threshold=2.5,
                        bg_map=fsaverage.sulc_right,
                        vmax=6,colorbar=False,
                        output_file=os.path.join(FMRIPREP_PATH, 'figures',
                        f'sub-{subj:02}_ses-{ses}_surf-right_{contrast}_{space}_{moco}'))
        
def display_tissues(gm_core, wm_core, csf_core,
                    gm_wm_interface, gm_csf_interface,
                    affine, anat_img, title="Tissues and Interfaces"):
    
    # Convert arrays to NIfTI images
    gm_img = nib.Nifti1Image(gm_core.astype(np.int8), affine)
    wm_img = nib.Nifti1Image(wm_core.astype(np.int8), affine)
    csf_img = nib.Nifti1Image(csf_core.astype(np.int8), affine)
    gm_wm_interface_img = nib.Nifti1Image(gm_wm_interface.astype(np.int8), affine)
    gm_csf_interface_img = nib.Nifti1Image(gm_csf_interface.astype(np.int8), affine)

    # Display all overlays
    display = plotting.plot_anat(anat_img, title=title, display_mode='ortho', draw_cross=False, annotate=False)

    display.add_contours(gm_img, cmap='Blues', alpha=1, linewidths=0.2)                  # GM → blue
    display.add_contours(gm_csf_interface_img, cmap='Greens', alpha=1, linewidths=0.5)   # Pial surface → lightgreen
    display.add_contours(gm_wm_interface_img, cmap='Oranges', alpha=1, linewidths=0.5)   # GM/WM interface → orange
    display.add_contours(wm_img, cmap='YlOrBr', alpha=1, linewidths=1)                   # WM → yellowish
    display.add_contours(csf_img, cmap='Blues_r', alpha=1, linewidths=1)  
    plotting.show()

def map_labels_fsavg_to_native(labels_fsavg, coords_fsavg, coords_native):
    tree = cKDTree(coords_fsavg)
    _, idx = tree.query(coords_native)
    return labels_fsavg[idx]

def compute_surface_borders(labels, faces):
    border = np.zeros(len(labels), dtype=bool)
    for tri in faces:
        if len({labels[tri[0]], labels[tri[1]], labels[tri[2]]}) > 1:
            border[tri] = True
    return border

def project_atlas(DATA_DIR,subj):
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
    return(lh_native_inflated,lh_labels_native ,lh_native_sulc, lh_native_pial, 
           rh_native_inflated,rh_labels_native, rh_native_sulc,rh_native_pial )

def fsaverage_annot_to_native_surf(subj, FREESURFER_PATH, annotid='HCPMMP1'):
    HEMIS = ['lh', 'rh']
    for hemi in HEMIS:
        fname = f'{hemi}.{annotid}.annot'
        srcpath = op.join(FREESURFER_PATH, 'fsaverage', 'label', fname)
        outpath = op.join(FREESURFER_PATH, subj, 'label', fname)
            
        annot2native = fs.SurfaceTransform()
        annot2native.inputs.hemi = hemi
        annot2native.inputs.source_annot_file = srcpath
        annot2native.inputs.source_subject = 'fsaverage'
        annot2native.inputs.target_subject = subj
        annot2native.inputs.out_file = outpath
        
        annot2native.run()
        

def native_annot_surf_to_vol(subid, DATA_DIR, annotid='HCPMMP1'):
    datadir = DATA_DIR
    """ datadir is the bids/derivatives/fmriprep-xx.x.x dir (sub-xx inside) """
    # (Note: uses ANATDIR and VOL_EXT constants) 
    # ANATDIR = 'anat'  # relative path inside the subject's folder
    # VOL_EXT = 'nii.gz'
    # Freesurfer dir not specified here but assumed by the fs. interface to 
    # be pre-exported with export SUBJECTS_DIR=...
    
    vol_outdir = op.join(datadir, subid, 'anat')
    if not op.exists(vol_outdir):
        os.mkdir(vol_outdir)
    outpath = op.join(vol_outdir, f'{annotid}.nii.gz')
    annot2vol_command = fs.FSCommand(command='mri_aparc2aseg')
    annot2vol_command.inputs.args = " ".join([
        f'--s {subid}',
        f'--annot {annotid}',
        f'--o {outpath}',
        ])
    annot2vol_command.run()
    
    return outpath        