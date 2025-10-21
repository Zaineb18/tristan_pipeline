import matplotlib.pyplot as plt
import numpy as np 
import os 
import nibabel as nib 
from nilearn.plotting import plot_stat_map
from nilearn import datasets, surface, plotting,image 


#####PLOTTING SETTINGS#####
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

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
    disp = plot_stat_map(z_map, anat_file,colorbar=False,
    title=f'Custom contrast: {contrast} \n {moco} - {space}', vmax=10,
    display_mode='z',threshold=threshold,cut_coords=cut_coords)
    disp.add_contours(gm_file, levels=[0.5], colors='cyan',linewidths=0.5, alpha=0.5)
    plt.savefig(os.path.join(FMRIPREP_PATH, 'figures',
    f'sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco}_{thresh_strag}'))
    plt.show()
    plt.close() 

def disp_surf_activations(space, z_map, FREESURFER_PATH,FMRIPREP_PATH, contrast, moco, subj, ses ):
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
