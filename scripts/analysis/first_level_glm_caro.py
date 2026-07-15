from tristan_pipeline.io.params import *
from tristan_pipeline.utils.loading_utils import *
from tristan_pipeline.utils.preproc_utils import *
from tristan_pipeline.utils.glm_utils import *
from tristan_pipeline.utils.plotting_utils import *
from nilearn.glm.first_level import FirstLevelModel
import pandas as pd 
from nilearn import image 
from nilearn.glm import threshold_stats_img
from nilearn.image import index_img
import ants

print(dims)
subjects=[1,2,3,4,5,7,8,9]
def plot_activations(z_map, anat_file, gm_file, threshold, contrast, space, cut_coords, subj, ses, FMRIPREP_PATH, thresh_strag='fpr'): 
    disp = plot_stat_map(z_map, anat_file, colorbar=False,
    title=f'Custom contrast: {contrast} \n sub-{subj:02}', vmax=10, dim=dims[subj], 
    display_mode='z',threshold=threshold,cut_coords=cut_coords, interpolation='none')
    disp.add_contours(gm_file, levels=[0.5], colors='cyan',linewidths=0.5, alpha=0.5)
    plt.savefig(os.path.join(grp_dir,
    f'sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{thresh_strag}_ISMRM.png'))
    plt.show()
    plt.close() 
    
for subj in subjects:
    for ses in sessions: 
        FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
        FREESURFER_PATH =os.path.join(DATA_DIR, 'derivatives', 'freesurfer', f'sub-{subj:02}', 'surf' )         
        for space in spaces: 
            FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,_,_,xfm_MNItoT1,xfm_T1toMNI = load_fmriprepdata(FMRIPREP_PATH,subj,ses,space,None)
            bold_file,mask_file,anat_file,gm_file = FUNC_PATH[0],MASK_PATH[0],ANAT_PATH[0],GM_PATH[0]
            print('SUBJECT', subj, ' SPACE', space)
            print(bold_file,'\n')
            print(mask_file,'\n')
            print(anat_file,'\n')
            print(gm_file,'\n')
            confounds_file = confounds_files[0]
            xfm_MNItoT1_file,xfm_T1toMNI_file = xfm_MNItoT1[0],xfm_T1toMNI[0]
            print(xfm_MNItoT1_file,'\n')
            print(xfm_T1toMNI_file,'\n')
            
            confounds, _ = load_confounds(bold_file, strategy=('motion',), motion='basic',scrub=0)            
            confounds_names = confounds.keys()
            ###########MAKE DESIGN MATRIX AND FIT GLM###########
            design_matrix =  make_design_matrix(stimfile,confounds,confounds_names,minonset=min_onsets[subj],
            delay_volumes=d_vols[subj],hrf_model='glover',tr=trs[subj],n_scans=n_vols[subj]) 
            level1_glm = FirstLevelModel(t_r=trs[subj], n_jobs=80, noise_model='ar1',mask_img=mask_file)
            level1_glm_fitted = level1_glm.fit(index_img(nib.load(bold_file),slice(min_onsets[subj], None)), design_matrices=design_matrix)
            ###########MAKE CONTRASTS VECTORS###########
            contrasts = custom_contrast(design_matrix.keys())
            ###########CUT COORDS###########
            cut_coords = [(-15,-10,-4,0,4,8,12,18,20,24,30,39,42,54)]
            if space == "T1w":
                xfm = ants.read_transform(xfm_T1toMNI_file)
                fixed_x, fixed_y = 0, 0  # Dummy fixed x,y for transformation
                for i in range(len(cut_coords)):
                    cut_coords[i] = tuple([xfm.apply_to_point([fixed_x, fixed_y, z])[2]
                for z in cut_coords[i]])
                    print(cut_coords)    
            ###########MEAN BOLD IMAGE AND MAKE FIGURES AND STAT FOLDERS###########
            mean_bold = image.mean_img(image.index_img(bold_file, slice(10, None)))
            os.makedirs(os.path.join(FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}','stats'), exist_ok=True)
            os.makedirs(os.path.join(FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}', 'figures'), exist_ok=True)
            i=0
            ###########COMPUTE THE CONTRASTS###########
            for contrast in contrasts_names:
                contrast_vector = contrasts[contrast]
                z_map = level1_glm_fitted.compute_contrast(contrast_vector, output_type='z_score')
                nib.save(z_map, os.path.join(FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}','stats',f'sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}.nii'))
                ###########FPR at 0.001###########
                thresholded_map, threshold = threshold_stats_img(z_map,alpha=0.001,
                height_control='fpr',two_sided=True)
                print('Threshold', threshold)
                plot_activations(z_map, anat_file, gm_file, threshold, contrast, space,
                                     cut_coords[i], subj, ses,
                                     os.path.join(FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}'),)            
                ###########FDR at 0.05###########
                #thresholded_map, threshold = threshold_stats_img(z_map,alpha=0.05,
                #height_control='fdr', two_sided=True)
                #plot_activations(z_map, anat_file, gm_file, threshold, contrast, moco, space, cut_coords[i], subj, ses, grp_dir, thresh_strag='fdr')
                ###########Surface-based visualization###########
                i=i+1
