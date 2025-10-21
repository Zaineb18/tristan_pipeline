from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
from tristan_pipeline.analysis.glm_utils import *
from tristan_pipeline.plotting.plotting_utils import *
from nilearn.glm.first_level import FirstLevelModel
import pandas as pd 
from nilearn import image 
from nilearn.glm import threshold_stats_img
import ants

##########CONFIG###########
stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
#stimfile = "/home/zamor/Documents/TRISTAN/data_Caro/session1_localizer_standard.csv"
contrasts_names = ['calculations','checkerboard vs the others','clic right vs clic left']
mocos = ["ONAVonPEERSon"]
spaces = ["MNI152NLin2009cAsym", "T1w"]#
onav = True
subjects = [1,2,3,4]
sessions = [1]
onav_files = {1:"Y_B0_sent_2025-04-2311_47_37.753099.npy",
             2:"Y_B0_sent_2025-05-2714_30_55.782043.npy", 
             3:"Y_B0_sent_2025-09-0311-17-46.993207.npy", 
             4:"Y_B0_sent_2025-06-1111_13_47.267598.npy"}
d_vols= {1:0, 2:2, 3:2, 4:2}
n_vols = {1:153, 2:155, 3:155, 4:155}
trs= {1:2.12, 2:2.12, 3:2.16, 4:2.12}
min_onsets= {1:0, 2:0, 3:0, 4:0}

#d_vols= {1:0, 2:0, 3:0, 4:0}
#n_vols = {1:263, 2:263, 3:263, 4:263}
#trs= {1:1.2, 2:1.2, 3:1.2, 4:1.2}
#min_onsets= {1:0, 2:0, 3:0, 4:0} 
##########CONFIG###########

for subj in subjects: 
    for ses in sessions: 
        for moco in mocos:
            ###########READ FMRIPREP FILES AND MOTION FILES###########
            DATA_DIR = f"/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/data_{moco}"
            #DATA_DIR = f"/home/zamor/Documents/TRISTAN/data_Caro"
            FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
            FREESURFER_PATH =os.path.join(DATA_DIR, 'derivatives', 'freesurfer', f'sub-{subj:02}', 
            'surf' )
            for space in spaces: 
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,_,_,xfm_MNItoT1,xfm_T1toMNI = load_fmriprepdata(FMRIPREP_PATH, subj, ses, space)
                bold_file,mask_file,anat_file,gm_file = FUNC_PATH[0],MASK_PATH[0],ANAT_PATH[0],GM_PATH[0]
                confounds_file = confounds_files[0]
                xfm_MNItoT1_file,xfm_T1toMNI_file = xfm_MNItoT1[0],xfm_T1toMNI[0]
                if onav==False:
                    confounds, _ = load_confounds(bold_file, strategy=('motion',), motion='basic',scrub=0)            
                else:     
                    motion_reg, motion_labels = load_onav_reg(
                    filepath=os.path.join(f'/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/', 'onav_data',onav_files[subj]) ,
                    labels = ["Rx ", "Ry ", "Rz ", "x ", "y ", "z ", r"$\phi$ ", "f0 ", "G$_x$", "G$_y$", "G$_z$"],
                    y_labels = [" / °", " / °", " / °", " / mm", " / mm", " / mm", " / rad", " / Hz", " / µT/m", " / µT/m", "/ µT/m"])
                    motion_df = pd.DataFrame(motion_reg, columns=motion_labels)
                    confounds = motion_df
                confounds_names = confounds.keys()
                ###########MAKE DESIGN MATRIX AND FIT GLM###########
                design_matrix =  make_design_matrix(stimfile,confounds,confounds_names,minonset=min_onsets[subj],
                delay_volumes=d_vols[subj],hrf_model='glover',tr=trs[subj],n_scans=n_vols[subj]) 
                level1_glm = FirstLevelModel(t_r=trs[subj], n_jobs=80, noise_model='ar1',mask_img=mask_file)
                level1_glm_fitted = level1_glm.fit(bold_file, design_matrices=design_matrix)
                ###########MAKE CONTRASTS VECTORS###########
                contrasts = custom_contrast(design_matrix.keys())
                ###########CUT COORDS###########
                cut_coords = [(30,39,42,54),(4,12,20,24),(50,52,58,60)]
                if space == "T1w":
                    xfm = ants.read_transform(xfm_T1toMNI_file)
                    fixed_x, fixed_y = 0, 0  # Dummy fixed x,y for transformation
                    for i in range(3):
                        cut_coords[i] = tuple([xfm.apply_to_point([fixed_x, fixed_y, z])[2]
                        for z in cut_coords[i]])
                print(cut_coords)    
                ###########MEAN BOLD IMAGE AND MAKE FIGURES AND STAT FOLDERS###########
                mean_bold = image.mean_img(image.index_img(bold_file, slice(10, None)))
                os.makedirs(os.path.join(FMRIPREP_PATH, 'stat'), exist_ok=True)
                os.makedirs(os.path.join(FMRIPREP_PATH, 'figures'), exist_ok=True)
                i=0
                ###########COMPUTE THE CONTRASTS###########
                for contrast in contrasts_names:
                    contrast_vector = contrasts[contrast]
                    z_map = level1_glm_fitted.compute_contrast(contrast_vector, output_type='z_score')
                    nib.save(z_map, os.path.join(FMRIPREP_PATH, 'stat',f'sub-{subj:02}_ses-{ses}_zmap_{contrast}_{space}_{moco}.nii'))
                    ###########FPR at 0.001###########
                    thresholded_map, threshold = threshold_stats_img(z_map,alpha=0.001,
                    height_control='fpr',two_sided=True)
                    plot_activations(z_map, anat_file, gm_file, threshold, contrast, moco, space, cut_coords[i], subj, ses, FMRIPREP_PATH, thresh_strag='fpr')            
                    ###########FDR at 0.05###########
                    thresholded_map, threshold = threshold_stats_img(z_map,alpha=0.05,
                    height_control='fdr', two_sided=True)
                    plot_activations(z_map, anat_file, gm_file, threshold, contrast, moco, space, cut_coords[i], subj, ses, FMRIPREP_PATH, thresh_strag='fdr')
                    ###########Surface-based visualization###########
                    disp_surf_activations(space, z_map, FREESURFER_PATH,FMRIPREP_PATH, contrast, moco, subj, ses )
                    i=i+1