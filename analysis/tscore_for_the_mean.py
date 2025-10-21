import os
import nibabel as nib
import numpy as np
from nilearn import image, plotting
from nilearn.maskers import NiftiMasker
from nilearn.signal import clean
from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
from tristan_pipeline.analysis.glm_utils import *

##########CONFIG###########
stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session1_localizer_standard.csv"
mocos = ["ONAVonPEERSon"]
spaces = ["MNI152NLin2009cAsym", "T1w", "native bold"]
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
##########CONFIG###########

for subj in subjects:
    for ses in sessions:
        for moco in mocos: 
                ###########READ FMRIPREP FILES###########
                DATA_DIR = f"/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/data_{moco}"
                FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
                for space in spaces:
                    if space == "native bold":
                        FUNC_PATH, MASK_PATH = load_funcdata(FMRIPREP_PATH, subj, ses)
                    else: 
                        FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,_,_,xfm_MNItoT1,xfm_T1toMNI = load_fmriprepdata(FMRIPREP_PATH, subj, ses, space)        
                    bold_file,mask_file = FUNC_PATH[0],MASK_PATH[0]
                    ######MAKE DESIGN MATRIX WITH TASK, DRIFTS AND CONSTANT REGRESSORS ONLY######
                    design_matrix = make_design_matrix(stimfile,None,None,minonset=min_onsets[subj],
                    delay_volumes=d_vols[subj],hrf_model='glover',tr=trs[subj],n_scans=n_vols[subj])
                    design_matrix_noconstant = design_matrix.loc[:, design_matrix.columns != 'constant']
                    ######MASK BOLD DATA######
                    masker = NiftiMasker(mask_img=mask_file, standardize=False)
                    bold_data_2d = masker.fit_transform(bold_file)
                    ######REGRESS OUT TASK, DRIFTS FROM BOLD DATA AD COMPUTE TSNR#####
                    mean_signal = np.mean(clean(bold_data_2d,confounds=design_matrix_noconstant.values,
                    detrend=False,standardize=False,filter=False), axis=0)
                    std_signal = np.std(clean(bold_data_2d,confounds=design_matrix_noconstant.values,
                    detrend=True,standardize=False,filter=False), axis=0)    
                    tsnr_values = mean_signal / std_signal
                    ######SAVE AND PLOT######
                    np.save(os.path.join(FMRIPREP_PATH, 'stat',
                    f'sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco}'), tsnr_values)
                    tsnr_img = masker.inverse_transform(tsnr_values)
                    nib.save(tsnr_img, os.path.join(FMRIPREP_PATH, 'stat',
                    f'sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco}'))
                    mean_bold = image.mean_img(image.index_img(bold_file, slice(10, None)))
                    tsnr_img = image.math_img("img * (img > 0)", img=tsnr_img)

                    disp = plotting.plot_stat_map(tsnr_img, bg_img=mean_bold, threshold=0, vmax=130,
                    title=f"tSNR after task and drifts regression ({moco} - {space})",
                    display_mode='z',
                    cut_coords=(-15,-10,-5,4,17,29,36,44,52), annotate=False,colorbar=False, 
                            symmetric_cbar=False,)
                    plt.savefig(os.path.join(FMRIPREP_PATH, 'figures',f'sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco}'))
                    plt.show()