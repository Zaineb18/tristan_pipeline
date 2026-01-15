import os
import nibabel as nib
import numpy as np
from nilearn import image, plotting
from nilearn.maskers import NiftiMasker
from nilearn.signal import clean
from tristan_pipeline.io.params import *
from tristan_pipeline.utils.loading_utils import *
from tristan_pipeline.utils.preproc_utils import *
from tristan_pipeline.utils.glm_utils import *

spaces = ["MNI152NLin2009cAsym", "T1w", "native bold"]

for subj in subjects:
    for ses in sessions:
        FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
        for moco in list(mocos.keys()): 
                ###########READ FMRIPREP FILES###########
                for space in spaces:
                    if space == "native bold":
                        FUNC_PATH, MASK_PATH = load_funcdata(FMRIPREP_PATH, subj, ses,moco)
                    else: 
                        FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH,_,_,xfm_MNItoT1,xfm_T1toMNI = load_fmriprepdata(FMRIPREP_PATH, subj, ses, space,moco)        
                    bold_file,mask_file = FUNC_PATH[0],MASK_PATH[0]
                    ######MAKE DESIGN MATRIX WITH TASK, DRIFTS AND CONSTANT REGRESSORS ONLY######
                    design_matrix = make_design_matrix(stimfile,None,None,minonset=min_onsets[subj],
                    delay_volumes=d_vols[subj],hrf_model='glover',tr=trs[subj],n_scans=n_vols[subj])
                    design_matrix_noconstant = design_matrix.loc[:, design_matrix.columns != 'constant']
                    ######MASK BOLD DATA######
                    masker = NiftiMasker(mask_img=mask_file, standardize=False)
                    bold_data_2d = masker.fit_transform(bold_file)
                    ######REGRESS OUT TASK, DRIFTS FROM BOLD DATA AD COMPUTE TSNR#####
                    mean_signal = np.mean(clean(bold_data_2d[2:,:],
                                                confounds=design_matrix_noconstant.values[2:,:],
                                                detrend=False,standardize=False,filter=False), axis=0)
                    std_signal = np.std(clean(bold_data_2d[2:,:],
                                              confounds=design_matrix_noconstant.values[2:,:],
                                                detrend=True,standardize=False,filter=False), axis=0)    
                    tsnr_values = mean_signal / std_signal
                    ######SAVE AND PLOT######
                    np.save(os.path.join(FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}','stats',
                    f'sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco}'), tsnr_values)
                    tsnr_img = masker.inverse_transform(tsnr_values)
                    nib.save(tsnr_img, os.path.join(FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}', 'stats',
                    f'sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco}'))
                    mean_bold = image.mean_img(image.index_img(bold_file, slice(10, None)))
                    tsnr_img = image.math_img("img * (img > 0)", img=tsnr_img)

                    disp = plotting.plot_stat_map(tsnr_img, bg_img=mean_bold,
                                                   threshold=0, vmax=100,cmap='jet', 
                    cbar_tick_format='%d',resampling_interpolation='nearest',
                    title=f"tSNR after task and drifts regression (sub-{subj:02} - {moco} - {space})",
                    display_mode='z',
                    cut_coords=(-15,-10,-5,4,17,29,36,44,52),
                      annotate=False,colorbar=False, 
                            symmetric_cbar=False,)
                    plt.savefig(os.path.join(FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}','figures',f'sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco}.pdf'))
                    plt.show()