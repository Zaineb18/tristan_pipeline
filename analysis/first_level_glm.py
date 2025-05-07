from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
from tristan_pipeline.analysis.glm_utils import *
from nilearn.glm.first_level import FirstLevelModel
import pandas as pd 
from nilearn.plotting import plot_stat_map

for subj in subjects: 
    for ses in sessions: 
        FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH = load_fmriprepdata(FMRIPREP_PATH, subj, ses, space)

        bold_file = FUNC_PATH[0]
        mask_file = MASK_PATH[0]
        confounds_file = confounds_files[0]
        anat_file = ANAT_PATH[0]
        gm_file = GM_PATH[0]

        confounds, _ = load_confounds(bold_file, strategy=('motion','global_signal','compcor','high_pass'), 
                                      motion='power2', global_signal='power2',compcor="temporal_anat_combined",
                                      n_compcor=4,scrub=0)
        confounds_names = confounds.keys()
        stimfile = "/home/zamor/nasShare/INM-GlobalShare/Boulantetal_Tristan_2025/stimfiles/session2_localizer_standard.csv"
        design_matrix =  make_design_matrix(stimfile,confounds,confounds_names,hrf_model='glover',tr=2.12, n_scans=153) 
        elem_contrasts = elementary_contrast(design_matrix.keys() )
        contrasts = custom_contrast(design_matrix.keys())

        level1_glm = FirstLevelModel(t_r=2.12, n_jobs=80, noise_model='ar1', mask_img=mask_file, smoothing_fwhm=2)
        level1_glm_fitted = level1_glm.fit(bold_file, design_matrices=design_matrix)

        contrast_vector = contrasts['constant']
        z_map = level1_glm_fitted.compute_contrast(contrast_vector, output_type='z_score')
        plot_stat_map(z_map, anat_file, title='Custom contrast: calculations vs clic', vmax=10, display_mode='z', 
                    #cut_coords=(52,54,56,58,60,62,64,66,68),
                    #cut_coords=(8,10,12,14,45,48,50,53),
                      #threshold=
                      )
        plt.show()
        plot_stat_map(z_map, anat_file, title='Custom contrast: clic right vs clic left',  vmax= 10, display_mode='z',
                       cut_coords=(0,2,4,6,8,10,12,14), 
                       threshold=3.1)
        plt.show() 