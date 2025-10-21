#####DO NOT REVIEW, CHANGES IN PATHS#####
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.image import load_img, concat_imgs
import pandas as pd
import nibabel as nib
import os
from nilearn.plotting import plot_stat_map

subjects = [1, 3, 5]
contrast_name = 'calculations'
beta_maps = []

cut_coords = [(30,39,42,54),
                      (4,12,20,24),
                      (50,52,58,60)]
for subj in subjects:
    DATA_DIR = f'/home/zamor/Documents/TRISTAN/sub-{subj:02}/data_onavONPEERSON'
    if subj in[1,3,5]:
        FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
    else: 
        FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep_reconall')

    moco = "mocopeers"
    moco_= "ONAVonPEERSon"
    beta_path = FMRIPREP_PATH+f"/stat/sub-{subj:02}_ses-1_bmap_{contrast_name}_{moco}.nii"
    beta_maps.append(beta_path)


design_matrix = pd.DataFrame([1] * len(subjects), columns=["intercept"])
second_level_model = SecondLevelModel(smoothing_fwhm=3)
second_level_model = second_level_model.fit(beta_maps, design_matrix=design_matrix)
z_map = second_level_model.compute_contrast(second_level_contrast='intercept',
                                             output_type='z_score',
                                             )
thresholded_map, threshold = threshold_stats_img(
    z_map, alpha=0.05, height_control='fpr', cluster_threshold=10
)
plot_stat_map(z_map, title=f"Second-level: {contrast_name} - {moco_}", #threshold=1.,
              display_mode="z", cut_coords=(30,39,42,54),
output_file=os.path.join('/home/zamor/Documents/TRISTAN/', 'group_figures',
                         f'groupzmap_{contrast_name}_{moco}')
)
nib.save(thresholded_map,
        '/home/zamor/Documents/TRISTAN/'+f"group_stat/groupzmap_{contrast_name}_{moco}.nii")

