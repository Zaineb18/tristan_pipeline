import os
import nibabel as nib
from nilearn import image
from tristan_pipeline.io.params import *

space = "MNI152NLin2009cAsym"
os.makedirs(grp_dir, exist_ok=True)

for moco_label in list(mocos.keys()):
    subject_maps = []
    #########LOOP OVER SUBJECTS AND SESSIONS#########
    for subj in subjects:
        for ses in sessions:
            FMRIPREP_PATH =os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
            tsnr_file = os.path.join(FMRIPREP_PATH,f'sub-{subj:02}',f'ses-{ses}','stats',f"sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco_label}.nii")
            if not os.path.exists(tsnr_file):
                print(f"Missing file: {tsnr_file}")
                continue
            img = nib.load(tsnr_file)
            subject_maps.append(img)
    if not subject_maps:
        print(f"No maps found for {moco_label}")
        continue
    #########COMPUTE GROUP AVERAGE#########
    group_img = image.mean_img(subject_maps)
    group_img = image.math_img("img * (img > 0)", img=group_img)
    #########SAVE NII#########
    os.makedirs(os.path.join(grp_dir,'stats'), exist_ok=True)
    group_file = os.path.join(os.path.join(grp_dir,'stats'),f"group_tSNR_space-{space}_{moco_label}.nii.gz")
    nib.save(group_img, group_file)
