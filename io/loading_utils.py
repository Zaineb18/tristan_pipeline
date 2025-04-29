from tristan_pipeline.io.params import *
import glob
import nibabel as nib
#from nilearn.image import mean_img 

def load_rawdata(GLOB_DIR, subj, ses):
    RFUNC_PATH = glob.glob(os.path.join(RAW_PATH, f'sub-{subj:02}', f'ses-{ses}', 'func', f'sub-{subj:02}_ses-{ses}*_bold.nii'))
    RFMAP_PATH = glob.glob(os.path.join(RAW_PATH, f'sub-{subj:02}', f'ses-{ses}', 'fmap', f'sub-{subj:02}_ses-{ses}*_epi.nii'))
    return(RFUNC_PATH, RFMAP_PATH)
