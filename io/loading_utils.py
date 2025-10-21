from tristan_pipeline.io.params import *
import glob
import nibabel as nib
#from nilearn.image import mean_img 

def load_rawdata(GLOB_DIR, subj, ses):
    RFUNC_PATH = glob.glob(os.path.join(RAW_PATH, f'sub-{subj:02}', f'ses-{ses}', 'func', f'sub-{subj:02}_ses-{ses}*_bold.nii'))
    RFMAP_PATH = glob.glob(os.path.join(RAW_PATH, f'sub-{subj:02}', f'ses-{ses}', 'fmap', f'sub-{subj:02}_ses-{ses}*_epi.nii'))
    return(RFUNC_PATH, RFMAP_PATH)

def load_fmriprepdata(FMRIPREP_PATH, subj, ses, space):
    FUNC_PATH = sorted(glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'func', f'*space-{space}*bold.nii.gz')))
    MASK_PATH = sorted(glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'func', f'*space-{space}*brain_mask.nii.gz')))
    confounds_file = sorted(glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'func', f"*_desc-confounds_timeseries.tsv")))
    if space =="T1w":
        ANAT_PATH = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'sub-{subj:02}_ses-{ses}_desc-preproc_T1w.nii.gz'))
        GM_PATH = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'sub-{subj:02}_ses-{ses}_label-GM_probseg.nii.gz'))
        WM_PATH = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'sub-{subj:02}_ses-{ses}_label-WM_probseg.nii.gz'))
        CSF_PATH = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'sub-{subj:02}_ses-{ses}_label-CSF_probseg.nii.gz'))

    else:    
        ANAT_PATH = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'*space-{space}*_T1w.nii.gz'))
        GM_PATH = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'*space-{space}*_label-GM_probseg.nii.gz'))
        WM_PATH = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'*space-{space}*_label-WM_probseg.nii.gz'))
        CSF_PATH = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'*space-{space}*_label-CSF_probseg.nii.gz'))

    xfm_MNItoT1 = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'*from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'))
    xfm_T1toMNI = glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'anat', f'*from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'))

    return(FUNC_PATH, MASK_PATH, confounds_file, ANAT_PATH, GM_PATH, WM_PATH, CSF_PATH, xfm_MNItoT1,xfm_T1toMNI)

def load_funcdata(FMRIPREP_PATH, subj, ses):
    FUNC_PATH = sorted(glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'func', f'*_desc-preproc_bold.nii.gz')))
    MASK_PATH = sorted(glob.glob(os.path.join(FMRIPREP_PATH, f'sub-{subj:02}', f'ses-{ses}', 'func', f'*desc-brain_mask.nii.gz')))
    return(FUNC_PATH, MASK_PATH)

def load_onav_reg(filepath ,labels = ["Rx ", "Ry ", "Rz ", "x ", "y ", "z ", r"$\phi$ ", "f0 ", "G$_x$", "G$_y$", "G$_z$"],
                      y_labels = [" / °", " / °", " / °", " / mm", " / mm", " / mm", " / rad", " / Hz", " / µT/m", " / µT/m", "/ µT/m"]):
    onav_reg = np.load(filepath)
    motion_reg = onav_reg[::40,:6]
    return(motion_reg, labels[:6])
