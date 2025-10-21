from tristan_pipeline.io.params import *
from tristan_pipeline.utils.loading_utils import *
from tristan_pipeline.utils.preproc_utils import *
from tristan_pipeline.utils.analysis_utils import *
from tristan_pipeline.utils.plotting_utils import *

from nilearn.plotting import plot_stat_map
from nilearn import plotting, image
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
##########CONFIG###########
mocos = ["ONAVonPEERSon"]
spaces = ["T1w"]
subjects = [1, 2, 3, 4]
sessions = [1]
# Unique styles per subject
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
markers = ['o', 's', '^', '*']
alphas = [1.0, 1.0, 0.7, 0.5]
##########CONFIG###########
#######INITIALIZE COMBINED DATAFRAME 
#Initialize combined DataFrame for all subjects
combined_tsnr_df = pd.DataFrame(columns=['Subject', 'Tissue', 'tSNR'])

for subj_idx, subj in enumerate(subjects):
    for ses in sessions:
        for moco in mocos:
            ########### READ FMRIPREP FILES ###########
            DATA_DIR = f"/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-{subj:02}/data_{moco}"
            FMRIPREP_PATH = os.path.join(DATA_DIR, 'derivatives', 'fmriprep')
            for space in spaces:
                FUNC_PATH, MASK_PATH, confounds_files, ANAT_PATH, GM_PATH, WM_PATH, CSF_PATH, xfm_MNItoT1, xfm_T1toMNI = load_fmriprepdata(FMRIPREP_PATH, subj, ses, space)
                mask_file, anat_file, gm_file, wm_file, csf_file = MASK_PATH[0], ANAT_PATH[0], GM_PATH[0], WM_PATH[0], CSF_PATH[0]
                tsnr_file = os.path.join(FMRIPREP_PATH, 'stat', f"sub-{subj:02}_ses-{ses}_tSNRmap_space-{space}_{moco}.nii")
                

                t1_img_res = image.resample_to_img(nib.load(anat_file),
                                                   nib.load(tsnr_file), interpolation='continuous')

                gm,wm,csf,tsnr_data, brain_mask=prep_stats_anats_tissues(mask_file,
                                                gm_file, wm_file, csf_file, tsnr_file)
                wm_core, gm_core, csf_core, gm_wm_interface, gm_csf_interface,_ = make_tissues(wm, gm,
                                                                         csf, min_thresh=0.7)
                gm_core = gm_core.astype(bool)
                wm_core = wm_core.astype(bool)
                csf_core = csf_core.astype(bool)
                gm_wm_interface = gm_wm_interface.astype(bool)
                gm_csf_interface = gm_csf_interface.astype(bool)


                display_tissues(gm_core, wm_core, csf_core,
                    gm_wm_interface, gm_csf_interface,
                    nib.load(tsnr_file).affine, t1_img_res, title="Tissues and Interfaces")
                
                
                # Extract voxel-wise tSNR values
                tsnr_values = {
                    "GM": tsnr_data[gm_core],
                    "WM": tsnr_data[wm_core],
                    #"CSF": tsnr_data[csf_core],
                    "GM/WM interface": tsnr_data[gm_wm_interface],
                    #"Pial surface": tsnr_data[gm_csf_interface],
                }
                # Convert to long-format DataFrame for seaborn
                tsnr_long = []
                for tissue, values in tsnr_values.items():
                    for v in values:
                        tsnr_long.append({'Subject': f"sub-{subj:02}", 'Tissue': tissue, 'tSNR': v})
                tsnr_df = pd.DataFrame(tsnr_long)
                # Append to combined DataFrame
                combined_tsnr_df = pd.concat([combined_tsnr_df, tsnr_df], ignore_index=True)

# ----------------- Combined Violin Plot -----------------
plt.figure(figsize=(14, 6))
sns.violinplot(x='Tissue', y='tSNR', hue='Subject', data=combined_tsnr_df, palette=colors, inner='quartile')
plt.title("Temporal SNR distribution (all subjects)", size=20)
plt.ylabel("tSNR")
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Subject")
plt.tight_layout()
plt.show()
