import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
from nilearn import image 

def consecutive_blocks(task_vector):
    blocks = []
    in_block = False
    start = None
    for i, val in enumerate(task_vector):
        if val and not in_block:
            in_block = True
            start = i
        elif not val and in_block:
            in_block = False
            blocks.append((start, i-1))
    if in_block:
        blocks.append((start, len(task_vector)-1))
    return blocks

def events_task_vectors(stimfile,n_scans=155,delay_volumes=2,tr=2.12):
    df = pd.read_csv(stimfile, sep='\t', header=None)
    df.columns = ["trial_type", "onset_ms", "event_type", "description"]    
    delay = delay_volumes * tr
    df["onset"] = (df["onset_ms"] / 1000.0) + delay
    df["duration"] = 1.3
    events = df[["onset", "duration", "trial_type"]]
    task_vector_right = np.zeros(n_scans, dtype=bool)
    task_vector_left  = np.zeros(n_scans, dtype=bool)
    task_vector_calc = np.zeros(n_scans, dtype=bool)
    right_keywords = ['clicdvideo']
    left_keywords  = ['clicgvideo']
    calc_keywords = ['calculvideo']

    for _, row in events.iterrows():
        onset_vol = int(np.floor(row['onset'] / tr))
        duration_vols = max(1, int(np.ceil(row['duration'] / tr)))
        end_vol = min(onset_vol + duration_vols, n_scans)
        ttype = row['trial_type'].strip().lower()
        if any(k == ttype for k in right_keywords):
            task_vector_right[onset_vol:end_vol] = True
        elif any(k == ttype for k in left_keywords):
            task_vector_left[onset_vol:end_vol] = True
        elif any(k == ttype for k in calc_keywords):
            task_vector_calc[onset_vol:end_vol] = True
    return(events, task_vector_right, task_vector_left, task_vector_calc)        

def prep_stats_anats_tissues(mask_file, gm_file, wm_file, csf_file, stats_file):
    
    gm_img = nib.load(gm_file)
    wm_img = nib.load(wm_file)
    csf_img = nib.load(csf_file)
    stats_img = nib.load(stats_file)
    brain_mask = nib.load(mask_file).get_fdata().astype(bool)

    gm_img_res = image.resample_to_img(gm_img, stats_img, interpolation='continuous')
    wm_img_res = image.resample_to_img(wm_img, stats_img, interpolation='continuous')
    csf_img_res = image.resample_to_img(csf_img, stats_img, interpolation='continuous')
    gm = gm_img_res.get_fdata()
    wm = wm_img_res.get_fdata()
    csf = csf_img_res.get_fdata()
    
    stats_data = stats_img.get_fdata()    
    stats_data[stats_data < 0] = 0
    #brain_mask = (gm + wm + csf) > 0
    #stats_data = stats_data * brain_mask
    return(gm,wm,csf,stats_data, brain_mask)

def make_tissues(wm, gm, csf, min_thresh=0.7): 
    probs = np.stack([gm, wm, csf])
    tissue_labels = np.argmax(probs, axis=0)
    max_prob = np.max(probs, axis=0)
    ambiguous = max_prob < min_thresh
    tissue_labels[ambiguous] = -1

    gm_core = (tissue_labels == 0)
    wm_core = (tissue_labels == 1)
    csf_core = (tissue_labels == 2)
    
    struct = np.ones((3,3,3))
    wm_shell = ( binary_dilation(wm_core, structure=struct) & (~wm_core) & (~gm_core) & (~csf_core)) 
    gm_shell = ( binary_dilation(gm_core, structure=struct) & (~gm_core)  & (~wm_core) & (~csf_core))
    csf_shell = ( binary_dilation(csf_core, structure=struct) & (~csf_core) & (~wm_core) & (~gm_core))

    #gm_wm_interface = gm_shell & wm_shell
    #gm_csf_interface = gm_shell & csf_shell
    gm_wm_interface = (gm_shell & wm_shell) #| (ambiguous & (gm_shell | wm_shell))
    gm_csf_interface = (gm_shell & csf_shell) #| (ambiguous & (gm_shell | csf_shell))
    ambiguous = ambiguous & (~wm_core) & (~gm_core) & (~csf_core)  & (~wm_shell) & (~gm_shell) & (~csf_shell) & (~gm_wm_interface) & (~gm_csf_interface)
    return(wm_core, gm_core, csf_core, gm_wm_interface, gm_csf_interface, ambiguous)