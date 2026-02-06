import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
from nilearn import image 
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests
from scipy.stats import permutation_test

def consecutive_blocks_(task_vector):
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

def consecutive_blocks(task_vector):
    blocks = []
    for i, val in enumerate(task_vector):
        if val:  # event at each single timepoint
            blocks.append((i, i))
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
    task_vector_lang = np.zeros(n_scans, dtype=bool)
    task_vector_visu = np.zeros(n_scans, dtype=bool)

    right_keywords = ['clicdvideo']
    left_keywords  = ['clicgvideo']
    calc_keywords = ['calculvideo']
    lang_keywords = ['phraseVideo']
    visu_keywords = ['CboardH', 'CboardV']

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
        elif any(k == ttype for k in lang_keywords):
            task_vector_lang[onset_vol:end_vol] = True
        elif any(k == ttype for k in visu_keywords):
            task_vector_visu[onset_vol:end_vol] = True                        
    return(events, task_vector_right, task_vector_left, task_vector_calc, task_vector_lang, task_vector_visu)        

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

def compute_stars(values_list):
    baseline_data = values_list[0]
    p_values = [np.nan]
    stats_values = [np.nan] # U statistics
    effect_sizes = [np.nan]           
    for data in values_list[1:]:
        if len(baseline_data) == 0 or len(data) == 0:
            p_values.append(np.nan)
            stats_values.append(np.nan)
            continue
        #stat, p = mannwhitneyu(baseline_data, data, alternative='two-sided')
        stat, p = ttest_ind(baseline_data, data, equal_var=False) #Welch: unpaired t-test
        #stat, p = ttest_rel(baseline_data, data) #paired t-test
        p_values.append(p)
        stats_values.append(stat)
        print('p')
        #n1, n2 = len(baseline_data), len(data)
        #r_rb = 1 - (2 * stat) / (n1 * n2)
        #effect_sizes.append(r_rb)       
    # Bonferroni correction
    _, p_corrected, _, _ = multipletests(p_values[1:], method='bonferroni')
    p_corrected = [np.nan] + list(p_corrected)
    # assign stars
    stars = []
    baseline_median = np.median(baseline_data) if len(baseline_data) > 0 else np.nan
    #for vals, p in zip(values_list, p_corrected):
    for vals, p in zip(values_list, p_corrected):
        
        if np.isnan(p):
            stars.append("")
            continue
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = ""
        # add arrow
        if star:
            current_median = np.median(vals)
            if current_median > baseline_median:
                star += "↑"
            elif current_median < baseline_median:
                star += "↓"
        stars.append(star)
    return stars, p_values, p_corrected, stats_values#, effect_sizes

def _perm_test_mean(x, y, n_perm=10000):
    res = permutation_test(
        (x, y),
        statistic=lambda a, b: np.mean(a) - np.mean(b),
        permutation_type='independent',
        n_resamples=n_perm,
        alternative='two-sided',
        random_state=42
    )
    print('lll')
    return res.statistic, res.pvalue


def compute_stars_z(values_list, n_perm=10000):
    """
    Compare suprathreshold z-scores using permutation tests.
    Tests difference in mean(z | z > threshold).

    values_list[0] = baseline
    """
    baseline = values_list[0]
    p_values = [np.nan]
    stats_values = [np.nan]
    for vals in values_list[1:]:
        if len(baseline) == 0 or len(vals) == 0:
            p_values.append(np.nan)
            stats_values.append(np.nan)
            continue

        stat, p = _perm_test_mean(baseline, vals, n_perm=n_perm)

        stats_values.append(stat)
        p_values.append(p)

    # ----- Bonferroni correction -----
    _, p_corrected, _, _ = multipletests(p_values[1:], method="bonferroni")
    p_corrected = [np.nan] + list(p_corrected)

    # ----- stars + direction -----
    stars = []
    baseline_mean = np.mean(baseline) if len(baseline) else np.nan
    for vals, p in zip(values_list, p_corrected):
        if np.isnan(p):
            stars.append("")
            continue

        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = ""

        if star and len(vals) > 0:
            current_mean = np.mean(vals)
            if current_mean > baseline_mean:
                star += "↑"
            elif current_mean < baseline_mean:
                star += "↓"

        stars.append(star)

    return stars, p_values, p_corrected, stats_values
