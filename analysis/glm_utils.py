from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
import pandas as pd 
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt 

def make_design_matrix(stimfile,confounds,confounds_names,minonset,delay_volumes=2, 
                       hrf_model='glover',tr=2.12, n_scans=153, drift_model='cosine'): 
        df = pd.read_csv(stimfile, sep ='\t', header=None)
        df.columns = ["trial_type", "onset_ms", "event_type", "description"]
        delay = delay_volumes * tr
        df["onset"] = (df["onset_ms"] / 1000.0) + delay
        df["duration"] = 1.3
        events = df[["onset", "duration", "trial_type"]]
        frame_times = np.arange(n_scans) * tr
        design_matrix = make_first_level_design_matrix(frame_times, events, min_onset=minonset,
                                                        hrf_model='glover', 
                                                        drift_model=drift_model,
                                                       add_regs=confounds,
                                                       add_reg_names=confounds_names
                                                       )
        # Plot it
        plot_design_matrix(design_matrix)
        plt.show()
        return(design_matrix)

def elementary_contrast(design_matrix_columns): 
    elem_contrast = {}
    n_columns = len(design_matrix_columns)
    # simple contrasts
    for i in range(n_columns):
        
        elem_contrast[design_matrix_columns[i]] = np.eye(n_columns)[i]
    return elem_contrast


def custom_contrast_(design_matrix_columns):
    elem_contrast = elementary_contrast(design_matrix_columns)
    contrasts = {            
                 'mean signal': elem_contrast['constant'],
                 'CboardH': elem_contrast['CboardH'],
                 'CboardV': elem_contrast['CboardV'],
                 'calculations': elem_contrast['calculvideo']+elem_contrast['calculaudio'],
                 'checkerboard vs the others': elem_contrast['CboardH']+elem_contrast['CboardV']-elem_contrast['calculvideo']-elem_contrast['calculaudio']-elem_contrast['phraseVideo']-elem_contrast['phraseAudio']-elem_contrast['clicDvideo']-elem_contrast['clicGvideo']-elem_contrast['clicDaudio']-elem_contrast['clicGaudio'],
                 'clic right vs clic left': elem_contrast['clicDvideo'] - elem_contrast['clicGvideo'] + elem_contrast['clicDaudio'] - elem_contrast['clicGaudio'],
                 }
    return contrasts


def custom_contrast(design_matrix_columns):
    elem_contrast = elementary_contrast(design_matrix_columns)
    contrasts = {            
                 'mean signal': elem_contrast['constant'],
                 'CboardH': elem_contrast['CboardH'],
                 'CboardV': elem_contrast['CboardV'],
                 'calculations': elem_contrast['calculvideo'],
                 'sentences': elem_contrast['phraseVideo'], 
                 'clicDvideo': elem_contrast['clicDvideo'],
                 'clicGvideo': elem_contrast['clicGvideo'],
                 'checkerboard vs the others': elem_contrast['CboardH']+elem_contrast['CboardV']-elem_contrast['calculvideo']- elem_contrast['phraseVideo']-elem_contrast['clicDvideo']-elem_contrast['clicGvideo'],
                 'clic right vs clic left': elem_contrast['clicDvideo'] - elem_contrast['clicGvideo'],
                 'vertical checkerboard vs horizontal checkerboard': elem_contrast['CboardV'] - elem_contrast['CboardH'],
                 'calculations vs phrases': elem_contrast['calculvideo'] -  elem_contrast['phraseVideo'],
                 'calculations vs checkerboard': elem_contrast['calculvideo'] - elem_contrast['CboardH'] - elem_contrast['CboardV'],
                 'phrases vs checkerboard': elem_contrast['phraseVideo'] - elem_contrast['CboardH'] - elem_contrast['CboardV'],
                 'calculations vs clic': elem_contrast['calculvideo'] - elem_contrast['clicDvideo'] - elem_contrast['clicGvideo'],
                 'phrases vs clic': elem_contrast['phraseVideo'] - elem_contrast['clicDvideo'] - elem_contrast['clicGvideo'],
                 }
    return contrasts