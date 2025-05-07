from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *
import pandas as pd 
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt 

def make_design_matrix(stimfile,confounds,confounds_names,hrf_model='glover',tr=2.12, n_scans=153): 
        df = pd.read_csv(stimfile, sep ='\t', header=None)
        df.columns = ["trial_type", "onset_ms", "event_type", "description"]
        df["onset"] = df["onset_ms"] / 1000.0
        df["duration"] = 1.3
        events = df[["onset", "duration", "trial_type"]]
        frame_times = np.arange(n_scans) * tr
        design_matrix = make_first_level_design_matrix(frame_times, events, hrf_model='glover', drift_model='cosine',
                                                       add_regs=confounds,add_reg_names=confounds_names)
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

def custom_contrast(design_matrix_columns):
    elem_contrast = elementary_contrast(design_matrix_columns)
    contrasts = {            
                 'constant': elem_contrast['constant'],
                 'CboardH': elem_contrast['CboardH'],
                 'CboardV': elem_contrast['CboardV'],
                 'calculvideo': elem_contrast['calculvideo'],
                 'phraseVideo': elem_contrast['phraseVideo'], 
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


