from tristan_pipeline.io.params import *
from tristan_pipeline.io.loading_utils import *
from tristan_pipeline.preproc.preproc_utils import *

import glob
import nibabel as nib
from nilearn.image import mean_img, load_img, index_img
import os
import shutil, json

subjects = [5]
sessions = [1]

#####First: prep the ap/pa files and match them to the correct bold files
for subj in subjects: 
    for ses in sessions:
        #subj = 2
        #ses = 1 
        RFUNC_PATH, RFMAP_PATH = load_rawdata(GLOB_DIR, subj, ses)    
        for fmap_file in RFMAP_PATH:
            try:
                fmap_task = fmap_file.split('task-')[1].split('_')[0]
                fmap_acq = fmap_file.split('acq-')[1].split('_')[0]
            except IndexError:
                print(f"Could not extract tags from fmap: {fmap_file} \n")
                continue
            print(f"Fmap file: {fmap_file}, task: {fmap_task}, acq: {fmap_acq} \n")
            matching_func = None
            for bold_file in RFUNC_PATH:
                try:
                    bold_task = bold_file.split('task-')[1].split('_')[0]
                    bold_acq = bold_file.split('acq-')[1].split('_')[0]
                except IndexError:
                    continue
                #print(f"  Bold file: {bold_file}, task: {bold_task}, acq: {bold_acq}")
                if bold_task == fmap_task and bold_acq == fmap_acq:
                    #matching_func = os.path.relpath(bold_file, start=os.path.dirname(fmap_file))
                    matching_func = bold_file
                    break   
            if matching_func is None:
                print(f"No matching func file found for {fmap_file} \n")
            else:
                print(f"Matched fmap {fmap_file} with func {matching_func} \n")
            #Take first volume from funtional run for AP 
            
            #matching_func= "/home/zamor/Documents/TRISTAN/data_Caro/rawdata/sub-03/ses-1/fmap/sub-03_ses-1_dir-PA_epi.nii"
            bold_img = nib.load(matching_func)
            ap_img = index_img(bold_img, 0)
            #ap_img.to_filename("/home/zamor/Documents/TRISTAN/data_Caro/rawdata/sub-03/ses-1/fmap/sub-03_ses-1_dir-PA_epi__.nii")
            
            ap_img.to_filename(fmap_file.replace("dir-PA","dir-AP"))
            json_bold_path = matching_func.replace('nii','json')
            json_pa_path = fmap_file.replace("nii","json")
            json_ap_path = json_pa_path.replace("dir-PA","dir-AP")
            shutil.copy(json_bold_path,json_ap_path)
            #In PA json
            with open(json_pa_path, 'r') as f:
                fmap_data_pa = json.load(f)
            fmap_data_pa['IntendedFor'] = [matching_func.split(f'sub-{subj:02}/')[1]]
            fmap_data_pa["PhaseEncodingDirection"]= "j"
            with open(json_pa_path, 'w') as f:
                json.dump(fmap_data_pa, f, indent=4) 
            #In AP json
            with open(json_ap_path, 'r') as f:
                fmap_data_ap = json.load(f)
            fmap_data_ap['IntendedFor'] = [matching_func.split(f'sub-{subj:02}/')[1]]
            fmap_data_ap["PhaseEncodingDirection"]= "j-"
            with open(json_ap_path, 'w') as f:
                json.dump(fmap_data_ap, f, indent=4) 
            #In bold json
            with open(json_bold_path, 'r') as f:
                fmap_data = json.load(f)
            fmap_data["PhaseEncodingDirection"]= "j-"
            with open(json_bold_path, 'w') as f:
                json.dump(fmap_data, f, indent=4) 

#####Second: make the fmaps bids compliant
import re
import os
import shutil
from collections import defaultdict

for subj in subjects: 
    for ses in sessions:
        RFUNC_PATH, RFMAP_PATH = load_rawdata(GLOB_DIR, subj, ses)
        # Group fieldmaps by task/acq (to match AP/PA pairs)
        pair_groups = defaultdict(list)
        for fmap_file in RFMAP_PATH:
            match = re.search(r'task-[^_]+_acq-[^_]+', fmap_file)
            if match:
                key = match.group()
                pair_groups[key].append(fmap_file)

        run = 1
        for key, files in sorted(pair_groups.items()):
            for fmap_file in files:
                # Determine extensions
                for ext in ['.nii', '.nii.gz']:
                    if fmap_file.endswith(ext):
                        base_file = fmap_file
                        json_file = fmap_file.replace(ext, '.json')
                        break
                else:
                    continue  # Skip if extension not found

                # Rename using run-XX
                new_base = re.sub(r'task-[^_]+_acq-[^_]+', f'run-{run:02}', base_file)
                new_json = re.sub(r'task-[^_]+_acq-[^_]+', f'run-{run:02}', json_file)

                print(f"Renaming: {base_file} → {new_base}")
                print(f"Renaming: {json_file} → {new_json} \n")

                # Uncomment to actually rename
                shutil.copy(base_file, new_base)
                shutil.copy(json_file, new_json)
                os.remove(base_file)
                os.remove(json_file)
                compress_nii_to_niigz(new_base)

            run += 1