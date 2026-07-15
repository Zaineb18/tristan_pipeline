import glob
import json
import os
import shutil
import nibabel as nib
from nilearn.image import index_img

# ==============================================================================
# Fmap preparation for Caroline_Full_Dataset
# ==============================================================================
# Each subject has:
#   fmap/  : one PA epi  (dir-PA_epi.nii.gz + .json)
#   func/  : one or two AP bold runs
#
# This script:
#   1. Extracts the first volume of each AP bold run → dir-AP_epi (run-01/02 if needed)
#   2. Updates PhaseEncodingDirection in all relevant JSONs
#   3. Sets IntendedFor in both AP and PA fmap JSONs
# ==============================================================================

RAW_PATH = '/home/zamor/Documents/TRISTAN/dataset_Caroline/rawdata'
#subjects = list(range(1, 10))   # sub-01 … sub-09
subjects= [1,2,3,4,5,6,7,8,9]
sessions = [1]

for subj in subjects:
    for ses in sessions:
        sub_label = f"sub-{subj:02d}"
        ses_label = f"ses-{ses}"

        func_dir = os.path.join(RAW_PATH, sub_label, ses_label, "func")
        fmap_dir = os.path.join(RAW_PATH, sub_label, ses_label, "fmap")

        if not os.path.isdir(func_dir) or not os.path.isdir(fmap_dir):
            print(f"[SKIP] Missing func or fmap dir for {sub_label} {ses_label}")
            continue

        # ------------------------------------------------------------------
        # 1. Collect AP bold runs (sorted → run-01, run-02, ...)
        # ------------------------------------------------------------------
        ap_bold_files = sorted(glob.glob(
            os.path.join(func_dir, f"{sub_label}_{ses_label}_task-localizer*_bold.nii.gz")
        ))

        if not ap_bold_files:
            print(f"[WARN] No AP bold files found for {sub_label}")
            continue

        multi_run = len(ap_bold_files) > 1
        print(f"\n{'='*60}")
        print(f"{sub_label} | {ses_label} | {len(ap_bold_files)} AP bold run(s)")

        # ------------------------------------------------------------------
        # 2. Collect the single PA fmap
        # ------------------------------------------------------------------
        pa_files = glob.glob(
            os.path.join(fmap_dir, f"{sub_label}_{ses_label}_dir-PA_epi.nii.gz")
        )
        if len(pa_files) != 1:
            print(f"[WARN] Expected exactly 1 PA fmap, found {len(pa_files)} – skipping {sub_label}")
            continue

        pa_nii_orig  = pa_files[0]
        pa_json_orig = pa_nii_orig.replace(".nii.gz", ".json")

        # ------------------------------------------------------------------
        # 3. For each AP bold run: extract vol-0, write AP fmap, handle PA
        # ------------------------------------------------------------------
        for run_idx, bold_nii in enumerate(ap_bold_files, start=1):

            run_label = f"run-{run_idx:02d}" if multi_run else None

            # Relative path for IntendedFor (relative to subject root)
            rel_bold = os.path.join(ses_label, "func", os.path.basename(bold_nii))

            # -- Build filenames -------------------------------------------
            if run_label:
                ap_nii  = os.path.join(fmap_dir,
                    f"{sub_label}_{ses_label}_{run_label}_dir-AP_epi.nii.gz")
                pa_nii  = os.path.join(fmap_dir,
                    f"{sub_label}_{ses_label}_{run_label}_dir-PA_epi.nii.gz")
            else:
                ap_nii  = os.path.join(fmap_dir,
                    f"{sub_label}_{ses_label}_dir-AP_epi.nii.gz")
                pa_nii  = pa_nii_orig   # single run: edit in place

            ap_json = ap_nii.replace(".nii.gz", ".json")
            pa_json = pa_nii.replace(".nii.gz", ".json")

            # -- AP: extract first volume ----------------------------------
            bold_img = nib.load(bold_nii)
            ap_img   = index_img(bold_img, 0)
            ap_img.to_filename(ap_nii)
            print(f"  [AP]   Extracted vol-0 → {os.path.basename(ap_nii)}")

            bold_json = bold_nii.replace(".nii.gz", ".json")
            shutil.copy(bold_json, ap_json)

            with open(ap_json, "r") as f:
                ap_data = json.load(f)
            ap_data["PhaseEncodingDirection"] = "i-"
            ap_data["IntendedFor"] = [rel_bold]
            with open(ap_json, "w") as f:
                json.dump(ap_data, f, indent=4)
            print(f"  [AP]   JSON updated → {os.path.basename(ap_json)}")

            # -- PA: duplicate per run (or edit in place for single run) --
            if multi_run:
                shutil.copy(pa_nii_orig, pa_nii)
                shutil.copy(pa_json_orig, pa_json)
                print(f"  [PA]   Duplicated   → {os.path.basename(pa_nii)}")

            with open(pa_json, "r") as f:
                pa_data = json.load(f)
            pa_data["PhaseEncodingDirection"] = "i"
            pa_data["IntendedFor"] = [rel_bold]   # points only to its own run
            with open(pa_json, "w") as f:
                json.dump(pa_data, f, indent=4)
            print(f"  [PA]   JSON updated → {os.path.basename(pa_json)}")
            print(f"         IntendedFor  → {rel_bold}")

            # -- Bold JSON: just fix PhaseEncodingDirection ----------------
            with open(bold_json, "r") as f:
                bold_data = json.load(f)
            bold_data["PhaseEncodingDirection"] = "i-"
            with open(bold_json, "w") as f:
                json.dump(bold_data, f, indent=4)
            print(f"  [BOLD] JSON updated → {os.path.basename(bold_json)}")

        # ------------------------------------------------------------------
        # 4. Remove the original unrun-labeled PA if we duplicated it
        # ------------------------------------------------------------------
        if multi_run:
            os.remove(pa_nii_orig)
            os.remove(pa_json_orig)
            print(f"\n  [CLEAN] Removed original: {os.path.basename(pa_nii_orig)}")
            print(f"  [CLEAN] Removed original: {os.path.basename(pa_json_orig)}")

print("\nDone.")