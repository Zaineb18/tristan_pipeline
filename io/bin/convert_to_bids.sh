
#!/bin/bash

# ==============================================================================
# DICOM → NIfTI conversion + BIDS organization for Caroline_Full_Dataset
# ==============================================================================
# Input  : sourcedata/sub-*/ses-1/<series>/
# Output : rawdata/sub-*/ses-1/{anat,fmap,func}/
#
# Series naming conventions:
#   *MPRAGE*              → anat/  T1w
#   *CAIPI_PA*            → fmap/  dir-PA_epi   (single reverse-PE volume)
#   *CAIPI_AP*            → func/  task-rest_bold (AP = main bold direction)
#                           Multiple AP runs are indexed as run-1, run-2, ...
#
# Usage  : bash convert_to_bids.sh /path/to/Caroline_Full_Dataset
# ==============================================================================

set +e
set +u

# ---------- argument ----------------------------------------------------------
ROOT_DIR="${1}"
if [[ ! -d "$ROOT_DIR" ]]; then
    echo "[ERROR] Usage: $0 /path/to/Caroline_Full_Dataset" >&2
    exit 1
fi

SOURCEDATA="${ROOT_DIR}/sourcedata"
RAWDATA="${ROOT_DIR}/rawdata"

# ---------- helpers -----------------------------------------------------------
log()     { echo "[INFO]    $*"; }
verbose() { echo "[VERBOSE] $*" >&2; }
warn()    { echo "[WARN]    $*"; }

log "ROOT      : $ROOT_DIR"
log "SOURCEDATA: $SOURCEDATA"
log "RAWDATA   : $RAWDATA"
echo ""

# ---------- convert one series ------------------------------------------------
convert_series() {
    local dicom_dir="$1"
    local out_dir="$2"
    local label="$3"
    local bids_name="$4"       # full BIDS filename stem (no extension)

    verbose "  dicom_dir : $dicom_dir"
    verbose "  out_dir   : $out_dir"
    verbose "  bids_name : $bids_name"

    local n_dcm
    n_dcm=$(find "$dicom_dir" -type f \( -iname "*.dcm" -o -iname "*.ima" \) | wc -l)
    verbose "  DICOM files found: $n_dcm"

    if [ "$n_dcm" -eq 0 ]; then
        warn "No DICOM files in: $dicom_dir – skipping [$label]"
        return
    fi

    mkdir -p "$out_dir"

    log "▶ Converting [$label] → ${bids_name}.nii.gz  ($n_dcm files)"

    cd /home/team/
    ./dcm2niix -z y -f "$bids_name" -o "$out_dir" "$dicom_dir" || true

    log "✔ [$label] done → ${out_dir}/${bids_name}.nii.gz"
    echo ""
}

# ---------- main loop ---------------------------------------------------------
total=0
skipped=0

log "Scanning sourcedata..."
echo ""

for sub_src in "${SOURCEDATA}"/sub-*/; do
    [[ ! -d "$sub_src" ]] && continue

    sub_id=$(basename "$sub_src")           # e.g. sub-01
    out_base="${RAWDATA}/${sub_id}/ses-1"

    log "══════════════════════════════════"
    log "Subject : $sub_id"
    log "src     : $sub_src"
    log "out     : $out_base"
    echo ""

    # We process ses-1 directly (no hash level in this dataset)
    ses_dir="${sub_src}ses-1/"
    [[ ! -d "$ses_dir" ]] && { warn "No ses-1 dir for $sub_id – skipping."; continue; }

    # -- Count AP bold runs first so we can index them -------------------------
    # Collect all AP series dirs sorted by their numeric prefix
    mapfile -t ap_dirs < <(
        find "$ses_dir" -maxdepth 1 -type d -name "*CAIPI_AP*" \
        | sort -t'/' -k7 -V
    )
    n_ap_runs=${#ap_dirs[@]}
    verbose "  AP bold runs found: $n_ap_runs"

    ap_run_index=0   # incremented as we encounter AP dirs below

    # -- Iterate over series dirs sorted numerically by leading index ----------
    while IFS= read -r -d '' series_dir; do
        [[ ! -d "$series_dir" ]] && continue

        series_name=$(basename "$series_dir")
        name_lower=$(echo "$series_name" | tr '[:upper:]' '[:lower:]')

        log "── Series: '$series_name'"
        verbose "  full path: $series_dir"

        # ---- Route to BIDS folder/suffix ------------------------------------
        if echo "$name_lower" | grep -q "mprage"; then
            # ---- T1w anatomical ---------------------------------------------
            folder="anat"
            bids_name="${sub_id}_ses-1_T1w"
            convert_series "$series_dir" "${out_base}/${folder}" "$series_name" "$bids_name"
            total=$((total + 1))

        elif echo "$name_lower" | grep -iq "caipi_pa"; then
            # ---- Reverse-PE fieldmap (PA) ------------------------------------
            folder="fmap"
            bids_name="${sub_id}_ses-1_dir-PA_epi"
            convert_series "$series_dir" "${out_base}/${folder}" "$series_name" "$bids_name"
            total=$((total + 1))

        elif echo "$name_lower" | grep -iq "caipi_ap"; then
            # ---- Main BOLD run (AP) ------------------------------------------
            folder="func"
            ap_run_index=$((ap_run_index + 1))

            if [ "$n_ap_runs" -gt 1 ]; then
                # pad run index to 2 digits for consistency
                run_label=$(printf "run-%02d" "$ap_run_index")
                bids_name="${sub_id}_ses-1_task-localizer_${run_label}_bold"
            else
                bids_name="${sub_id}_ses-1_task-localizer_bold"
            fi

            convert_series "$series_dir" "${out_base}/${folder}" "$series_name" "$bids_name"
            total=$((total + 1))

        else
            warn "  No BIDS match for '$series_name' – skipping."
            skipped=$((skipped + 1))
        fi

    done < <(find "$ses_dir" -maxdepth 1 -mindepth 1 -type d -print0 | sort -zV)

done

echo ""
log "══════════════════════════════════"
log "Done.  Converted : $total | Skipped: $skipped"
log "NIfTIs are in    : ${RAWDATA}"