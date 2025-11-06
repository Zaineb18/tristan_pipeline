#!/bin/bash

#WORKDIR="/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-02/data_ONAVoffPEERSoff"
#WORKDIR="/home/zamor/Documents/TRISTAN/data_Caro"

#subjects=("sub-02")
#for sub in "${subjects[@]}"
#do
#	for ses in 1
#	do 
#		SUBID="${sub}"
#		SESID="ses-${ses%}"
#		WORKSUBDIR="${WORKDIR%}/${SUBID%}/${SESID%}"
#		mkdir $WORKSUBDIR
#	singularity run --cleanenv \
#		--bind /home/team/freesurfer/7.4.1/license.txt:/freesurfer-license.txt:ro \
#		--bind /home/zamor/Documents/TRISTAN/ismrm_dataset/sub-02/data_ONAVoffPEERSoff/rawdata:/rawdata:ro \
#		--bind /home/zamor/Documents/TRISTAN/ismrm_dataset/sub-02/data_ONAVoffPEERSoff/derivatives/fmriprep:/out:rw \
#		--bind /home/zamor/Documents/TRISTAN/ismrm_dataset/sub-02/data_ONAVoffPEERSoff/tmp:/tmpdir:rw \
#		--bind /home/zamor/Documents/TRISTAN/ismrm_dataset/sub-02/data_ONAVoffPEERSoff/derivatives/freesurfer:/fsdir:ro \

#		/home/team/FMRIPREP/fmriprep-23.2.1.simg /rawdata /out  participant \
#		--skip_bids_validation \
#		--work-dir=/tmpdir --fs-license-file=/freesurfer-license.txt \
#	    --output-spaces func anat MNI152NLin2009cAsym \
#		--dummy-scans 0 \
#       	--ignore slicetiming \
#		--fs-subjects-dir /fsdir \
#      	--fs-no-reconall \ 
		 

#	rm -rf "$tmpdir"
#	done
#done


#!/bin/bash

WORKDIR="/home/zamor/Documents/TRISTAN/ismrm_dataset/sub-04/data_ONAVoffPEERSoff"

subjects=("sub-04")

for sub in "${subjects[@]}"; do
  for ses in 1; do 
    SUBID="${sub}"
    SESID="ses-${ses}"
    WORKSUBDIR="${WORKDIR}/${SUBID}/${SESID}"

    mkdir -p "$WORKSUBDIR"

    singularity run --cleanenv \
      --bind /home/team/freesurfer/7.4.1/license.txt:/freesurfer-license.txt:ro \
      --bind /home/zamor/Documents/TRISTAN/ismrm_dataset/sub-04/data_ONAVoffPEERSoff/rawdata:/rawdata:ro \
      --bind /home/zamor/Documents/TRISTAN/ismrm_dataset/sub-04/data_ONAVoffPEERSoff/derivatives/fmriprep:/out:rw \
      --bind /home/zamor/Documents/TRISTAN/ismrm_dataset/sub-04/data_ONAVoffPEERSoff/tmp:/tmpdir:rw \
      --bind /home/zamor/Documents/TRISTAN/ismrm_dataset/sub-04/data_ONAVoffPEERSoff/derivatives/freesurfer:/fsdir:ro \
      /home/team/FMRIPREP/fmriprep-23.2.1.simg \
      /rawdata /out participant \
      --skip_bids_validation \
      --work-dir /tmpdir \
      --fs-license-file /freesurfer-license.txt \
      --output-spaces func anat MNI152NLin2009cAsym \
      --dummy-scans 0 \
      --ignore slicetiming \
      --fs-subjects-dir /fsdir 

    # cleanup temporary directory
    rm -rf "$tmpdir"
  done
done
