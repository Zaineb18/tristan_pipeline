#!/bin/bash

WORKDIR="/home/zamor/Documents/TRISTAN/data_onavOFF"
subjects=("sub-01")
for sub in "${subjects[@]}"
do
	for ses in 1
	do 
		SUBID="${sub}"
		SESID="ses-${ses%}"
		WORKSUBDIR="${WORKDIR%}/${SUBID%}/${SESID%}"
		mkdir $WORKSUBDIR
	singularity run --cleanenv \
		--bind /home/team/freesurfer/7.4.1/license.txt:/freesurfer-license.txt:ro \
		--bind /home/zamor/Documents/TRISTAN/data_onavOFF/rawdata:/rawdata:ro \
		--bind /home/zamor/Documents/TRISTAN/data_onavOFF/derivatives/fmriprep:/out:rw \
		--bind /home/zamor/Documents/TRISTAN/data_onavOFF/tmp:/tmpdir:rw \
		/home/team/FMRIPREP/fmriprep-23.2.1.simg /rawdata /out  participant \
		--skip_bids_validation \
		--work-dir=/tmpdir --fs-license-file=/freesurfer-license.txt \
	        --output-spaces func anat MNI152NLin2009cAsym \
		--dummy-scans 0 \
       		--ignore slicetiming \
       		--fs-no-reconall  
       		

	rm -rf "$tmpdir"
	done
done
