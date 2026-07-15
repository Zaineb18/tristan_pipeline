#!/bin/bash

WORKDIR="/home/zamor/Documents/TRISTAN/dataset_recaroline/dataset-9"

singularity run --cleanenv \
		--bind /home/team/freesurfer/7.4.1/license.txt:/freesurfer-license.txt:ro \
		--bind /home/zamor/Documents/TRISTAN/dataset_recaroline/dataset-9/rawdata:/rawdata:ro \
		--bind /home/zamor/Documents/TRISTAN/dataset_recaroline/dataset-9/derivatives/fmriprep:/out:rw \
		--bind /home/zamor/Documents/TRISTAN/dataset_recaroline/dataset-9/tmp:/tmpdir:rw \
		/home/team/FMRIPREP/fmriprep-23.2.1.simg /rawdata /out  participant \
		--skip_bids_validation \
		--work-dir=/tmpdir --fs-license-file=/freesurfer-license.txt \
	    --output-spaces func anat MNI152NLin2009cAsym \
		--dummy-scans 0 \
       	--ignore slicetiming \
       	--fs-no-reconall  
		
	rm -rf "$tmpdir"

