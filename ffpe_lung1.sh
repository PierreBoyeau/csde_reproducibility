#!/bin/bash
#BSUB -q medium
#BSUB -R "rusage[mem=200GB]"
#BSUB -R "affinity[thread*64]"
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

module load miniconda
DATA_DIR=${MERFISH_DATA_DIR:-/path/to/merfish_pancancer}

conda activate ppi-spatial

transcript_file="$DATA_DIR/HumanLungCancerPatient1/detected_transcripts.csv"
processed_transcripts_file="$DATA_DIR/HumanLungCancerPatient1/detected_transcripts.processed.csv"


python workflow/ffpe-assign-transcripts-to-cells.py \
"$transcript_file" \
"$processed_transcripts_file"