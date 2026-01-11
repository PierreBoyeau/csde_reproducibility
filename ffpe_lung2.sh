#!/bin/bash
#BSUB -q medium
#BSUB -R "rusage[mem=200GB]"
#BSUB -R "affinity[thread*64]"
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

module load miniconda
conda activate ppi-spatial

transcript_file=~/data/spatial_data/merfish_pancancer/HumanLungCancerPatient2/detected_transcripts.csv
processed_transcripts_file=~/data/spatial_data/merfish_pancancer/HumanLungCancerPatient2/detected_transcripts.processed.csv


python workflow/ffpe-assign-transcripts-to-cells.py \
"$transcript_file" \
"$processed_transcripts_file"