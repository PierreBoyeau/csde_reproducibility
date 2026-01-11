#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=500GB]"
#BSUB -R "affinity[thread*64]"
#BSUB -gpu num=1:j_exclusive=no
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

module load miniconda
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc

conda activate proseg

transcript_file=~/data/spatial_data/merfish_pancancer/HumanColonCancerPatient2/detected_transcripts.processed.csv
output_dir=~/data/spatial_data/merfish_pancancer/HumanColonCancerPatient2/proseg_results

proseg "$transcript_file" \
    --output-expected-counts "$output_dir/expected-counts.csv.gz" \
    --output-cell-metadata "$output_dir/cell-metadata.csv.gz" \
    --output-transcript-metadata "$output_dir/transcript-metadata.csv.gz" \
    --output-cell-polygons "$output_dir/cell-polygons.geojson.gz" \
    --output-cell-polygon-layers "$output_dir/cell-polygons-layers.geojson.gz" \
    --merfish

cd $output_dir
gunzip -k cell-polygons.geojson.gz