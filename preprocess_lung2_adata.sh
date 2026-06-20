#!/bin/bash
#BSUB -q medium
#BSUB -R "rusage[mem=150GB]"
#BSUB -R "affinity[thread*32]"
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

DATA_DIR=${MERFISH_DATA_DIR:-/path/to/merfish_pancancer}

module load miniconda
conda activate ppi-spatial

python preprocess_data.py \
"$DATA_DIR/HumanLungCancerPatient2/proseg_results" \
"$DATA_DIR/lung2_adata.h5ad"