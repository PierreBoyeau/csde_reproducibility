#!/bin/bash
#BSUB -q gsla_high_gpu
#BSUB -R "rusage[mem=250GB]"
#BSUB -R "affinity[thread*64]"
#BSUB -gpu num=1:j_exclusive=no
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

DATA_DIR=${MERFISH_DATA_DIR:-/path/to/merfish_pancancer}

module load miniconda
conda activate ppi-spatial

python preprocess_data.py \
"$DATA_DIR/HumanLungCancerPatient1/proseg_results" \
"$DATA_DIR/HumanLungCancerPatient2/proseg_results" \
"$DATA_DIR/lung_adata.h5ad"