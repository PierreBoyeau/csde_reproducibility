#!/bin/bash
#BSUB -q medium
#BSUB -R "rusage[mem=150GB]"
#BSUB -R "affinity[thread*32]"
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

module load miniconda
conda activate ppi-spatial

python preprocess_data.py \
~/data/spatial_data/merfish_pancancer/HumanLungCancerPatient1/proseg_results \
~/data/spatial_data/merfish_pancancer/lung1_adata.h5ad