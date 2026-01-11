#!/bin/bash
#BSUB -q gsla_high_gpu
#BSUB -R "rusage[mem=250GB]"
#BSUB -R "affinity[thread*64]"
#BSUB -gpu num=1:j_exclusive=no
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

module load miniconda
conda activate ppi-spatial

python preprocess_data.py \
~/data/spatial_data/merfish_pancancer/HumanColonCancerPatient1/proseg_results \
~/data/spatial_data/merfish_pancancer/HumanColonCancerPatient2/proseg_results \
~/data/spatial_data/merfish_pancancer/colon_adata.h5ad