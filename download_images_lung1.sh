#!/bin/bash
#BSUB -q short
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "affinity[thread*8]"
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

module load miniconda
conda activate gcloud

cd /home/labs/nyosef/pierrebo/data/spatial_data/merfish_pancancer/HumanLungCancerPatient1
mkdir images
cd images

gsutil -m cp \
  "gs://vz-ffpe-showcase/HumanLungCancerPatient1/images/micron_to_mosaic_pixel_transform.csv" \
  "gs://vz-ffpe-showcase/HumanLungCancerPatient1/images/mosaic_Cellbound1_z3.tif" \
  "gs://vz-ffpe-showcase/HumanLungCancerPatient1/images/mosaic_Cellbound2_z3.tif" \
  "gs://vz-ffpe-showcase/HumanLungCancerPatient1/images/mosaic_Cellbound3_z3.tif" \
  "gs://vz-ffpe-showcase/HumanLungCancerPatient1/images/mosaic_DAPI_z3.tif" \
  "gs://vz-ffpe-showcase/HumanLungCancerPatient1/images/mosaic_PolyT_z3.tif" \
  .
