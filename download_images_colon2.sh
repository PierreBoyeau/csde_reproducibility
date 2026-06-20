#!/bin/bash
#BSUB -q short
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "affinity[thread*8]"
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err


# Set DATA_DIR to the root of the merfish_pancancer dataset directory.
DATA_DIR=${MERFISH_DATA_DIR:-/path/to/merfish_pancancer}

mkdir -p "$DATA_DIR/HumanColonCancerPatient2/images"
cd "$DATA_DIR/HumanColonCancerPatient2/images"

gsutil -m cp \
  "gs://vz-ffpe-showcase/HumanColonCancerPatient2/images/micron_to_mosaic_pixel_transform.csv" \
  "gs://vz-ffpe-showcase/HumanColonCancerPatient2/images/mosaic_Cellbound1_z3.tif" \
  "gs://vz-ffpe-showcase/HumanColonCancerPatient2/images/mosaic_Cellbound2_z3.tif" \
  "gs://vz-ffpe-showcase/HumanColonCancerPatient2/images/mosaic_Cellbound3_z3.tif" \
  "gs://vz-ffpe-showcase/HumanColonCancerPatient2/images/mosaic_DAPI_z3.tif" \
  "gs://vz-ffpe-showcase/HumanColonCancerPatient2/images/mosaic_PolyT_z3.tif" \
  .
