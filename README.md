# CSDE: Correcting Spatial DE Analysis

This repository contains the code required to reproduce the results described in the paper "Correcting Spatial Differential Expression Analysis".

## Package description & installation

The `spatial-correction` folder contains the source code for CSDE utils and core statistical components. It is a Python package managed by poetry, but can be installed using pip in your environment.

To install the package:

```bash
pip install ./spatial-correction
```

**Note:** The scripts and notebooks in this repository contain hardcoded file paths (e.g., to `/data1/datasets/...` or `~/data/...`). You **must** replace these paths with the appropriate locations on your system where you have stored the data. Alternatively, you can modify the scripts to use relative paths if your data structure matches the repository layout.

## Pipeline walkthrough

The analysis pipeline consists of several stages, from raw data acquisition to the application of the CSDE method.

### 1. Raw data downloads
The raw data for this analysis comes from the MERSCOPE FFPE Human Immuno-Oncology Data Release.
You can access the data at the [Vizgen FFPE Showcase](https://info.vizgen.com/ffpe-showcase).
Specifically, we used the **Lung Cancer** (Patient 1 & 2) and **Colon Cancer** (Patient 1 & 2) datasets.

The downloading process is automated in the scripts:
- `download_images_lung1.sh`
- `download_images_lung2.sh`
- `download_images_colon1.sh`
- `download_images_colon2.sh`

### 2. Segmentation
We utilize **ProSeg** for cell segmentation, a probabilistic segmentation method for in situ transcriptomics.
Please refer to the [ProSeg GitHub repository](https://github.com/dcjones/proseg) for installation instructions and detailed usage documentation.

The segmentation is performed on the raw transcript files downloaded in the previous step. The scripts are:
- `proseg_lung1.sh`
- `proseg_lung2.sh`
- `proseg_colon1.sh`
- `proseg_colon2.sh`

### 3. Rough annotation
Following segmentation, an initial round of automated annotation is performed to classify cells into major cell types. This provides the "Automated" baseline for our comparisons and serves as the starting point for manual curation.

This involves two steps:
1. **Preprocessing:** Creating an `AnnData` object from ProSeg outputs.
   - Scripts: `preprocess_lung1_adata.sh`, `preprocess_lung2_adata.sh`, `preprocess_colon1_adata.sh`, `preprocess_colon2_adata.sh`.
   - Uses `preprocess_data.py` internally.
2. **Annotation:** Running notebooks to cluster and label cells based on marker genes.
   - Notebooks: `annotate_lung1.ipynb`, `annotate_lung2.ipynb`, `annotate_colon1.ipynb`, `annotate_colon2.ipynb`.

### 4. Export of a random subset of cells
To facilitate manual curation, we export a random subset of cells for verification. This process is covered in the notebook `export_data.ipynb`. This notebook:
- Loads the segmented and roughly annotated data.
- Subsamples cells (with importance sampling for rare cell types like T cells).
- Exports images and metadata for manual review.

These images are then manually annotated using [CVAT](https://www.cvat.ai/).

### 5. Application of CSDE
The core analysis involves correcting differential expression estimates using CSDE. The following script runs the experiment for T-cell subsets.

```bash
python /data1/proseg_pipeline/t_cell_subset_experiment.py \
    --adata-path /data1/datasets/merfish_pancancer/lung2_adata.annotated.h5ad \
    --annotations-path /data1/datasets/merfish_pancancer/HumanLungCancerPatient2/manual_annotations/annotations.json \
    --output-path /data1/proseg_pipeline/results/lung2_experiment_${subset}.csv \
    --subset cd8
```
