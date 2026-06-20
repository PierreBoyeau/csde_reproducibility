# CSDE: Correcting Spatial DE Analysis

This repository contains the code required to reproduce the results described in the paper "Correcting Spatial Differential Expression Analysis".

## Package description & installation

The pipeline relies on two separate conda environments:

- **`ppi-spatial`** — runs the CSDE package and all preprocessing, analysis,
  and annotation scripts/notebooks. The `spatial-correction` folder contains
  the source code for CSDE utils and core statistical components; it is a
  Python package managed by poetry, but can be installed with pip:

  ```bash
  conda create -n ppi-spatial python=3.10
  conda activate ppi-spatial
  pip install ./spatial-correction
  ```

- **`proseg`** — runs the [ProSeg](https://github.com/dcjones/proseg) cell
  segmentation step. ProSeg is a separate tool; follow its documentation for
  installation:

  ```bash
  conda create -n proseg
  conda activate proseg
  # then install ProSeg following https://github.com/dcjones/proseg
  ```

The shell scripts activate the relevant environment automatically
(`conda activate ppi-spatial` or, for segmentation, `conda activate proseg`).

## Configuration

The shell scripts resolve data paths from the `MERFISH_DATA_DIR` environment
variable, which should point to the root directory where you store the
MERFISH pan-cancer data. Set it once before running anything:

```bash
export MERFISH_DATA_DIR=/path/to/merfish_pancancer
```

The analysis utilities in `workflow/scripts/benchmark_utils.py` read the same
variable. The notebooks load their inputs from `$MERFISH_DATA_DIR` as well; if
you run them interactively, make sure the variable is set in the kernel
environment (or edit the path at the top of the notebook to match your layout).

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

### 5. Import of manual annotations

Once the exported cell images have been reviewed in CVAT, the annotation
results must be brought back into the analysis pipeline. `export_data.ipynb`
writes one image per cell to
`$MERFISH_DATA_DIR/<FullDatasetName>/manual_annotations/images/cell_<idx>.png`,
where `<idx>` is the cell's index in `adata.obs`.

The annotator records a cell-type label for each image and exports a two-column
CSV (`cell_id`, `annotation_name`), where `cell_id` matches the `cell_<idx>.png`
suffix. The script `workflow/scripts/import_annotations.py` converts that CSV
into the `annotations.json` format consumed by the downstream experiments
(`MerfishDataNavigator.static_load_annotations`):

```bash
python workflow/scripts/import_annotations.py \
    --annotations-csv /path/to/annotated.csv \
    --output-path "$MERFISH_DATA_DIR/HumanLungCancerPatient1/manual_annotations/annotations.json"
```

See the module docstring in `import_annotations.py` for the full annotation
workflow and the expected label set.

### 6. Application of CSDE
The core analysis involves correcting differential expression estimates using CSDE. The following script runs the experiment for T-cell subsets.

```bash
python workflow/scripts/t_cell_subset_experiment.py \
    --adata-path "$MERFISH_DATA_DIR/lung2_adata.annotated.h5ad" \
    --annotations-path "$MERFISH_DATA_DIR/HumanLungCancerPatient2/manual_annotations/annotations.json" \
    --output-path results/lung2_experiment_cd8.csv \
    --subset cd8
```

## Revision experiments

Additional experiments conducted during peer review are available on GitHub at https://github.com/PierreBoyeau/csde_reproducibility_revision/tree/v1.0.2 (DOI: 10.5281/zenodo.20767181)
