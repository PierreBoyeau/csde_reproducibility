# Workflow commands

Paths below resolve from the `MERFISH_DATA_DIR` environment variable, which
should point to the root directory holding the MERFISH pan-cancer data:

```bash
export MERFISH_DATA_DIR=/path/to/merfish_pancancer
```

```bash
python colocalization_experiment.py \
--adata-path "$MERFISH_DATA_DIR/HumanLungCancerPatient1/proseg_results/adata_cell_type2.h5ad" \
--cell-type-key "cell_type" \
--cell-type-value "t cell" \
--annotations-path "$MERFISH_DATA_DIR/HumanLungCancerPatient1/manual_annotations/annotations.json" \
--spatial-group-key "spatial_group" \
--spatial-neighbor-name "tumor" \
--spatial-dist-threshold 20 \
--output-path "results/lung1/colocalization_experiment.csv"

python colocalization_experiment.py \
--adata-path "$MERFISH_DATA_DIR/HumanUterineCancerPatient2-RACostain/proseg_results/cluster_0/adata_final.h5ad" \
--cell-type-key "cell_type" \
--cell-type-value "t cell" \
--annotations-path "$MERFISH_DATA_DIR/HumanUterineCancerPatient2-RACostain/manual_annotations/annotations.json" \
--spatial-group-key "is_close_to_cancer" \
--spatial-dist-threshold 20 \
--output-path "results/RACostain/colocalization_experiment.csv"
```
