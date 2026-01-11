# Workflow commands

```
python colocalization_experiment.py \
--adata-path "/data1/spatial-correction/data/merfish_pancancer/HumanLungCancerPatient1/proseg_results/adata_cell_type2.h5ad" \
--cell-type-key "cell_type" \
--cell-type-value "t cell" \
--annotations-path "/data1/spatial-correction/notebooks/analysis/Lung1_proseg_export2/default.json" \
--spatial-group-key "spatial_group" \
--spatial-neighbor-name "tumor" \
--spatial-dist-threshold 20 \
--output-path "/data1/spatial-correction/results/lung1/colocalization_experiment.csv"

python colocalization_experiment.py \
--adata-path "/data1/spatial-correction/data/merfish_pancancer/HumanUterineCancerPatient2-RACostain/proseg_results/cluster_0/adata_final.h5ad" \
--cell-type-key "cell_type" \
--cell-type-value "t cell" \
--annotations-path "/data1/spatial-correction/notebooks/analysis/RACostain_proseg_export1/default.json" \
--spatial-group-key "is_close_to_cancer" \
--spatial-dist-threshold 20 \
--output-path "/data1/spatial-correction/results/RACostain/colocalization_experiment.csv"
```
