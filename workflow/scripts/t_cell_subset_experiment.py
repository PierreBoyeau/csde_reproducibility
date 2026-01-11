import numpy as np
import pandas as pd
import scanpy as sc
from spatial_correction.datanavigation import MerfishDataNavigator
from benchmark_utils import compute_dist_to_nn, run_benchmark
import click


def extract_group(adata, ct_key, ct_value, spatial_group_key):
    is_right_ct = (adata.obs[ct_key] == ct_value).values
    is_infiltrating = (adata.obs[spatial_group_key] == 1).values

    is_infiltrating_rightct = is_right_ct & is_infiltrating
    is_other_rightct = is_right_ct & ~is_infiltrating
    is_other_non_rightct = ~is_right_ct

    y_ = np.zeros(adata.shape[0], dtype=int)
    y_[is_other_rightct] = 0
    y_[is_infiltrating_rightct] = 1
    y_[is_other_non_rightct] = 2
    return y_


@click.command()
@click.option(
    "--adata-path", type=click.Path(exists=True), help="Path to the adata file"
)
@click.option(
    "--annotations-path",
    type=click.Path(exists=True),
    help="Path to the annotations file",
)
@click.option(
    "--output-path",
    type=click.Path(),
    help="Path to save the output",
)
@click.option("--subset", type=str, help="subset to use (pos or neg)", default="pos")
def main(
    adata_path: str,
    annotations_path: str,
    output_path: str,
    subset: str,
):
    """
    Run the colocalization experiment.

    Parameters
    ----------
    adata_path: str
        Path to the adata file.
    annotations_path
        Path to the json with manual annotations.
    output_path: str
        Path to save the output.
    """
    cell_type_key = "cell_type"
    cell_type_value = "t cell"
    spatial_group_key = "spatial_group_str"
    spatial_neighbor_name = "cancer"
    spatial_dist_threshold = 20

    adata = sc.read_h5ad(adata_path)
    adata.obsm["pos"] = adata.obs[["centroid_x", "centroid_y"]].values
    # breakpoint()
    _, d_to_nn = compute_dist_to_nn(
        adata, group_key=cell_type_key, group_value=spatial_neighbor_name
    )
    adata.obs["d_to_nn"] = d_to_nn

    # breakpoint()
    adata.obs[spatial_group_key] = (d_to_nn < spatial_dist_threshold).astype(int)

    # loading annotations
    annotations = MerfishDataNavigator.static_load_annotations(
        path_to_annotations=annotations_path
    )

    annotated_cells = annotations.index.values
    annotations_ = annotations.loc[annotated_cells, "annotation_name"].values
    adata.obs.loc[annotated_cells, "hand_annot"] = annotations_

    adata_gt = adata[annotated_cells].copy()

    adata_t_cells = adata[adata.obs["cell_type"] == "t cell"]
    adata_t_cells.X = adata_t_cells.layers["znormalized"].copy()
    sc.pp.pca(adata_t_cells, n_comps=50)
    adata_t_cells.obs["PC1"] = adata_t_cells.obsm["X_pca"][:, 0]
    adata.obs.loc[adata_t_cells.obs.index, "PC1"] = adata_t_cells.obs["PC1"].values

    adata.obs.loc[:, "spatial_group2_pred"] = 2
    adata.obs.loc[:, "spatial_group2"] = 2

    adata.obs.loc[:, "CD8A"] = adata[:, "CD8A"].layers["counts"].toarray().flatten()
    adata.obs.loc[:, "CD4"] = adata[:, "CD4"].layers["counts"].toarray().flatten()
    adata_gt.obs.loc[:, "CD8A"] = adata_gt[:, "CD8A"].layers["counts"].toarray().flatten()
    adata_gt.obs.loc[:, "CD4"] = adata_gt[:, "CD4"].layers["counts"].toarray().flatten()

    if subset == "pos":
        is_relevant_t_cell = lambda x: (x["PC1"] > 0.0)
    elif subset == "neg":
        is_relevant_t_cell = lambda x: (x["PC1"] <= 0.0)
    elif subset == "cd8":
        is_relevant_t_cell = lambda x: (x["CD8A"] >= 0.5)
    elif subset == "cd4":
        is_relevant_t_cell = lambda x: (x["CD4"] >= 0.5)
    elif subset == "noncd8":
        is_relevant_t_cell = lambda x: (x["CD8A"] <= 0.5)
    elif subset == "cd4noncd8":
        is_relevant_t_cell = lambda x: (x["CD4"] >= 0.5) & (x["CD8A"] <= 0.5)
    elif subset == "cd8noncd4":
        is_relevant_t_cell = lambda x: (x["CD8A"] >= 0.5) & (x["CD4"] <= 0.5)
    else:
        raise ValueError(f"Invalid subset: {subset}")

    is_relevant_t_cell_inside_pred = (
        lambda x: (is_relevant_t_cell(x))
        & (x["spatial_group_str"] == 1)
        & (x["cell_type"] == "t cell")
    )
    is_relevant_t_cell_inside = (
        lambda x: (is_relevant_t_cell(x))
        & (x["spatial_group_str"] == 1)
        & (x["hand_annot"] == "t cell")
    )
    is_t_cell_outside_pred = lambda x: (x["spatial_group_str"] == 0) & (
        x["cell_type"] == "t cell"
    )
    is_t_cell_outside = lambda x: (x["spatial_group_str"] == 0) & (
        x["hand_annot"] == "t cell"
    )

    adata.obs.loc[is_relevant_t_cell_inside_pred, "spatial_group2_pred"] = 1
    adata.obs.loc[is_t_cell_outside_pred, "spatial_group2_pred"] = 0
    adata.obs.loc[is_relevant_t_cell_inside, "spatial_group2"] = 1
    adata.obs.loc[is_t_cell_outside, "spatial_group2"] = 0

    # adata_right_cells = adata_gt[
    #     adata_gt.obs["hand_annot"].str.contains(cell_type_value)
    # ].copy()
    adata_right_cells = adata_gt[
        is_relevant_t_cell_inside_pred(adata_gt.obs) | is_t_cell_outside_pred(adata_gt.obs)
    ]
    adata_right_cells.X = adata_right_cells.layers["counts"].astype(int).astype(float)
    n_cells_expressing = (adata_right_cells.X >= 1).sum(0)
    
    # adata_right_cells_imputed = adata[
    #     adata.obs[cell_type_key] == cell_type_value
    # ].copy()
    adata_right_cells_imputed = adata[
        is_relevant_t_cell_inside_pred(adata.obs) | is_t_cell_outside_pred(adata.obs)
    ]
    adata_right_cells_imputed.X = (
        adata_right_cells_imputed.layers["counts"].astype(int).astype(float)
    )
    n_cells_expressing_imputed = (adata_right_cells_imputed.X >= 1).sum(0)
    ncells_expressing = pd.DataFrame(
        {
            "n_cells_expressing": n_cells_expressing,
            "n_cells_expressing_imputed": n_cells_expressing_imputed,
            "gene_name": adata_right_cells.var_names,
        }
    )

    is_gene_expressed_by_ct = n_cells_expressing >= 10
    gene_names = adata.var_names[is_gene_expressed_by_ct]

    adata_gt = adata[annotated_cells].copy()
    adata_gt_pred = adata[adata_gt.obs.index].copy()
    adata_other = adata[~adata.obs.index.isin(adata_gt.obs.index)].copy()

    y_gt = adata_gt.obs["spatial_group2"].values
    y_gt_pred = adata_gt.obs["spatial_group2_pred"].values
    x_gt_ = adata_gt.layers["counts"]
    x_gt_pred_ = adata_gt.layers["counts"]

    x_pred_ = adata_other.layers["counts"]
    y_pred = adata_other.obs["spatial_group2_pred"].values

    x_gt = x_gt_[:, is_gene_expressed_by_ct]
    x_gt_pred = x_gt_pred_[:, is_gene_expressed_by_ct]
    x_pred = x_pred_[:, is_gene_expressed_by_ct]

    importance_weights = 1.0 / adata_gt.obs["sampling_weight"].values
    normalized_weights = (
        float(adata_gt.shape[0]) * importance_weights / importance_weights.sum()
    )
    SHARED_KWARGS = dict(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt_pred, y_gt_pred),
        inputs_unl=(x_pred, y_pred),
        importance_weights=normalized_weights,
        family="poisson",
        optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
    )

    res, _ = run_benchmark(
        adata_gt,
        adata_right_cells_imputed,
        spatial_group_key,
        gene_names,
        SHARED_KWARGS,
        full_run=True,
    )
    res = res.merge(ncells_expressing, on="gene_name", how="left")
    res.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
