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
    "--cell-type-key", type=str, help="cell-type key in adata.obs", default="cell_type"
)
@click.option(
    "--cell-type-value",
    type=str,
    help="Name of the cell-type of interest",
)
@click.option(
    "--annotations-path",
    type=click.Path(exists=True),
    help="Path to the annotations file",
)
@click.option(
    "--spatial-group-key",
    type=str,
    help="Key in adata.obs that defines the spatial groups. If not provided, the groups are computed based on the distance to the nearest neighbor",
    default=None,
)
@click.option(
    "--spatial-neighbor-name",
    type=str,
    help="Name of the neighbor to compute the distance to the nearest neighbor",
    default="tumor",
)
@click.option(
    "--spatial-dist-threshold",
    type=float,
    help="Distance threshold to consider a cell as close to a neighbor",
    default=20,
)
@click.option(
    "--n",
    type=int,
    help="Number of manual annotations to use",
    default=None,
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed",
    default=0,
)
@click.option(
    "--output-path",
    type=click.Path(),
    help="Path to save the output",
)
def main(
    adata_path: str,
    cell_type_key: str,
    cell_type_value: str,
    annotations_path: str,
    spatial_group_key: str,
    spatial_neighbor_name: str,
    spatial_dist_threshold: float,
    n: int,
    random_seed: int,
    output_path: str,
):
    """
    Run the colocalization experiment.

    Parameters
    ----------
    adata_path: str
        Path to the adata file.
    cell_type_key: str
        Key in adata.obs that defines (predicted) cell-types.
    cell_type_value: str
        Name of the cell-type of interest.
    annotations_path
        Path to the json with manual annotations.
    spatial_group_key: str
        Key in adata.obs that defines the spatial groups (0/1), e.g., inside/outside the tumor.
    spatial_neighbor_name: str
        Name of the secondary cell-type of interest to define the spatial groups.
    spatial_dist_threshold: float
        Distance threshold to consider a cell as close to a neighbor.
    n: int
        Number of manual annotations to use.
    random_seed: int
        Random seed.
    output_path: str
        Path to save the output.
    """
    adata = sc.read_h5ad(adata_path)
    if spatial_group_key not in adata.obs.columns:
        adata.obsm["pos"] = adata.obs[["centroid_x", "centroid_y"]].values
        _, d_to_nn = compute_dist_to_nn(
            adata, group_key=cell_type_key, group_value=spatial_neighbor_name
        )
        adata.obs["d_to_nn"] = d_to_nn
        adata.obs[spatial_group_key] = (d_to_nn < spatial_dist_threshold).astype(int)
    adata_all = adata.copy()

    # loading annotations
    annotations = MerfishDataNavigator.static_load_annotations(
        path_to_annotations=annotations_path
    )
    if n is not None:
        annotations = annotations.sample(n=n, replace=False, random_state=random_seed)
    annotated_cells = annotations.index.values
    annotations_ = annotations.loc[annotated_cells, "annotation_name"].values
    adata.obs.loc[annotated_cells, "hand_annot"] = annotations_

    # prepare data
    adata_gt = adata[annotated_cells].copy()

    adata_right_cells = adata_gt[
        adata_gt.obs["hand_annot"].str.contains(cell_type_value)
    ].copy()
    adata_right_cells.X = adata_right_cells.layers["counts"].astype(int).astype(float)
    n_cells_expressing = (adata_right_cells.X >= 1).sum(0)

    adata_right_cells_imputed = adata[
        adata.obs[cell_type_key] == cell_type_value
    ].copy()
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
    adata_gt = adata[annotated_cells].copy()
    adata_gt_pred = adata[adata_gt.obs.index].copy()
    adata_other = adata_all[~adata_all.obs.index.isin(adata_gt.obs.index)].copy()

    y_gt = extract_group(adata_gt, "hand_annot", cell_type_value, spatial_group_key)
    y_gt_pred = extract_group(
        adata_gt_pred, "cell_type", cell_type_value, spatial_group_key
    )
    y_pred = extract_group(adata_other, "cell_type", cell_type_value, spatial_group_key)
    importance_weights = 1.0 / adata_gt.obs["sampling_weight"].values
    normalized_weights = (
        float(adata_gt.shape[0]) * importance_weights / importance_weights.sum()
    )
    # int and float to avoid non-integer values
    x_gt_ = adata_gt.layers["counts"].astype(int).astype(float).copy()
    x_gt_pred_ = adata_gt_pred.layers["counts"].astype(int).astype(float).copy()
    x_pred_ = adata_other.layers["counts"].astype(int).astype(float).copy()

    # # filtering for numerical stability
    is_gene_expressed_by_ct = n_cells_expressing >= 10
    x_gt = x_gt_[:, is_gene_expressed_by_ct]
    x_gt_pred = x_gt_pred_[:, is_gene_expressed_by_ct]
    x_pred = x_pred_[:, is_gene_expressed_by_ct]
    gene_names = adata.var_names[is_gene_expressed_by_ct]

    SHARED_KWARGS = dict(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt_pred, y_gt_pred),
        inputs_unl=(x_pred, y_pred),
        importance_weights=normalized_weights,
        family="poisson",
        optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
    )

    all_res, losses = run_benchmark(
        adata_gt,
        adata_right_cells_imputed,
        spatial_group_key=spatial_group_key,
        gene_names=gene_names,
        shared_kwargs=SHARED_KWARGS,
        full_run=True,
    )

    losses.to_csv(output_path.replace(".csv", "_losses.csv"), index=False)

    n_focus_cells_gt = adata_right_cells.shape[0]
    n_focus_cells_pred = adata_right_cells_imputed.shape[0]
    all_res = all_res.merge(ncells_expressing, on="gene_name", how="left").assign(
        n_focus_cells_gt=n_focus_cells_gt,
        n_focus_cells_pred=n_focus_cells_pred,
        n=n,
        random_seed=random_seed,
    )
    all_res.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
