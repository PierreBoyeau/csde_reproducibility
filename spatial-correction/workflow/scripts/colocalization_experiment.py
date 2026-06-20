import numpy as np
import pandas as pd
import scanpy as sc
from spatial_correction.baselines import glm_test
from spatial_correction.datanavigation import MerfishDataNavigator
from spatial_correction.ppi import InterceptRegression
import click

from sklearn.neighbors import NearestNeighbors


def compute_dist_to_nn(adata, group_key, group_value):
    """
    Compute the distance to the nearest neighbor for a given group of cells.
    """
    pop_ref = adata.obs[group_key].str.contains(group_value).values.astype(bool)
    ref_adata = adata[pop_ref].copy()
    ref_ = NearestNeighbors(n_neighbors=1)
    ref_.fit(ref_adata.obsm["pos"])
    d_to_nn = ref_.kneighbors(adata.obsm["pos"], return_distance=True)[0].flatten()
    return ref_, d_to_nn


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
    default="spatial_group",
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
    output_path: str,
):
    adata = sc.read_h5ad(adata_path)

    if spatial_group_key not in adata.obs.columns:
        adata.obsm["pos"] = adata.obs[["centroid_x", "centroid_y"]].values
        _, d_to_nn = compute_dist_to_nn(
            adata, group_key=cell_type_key, group_value=spatial_neighbor_name
        )
        adata.obs["d_to_nn"] = d_to_nn
        adata.obs[spatial_group_key] = (d_to_nn < spatial_dist_threshold).astype(int)

    gp1 = lambda x: x[cell_type_key].str.contains(cell_type_value) & (
        x[spatial_group_key] == 1
    )
    gp2 = lambda x: x[cell_type_key].str.contains(cell_type_value) & (
        x[spatial_group_key] == 0
    )
    gp3 = lambda x: (~x[cell_type_key].astype(str).str.contains(cell_type_value))

    # gp1_all = adata.obs.loc[gp1]
    # gp2_all = adata.obs.loc[gp2]
    # gp3_all = adata.obs.loc[gp3].sample(n=10000)

    gp1_all = adata.obs.loc[gp1]
    gp2_all = adata.obs.loc[gp2]
    gp3_all = adata.obs.loc[gp3]
    min_pop_size = min(gp1_all.shape[0], gp2_all.shape[0], gp3_all.shape[0])
    gp1_all = gp1_all.sample(n=min_pop_size, replace=False)
    gp2_all = gp2_all.sample(n=min_pop_size, replace=False)
    gp3_all = gp3_all.sample(n=min_pop_size, replace=False)

    all_idx = np.concatenate(
        [gp1_all.index.values, gp2_all.index.values, gp3_all.index.values]
    )

    annotations = MerfishDataNavigator.static_load_annotations(
        path_to_annotations=annotations_path
    )
    annotated_cells = annotations.index.values
    annotations_ = annotations.loc[annotated_cells, "annotation_name"].values
    adata.obs.loc[annotated_cells, "hand_annot"] = annotations_

    adata_gt = adata[annotated_cells].copy()
    adata_right_cells = adata_gt[
        adata_gt.obs["hand_annot"].str.contains(cell_type_value)
    ].copy()
    adata_right_cells.X = adata_right_cells.layers["counts"].astype(int).astype(float)
    n_cells_expressing = (adata_right_cells.X >= 1).sum(0)
    ncells_expressing = pd.DataFrame(
        {
            "n_cells_expressing": n_cells_expressing,
            "gene_name": adata_right_cells.var_names,
        }
    )

    adata_all = adata[all_idx].copy()
    adata_gt = adata[annotated_cells].copy()
    adata_gt_pred = adata[adata_gt.obs.index].copy()
    adata_other = adata_all[~adata_all.obs.index.isin(adata_gt.obs.index)].copy()

    y_gt = extract_group(adata_gt, "hand_annot", "t cell", spatial_group_key)
    y_pred = extract_group(adata_other, "cell_type", "t cell", spatial_group_key)
    y_gt_pred = extract_group(adata_gt_pred, "cell_type", "t cell", spatial_group_key)

    # int and float to avoid non-integer values
    # (proseg returns expected counts as floats)
    x_gt_ = adata_gt.layers["counts"].astype(int).astype(float).copy()
    x_pred_ = adata_other.layers["counts"].astype(int).astype(float).copy()
    x_gt_pred_ = adata_gt_pred.layers["counts"].astype(int).astype(float).copy()

    # # filtering for numerical stability
    is_gene_expressed_by_ct = n_cells_expressing >= 10
    x_gt = x_gt_[:, is_gene_expressed_by_ct]
    x_gt_pred = x_gt_pred_[:, is_gene_expressed_by_ct]
    x_pred = x_pred_[:, is_gene_expressed_by_ct]
    gene_names = adata.var_names[is_gene_expressed_by_ct]

    print(pd.Series(y_gt).value_counts())
    print(pd.Series(y_gt_pred).value_counts())
    print(pd.Series(y_pred).value_counts())

    SHARED_KWARGS = dict(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt_pred, y_gt_pred),
        inputs_unl=(x_pred, y_pred),
        family="poisson",
        optimizer_kwargs=dict(tol=1e-3),
    )

    # classic
    classic_model = InterceptRegression(**SHARED_KWARGS)
    classic_model.fit(lambd_=0.0)
    classic_model.get_asymptotic_distribution()
    res_classic = (
        classic_model.differential_expression_ew(1)
        .assign(gene_name=gene_names)
        .set_index("gene_name")
        .reindex(adata_gt.var_names)
        .fillna(
            {
                "padj": 1,
                "pval": 1,
                "is_significant_005": False,
                "beta": 0,
            }
        )
        .assign(model="classic")
        .reset_index(names="gene_name")
    )

    # ppi
    ppi_model = InterceptRegression(**SHARED_KWARGS)
    ppi_model.fit(lambd_=None)
    ppi_model.get_asymptotic_distribution()
    res_ppi = (
        ppi_model.differential_expression_ew(1)
        .assign(gene_name=gene_names)
        .set_index("gene_name")
        .reindex(adata_gt.var_names)
        .fillna(
            {
                "padj": 1,
                "pval": 1,
                "is_significant_005": False,
                "beta": 0,
            }
        )
        .assign(model="ppi")
        .reset_index(names="gene_name")
    )

    # ppi_ew
    ppi_ew_model = InterceptRegression(lambd_mode="element", **SHARED_KWARGS)
    ppi_ew_model.fit(lambd_=None)
    ppi_ew_model.get_asymptotic_distribution()
    res_ppi_ew = (
        ppi_ew_model.differential_expression_ew(1)
        .assign(gene_name=gene_names)
        .set_index("gene_name")
        .reindex(adata_gt.var_names)
        .fillna(
            {
                "padj": 1,
                "pval": 1,
                "is_significant_005": False,
                "beta": 0,
            }
        )
        .assign(model="ppi_ew")
        .reset_index(names="gene_name")
    )

    # imputation
    res_imput = glm_test(
        adata_right_cells,
        label_key=spatial_group_key,
        family="poisson",
    )
    res_imput = (
        res_imput.drop(columns=["beta0", "lfc", "e0", "e1"])
        .rename(
            columns={
                "beta1": "beta",
            }
        )
        .assign(model="imputation")
    )

    all_res = pd.concat([res_classic, res_imput, res_ppi, res_ppi_ew]).merge(
        ncells_expressing, on="gene_name", how="left"
    )
    all_res.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
