import pandas as pd
import scanpy as sc
import os
import numpy as np
from tqdm import tqdm
import click


def build_h5ad_from_csv(dir_path):
    counts_path = os.path.join(dir_path, "expected-counts.csv.gz")
    cell_metadata_path = os.path.join(dir_path, "cell-metadata.csv.gz")
    counts = pd.read_csv(counts_path)
    cell_metadata = pd.read_csv(cell_metadata_path)
    adata = sc.AnnData(
        X=counts.values,
        obs=cell_metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    ncounts = adata.X.sum(axis=1)
    adata.obs["ncounts"] = ncounts
    adata.obs["original_index"] = adata.obs.index.copy()
    return adata

def preprocess_data_counts(adata):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.layers["normalized"] = adata.X.copy()
    sc.pp.scale(adata)
    adata.layers["znormalized"] = adata.X.copy()

    adata.X = adata.layers["counts"].copy()
    adata.X.sum(1).min(), adata.X.sum(1).max()

    adata.obsm["pos"] = np.array(adata.obs[["centroid_x", "centroid_y"]])
    sc.pp.neighbors(adata, use_rep="pos", key_added="pos_neighbors", n_neighbors=3)

    # x_ref = adata.layers["normalized"]
    # g_exp_norm = np.zeros_like(adata.X)
    # for i, neigh in enumerate(tqdm(adata.obsp["pos_neighbors_connectivities"])):
    #     is_neighbor = neigh.toarray().flatten() > 0
    #     g_exp_norm[i, :] = x_ref[i, :] - x_ref[is_neighbor, :].mean(0)
    # adata.layers["normalized_neighbors"] = g_exp_norm

    # x_ref = adata.layers["znormalized"].copy()
    # g_exp_norm = np.zeros_like(adata.X)
    # for i, neigh in enumerate(tqdm(adata.obsp["pos_neighbors_connectivities"])):
    #     is_neighbor = neigh.toarray().flatten() > 0
    #     g_exp_norm[i, :] = x_ref[i, :] - x_ref[is_neighbor, :].mean(0)
    # adata.layers["znormalized_neighbors"] = g_exp_norm
    return adata

@click.command()
@click.argument("input_dir1", type=click.Path(exists=True))
@click.argument("output_path")
@click.option("--input_dir2", type=click.Path(exists=True), default=None)
def main(input_dir1, output_path, input_dir2):
    adata1 = build_h5ad_from_csv(input_dir1)
    adata1.obs["batch_id"] = "slice1"
    if input_dir2 is not None:
        adata2 = build_h5ad_from_csv(input_dir2)
        adata2.obs["batch_id"] = "slice2"
        adata = adata1.concatenate(adata2)
    else:
        adata = adata1
    adata = preprocess_data_counts(adata)
    adata = adata[adata.obs["ncounts"] > 1e-5]
    
    sc.tl.pca(adata, n_comps=50, layer="normalized")
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.25, key_added="leiden_0.25", flavor="igraph", n_iterations=2)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden_0.5", flavor="igraph", n_iterations=2)
    sc.tl.leiden(adata, resolution=1, key_added="leiden_1", flavor="igraph", n_iterations=2)
    sc.tl.leiden(adata, resolution=2, key_added="leiden_2", flavor="igraph", n_iterations=2)
    adata.write_h5ad(output_path)

if __name__ == "__main__":
    main()