import os

import pandas as pd

from sklearn.neighbors import NearestNeighbors
from spatial_correction.ppi import InterceptRegression
from spatial_correction.baselines import glm_test
import plotnine as gg
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
import scanpy as sc
import scipy.stats as stats

from matplotlib.colors import ListedColormap
from matplotlib.colors import rgb2hex
import numpy as np

DEFAULT_LEGEND = gg.theme_classic() + gg.theme(
    text=gg.element_text(family="Helvetica Neue"),
    axis_text=gg.element_text(size=6),
    axis_title=gg.element_text(size=6.5),
    axis_line=gg.element_line(linewidth=0.35),
    legend_position="none",
    axis_ticks_major=gg.element_line(linewidth=0.25),
)

DEFAULT_SIZE = gg.theme(figure_size=(1.21, 1.08))
MODEL_COLORS = {
    "autom.": "#FDBF64",
    "CSDE": "#7EB8FF",
    "manual": "#90D989",
}


def import_adata(dataset_name, spatial_neighbor_name, data_dir=None):
    if data_dir is None:
        data_dir = os.environ["MERFISH_DATA_DIR"]
    path_to_adata = f"{data_dir}/{dataset_name}_adata.annotated.h5ad"
    cell_type_key = "cell_type"
    spatial_dist_threshold = 20

    adata = sc.read_h5ad(path_to_adata)
    adata.obsm["pos"] = adata.obs[["centroid_x", "centroid_y"]].values

    _, d_to_nn = compute_dist_to_nn(
        adata, group_key=cell_type_key, group_value=spatial_neighbor_name
    )
    adata.obs["d_to_nn"] = d_to_nn
    adata.obs["spatial_group_str"] = (d_to_nn < spatial_dist_threshold).astype(str)
    return adata


def run_benchmark(
    adata_gt,
    adata_right_cells_imputed,
    spatial_group_key,
    gene_names,
    shared_kwargs,
    full_run=True,
):
    all_res = []
    all_losses = []

    classic_model = InterceptRegression(**shared_kwargs)
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
    all_res.append(res_classic)
    all_losses.append(classic_model.losses.assign(model="classic"))

    # ppi_ew
    ppi_ew_model = InterceptRegression(lambd_mode="element", **shared_kwargs)
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
    all_res.append(res_ppi_ew)
    all_losses.append(ppi_ew_model.losses.assign(model="ppi_ew"))
    if full_run:
        # ppi
        ppi_model = InterceptRegression(**shared_kwargs)
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
        all_res.append(res_ppi)
        all_losses.append(ppi_model.losses.assign(model="ppi"))

    all_res = pd.concat(all_res)
    losses = pd.concat(all_losses)
    return all_res, losses


def compute_neighbor_proportions(adata, group_key, k):
    """
    Compute the proportion of different cell types among the k nearest neighbors for each cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Requires 'pos' in obsm.
    group_key : str
        Column name in adata.obs containing cell type annotations.
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    pd.DataFrame
        DataFrame (cells x cell_types) with proportions of neighbor cell types.
    """
    positions = adata.obsm["pos"]
    cell_types = adata.obs[group_key]
    all_cell_types = cell_types.unique()

    # Fit KNN
    nn = NearestNeighbors(
        n_neighbors=k + 1
    )  # +1 because the cell itself is the closest neighbor
    nn.fit(positions)
    distances, indices = nn.kneighbors(positions)

    # Exclude self from neighbors
    neighbor_indices = indices[:, 1:]

    # Get neighbor cell types
    neighbor_cell_types = cell_types.iloc[neighbor_indices.flatten()].values.reshape(
        neighbor_indices.shape
    )

    # Calculate proportions efficiently
    neighbor_df = pd.DataFrame(neighbor_cell_types, index=adata.obs_names)
    # Use apply with value_counts for proportions directly
    proportions = neighbor_df.apply(
        lambda x: x.value_counts(normalize=True), axis=1
    ).fillna(0)

    # Ensure all cell types are present as columns and the index matches adata
    proportions_df = proportions.reindex(columns=all_cell_types, fill_value=0.0)
    proportions_df = proportions_df.reindex(adata.obs_names, fill_value=0.0)
    adata.obsm["spatial_neighbor_proportions"] = proportions_df
    return proportions_df


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


def plot_ess(all_results, return_df=False):
    pivoted = all_results.pivot(
        index=["gene_name", "dataset", "cell_type"], columns="model", values=["cov"]
    )
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]

    ess = (600 * pivoted["cov_classic"] / pivoted["cov_ppi_ew"]).dropna().values
    print()
    print("ESS (PPI++ vs classic): ", np.mean(ess))
    print()


def plot_spatial_group(dataset_name, spatial_neighbor_name):
    renamer = {
        "epithelial": "cancer",
        "myeloid": "other",
        "immune cells": "other",
    }
    element_to_hex = {
        "immune cells": "#fbb4ae",
        "macrophage": "#b3cde3",
        "t cell": "#ccebc5",
        "b cell": "#decbe4",
        "other": "#fed9a6",
        "fibroblast": "#ffffcc",
        "dendritic cell": "#e5d8bd",
        "cancer": "#fddaec",
        "endothelial": "#f2f2f2",
    }
    unique_elements = list(element_to_hex.keys())
    pastel1 = plt.cm.tab10
    colors = pastel1(np.linspace(0, 1, len(unique_elements)))
    element_to_color = {element: colors[i] for i, element in enumerate(unique_elements)}
    element_to_hex = {key: rgb2hex(value) for key, value in element_to_color.items()}

    adata = import_adata(dataset_name, spatial_neighbor_name)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str).replace(renamer)

    cmap = {
        "False": "#BFC0D3",
        "True": "#5A5B69",
    }

    fig = (
        gg.ggplot(
            adata.obs, gg.aes(x="centroid_x", y="centroid_y", color="spatial_group_str")
        )
        + gg.geom_point(stroke=0.0, size=0.25)
        + gg.geom_point(
            gg.aes(x="centroid_x", y="centroid_y"),
            data=adata.obs.query("cell_type == 't cell'"),
            size=0.4,
            stroke=0.0,
            alpha=0.8,
            inherit_aes=False,
            color="red",
        )
        + gg.theme_void()
        + gg.scale_color_manual(values=cmap)
        + gg.coord_cartesian()
        + gg.theme(
            legend_position="none",
            dpi=1500,
        )
    )
    fig.save(f"figures/{dataset_name}_spatial_group.png", dpi=1500)

    fig = (
        gg.ggplot(adata.obs, gg.aes(x="centroid_x", y="centroid_y", color="cell_type"))
        + gg.geom_point(stroke=0.0, size=0.25)
        + gg.theme_void()
        + gg.scale_color_manual(element_to_hex)
        + gg.coord_cartesian()
        + gg.theme(
            legend_position="none",
            dpi=1500,
        )
    )
    fig.save(f"figures/{dataset_name}_cell_type.png", dpi=500)

    fig = (
        gg.ggplot(
            adata.obs.sample(10000),
            gg.aes(x="centroid_x", y="centroid_y", color="cell_type"),
        )
        + gg.geom_point(stroke=0.0, size=0.25)
        + gg.theme_void()
        + gg.scale_color_manual(element_to_hex)
        + gg.coord_cartesian()
        + gg.theme(
            # legend_position="none",
            dpi=1500,
        )
        + gg.guides(color=gg.guide_legend(override_aes={"size": 5}))
    )
    fig.save(f"figures/{dataset_name}_cell_type_legend.svg")


class BenchmarkPlotter:
    def __init__(self, model_colors=None, legend=None, size=None):
        self.model_colors = model_colors if model_colors is not None else MODEL_COLORS
        self.legend = legend if legend is not None else DEFAULT_LEGEND
        self.size = size if size is not None else DEFAULT_SIZE

    def get_n_discoveries(self, all_results, return_df=False):
        plot_df = (
            all_results.groupby("name_display")["is_de"]
            .sum()
            .to_frame("# of discoveries")
            .astype(int)
            .reset_index()
        )
        if return_df:
            return plot_df
        else:
            fig = (
                gg.ggplot(
                    plot_df,
                    gg.aes(x="name_display", y="# of discoveries", fill="name_display"),
                )
                + gg.geom_col(width=0.5)
                + self.legend
                + gg.theme(figure_size=(0.79, 1.08))
                + gg.scale_y_continuous(expand=(0, 0))
                + gg.scale_fill_manual(values=self.model_colors)
                + gg.labs(
                    x="",
                )
            )
            return fig

    def plot_ref_expr_discoveries(self, all_results, return_df=False, log_scale=False):
        discoveries = all_results.query("is_de").dropna(subset=["mean_exp"])
        if log_scale:
            discoveries["mean_exp"] = discoveries["mean_exp"] + 1e-4

        discoveries_ppipp = discoveries.query("name_display == 'CSDE'")
        discoveries_classic = discoveries.query("name_display == 'manual'")
        discoveries_imput = discoveries.query("name_display == 'autom.'")

        print("Ref expression of discoveries across models")
        print("PPI++ vs imput")
        print(
            stats.ttest_ind(
                discoveries_ppipp["mean_exp"],
                discoveries_imput["mean_exp"],
                alternative="greater",
            )
        )
        print("PPI++ vs classic")
        print(
            stats.ttest_ind(
                discoveries_ppipp["mean_exp"],
                discoveries_classic["mean_exp"],
                alternative="greater",
            )
        )
        print("imput vs classic")
        print(
            stats.ttest_ind(
                discoveries_classic["mean_exp"],
                discoveries_imput["mean_exp"],
                alternative="greater",
            )
        )
        print()

        if return_df:
            return discoveries
        else:
            fig = (
                gg.ggplot(discoveries, gg.aes(x="name_display", y="mean_exp"))
                + gg.geom_boxplot(gg.aes(fill="name_display"), outlier_alpha=0)
                # + gg.geom_violin(gg.aes(fill="name_display"))
                + gg.geom_jitter(width=0.2, alpha=0.5, size=0.1)
                + self.legend
                + self.size
                + gg.scale_fill_manual(values=self.model_colors)
                + gg.labs(
                    x="",
                    y="ref expr.",
                )
            )
            if log_scale:
                fig = fig + gg.scale_y_log10()
            return fig

    def plot_ref_lfc_discoveries(self, all_results, return_df=False):
        discoveries = all_results.query("is_de").dropna(subset=["mean_exp"])

        discoveries_ppipp = discoveries.query("name_display == 'CSDE'")
        discoveries_classic = discoveries.query("name_display == 'manual'")
        discoveries_imput = discoveries.query("name_display == 'autom.'")

        print("Ref LFC of discoveries across models")
        print("PPI++ vs imput")
        print(
            stats.ttest_ind(
                discoveries_ppipp["lfc"],
                discoveries_imput["lfc"],
                alternative="greater",
            )
        )
        print("PPI++ vs classic")
        print(
            stats.ttest_ind(
                discoveries_ppipp["lfc"],
                discoveries_classic["lfc"],
                alternative="greater",
            )
        )
        print("imput vs classic")
        print(
            stats.ttest_ind(
                discoveries_classic["lfc"],
                discoveries_imput["lfc"],
                alternative="greater",
            )
        )
        print()

        if return_df:
            return discoveries
        else:
            fig = (
                gg.ggplot(discoveries, gg.aes(x="name_display", y="lfc"))
                + gg.geom_boxplot(gg.aes(fill="name_display"), outlier_alpha=0)
                + gg.geom_jitter(width=0.2, alpha=0.5, size=0.1)
                + self.legend
                + self.size
                + gg.scale_fill_manual(values=self.model_colors)
                + gg.labs(
                    x="",
                    y="LFC (T cell vs all, scRNA-seq)",
                )
            )
            return fig

    def plot_ref_pct_cells_discoveries(
        self, all_results, return_df=False, plot_log=False
    ):
        discoveries = all_results.query("is_de").dropna(subset=["mean_exp"])
        if plot_log:
            discoveries["pct_exp_log"] = np.log10(1e-4 + discoveries["pct_exp"])
            koi = "pct_exp_log"
        else:
            koi = "pct_exp"

        discoveries_ppipp = discoveries.query("name_display == 'CSDE'")
        discoveries_classic = discoveries.query("name_display == 'manual'")
        discoveries_imput = discoveries.query("name_display == 'autom.'")

        print(f"Ref {koi} of discoveries across models")
        print("PPI++ vs imput")
        print(
            stats.ttest_ind(
                discoveries_ppipp[koi],
                discoveries_imput[koi],
                alternative="greater",
            )
        )
        print("PPI++ vs classic")
        print(
            stats.ttest_ind(
                discoveries_ppipp[koi],
                discoveries_classic[koi],
                alternative="greater",
            )
        )
        print("imput vs classic")
        print(
            stats.ttest_ind(
                discoveries_classic[koi],
                discoveries_imput[koi],
                alternative="greater",
            )
        )
        print()

        if return_df:
            return discoveries
        else:
            fig = (
                gg.ggplot(discoveries, gg.aes(x="name_display", y=koi))
                + gg.geom_boxplot(gg.aes(fill="name_display"), outlier_alpha=0)
                + gg.geom_jitter(width=0.2, alpha=0.5, size=0.1)
                + self.legend
                + self.size
                + gg.scale_fill_manual(values=self.model_colors)
                + gg.labs(
                    x="",
                    y="% cells expressing (T cells, scRNA-seq)",
                )
            )
            return fig

    def plot_biologial_relevance(
        self, all_results_1, all_results_2, return_df=False, ref_key="lfc"
    ):
        # biological relevance
        corrs_ref = (
            all_results_1.dropna(subset=[ref_key])
            .groupby("name_display")
            .apply(lambda x: stats.spearmanr(np.abs(x["beta"]), x[ref_key])[0])
            .to_frame("bio. consistency score")
            .reset_index()
        )
        # reproducibility
        merged_df = all_results_1.merge(
            all_results_2, on=["gene_name", "name_display"], suffixes=("_1", "_2")
        )
        corrs_rep = (
            merged_df.groupby("name_display")
            # .apply(lambda x: stats.pearsonr(x["beta_1"], x["beta_2"])[0])
            .apply(lambda x: stats.spearmanr(np.abs(x["beta_1"]), np.abs(x["beta_2"]))[0])
            .to_frame("reproducibility score")
            .reset_index()
        )

        corrs_ = corrs_ref.merge(corrs_rep, on="name_display")
        if return_df:
            return corrs_
        else:
            fig = (
                gg.ggplot(
                    corrs_,
                    gg.aes(
                        x="bio. consistency score",
                        y="reproducibility score",
                        fill="name_display",
                    ),
                )
                + gg.geom_point(stroke=0.0, size=2.5)
                + gg.geom_text(gg.aes(label="name_display"), size=6)
                + gg.geom_hline(yintercept=0, color="black", size=0.25)
                + gg.geom_vline(xintercept=0, color="black", size=0.25)
                + self.legend
                + self.size
                + gg.scale_fill_manual(values=self.model_colors)
                + gg.labs(
                    x="",
                )
            )
            return fig

    def discovery_comparison(self, all_results, return_df=False):
        valid_discoveries = (
            all_results.query("is_de")
            .dropna(subset=["mean_exp"])
            .loc[lambda x: x["name_display"].isin(["CSDE", "manual"])]
        )
        geneset_ppi = set(
            valid_discoveries.query("name_display == 'CSDE'")["gene_name"].unique()
        )
        geneset_classic = set(
            valid_discoveries.query("name_display == 'manual'")["gene_name"].unique()
        )

        inter_set = geneset_ppi & geneset_classic
        union_set = geneset_ppi | geneset_classic
        ppi_only = geneset_ppi - geneset_classic
        classic_only = geneset_classic - geneset_ppi

        print("PPI++ only: ", len(ppi_only))
        print(", ".join(list(ppi_only)))
        print("classic only: ", len(classic_only))
        print(", ".join(list(classic_only)))
        print("inter_set: ", len(inter_set))
        print(", ".join(list(inter_set)))

        genes_to_plot = list(ppi_only | inter_set)
        plot_df = valid_discoveries.query(
            "gene_name in @genes_to_plot"
        ).drop_duplicates(subset=["gene_name"])
        plot_df["gene_color"] = "shared"
        plot_df.loc[plot_df["gene_name"].isin(ppi_only).values, "gene_color"] = (
            "PPI++ only"
        )
        plot_df.loc[plot_df["gene_name"].isin(classic_only).values, "gene_color"] = (
            "classic only"
        )
        plot_df["beta_clipped"] = plot_df["beta"]

        cat_to_color = {
            "shared": "#000000",
            "PPI++ only": "#7EB8FF",
            "classic only": "#90D989",
        }

        y_key = "lfc"
        fig = (
            gg.ggplot(plot_df, gg.aes(x="beta_clipped", y=y_key, fill="gene_color"))
            + gg.geom_text(gg.aes(label="gene_name", color="gene_color"), size=6)
            + gg.scale_fill_manual(values=cat_to_color)
            + gg.scale_color_manual(values=cat_to_color)
            + gg.labs(
                x="LFC (spatial)",
                y="ref expr.",
            )
            + self.legend
            + gg.theme(figure_size=(3.0, 1.5))
        )
        return fig

    def plot_all_figures(self, all_results_rep1, all_results_rep2=None):
        fig_ndiscoveries = self.get_n_discoveries(all_results_rep1)
        fig_refexpr = self.plot_ref_expr_discoveries(all_results_rep1)
        fig_refexpr_log = self.plot_ref_expr_discoveries(
            all_results_rep1, log_scale=True
        )

        fig_reflfc = self.plot_ref_lfc_discoveries(all_results_rep1)
        plot_ess(all_results_rep1)
        fig_refpct = self.plot_ref_pct_cells_discoveries(all_results_rep1)
        fig_refpct_log = self.plot_ref_pct_cells_discoveries(
            all_results_rep1, plot_log=True
        )
        fig_biological_relevance_lfc = self.plot_biologial_relevance(
            all_results_rep1, all_results_rep2, ref_key="lfc"
        )
        fig_biological_relevance_meanexp = self.plot_biologial_relevance(
            all_results_rep1, all_results_rep2, ref_key="mean_exp"
        )

        fig_comp = self.discovery_comparison(all_results_rep1)
        fig_venn = self.plot_venn(all_results_rep1)

        display(
            fig_ndiscoveries,
            fig_refexpr,
            fig_refexpr_log,
            fig_reflfc,
            fig_refpct,
            fig_refpct_log,
            fig_biological_relevance_lfc,
            fig_biological_relevance_meanexp,
            fig_comp,
            fig_venn,
        )
        return {
            "ndiscoveries": fig_ndiscoveries,
            "refexpr": fig_refexpr,
            "refexpr_log": fig_refexpr_log,
            "reflfc": fig_reflfc,
            "refpct": fig_refpct,
            "refpct_log": fig_refpct_log,
            "biological_relevance_lfc": fig_biological_relevance_lfc,
            "biological_relevance_meanexp": fig_biological_relevance_meanexp,
            "compwithclassic": fig_comp,
            "venn": fig_venn,
        }

    def plot_venn(self, res):
        genes_imput = (
            res.query("model == 'imputation'").query("is_de")["gene_name"].unique()
        )
        genes_classic = (
            res.query("model == 'classic'").query("is_de")["gene_name"].unique()
        )
        genes_ppi = res.query("model == 'ppi_ew'").query("is_de")["gene_name"].unique()

        set_imput = set(genes_imput)
        set_classic = set(genes_classic)
        set_ppi = set(genes_ppi)

        fig = venn3([set_imput, set_classic, set_ppi], ("imput.", "classic", "PPI++"))
        return fig

    def get_imputation_only(self, res):
        genes_in_ppi = (
            res.query("model == 'ppi_ew'").query("is_de")["gene_name"].unique()
        )
        genes_in_imputation = (
            res.query("model == 'imputation'")
            .query("is_de")
            .loc[lambda x: ~x["gene_name"].isin(genes_in_ppi)]["gene_name"]
        )
        print("genes in imputation only")
        print(" ".join(genes_in_imputation))
        print()

    def get_updown(self, res):
        res_ppi = res.query("model == 'ppi_ew'").query("is_de")
        res_classic = (
            res.query("model == 'classic'").query("is_de")["gene_name"].unique()
        )
        res_imputation = res.query("model == 'imputation'").query("is_de")
        res_inter = res_ppi.loc[lambda x: x["gene_name"].isin(res_classic)]
        res_ppi_only = res_ppi.loc[lambda x: ~x["gene_name"].isin(res_classic)]
        res_imputation_only = res_imputation.loc[
            lambda x: ~x["gene_name"].isin(res_ppi)
        ]
        # inter up
        print("Genes upregulated in the tumor detected by both CSDE and manual")
        print(" ".join(res_inter.query("beta > 0")["gene_name"].unique()))
        print("Genes downregulated in the tumor detected by both CSDE and manual")
        print(" ".join(res_inter.query("beta < 0")["gene_name"].unique()))

        print()
        print("Genes upregulated in the tumor detected by CSDE only")
        print(" ".join(res_ppi_only.query("beta > 0")["gene_name"].unique()))
        print("Genes downregulated in the tumor detected by CSDE only")
        print(" ".join(res_ppi_only.query("beta < 0")["gene_name"].unique()))
