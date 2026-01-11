import glob
import json
import os
import pickle
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc
import shapely.wkb
import geojson
from matplotlib.lines import Line2D
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


class MicronPixelConverter:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def micron_to_mosaic(self, pos):
        return (self.alpha * pos) + self.beta

    def mosaic_to_micron(self, pos):
        res = pos - self.beta
        res = res / self.alpha
        return res


class MerfishDataNavigator:
    def __init__(
        self,
        path_to_assay_data: str,
        path_to_adata: Union[str, sc.AnnData],
        path_to_images: str,
        path_to_raw_transcripts: str = None,
        boundary_parquet_path: str = None,
        boundary_hdf5_dir: str = None,
        boundary_geojson_path: str = None,
    ):
        if isinstance(path_to_adata, str):
            self.adata = sc.read_h5ad(path_to_adata)
        else:
            self.adata = path_to_adata
        self.preprocess_adata()
        self.adata_info = self.adata.obs
        if path_to_raw_transcripts is not None:
            self.raw_transcripts = pd.read_csv(
                path_to_raw_transcripts,
                index_col=0,
            )
        else:
            self.raw_transcripts = pd.read_csv(
                os.path.join(path_to_assay_data, "detected_transcripts.csv"),
                index_col=0,
            )
        # micron / pixel conversions
        micron_to_mosaic_scale = pd.read_csv(
            os.path.join(
                path_to_assay_data, "images", "micron_to_mosaic_pixel_transform.csv"
            ),
            header=None,
            sep="\s",
        ).values
        alpha = np.array([micron_to_mosaic_scale[0, 0], micron_to_mosaic_scale[1, 1]])
        beta = np.array([micron_to_mosaic_scale[0, 2], micron_to_mosaic_scale[1, 2]])
        self.coord_converter = MicronPixelConverter(alpha, beta)

        # cell boundaries
        if boundary_parquet_path is not None:
            cells_bounds_ = self._open_boundaries_parquet(boundary_parquet_path)
            cell_metadata = None
        elif boundary_hdf5_dir is not None:
            boundaries = glob.glob(os.path.join(boundary_dir, "*.hdf5"))
            cells_bounds_, cell_metadata = self._open_boundaries_h5ad(boundaries)
        elif boundary_geojson_path is not None:
            cells_bounds_ = self._open_boundaries_geojson(boundary_geojson_path)
            cell_metadata = None
        else:
            raise ValueError("No cell boundaries found")
        self.cell_boundaries = cells_bounds_
        self.cell_metadata = pd.DataFrame(cell_metadata)

        # boundary_dir = os.path.join(path_to_assay_data, "cell_boundaries")
        # boundaries = glob.glob(os.path.join(boundary_dir, "*.hdf5"))
        # open_h5ad = len(boundaries) != 0
        # boundary_parquet_path = os.path.join(path_to_assay_data, "cell_boundaries.parquet")
        # open_parquet = os.path.exists(boundary_parquet_path)
        # if open_parquet:
        #     cells_bounds_ = self._open_boundaries_parquet(boundary_parquet_path)
        #     cell_metadata = None
        # elif open_h5ad:
        #     cells_bounds_, cell_metadata = self._open_boundaries_h5ad(boundaries)
        # else:
        #     raise ValueError("No cell boundaries found")
        # self.cell_boundaries = cells_bounds_
        # self.cell_metadata = pd.DataFrame(cell_metadata)

        # images
        self.imgs = [Image.open(img_path) for img_path in path_to_images]

    def _open_boundaries_geojson(self, geojson_path):
        with open(geojson_path) as f:
            gj = geojson.load(f)
        cell_bounds_ = {}
        for feature in gj["features"]:
            cell_id = str(feature["properties"]["cell"])
            polygon = feature["geometry"]["coordinates"][0]
            polygon = np.array(polygon).squeeze()
            cell_bounds_[cell_id] = polygon
        return cell_bounds_

    def _open_boundaries_h5ad(self, boundary_paths):
        cells_bounds_ = {}
        cell_metadata = []
        for boundary in tqdm(boundary_paths):
            fov_id = os.path.basename(boundary).split(".")[0].split("_")[-1]
            with h5py.File(boundary, "r") as f:
                data = f["featuredata"]
                cell_ids = data.keys()
                for cell_id in cell_ids:
                    data_ = data[cell_id]["zIndex_3"]
                    if len(data_) != 0:
                        cell_bounds = np.array(data_["p_0"]["coordinates"]).squeeze()
                        if cell_id in cells_bounds_:
                            print(f"Cell {cell_id} already in cells_bounds")
                        cells_bounds_[cell_id] = cell_bounds
                    cell_metadata.append(
                        {
                            "cell_id": cell_id,
                            "fov_id": fov_id,
                            "boundary_path": boundary,
                        }
                    )
        return cells_bounds_, pd.DataFrame(cell_metadata)

    def _open_boundaries_parquet(self, parquet_path):
        boundaries = pq.read_table(parquet_path)
        cell_indices = boundaries["EntityID"]
        geom_col = boundaries["Geometry"]
        geometries = [
            shapely.wkb.loads(geometry.as_py()) for geometry in tqdm(geom_col)
        ]

        cell_boundaries = {}
        for geom, cell_index in zip(geometries, tqdm(cell_indices)):
            geom_ = geom.geoms[0]
            x, y = geom_.exterior.coords.xy
            cell_index_str = str(cell_index)
            cell_boundaries[cell_index_str] = np.array([x, y]).T
        return cell_boundaries

    def extract_gene_and_img(
        self,
        min_x,
        max_x,
        min_y,
        max_y,
    ):
        """
        Extracts transcripts and image corresponding to a given area (in microns).

        min_x: Minimum x-coordinate (in microns).
        max_x: Maximum x-coordinate (in microns).
        min_y: Minimum y-coordinate (in microns).
        max_y: Maximum y-coordinate (in microns).
        """
        adata_obs = self.adata.obs
        adata_sel = self.adata[
            (adata_obs["min_x"] >= min_x)
            & (adata_obs["min_x"] <= max_x)
            & (adata_obs["min_y"] >= min_y)
            & (adata_obs["min_y"] <= max_y)
        ]
        raw_transcripts_sel = self.raw_transcripts.loc[
            (self.raw_transcripts["global_x"] >= min_x)
            & (self.raw_transcripts["global_x"] <= max_x)
            & (self.raw_transcripts["global_y"] >= min_y)
            & (self.raw_transcripts["global_y"] <= max_y)
        ]

        pos_min = np.array([min_x, min_y])
        pos_max = np.array([max_x, max_y])
        pos_min = self.coord_converter.micron_to_mosaic(pos_min)
        pos_max = self.coord_converter.micron_to_mosaic(pos_max)
        extent_pix = (pos_min[0], pos_min[1], pos_max[0], pos_max[1])

        imgs_sel = []
        for img in self.imgs:
            img_sel = img.crop(extent_pix)
            img_sel = self.normalize_image(img_sel)
            imgs_sel.append(img_sel)
        img_sel = np.stack(imgs_sel, axis=2)

        cells_bounds_ = [
            self.cell_boundaries[cell_idx] for cell_idx in adata_sel.obs.index
        ]
        return {
            "imgs": imgs_sel,
            "pos_min": pos_min,
            "pos_max": pos_max,
            "raw_transcripts": raw_transcripts_sel,
            "adata": adata_sel,
            "cells_bounds": cells_bounds_,
        }

    def plot_cell_info(
        self,
        cell_idx,
        de_res_markers,
        gene_color_pairs,
        delta=10,
        marker_size=10,
        key_to_plot=None,
        suptitle=None,
        normalize_range=5,
    ):
        """
        Plot an area aroud a cell along with relevant information.

        Args:
            cell_idx: index of the cell to plot.
            de_res_markers: Dataframe to color top expressed genes by.
            For instance, this can correspond to the LFC of the gene between the cell-type
            the observation belongs to and the rest of the cells, computed on scRNA-seq.
            gene_color_pairs: List of gene name, color pairs, of the form (gene_name, color).
            Here, `color` can be any matplotlib color.
            delta: Delta defining the area to plot around the cell, in microns.
            key_to_plot: Key to plot in the adata.obs.
        """
        cell_info = self.adata_info.loc[cell_idx]
        xmin = cell_info["min_x"].item() - delta
        xmax = cell_info["max_x"].item() + delta
        ymin = cell_info["min_y"].item() - delta
        ymax = cell_info["max_y"].item() + delta
        pos_pix = self.coord_converter.micron_to_mosaic(np.array([xmin, ymin]))

        outs = self.extract_gene_and_img(xmin, xmax, ymin, ymax)
        cell_bounds = outs["cells_bounds"]
        cell_bounds_fov_pix = [
            self.coord_converter.micron_to_mosaic(bounds) - pos_pix
            for bounds in cell_bounds
        ]
        cell_bounds_fov_assignment = [
            "#00b01d" if cell_id == cell_idx else "#800101"
            for cell_id in outs["adata"].obs.index
        ]  # plot central cell in green, other cells in red

        ct_info = outs["adata"].obs.copy()
        ct_info.loc[
            :, ["center_x_pix", "center_y_pix"]
        ] = self.coord_converter.micron_to_mosaic(
            ct_info.loc[:, ["center_x", "center_y"]].values
        )
        ct_info.loc[:, ["center_x_pix_norm", "center_y_pix_norm"]] = (
            ct_info.loc[:, ["center_x_pix", "center_y_pix"]] - pos_pix
        ).values

        sel_transcripts = outs["raw_transcripts"]
        sel_transcripts.loc[
            :, ["center_x_pix", "center_y_pix"]
        ] = self.coord_converter.micron_to_mosaic(
            sel_transcripts.loc[:, ["global_x", "global_y"]].values
        )
        sel_transcripts.loc[:, ["center_x_pix_norm", "center_y_pix_norm"]] = (
            sel_transcripts.loc[:, ["center_x_pix", "center_y_pix"]] - pos_pix
        ).values

        gexp = np.array(self.adata[self.adata.obs.index == cell_idx].X)
        gene_df = (
            pd.DataFrame(gexp, columns=self.adata.var_names)
            .T.sort_values(by=0, ascending=False)
            .head(15)
        )
        gene_df.columns = ["expression"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        if suptitle is not None:
            plt.suptitle(suptitle)
        
        # Fig 1. Cell boundaries info
        plt.sca(axes[0])
        plt.imshow(outs["imgs"][0])
        for bound, color in zip(cell_bounds_fov_pix, cell_bounds_fov_assignment):
            plt.plot(bound[:, 0], bound[:, 1], color=color)
        if key_to_plot is not None:
            for _, cell_info in ct_info.iterrows():
                plt.text(
                    cell_info["center_x_pix_norm"],
                    cell_info["center_y_pix_norm"],
                    cell_info[key_to_plot],
                    color="white",
                )
        legend_elements = []
        for gene_name, color in gene_color_pairs:
            sel_transcripts_ = sel_transcripts.query(f"gene == '{gene_name}'")
            plt.scatter(
                sel_transcripts_["center_x_pix_norm"],
                sel_transcripts_["center_y_pix_norm"],
                color=color,
                alpha=0.5,
                s=marker_size,
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=gene_name,
                    markerfacecolor=color,
                    markersize=10,
                )
            )
        plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.1, 1))
        plt.axis("off")
        
        # Fig 2. Top expressed genes
        plt.sca(axes[1])
        # ova_lfc = de_res_markers.loc[gene_df.index]["lfc"].values
        ova_lfc = de_res_markers.reindex(gene_df.index).fillna(0)["lfc"].values
        norm = plt.Normalize(-normalize_range, normalize_range)
        cmap = plt.cm.coolwarm
        plt.barh(
            y=range(len(gene_df)),
            width=gene_df["expression"],
            color=cmap(norm(ova_lfc)),
        )
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[1])
        plt.yticks(range(len(gene_df)), gene_df.index)
        plt.xlabel("Expression")
        plt.title("Top Expressed Genes")
        return fig

    @staticmethod
    def static_load_annotations(path_to_annotations):
        """
        Loads cell annotations from a json file to some adata.

        The cell annotations should be stored as a Datumaro annotation file.
        """
        with open(path_to_annotations, "r") as f:
            annotations = json.load(f)

        annotation_classes = {}
        for item_id, item in enumerate(annotations["categories"]["label"]["labels"]):
            annotation_classes[item_id] = item["name"]

        ids = []
        annots = []
        for item in annotations["items"]:
            try:
                id_ = item["id"]
                annot_ = item["annotations"][0]["label_id"]
                ids.append(id_)
                annots.append(annot_)
            except:
                print(item)

        ids = np.array(ids)
        annots = np.array(annots)
        annot_df = (
            pd.DataFrame(
                {
                    "id": ids,
                    "annot": annots,
                }
            )
            .assign(annotation_name=lambda x: x["annot"].map(annotation_classes))
            .drop_duplicates(subset=["id"])
        )
        # annotated_cells = annot_df["id"].values
        # hand_annots = (
        #     annot_df.set_index("id").loc[annotated_cells, "annotation_name"].values
        # )
        return annot_df.set_index("id")

    def load_annotations(self, path_to_annotations, annotation_key="hand_annot"):
        hand_annots = self.static_load_annotations(path_to_annotations)
        self.adata.obs.loc[hand_annots.index, annotation_key] = hand_annots

    @staticmethod
    def normalize_image(img):
        """
        Contrast adjustment for better visualization
        """
        img_arr = np.array(img)
        vmax = np.percentile(img_arr, 99)
        vmin = np.percentile(img_arr, 5)
        img_arr = np.clip(img_arr, vmin, vmax)
        img_arr = (img_arr - vmin) / (vmax - vmin)
        return img_arr

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def preprocess_adata(self):
        if "centroid_x" in self.adata.obs.columns:
            self.adata.obs["center_x"] = self.adata.obs["centroid_x"]
        if "centroid_y" in self.adata.obs.columns:
            self.adata.obs["center_y"] = self.adata.obs["centroid_y"]

        center_x_not_here = "center_x" not in self.adata.obs.columns
        center_y_not_here = "center_y" not in self.adata.obs.columns

        if center_x_not_here or center_y_not_here:
            self.adata.obs["center_x"] = (
                self.adata.obs["max_x"] + self.adata.obs["min_x"]
            ) / 2
            self.adata.obs["center_y"] = (
                self.adata.obs["max_y"] + self.adata.obs["min_y"]
            ) / 2

        if "min_x" not in self.adata.obs.columns:
            self.adata.obs.loc[:, "min_x"] = self.adata.obs["center_x"] - 1
            self.adata.obs.loc[:, "max_x"] = self.adata.obs["center_x"] + 1
            self.adata.obs.loc[:, "min_y"] = self.adata.obs["center_y"] - 1
            self.adata.obs.loc[:, "max_y"] = self.adata.obs["center_y"] + 1
