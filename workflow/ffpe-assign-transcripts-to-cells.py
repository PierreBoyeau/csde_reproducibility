#!/usr/bin/env python

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import argparse
import h5py
from glob import glob
import os.path

NZLAYERS = 7

def read_feature_data(filename: str):
    f = h5py.File(filename, "r")
    g = f["featuredata"]
    gdfs = dict()

    for z in range(NZLAYERS):
        lyrkey = f"zIndex_{z}"
        ps = []
        cell_ids = []
        for (cell_id, feature) in g.items():
            for polygon in feature[lyrkey].values():
                coords = np.array(polygon["coordinates"])
                ps.append(Polygon(coords[0,:,:]))
                cell_ids.append(cell_id)

        gdf = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=gpd.GeoSeries(ps))
        gdfs[z] = gdf
    return gdfs

def read_all_feature_data(cell_boundaries_path: str):
    print("Reading polygons...")
    gdfss = []
    for filename in glob(f"{cell_boundaries_path}/*.hdf5"):
        print(filename)
        gdfss.append(read_feature_data(filename))

    print("Concating polygons...")
    gdfs_concat = dict()
    for z in range(NZLAYERS):
        gdfs_concat[z] = pd.concat([gdfs[z] for gdfs in gdfss])

    return gdfs_concat


def write_results(results, gene_rev_index, output_filename):
    df = pd.DataFrame({
        "gene": [gene_rev_index[gene_id] for gene_id in results.gene],
        "x": np.fromiter((p.x for p in results.geometry), np.float32),
        "y": np.fromiter((p.y for p in results.geometry), np.float32),
        "z": results.z,
        "cell": results.cell_id,
    }).to_csv(output_filename, na_rep="NA", index=False)


def read_transcripts(filename):
    print("Reading transcripts...")
    df = pd.read_csv(filename)

    gene_index = dict()
    for gene in df.gene:
        if gene not in gene_index:
            gene_index[gene] = len(gene_index)

    gdfs = dict()
    for z in range(NZLAYERS):
        print(f"Building layer {z} transcript index")
        dfz = df[df.global_z == float(z)]
        ps = []
        for (x, y) in zip(dfz.global_x, dfz.global_y):
            ps.append(Point(x, y))

        genes = np.fromiter([gene_index[gene] for gene in dfz.gene], int)
        gdf = gpd.GeoDataFrame({"gene": genes}, geometry=gpd.GeoSeries(ps))
        gdfs[z] = gdf

    return gdfs, gene_index
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("transcripts_filename")
    argparser.add_argument("output_filename")
    args = argparser.parse_args()

    cell_boundaries_path = os.path.join(
        os.path.dirname(args.transcripts_filename),
        "cell_boundaries")

    transcripts, gene_index = read_transcripts(args.transcripts_filename)
    polygons = read_all_feature_data(cell_boundaries_path)

    ngenes = max(gene_index.values()) + 1
    gene_rev_index = np.full(ngenes, "", dtype=object)
    for (gene, gene_id) in gene_index.items():
        gene_rev_index[gene_id] = gene

    results = []
    for z in range(NZLAYERS):
        result = transcripts[z].sjoin(polygons[z], how="left")
        result.insert(0, "z", np.full(result.shape[0], z))
        print(f"Joining layer {z}...")
        results.append(result)

    print("Concatenating results...")
    results = pd.concat(results)
    print("Writing results...")
    write_results(results, gene_rev_index, args.output_filename)