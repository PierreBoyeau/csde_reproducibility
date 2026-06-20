#!/usr/bin/env python
"""
Import manual cell annotations into the analysis pipeline.

Annotation workflow
-------------------
1. Run export_data.ipynb for a dataset. It writes ~600 cell images to
       {DATA_DIR}/{FullDatasetName}/manual_annotations/images/cell_{idx}.png
   where {idx} is the cell's index in adata.obs. Each image shows the cell
   neighbourhood with marker-gene transcript dots overlaid.

2. An annotator reviews those images and records a cell-type label for each
   one. The expected labels are:
       "t cell", "fibroblast", "cancer", "endothelial",
       "myeloid", "b cell", "other"
   The annotator should produce a two-column CSV:

       cell_id,annotation_name
       1234,t cell
       5678,fibroblast
       9012,cancer
       ...

   cell_id must match the filename suffix (cell_{cell_id}.png) and the
   corresponding row in adata.obs.

3. Run this script to convert the CSV to the JSON format consumed by
   MerfishDataNavigator.static_load_annotations():

       python workflow/scripts/import_annotations.py \\
           --annotations-csv /path/to/annotated.csv \\
           --output-path "$DATA_DIR/HumanLungCancerPatient1/manual_annotations/annotations.json"

Usage
-----
    python workflow/scripts/import_annotations.py \\
        --annotations-csv ANNOTATIONS_CSV \\
        --output-path    OUTPUT_PATH
"""

import json
import click
import pandas as pd


@click.command()
@click.option(
    "--annotations-csv",
    type=click.Path(exists=True),
    required=True,
    help="CSV with columns: cell_id, annotation_name",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Destination path for the annotations JSON file",
)
def main(annotations_csv: str, output_path: str):
    df = pd.read_csv(annotations_csv)

    missing = {"cell_id", "annotation_name"} - set(df.columns)
    if missing:
        raise click.UsageError(f"CSV is missing required columns: {missing}")

    df = df.set_index("cell_id")

    # orient="index" produces {"cell_id": {"annotation_name": "t cell"}, ...}
    # which is the format read by MerfishDataNavigator.static_load_annotations().
    df.to_json(output_path, orient="index")
    click.echo(f"Wrote {len(df)} annotations to {output_path}")


if __name__ == "__main__":
    main()
