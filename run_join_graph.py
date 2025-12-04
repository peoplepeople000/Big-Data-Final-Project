from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from joinability_pipeline import compute_lazo_joinability
from join_graph import build_dataset_graph, export_graph_to_html
from lazo_sketch import ColumnSketch


def main():
    """
    End-to-end script:
    - Load precomputed column sketches.
    - Compute LAZO-based joinability.
    - Build a dataset-level join graph.
    - Export to an interactive HTML visualization.
    """
    sketches_path = Path("nyc_column_sketches.pkl")
    if not sketches_path.exists():
        print(f"Sketch file not found at {sketches_path}. Run build_sketches_all.py first.")
        return

    with sketches_path.open("rb") as f:
        sketches: Dict[Tuple[str, str], ColumnSketch] = pickle.load(f)

    print(f"Loaded {len(sketches)} column sketches.")

    # Optional: limit number of columns for a first test (to keep runtime manageable)
    max_columns = 200  # adjust as needed
    if len(sketches) > max_columns:
        print(f"Subsampling sketches down to {max_columns} columns for this run.")
        items = list(sketches.items())[:max_columns]
        sketches = dict(items)

    # Compute joinability; choose thresholds to reduce noise.
    js_threshold = 0.3
    jc_threshold = 0.5
    print(f"Computing LAZO joinability with js_threshold={js_threshold}, jc_threshold={jc_threshold} ...")
    join_df: pd.DataFrame = compute_lazo_joinability(sketches, js_threshold, jc_threshold)

    if join_df.empty:
        print("No joinable column pairs found with the current thresholds.")
        return

    print(f"Joinable column pairs found: {len(join_df)}")

    # Build dataset-level graph
    G = build_dataset_graph(join_df)

    # Export to HTML for visualization
    export_graph_to_html(G, output_html="nyc_join_graph.html", min_edge_weight=0.6)


if __name__ == "__main__":
    main()