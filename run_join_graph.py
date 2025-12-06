from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from joinability_pipeline import compute_lazo_joinability
from join_graph import build_dataset_graph
from lazo_sketch import ColumnSketch


def save_top_dataset_chart(G, output_path: Path, top_n: int = 15) -> None:
    """Aggregate node-level stats and plot the datasets with the most partners."""
    stats = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        num_partners = len(neighbors)
        total_join_cols = sum(G[node][nbr].get("num_joinable_columns", 0) for nbr in neighbors)
        max_jc = max((G[node][nbr].get("max_jc", 0) for nbr in neighbors), default=0)
        stats.append(
            {
                "dataset": node,
                "num_partners": num_partners,
                "total_joinable_columns": total_join_cols,
                "max_jc_with_partner": max_jc,
            }
        )

    df = pd.DataFrame(stats)
    if df.empty:
        return

    top = df.nlargest(top_n, "num_partners")
    plt.figure(figsize=(10, max(4, top_n * 0.35)))
    plt.barh(top["dataset"], top["num_partners"])
    plt.gca().invert_yaxis()
    plt.title("Top datasets by number of joinable partners")
    plt.xlabel("Number of partner datasets")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved partner count chart to: {output_path.resolve()}")


def save_top_pair_chart(join_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    """Plot dataset pairs with the most joinable column pairs."""
    pair_counts = (
        join_df.groupby(["left_dataset", "right_dataset"])
        .size()
        .reset_index(name="num_column_pairs")
    )
    if pair_counts.empty:
        return

    top = pair_counts.nlargest(top_n, "num_column_pairs")
    top["pair"] = top["left_dataset"] + " ↔ " + top["right_dataset"]
    plt.figure(figsize=(10, max(4, top_n * 0.35)))
    plt.barh(top["pair"], top["num_column_pairs"])
    plt.gca().invert_yaxis()
    plt.title("Top dataset pairs by joinable column pairs")
    plt.xlabel("Number of joinable column pairs")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved dataset pair chart to: {output_path.resolve()}")


def save_containment_hist(join_df: pd.DataFrame, output_path: Path) -> None:
    """Plot containment distribution to show how strong the detected joins are."""
    if join_df.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(
        join_df["jc_left_in_right"],
        bins=30,
        alpha=0.6,
        label="left in right",
    )
    plt.hist(
        join_df["jc_right_in_left"],
        bins=30,
        alpha=0.6,
        label="right in left",
    )
    plt.xlabel("Containment")
    plt.ylabel("Count of column pairs")
    plt.title("Containment distribution across joinable columns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved containment distribution chart to: {output_path.resolve()}")


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
    max_columns = None  # adjust as needed (orignally 200)
    if max_columns is not None and len(sketches) > max_columns:
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
    csv_path = Path("nyc_joinability_pairs.csv")
    join_df.to_csv(csv_path, index=False)
    print(f"Joinability pairs exported to CSV: {csv_path.resolve()}")

    # Print a quick summary to the terminal so users can inspect results without opening the HTML.
    top_n = 100
    print(f"\nTop {min(top_n, len(join_df))} column pairs by containment/similarity:\n")
    print(
        join_df.head(top_n).to_string(
            index=False,
            columns=[
                "left_dataset",
                "left_column",
                "right_dataset",
                "right_column",
                "js",
                "jc_left_in_right",
                "jc_right_in_left",
            ],
        )
    )
    print()

    # Build dataset-level graph and create static charts for reporting.
    G = build_dataset_graph(join_df)
    charts_dir = Path("reports")
    charts_dir.mkdir(exist_ok=True)
    save_top_dataset_chart(G, charts_dir / "top_datasets_by_partners.png")
    save_top_pair_chart(join_df, charts_dir / "top_dataset_pairs.png")
    save_containment_hist(join_df, charts_dir / "containment_distribution.png")


if __name__ == "__main__":
    main()
