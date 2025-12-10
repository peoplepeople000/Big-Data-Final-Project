from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_pairs(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Joinability CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {
        "left_dataset",
        "left_column",
        "right_dataset",
        "right_column",
        "js",
        "jc_left_in_right",
        "jc_right_in_left",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def plot_js_distribution(df: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.hist(df["js"], bins=40, alpha=0.8, color="#4a90e2")
    plt.title("Distribution of LAZO Jaccard Similarity")
    plt.xlabel("Jaccard similarity (JS)")
    plt.ylabel("Number of column pairs")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved JS distribution: {output.resolve()}")


def plot_containment_distribution(df: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df["jc_left_in_right"], bins=40, alpha=0.6, label="left in right")
    plt.hist(df["jc_right_in_left"], bins=40, alpha=0.6, label="right in left")
    plt.title("Containment Distribution")
    plt.xlabel("Containment")
    plt.ylabel("Number of column pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved containment distribution: {output.resolve()}")


def plot_top_dataset_pairs(df: pd.DataFrame, output: Path, top_n: int = 15) -> None:
    pair_counts = (
        df.groupby(["left_dataset", "right_dataset"])
        .size()
        .reset_index(name="num_column_pairs")
        .sort_values("num_column_pairs", ascending=False)
        .head(top_n)
    )
    if pair_counts.empty:
        print("No dataset pairs to plot.")
        return
    pair_counts["pair"] = (
        pair_counts["left_dataset"] + " ↔ " + pair_counts["right_dataset"]
    )
    plt.figure(figsize=(10, max(4, top_n * 0.35)))
    plt.barh(pair_counts["pair"], pair_counts["num_column_pairs"])
    plt.gca().invert_yaxis()
    plt.title("Top dataset pairs by joinable columns")
    plt.xlabel("Number of joinable column pairs")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved dataset pair chart: {output.resolve()}")


def plot_dataset_hubs(df: pd.DataFrame, output: Path, top_n: int = 15) -> None:
    partners = defaultdict(set)
    for row in df.itertuples(index=False):
        partners[row.left_dataset].add(row.right_dataset)
        partners[row.right_dataset].add(row.left_dataset)

    hub_df = (
        pd.DataFrame(
            [(ds, len(neigh)) for ds, neigh in partners.items()],
            columns=["dataset", "num_partners"],
        )
        .sort_values("num_partners", ascending=False)
        .head(top_n)
    )
    if hub_df.empty:
        print("No dataset hubs to plot.")
        return

    plt.figure(figsize=(10, max(4, top_n * 0.35)))
    plt.barh(hub_df["dataset"], hub_df["num_partners"])
    plt.gca().invert_yaxis()
    plt.title("Datasets with the most join partners")
    plt.xlabel("Number of partner datasets")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved dataset hub chart: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize LAZO joinability results stored in nyc_joinability_pairs.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("nyc_joinability_pairs.csv"),
        help="Path to the joinability CSV (default: nyc_joinability_pairs.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports"),
        help="Directory to store generated charts (default: reports/)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top datasets/pairs to display in bar charts",
    )
    args = parser.parse_args()

    df = load_pairs(args.csv)
    args.outdir.mkdir(exist_ok=True)

    plot_js_distribution(df, args.outdir / "js_distribution.png")
    plot_containment_distribution(df, args.outdir / "containment_distribution.png")
    plot_top_dataset_pairs(df, args.outdir / "top_dataset_pairs.png", top_n=args.top)
    plot_dataset_hubs(df, args.outdir / "top_datasets_by_partners.png", top_n=args.top)


if __name__ == "__main__":
    main()
