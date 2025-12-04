from __future__ import annotations

import random

import pandas as pd

from domain import Domain
from column_extraction import extract_columns_for_domain
from joinability_pipeline import build_all_sketches, compute_lazo_joinability

# Demo script: assumes some NYC datasets have already been downloaded to data.cityofnewyork.us/data/*.json
# Run with: python run_joinability_example.py
# This is a light, end-to-end example to illustrate how to build sketches and compute joinability.


def sample_columns(column_dict, max_datasets=3, max_cols_per_dataset=5):
    """
    Take a small sample to keep the demo fast.
    """
    # Group by dataset_id
    by_ds = {}
    for (ds, col), series in column_dict.items():
        by_ds.setdefault(ds, []).append((col, series))

    chosen = {}
    datasets = list(by_ds.keys())
    random.seed(123)
    random.shuffle(datasets)

    for ds in datasets[:max_datasets]:
        cols = by_ds[ds]
        random.shuffle(cols)
        for col, series in cols[:max_cols_per_dataset]:
            chosen[(ds, col)] = series
    return chosen


def main():
    domain_name = "data.cityofnewyork.us"
    domain = Domain(domain_name)  # Existing class; does NOT download here.

    # Extract columns from already-downloaded JSON files.
    column_dict = extract_columns_for_domain(domain.sanitized)

    if not column_dict:
        print("No data found. Please download some datasets with Domain.download_all_raw_dataset() first.")
        return

    sampled = sample_columns(column_dict)
    print(f"Building sketches for {len(sampled)} columns from {len(set(ds for ds, _ in sampled))} datasets...")

    sketches = build_all_sketches(sampled)

    # Compute joinability with modest thresholds
    results = compute_lazo_joinability(sketches, js_threshold=0.3, jc_threshold=0.5)

    if results.empty:
        print("No joinable pairs found with current thresholds.")
        return

    # Show top 20 by containment/similarity
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
