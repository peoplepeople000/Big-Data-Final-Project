from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm  # <<< NEW

from column_extraction import extract_columns_for_domain
from joinability_pipeline import build_all_sketches
from lazo_sketch import ColumnSketch, ColumnSketcher


def main():
    domain_folder = "data.cityofnewyork.us"
    print(f"Extracting columns from domain folder: {domain_folder}")
    column_dict: Dict[Tuple[str, str], pd.Series] = extract_columns_for_domain(domain_folder)
    print(f"Total (dataset, column) pairs extracted: {len(column_dict)}")

    if not column_dict:
        print("No columns found. Did you run download_nyc_all.py first?")
        return

    # Filter out small columns
    filtered_column_dict: Dict[Tuple[str, str], pd.Series] = {}
    for key, series in column_dict.items():
        if len(series.dropna()) >= 50:
            filtered_column_dict[key] = series

    print(f"Columns after basic filtering: {len(filtered_column_dict)}")

    print("Building sketches for all remaining columns...")

    sketcher = ColumnSketcher()
    sketches: Dict[Tuple[str, str], ColumnSketch] = {}

    # >>> HERE IS THE PROGRESS BAR
    for (dataset_id, col_name), series in tqdm(filtered_column_dict.items(), desc="Building sketches"):
        sketch = sketcher.build_sketch(series, column_name=col_name, dataset_id=dataset_id)
        sketches[(dataset_id, col_name)] = sketch

    print(f"Built sketches for {len(sketches)} columns.")

    out_path = Path("nyc_column_sketches.pkl")
    with out_path.open("wb") as f:
        pickle.dump(sketches, f)

    print(f"Saved column sketches to: {out_path.resolve()}")


if __name__ == "__main__":
    main()