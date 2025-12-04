from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Assumptions about JSON structure:
# - Each raw dataset file is an array of objects (list of dicts) as returned by Socrata.
# - Keys are column names; values are JSON primitives or nested structures (we keep only simple types).
# - Files are named like data_<dataset_id>.json under <domain>/data created by Domain.


def load_dataset_to_df(json_path: Path) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # If the file is not a list of dicts, wrap to avoid crash.
    if not isinstance(data, list):
        data = []
    return pd.DataFrame.from_records(data)


def extract_columns_for_domain(domain_dir: str) -> Dict[Tuple[str, str], pd.Series]:
    """
    Read all downloaded JSON files for a domain and return a dict:
        { (dataset_id, column_name): Series }
    Only string-like and numeric columns are kept.
    """
    base = Path(domain_dir)
    data_dir = base / "data"
    column_dict: Dict[Tuple[str, str], pd.Series] = {}

    if not data_dir.exists():
        return column_dict

    for json_file in data_dir.glob("data_*.json"):
        dataset_id = json_file.stem.replace("data_", "")
        df = load_dataset_to_df(json_file)
        if df.empty:
            continue

        # Select string-like and numeric columns
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        num_cols = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
        cols = list(dict.fromkeys(str_cols + num_cols))  # preserve order, avoid duplicates

        for col in cols:
            series = df[col]
            column_dict[(dataset_id, col)] = series

    return column_dict
