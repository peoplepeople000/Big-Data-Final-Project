# sample_loader.py

import json
from pathlib import Path


def load_column_samples(domain_folder: str, dataset_id: str, column: str, n: int = 25):
    """
    Reads up to n values for `column` from:
        {domain_folder}/data/data_<dataset_id>.json

    Example:
        data.cityofnewyork.us/data/data_erm2-nwe9.json

    Used when USE_FAKE_DATA = False.
    """

    base = Path(domain_folder)
    file_path = base / "data" / f"data_{dataset_id}.json"

    if not file_path.exists():
        # Real data not downloaded yet
        return []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    values = []
    for row in data:
        if isinstance(row, dict) and column in row:
            values.append(row[column])
            if len(values) >= n:
                break

    return values
