from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sodapy import Socrata
from tqdm import tqdm

from domain import Domain

ROW_BUCKETS: List[Tuple[int, Optional[int], str]] = [
    (0, 1_000, "0–1K"),
    (1_000, 10_000, "1K–10K"),
    (10_000, 100_000, "10K–100K"),
    (100_000, 1_000_000, "100K–1M"),
    (1_000_000, None, "1M+"),
]


def _to_datetime(value) -> Optional[pd.Timestamp]:
    """Convert Socrata timestamps (epoch seconds or ISO strings) to pandas Timestamp."""
    if value in (None, "", 0):
        return None
    try:
        if isinstance(value, (int, float)):
            return pd.to_datetime(int(value), unit="s", utc=True)
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _fetch_row_count(client: Socrata, dataset_id: str) -> Optional[int]:
    """
    Query count(*) for a dataset.
    Heavyweight call; should be optional.
    """
    try:
        resp = client.get(dataset_id, select="count(1)")
        if resp:
            first = resp[0]
            # Socrata names the aggregation count_1
            value = first.get("count_1") or first.get("count")
            if value is not None:
                return int(value)
    except Exception:
        return None
    return None


def _detect_special_columns(columns: Iterable[Dict]) -> Dict[str, bool]:
    """Heuristics to flag datasets that contain spatial/date/year columns."""
    has_location = False
    has_date = False
    has_year = False

    for col in columns or []:
        dtype = (col.get("dataTypeName") or "").lower()
        field = (col.get("fieldName") or "").lower()
        name = (col.get("name") or "").lower()

        if dtype in {"location", "point", "multipolygon", "polygon"}:
            has_location = True
        if any(keyword in field for keyword in ("latitude", "longitude", "geom", "geocode", "borough")):
            has_location = True

        if dtype in {"calendar_date", "floating_timestamp", "date", "meta_data", "timestamp"}:
            has_date = True
        if "date" in field or "date" in name:
            has_date = True

        if dtype in {"number", "text"} and "year" in field:
            has_year = True
        if "fiscal_year" in field or "school_year" in field:
            has_year = True

    return {
        "has_location_column": has_location,
        "has_date_column": has_date,
        "has_year_column": has_year,
    }


def _bucket_row_count(row_count: Optional[int]) -> str:
    if row_count is None:
        return "unknown"
    for lower, upper, label in ROW_BUCKETS:
        if upper is None and row_count >= lower:
            return label
        if lower <= row_count < upper:
            return label
    return "unknown"


def collect_metadata_records(
    domain: Domain,
    max_datasets: Optional[int] = None,
    fetch_row_counts: bool = False,
) -> List[Dict]:
    """
    Iterate through dataset IDs and collect a metadata snapshot per dataset.
    Returns a list of dictionaries ready to convert to a DataFrame.
    """
    dataset_ids = domain.city_datasets_ids()
    if max_datasets is not None:
        dataset_ids = dataset_ids[:max_datasets]

    records: List[Dict] = []
    iterator = dataset_ids
    if not isinstance(iterator, list):
        iterator = list(iterator)
    for dataset_id in tqdm(iterator, desc="Fetching metadata", unit="dataset"):
        try:
            metadata = domain.client.get_metadata(dataset_id)
        except Exception as exc:
            domain.log.error(f"{dataset_id}: failed to fetch metadata — {exc}")
            continue
        if not metadata:
            continue

        columns = metadata.get("columns") or []
        flags = _detect_special_columns(columns)
        created_at = _to_datetime(metadata.get("createdAt"))
        updated_at = _to_datetime(
            metadata.get("rowsUpdatedAt") or metadata.get("dataUpdatedAt")
        )
        row_count = metadata.get("metadata", {}).get("rowCount")
        if row_count is not None:
            try:
                row_count = int(row_count)
            except (ValueError, TypeError):
                row_count = None
        if row_count is None and fetch_row_counts:
            row_count = _fetch_row_count(domain.client, dataset_id)

        record = {
            "dataset_id": dataset_id,
            "name": metadata.get("name"),
            "category": metadata.get("category"),
            "tags": ",".join(metadata.get("tags") or []),
            "view_type": metadata.get("viewType"),
            "download_type": metadata.get("downloadType"),
            "n_columns": len(columns),
            "row_count": row_count,
            "row_count_bucket": _bucket_row_count(row_count),
            "created_at": created_at,
            "updated_at": updated_at,
            "view_count": metadata.get("viewCount"),
            "download_count": metadata.get("downloadCount"),
        }
        record.update(flags)
        records.append(record)

    return records


def build_metadata_dataframe(
    domain: Domain,
    max_datasets: Optional[int] = None,
    fetch_row_counts: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper: gather metadata and convert to DataFrame."""
    records = collect_metadata_records(domain, max_datasets, fetch_row_counts)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["created_year"] = df["created_at"].dt.year
    df["updated_days_ago"] = (
        datetime.now(timezone.utc) - df["updated_at"]
    ).dt.days
    return df


def summarize_metadata(df: pd.DataFrame) -> None:
    """Print headline stats similar to the Barbosa study."""
    total = len(df)
    print(f"Datasets analyzed: {total:,}")

    # View type / download type distributions
    if "view_type" in df:
        print("\nView type distribution:")
        print(df["view_type"].value_counts(dropna=False).head(10))
    if "download_type" in df:
        print("\nDownload type distribution:")
        print(df["download_type"].value_counts(dropna=False).head(10))

    # Categories
    if "category" in df:
        print("\nTop categories:")
        print(df["category"].value_counts(dropna=False).head(10))

    # Row buckets
    if "row_count_bucket" in df:
        print("\nRow count buckets:")
        print(df["row_count_bucket"].value_counts(dropna=False))

    # Created year
    if "created_year" in df:
        print("\nDatasets by creation year:")
        print(df["created_year"].value_counts(dropna=False).sort_index())

    # Updated recency
    if "updated_days_ago" in df:
        print("\nDatasets by recency (days since last update):")
        recency_bins = pd.cut(
            df["updated_days_ago"],
            bins=[-1, 30, 180, 365, 730, math.inf],
            labels=["<=30d", "31-180d", "181-365d", "1-2y", "2y+"],
        )
        print(recency_bins.value_counts(dropna=False))

    # Special columns
    print("\nSemantic coverage:")
    for column, label in [
        ("has_location_column", "With location columns"),
        ("has_date_column", "With date/datetime columns"),
        ("has_year_column", "With year columns"),
    ]:
        if column in df:
            pct = df[column].mean() * 100
            print(f"{label}: {pct:.1f}%")
