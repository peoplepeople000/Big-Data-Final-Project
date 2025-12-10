from __future__ import annotations

import argparse
from pathlib import Path

from domain import Domain
from metadata_analysis import build_metadata_dataframe, summarize_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Socrata metadata for NYC Open Data and print summary statistics."
    )
    parser.add_argument(
        "--domain",
        default="data.cityofnewyork.us",
        help="Target Socrata domain (default: data.cityofnewyork.us)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of datasets to scan (default: all available)",
    )
    parser.add_argument(
        "--fetch-row-counts",
        action="store_true",
        help="If set, perform count(*) queries per dataset to estimate row counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the metadata CSV. "
             "Default: <domain>/metadata_summary.csv",
    )
    args = parser.parse_args()

    domain = Domain(args.domain)
    print(f"Collecting metadata for domain: {args.domain}")
    df = build_metadata_dataframe(
        domain,
        max_datasets=args.limit,
        fetch_row_counts=args.fetch_row_counts,
    )
    if df.empty:
        print("No metadata records collected. Exiting.")
        return

    output_path = args.output
    if output_path is None:
        output_path = Path(domain.sanitized) / "metadata_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Metadata summary written to: {output_path.resolve()}")

    summarize_metadata(df)


if __name__ == "__main__":
    main()
