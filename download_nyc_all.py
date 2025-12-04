from __future__ import annotations

import time

from domain import Domain


def main():
    """
    Incrementally download raw JSON for all datasets under data.cityofnewyork.us.

    - This script can be run multiple times. The Domain class should skip datasets
      that already have a local JSON file.
    - Without an app_token, requests will be throttled; be gentle and sleep between requests.
    """
    domain_name = "data.cityofnewyork.us"
    d = Domain(domain_name)

    # Get all dataset IDs for this domain
    ids = d.city_datasets_ids()
    print(f"Total datasets found: {len(ids)}")

    # Optional: limit per run so you don't overload the API or your machine
    # Set this to None if you really want to attempt all in one go.
    max_per_run = 600

    num_downloaded = 0
    for dsid in ids:
        if max_per_run is not None and num_downloaded >= max_per_run:
            break

        print(f"[{num_downloaded+1}] Downloading dataset {dsid} ...")
        try:
            d.download_raw_dataset(dsid)
            num_downloaded += 1
        except Exception as e:
            print(f"  -> Error downloading {dsid}: {e}")

        # Be nice to the Socrata API; avoid hitting rate limits too hard.
        time.sleep(0.5)

    print(f"Finished this run. Downloaded (or attempted) {num_downloaded} datasets.")


if __name__ == "__main__":
    main()