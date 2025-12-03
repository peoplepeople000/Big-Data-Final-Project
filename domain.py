import re
import requests
import json
import logging
from pathlib import Path
from sodapy import Socrata

class Domain:
    def __init__(self, domain, token=None, timeout=10):
        self.domain = domain
        self.sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", domain)
        self.views = f"https://{domain}/views.json"
        self.client = Socrata(domain, token, timeout=timeout)
        self.identifiers_file = None
        self.base = Path(self.sanitized)
        self.datadir = self.base / "data"
        self.schemadir = self.base / "schema"
        self.base.mkdir(exist_ok=True)

        self.log = self.setup_logger()
    
    def setup_logger(self):
        self.logdir = self.base / "logs"
        self.logdir.mkdir(exist_ok=True)

        logger = logging.getLogger(self.sanitized)
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        # Handlers
        error_handler = logging.FileHandler(self.logdir / "errors.log", encoding="utf-8")
        error_handler.setLevel(logging.ERROR)

        warning_handler = logging.FileHandler(self.logdir / "warnings.log", encoding="utf-8")
        warning_handler.setLevel(logging.WARNING)

        info_handler = logging.FileHandler(self.logdir / "info.log", encoding="utf-8")
        info_handler.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        for h in (error_handler, warning_handler, info_handler):
            h.setFormatter(fmt)
            logger.addHandler(h)

        return logger

    def city_datasets_count(self):
        params = {"count": True}
        response = requests.get(self.views, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("count", -1) # Error (no count) represented by -1 rather than 0
    
    def city_datasets_ids(self):
        datasets = self.client.datasets()
        ids = [d["resource"]["id"] for d in datasets]
        return ids
    
    def write_dataset_ids_to_file(self, filepath=None):
        ids = self.city_datasets_ids()
        if filepath is None:
            filepath = f"{self.sanitized}_ids.txt"
        full_path = self.base / filepath
        with full_path.open("w", encoding="utf-8") as f:
            for dataset_id in ids:
                f.write(dataset_id + "\n")

        self.identifiers_file = full_path
        return full_path
    
    def _set_up_ids(self):
        if self.identifiers_file is None:
            self.identifiers_file = self.base / f"{self.sanitized}_ids.txt"
        if not self.identifiers_file.exists():
            self.write_dataset_ids_to_file()
        return True

    def download_all_raw_dataset(self):
        self._set_up_ids()
        with self.identifiers_file.open() as f:
            for line in f:
                dataset_id = line.strip()
                outfile = f"data_{dataset_id}.json"
                outpath = self.datadir / outfile
                if outpath.exists():
                    continue
                self.download_raw_dataset(dataset_id)
    
    def download_raw_dataset(self, dataset_id, outfile=None):
        url = f"https://{self.domain}/resource/{dataset_id}.json"
        try:
            resp = requests.get(url, timeout=15)
        except Exception as e:
            self.log.error(f"{dataset_id}: network error — {e}")
            return None

        # Try getting JSON first, even on 40x/50x errors
        payload = None
        try:
            payload = resp.json()
        except ValueError:
            pass   # Response body is not JSON

        # If Socrata returned a JSON error block:
        # e.g. {"error": true, "message": "...", "code": "..."}
        if isinstance(payload, dict) and payload.get("error"):
            self.log.error(f"{dataset_id}: Socrata error — {payload.get('message')}")
            return None

        # Now let HTTP errors raise only if it's NOT a Socrata dataset error
        try:
            resp.raise_for_status()
        except Exception as e:
            self.log.error(f"{dataset_id}: HTTP error — {e}")
            return None

        # Handle empty array
        if isinstance(payload, list) and all(not row for row in payload):
            self.log.warning(f"{dataset_id}: empty dataset")
            return None

        # If we get here, payload is valid — save it
        if outfile is None:
            outfile = f"data_{dataset_id}.json"
        self._ensure_data_dir()
        outpath = self.datadir / outfile
        outpath.write_bytes(resp.content)

    def fetch_all_schema(self):
        self._set_up_ids()
        with self.identifiers_file.open() as f:
            for line in f:
                dataset_id = line.strip()
                outfile = f"schema_{dataset_id}.txt"
                outpath = self.schemadir / outfile
                if outpath.exists():
                    continue
                self.fetch_schema(dataset_id)

    def fetch_schema(self, dataset_id, outfile=None):
        if outfile is None:
            outfile = f"schema_{dataset_id}.txt"
        data_file = self.datadir / f"{dataset_id}.json"
        schema = None
        if data_file.exists():
            try:
                with data_file.open() as f:
                    payload = json.load(f)
                if isinstance(payload, list) and payload:
                    # Find the first non-empty dict
                    record = next(
                        (row for row in payload if isinstance(row, dict) and row),
                        None
                    )
                    if record:
                        schema = list(record.keys())
                    else:
                        # All objects are empty
                        self.log.warning(
                            f"{dataset_id}: JSON file contains only empty objects — schema skipped"
                        )
                        return None
                else:
                    # JSON is [] or invalid structure
                    self.log.warning(
                        f"{dataset_id}: JSON file is empty or malformed — schema skipped"
                    )
                    return None
            except Exception as e:
                self.log.warning(f"{dataset_id}: failed to parse JSON — {e}")
                return None
        else:
            try:
                metadata = self.client.get_metadata(dataset_id)
                cols = metadata.get("columns") or []
                schema = [c.get("fieldName") for c in cols if c.get("fieldName")]
            except Exception as e:
                self.log.error(f"{dataset_id}: failed to fetch metadata — {e}")
                return None
        if not schema:
            self.log.warning(f"{dataset_id}: schema empty — skipping file write")
            return None
        self._ensure_schema_dir()
        outpath = self.schemadir / outfile
        outpath.write_text("\n".join(schema), encoding="utf-8")
        return outpath

    def _ensure_data_dir(self):
        self.datadir.mkdir(exist_ok=True)

    def _ensure_schema_dir(self):
        self.schemadir.mkdir(exist_ok=True)
