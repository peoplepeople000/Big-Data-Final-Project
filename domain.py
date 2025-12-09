import re
import requests
import json
import logging
import pandas as pd
from typing import Dict, Tuple
import time
from pathlib import Path
from sodapy import Socrata
import pandas as pd
from datetime import datetime
from requests.exceptions import Timeout as RequestsTimeout

class Domain:
    __slots__ = ("domain", "sanitized", "views", "token", "timeout", "client", "ids", "base", "parent_dir", "datadir", "schemadir", "metadatadir", "log", "logdir")
    
    def __init__(self, domain, token=None, timeout=10, parent_dir="all_city_data"):
        self.domain = domain
        self.sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", domain)
        self.views = f"https://{domain}/views.json"
        self.token = token
        self.timeout = timeout
        self.client = None
        self.ids = None
        self.parent_dir = Path(parent_dir)
        self.parent_dir.mkdir(exist_ok=True)
        self.base = self.parent_dir / self.sanitized
        self.datadir = self.base / "data"
        self.schemadir = self.base / "schema"
        self.metadatadir = self.base / "metadata"
        self.logdir = self.base / "logs"
        self.log = None
    
    def setup_logger(self):
        self.base.mkdir(exist_ok=True)
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

    def city_datasets_count(self, max_retries=3, max_timeout=60):
        """Get dataset count with retry logic for timeouts."""
        params = {"count": True}
        original_timeout = self.timeout
        
        for attempt in range(max_retries):
            try:
                # Increase timeout on retries
                self.timeout = min(original_timeout * (attempt + 1), max_timeout)
                
                response = requests.get(self.views, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                # Reset timeout and return
                self.timeout = original_timeout
                return data.get("count", -1)
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    self._ensure_logger()
                    self.log.warning(f"Count request timeout (attempt {attempt+1}/{max_retries}), retrying with {self.timeout}s timeout")
                    time.sleep(2)
                else:
                    self._ensure_logger()
                    self.log.error(f"Failed to get count after {max_retries} attempts: {e}")
                    self.timeout = original_timeout
                    return -1
                    
            except Exception as e:
                self._ensure_logger()
                self.log.error(f"Error getting dataset count: {e}")
                self.timeout = original_timeout
                return -1
        
        self.timeout = original_timeout
        return -1
    
    def city_datasets_ids(self):
        if not self.ids:
            self._setup_client()     
            try:
                datasets = self.client.datasets()
                ids = [d["resource"]["id"] for d in datasets]
            except Exception as e:
                self._ensure_logger()
                self.log.error(f"Failed to fetch dataset IDs via sodapy: {e}")
                ids = []
            
            new_ids = self._dataset_ids_backup(ids)
            self.ids = new_ids if new_ids else ids
        return self.ids

    def _dataset_ids_backup(self, ids):
        """Backup method using direct API calls if sodapy fails or returns wrong count."""
        try:
            expected_count = self.city_datasets_count()
        except Exception as e:
            self._ensure_logger()
            self.log.warning(f"Could not get dataset count: {e}")
            return ids  # Return what we have
        
        if len(ids) < expected_count:
            self._ensure_logger()
            self.log.warning(f"{self.domain}: sodapy returned {len(ids)} datasets but count is {expected_count}. Fetching via API.")
            ids = []
            page = 1
            
            while True:
                params = {"limit": 200, "page": page}
                
                try:
                    response = requests.get(self.views, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:  # No more pages
                        break
                    
                    # Extract just the ids
                    for item in data:
                        ids.append(item["id"])
                    
                    page += 1
                    
                except Exception as e:
                    self.log.error(f"Failed to fetch page {page}: {e}")
                    break  # Stop trying if we hit an error
        
        return ids
    
    def write_dataset_ids_to_file(self, filepath=None):
        self.ids = self.city_datasets_ids()
        if filepath is None:
            filepath = f"{self.sanitized}_ids.txt"
        full_path = self.base / filepath
        with full_path.open("w", encoding="utf-8") as f:
            for dataset_id in self.ids:
                f.write(dataset_id + "\n")
        return full_path

    def _set_up_ids(self):
        # If we already have ids in memory, yield them
        if self.ids:
            yield from self.ids
            return
        
        # Check if file exists
        identifiers_file = self.base / f"{self.sanitized}_ids.txt"
        if identifiers_file.exists():
            # Yield ids from file as we read them
            with identifiers_file.open() as f:
                for line in f:
                    dataset_id = line.strip()
                    yield dataset_id
        else:
            # Fetch from API and yield
            yield from self.city_datasets_ids()

    def download_all_raw_dataset(self):
        for dataset_id in self._set_up_ids():
            outfile = f"data_{dataset_id}.json"
            outpath = self.datadir / outfile
            if outpath.exists():
                continue
            self.download_raw_dataset(dataset_id)
    
    def download_raw_dataset(self, dataset_id, outfile=None):
        self._ensure_logger()
        url = f"https://{self.domain}/resource/{dataset_id}.json"
        try:
            resp = requests.get(url, timeout=self.timeout)
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

    def download_all_schema(self):
        for dataset_id in self._set_up_ids():
            outfile = f"schema_{dataset_id}.txt"
            outpath = self.schemadir / outfile
            if outpath.exists():
                continue
            self.download_schema(dataset_id)

    def download_schema(self, dataset_id, outfile=None):
        self._ensure_logger()
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
                self._setup_client()
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
    
    def load_dataset_to_df(self, dataset_id):
        self._ensure_logger()
        data_path = (self.datadir / f"{dataset_id}.json")
        if data_path.exists():
            with data_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # If the file is not a list of dicts, wrap to avoid crash.
            if not isinstance(data, list):
                data = []
            return pd.DataFrame.from_records(data)
        else: # If the file is not downloaded, fetch it via the api
            try:
                self._setup_client()
                table = self.client.get(dataset_id)
            except requests.exceptions.HTTPError as e:
                self.log.error(f"{dataset_id}: Socrata error — {e}")
                return pd.DataFrame([]) # return clean empty DataFrame
            except Exception as e:
                # broad fallback so unexpected errors never crash the loader
                self.log.error(f"{dataset_id}: Unexpected error — {e}")
                return pd.DataFrame([])
            if not isinstance(table, list):
                table = []
            return pd.DataFrame.from_records(table)
        
    def extract_column_from_dataset(self, dataset_id) -> Dict[Tuple[str, str], pd.Series]: 
        df = self.load_dataset_to_df(dataset_id)
        if df.empty:
            return None
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        num_cols = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
        cols = list(dict.fromkeys(str_cols + num_cols))  # preserve order, avoid duplicates
        column_dict = {}
        for col in cols:
            series = df[col]
            column_dict[(dataset_id, col)] = series
        return column_dict

    def extract_columns_for_domain(self):
        """
        Read all downloaded JSON files for a domain and return a dict:
            { (dataset_id, column_name): Series }
        Only string-like and numeric columns are kept.
        """
        for dataset_id in self._set_up_ids():
            yield self.extract_column_from_dataset(dataset_id)
    
    def get_relevant_metadata(self, dataset_id, outfile=None, retry_timeout=None):
        self._ensure_logger()
        if outfile is None:
            outfile = f"metadata_{dataset_id}.txt"
        
        # Use custom timeout if provided, otherwise use instance timeout
        timeout_to_use = retry_timeout if retry_timeout is not None else self.timeout
        
        try:
            self._setup_client()
            # Temporarily set timeout for this request
            original_timeout = self.client.timeout
            self.client.timeout = timeout_to_use
            
            metadata = self.client.get_metadata(dataset_id)
            
            # Reset timeout
            self.client.timeout = original_timeout
            
            # Build relevant metadata structure
            relevant_metadata = {
                "category": metadata.get("category"),
                "format": metadata.get("viewType"),
                "tags": metadata.get("tags"),
                "downloadCount": metadata.get("downloadCount"),
                "viewCount": metadata.get("viewCount"),
                "createdAt": metadata.get("createdAt"),
                "publicationDate": metadata.get("publicationDate"),
                "updatedAt": metadata.get("viewLastModified")
            }
            
            # Only process columns for tabular datasets that are queryable
            if relevant_metadata.get("format") == "tabular":
                # Check if this is a queryable dataset
                asset_type = metadata.get("assetType", "")
                
                # Skip column stats for non-queryable asset types
                if asset_type in ['filter', 'href', 'external', 'link', 'file', 'chart', 'map', 'story']:
                    self.log.info(f"{dataset_id}: Non-queryable asset type '{asset_type}', skipping column stats")
                    
                    # Still include basic schema info if available
                    cols = metadata.get("columns") or []
                    if cols:
                        relevant_columns = [
                            {
                                "name": c.get("fieldName"), 
                                "type": c.get("dataTypeName"),
                            }
                            for c in cols if c.get("fieldName")
                        ]
                        relevant_metadata["columns"] = relevant_columns
                        
                        # Use metadata's rowCount if available
                        if "rowCount" in metadata:
                            relevant_metadata["rowCount"] = int(metadata["rowCount"])
                    
                    return relevant_metadata
                
                # For queryable datasets (assetType == 'dataset'), proceed normally
                cols = metadata.get("columns") or []
                relevant_columns = [
                    {
                        "name": c.get("fieldName"), 
                        "type": c.get("dataTypeName"),
                    }
                    for c in cols if c.get("fieldName")
                ]
                
                if relevant_columns:
                    try:
                        counts = self.fetch_all_column_stats(dataset_id, cols)
                        
                        if counts and "total_rows" in counts:
                            relevant_metadata["rowCount"] = int(counts["total_rows"])
                            
                            for col in relevant_columns:
                                name = col["name"]
                                col["nulls"] = int(counts.get(f"{name}_nulls", 0))
                                col["semantic_nulls"] = int(counts.get(f"{name}_semantic_nulls", 0))
                            
                            relevant_metadata["columns"] = relevant_columns
                        else:
                            self.log.warning(f"{dataset_id}: column stats incomplete, skipping column info")
                            
                    except Exception as e:
                        self.log.warning(f"{dataset_id}: failed to fetch column stats — {e}")
                else:
                    self.log.warning(f"{dataset_id}: schema empty — skipping")
            
            return relevant_metadata
            
        except Exception as e:
            error_str = str(e).lower()
            is_timeout = 'timeout' in error_str or 'timed out' in error_str
            is_connection = 'connection' in error_str or 'remote' in error_str
            
            if is_timeout:
                self.log.error(f"{dataset_id}: timeout after {timeout_to_use}s — {e}")
            elif is_connection:
                self.log.error(f"{dataset_id}: connection error — {e}")
            else:
                self.log.error(f"{dataset_id}: failed to fetch metadata — {e}")
            
            return None
    
    def fetch_all_column_stats(self, dataset_id, cols):
        """
        Runs multiple smaller SoQL queries, skipping problematic column types.
        """
        results = {}
        try:
            results["total_rows"] = self._fetch_total_row_count(dataset_id)
        except Exception as e:
            self.log.error(f"{dataset_id}: failed to fetch total row count — {e}")
            return None  # Can't proceed without row count
        
        # Filter out problematic columns
        usable_cols = []
        SKIP_TYPES = {"point", "location", "polygon", "multipolygon", "line", "multiline", "multipoint"}
        
        for col in cols:
            field = col.get("fieldName", "")
            dtype = col.get("dataTypeName", "")
            
            # Skip computed columns (start with :@ or @)
            if field.startswith(":@") or field.startswith("@"):
                self.log.debug(f"{dataset_id}: Skipping computed column '{field}'")
                safe_alias = field.replace(":", "_").replace("@", "_").replace("-", "_")
                results[f"{safe_alias}_nulls"] = 0
                results[f"{safe_alias}_semantic_nulls"] = 0
            # Skip geometry/location columns
            elif dtype in SKIP_TYPES:
                self.log.debug(f"{dataset_id}: Skipping geometry column '{field}' (type: {dtype})")
                safe_alias = field.replace(":", "_").replace("@", "_").replace("-", "_")
                results[f"{safe_alias}_nulls"] = 0
                results[f"{safe_alias}_semantic_nulls"] = 0
            else:
                usable_cols.append(col)
        
        if not usable_cols:
            self.log.warning(f"{dataset_id}: No queryable columns found")
            return results
        
        # Process columns with dynamic chunking
        current_chunk = []
        url_base = f"https://{self.domain}/resource/{dataset_id}.json"
        MAX_URL_LENGTH = 1200  # More conservative
        
        for col in usable_cols:
            # Try adding this column to current chunk
            test_chunk = current_chunk + [col]
            select_clause = self._build_chunk_select_clause(test_chunk)
            
            # Estimate URL length (add overhead for encoding)
            estimated_length = len(url_base) + len(select_clause) + 100
            
            if estimated_length > MAX_URL_LENGTH and current_chunk:
                # Current chunk would be too long, process it first
                self._process_chunk_with_fallback(dataset_id, current_chunk, results)
                current_chunk = [col]  # Start new chunk
            else:
                current_chunk.append(col)
        
        # Process final chunk
        if current_chunk:
            self._process_chunk_with_fallback(dataset_id, current_chunk, results)
        
        return results

    def _process_chunk_with_fallback(self, dataset_id, chunk, results):
        """Process a chunk with fallback to individual columns."""
        try:
            self._fetch_chunk_stats(dataset_id, chunk, results)
        except Exception as e:
            self.log.warning(f"{dataset_id}: Chunk failed ({len(chunk)} cols), trying individually - {e}")
            # Try one column at a time
            for single_col in chunk:
                field = single_col["fieldName"]
                safe_alias = field.replace(":", "_").replace("@", "_").replace("-", "_")
                try:
                    self._fetch_chunk_stats(dataset_id, [single_col], results)
                except Exception as e2:
                    self.log.warning(f"{dataset_id}: Column '{field}' failed - {e2}")
                    # Set defaults for failed columns
                    results[f"{safe_alias}_nulls"] = 0
                    results[f"{safe_alias}_semantic_nulls"] = 0

    def _fetch_chunk_stats(self, dataset_id, chunk, results):
        """Helper to fetch stats for a single chunk using GET."""
        select_clause = self._build_chunk_select_clause(chunk)
        url = f"https://{self.domain}/resource/{dataset_id}.json"
        params = {"$select": select_clause}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        row = r.json()[0]
        results.update(row)

    def _build_chunk_select_clause(self, columns):
        """
        Builds a SELECT clause with simplified semantic null checks.
        """
        select_parts = []

        for col in columns:
            field = col["fieldName"]
            dtype = col["dataTypeName"]
            
            quoted_field = self._quote_field_name(field)
            safe_alias = field.replace(":", "_").replace("@", "_").replace("-", "_")
            
            # null count
            select_parts.append(
                f"(count(*) - count({quoted_field})) AS {safe_alias}_nulls"
            )
            
            # Simplified semantic nulls for text fields only
            TEXT_LIKE_TYPES = {"text", "url", "email", "phone", "html"}
            if dtype in TEXT_LIKE_TYPES:
                # Simpler check - just trim and empty string
                semantic = (
                    f"sum(CASE WHEN "
                    f"{quoted_field} IS NULL OR "
                    f"trim({quoted_field}) = '' "
                    f"THEN 1 ELSE 0 END) AS {safe_alias}_semantic_nulls"
                )
            else:
                semantic = f"0 AS {safe_alias}_semantic_nulls"
            select_parts.append(semantic)
        
        return ", ".join(select_parts)

    def _quote_field_name(self, field):
        """
        Quote field names that need it (contain special characters).
        """
        special_chars = {':', '@', '-', ' ', '.', '/', '\\', '(', ')'}
        if any(c in field for c in special_chars):
            escaped = field.replace('`', '``')
            return f"`{escaped}`"
        return field

    def _chunk(self, columns, size):
        for i in range(0, len(columns), size):
            yield columns[i:i+size]

    def _fetch_total_row_count(self, dataset_id):
        url = f"https://{self.domain}/resource/{dataset_id}.json"
        params = {"$select": "count(*) AS total_rows"}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return int(r.json()[0]["total_rows"])

    def download_all_relevant_metadata(self, max_retries=2, progressive_timeout=True, max_timeout=60):
        """
        Download metadata for all datasets with optional retry and progressive timeout.
        
        Args:
            max_retries: Number of times to retry failed downloads
            progressive_timeout: If True, increase timeout on retry
            max_timeout: Maximum timeout to use
        """
        self._ensure_meta_dir()
        
        failed_downloads = []
        successful = 0
        skipped = 0
        
        for dataset_id in self._set_up_ids():
            outfile = f"metadata_{dataset_id}.json"
            outpath = self.metadatadir / outfile
            
            # Skip if already exists and not empty
            if outpath.exists():
                try:
                    with outpath.open("r") as f:
                        existing = json.load(f)
                        if existing:  # Not empty
                            skipped += 1
                            continue
                except:
                    pass  # File exists but corrupted, re-download
            
            # Try downloading with retries
            metadata = None
            current_timeout = self.timeout
            
            for attempt in range(max_retries):
                if progressive_timeout and attempt > 0:
                    current_timeout = min(current_timeout * 1.5, max_timeout)
                
                metadata = self.get_relevant_metadata(dataset_id, retry_timeout=current_timeout)
                
                if metadata:
                    successful += 1
                    break
                
                # If failed and not last attempt, wait before retry
                if attempt < max_retries - 1:
                    time.sleep(1)
            
            # Save if we got metadata
            if metadata:
                with outpath.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            else:
                failed_downloads.append(dataset_id)
        
        # Log summary
        if hasattr(self, 'log'):
            self.log.info(f"Download complete: {successful} successful, {skipped} skipped, {len(failed_downloads)} failed")
            if failed_downloads:
                self.log.warning(f"Failed dataset IDs: {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")
        
        return {
            'successful': successful,
            'skipped': skipped,
            'failed': failed_downloads
        }

    def summarize_metadata(self): # relevant metadata must be downloaded first
        outpath = self.metadatadir / "summary"
        outpath.mkdir(exist_ok=True)
        
        # Check if there are any JSON files in metadatadir
        json_files = list(self.metadatadir.glob("*.json"))
        
        if not json_files:
            print(f"No metadata files found in {self.metadatadir}. Please run download_all_relevant_metadata() first.")
            return
        
        print(f"Found {len(json_files)} metadata files to analyze...")
        
        # Load all JSON files into a list
        metadata_list = []
        invalid_count = 0
        
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                    # Validate the data
                    if data is None:
                        invalid_count += 1
                        continue
                    
                    if not isinstance(data, dict):
                        invalid_count += 1
                        continue
                    
                    if not data:  # Empty dict
                        invalid_count += 1
                        continue
                    
                    # Optionally check for required fields
                    # if 'category' not in data and 'format' not in data:
                    #     invalid_count += 1
                    #     continue
                    
                    metadata_list.append(data)
                    
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            except Exception as e:
                print(f"Warning: Could not process {file}: {e}")
                invalid_count += 1
                continue
        
        if not metadata_list:
            print(f"No valid metadata found in {self.metadatadir}.")
            return
        
        if invalid_count > 0:
            print(f"Warning: Skipped {invalid_count} invalid/empty metadata files")
        
        print(f"Processing {len(metadata_list)} valid metadata files...")
        
        df = pd.DataFrame(metadata_list)
        
        # Tag Analysis
        tags = df['tags'].explode()
        tag_counts = tags.value_counts().reset_index()
        tag_counts.columns = ['tag', 'count']
        # tag_counts.to_json(outpath / "tag_counts.json", orient='records', indent=2)
        tag_dct = dict(zip(tag_counts["tag"], tag_counts["count"]))
        with open(outpath / "tag_counts.json", "w") as f:
            json.dump(tag_dct, f, indent=2)
        
        # Row Count Buckets
        df['row_bucket'] = pd.cut(
            df['rowCount'],
            bins=[0, 1000, 10000, 100000, 1000000, 10000000, float('inf')],
            labels=['0-1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M+'],
            right=False
        )
        bucket_counts = df['row_bucket'].value_counts().reset_index()
        bucket_counts.columns = ['bucket', 'count']
        bucket_counts = bucket_counts.sort_values('bucket')
        # bucket_counts.to_json(outpath / "row_buckets.json", orient='records', indent=2)
        row_dct = dict(zip(bucket_counts["bucket"], bucket_counts["count"]))
        with open(outpath / "row_buckets.json", "w") as f:
            json.dump(row_dct, f, indent=2)
        
        # View Count Buckets
        df['view_bucket'] = pd.cut(
            df['viewCount'],
            bins=[0, 100, 1000, 10000, float('inf')],
            labels=['0-100', '100-1K', '1K-10K', '10K+'],
            right=False
        )
        view_bucket_counts = df['view_bucket'].value_counts().reset_index()
        view_bucket_counts.columns = ['bucket', 'count']
        view_bucket_counts = view_bucket_counts.sort_values('bucket')
        # view_bucket_counts.to_json(outpath / "view_buckets.json", orient='records', indent=2)
        view_dct = dict(zip(view_bucket_counts["bucket"], view_bucket_counts["count"]))
        with open(outpath / "view_buckets.json", "w") as f:
            json.dump(view_dct, f, indent=2)

        # Download Count Buckets
        df['download_bucket'] = pd.cut(
            df['downloadCount'],
            bins=[0, 100, 1000, 10000, float('inf')],
            labels=['0-100', '100-1K', '1K-10K', '10K+'],
            right=False
        )
        download_bucket_counts = df['download_bucket'].value_counts().reset_index()
        download_bucket_counts.columns = ['bucket', 'count']
        download_bucket_counts = download_bucket_counts.sort_values('bucket')
        # download_bucket_counts.to_json(outpath / "download_buckets.json", orient='records', indent=2)
        download_dct = dict(zip(download_bucket_counts["bucket"], download_bucket_counts["count"]))
        with open(outpath / "download_buckets.json", "w") as f:
            json.dump(download_dct, f, indent=2)
        
        # Category distribution
        category_counts = df['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        # category_counts.to_json(outpath / "categories.json", orient='records', indent=2)
        category_dct = dict(zip(category_counts["category"], category_counts["count"]))
        with open(outpath / "categories.json", "w") as f:
            json.dump(category_dct, f, indent=2)
        
        # Format distribution
        format_counts = df['format'].value_counts().reset_index()
        format_counts.columns = ['format', 'count']
        # format_counts.to_json(outpath / "formats.json", orient='records', indent=2)
        format_dct = dict(zip(format_counts["format"], format_counts["count"]))
        with open(outpath / "formats.json", "w") as f:
            json.dump(format_dct, f, indent=2)
        
        # Age of publication in months
        current_time = datetime.now().timestamp()

        # Handle missing publicationDate values
        df['age_months'] = ((current_time - df['publicationDate']) / (30.44 * 24 * 3600))

        # Fill NaN values before converting to int
        df['age_months'] = df['age_months'].fillna(-1).astype(int)

        # Now create the counts, optionally filtering out invalid ages
        age_counts = df[df['age_months'] >= 0]['age_months'].value_counts().reset_index()
        age_counts.columns = ['age_months', 'count']
        age_counts = age_counts.sort_values('age_months')
        # age_counts.to_json(outpath / "publication_age.json", orient='records', indent=2)
        age_dct = dict(zip(age_counts["age_months"], age_counts["count"]))
        with open(outpath / "publication_age.json", "w") as f:
            json.dump(age_dct, f, indent=2)
        
        # Number of attributes (columns) per dataset
        # Handle missing columns (NaN values)
        df['num_attributes'] = df['columns'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        df['attr_bucket'] = pd.cut(
            df['num_attributes'],
            bins=[0, 10, 20, 30, 40, 50, float('inf')],
            labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50+'],
            right=False
        )
        attr_counts = df['attr_bucket'].value_counts().reset_index()
        attr_counts.columns = ['attribute_bucket', 'count']
        attr_counts = attr_counts.sort_values('attribute_bucket')
        # attr_counts.to_json(outpath / "attribute_counts.json", orient='records', indent=2)
        attr_dct = dict(zip(attr_counts["attribute_bucket"], attr_counts["count"]))
        with open(outpath / "attribute_counts.json", "w") as f:
            json.dump(attr_dct, f, indent=2)
        
        # Types of attributes (distribution of column types)
        all_column_types = []
        for columns_list in df['columns']:
            # Skip if columns is NaN (float)
            if isinstance(columns_list, list):
                for col in columns_list:
                    all_column_types.append(col['type'])

        type_counts = pd.Series(all_column_types).value_counts().reset_index()
        type_counts.columns = ['type', 'count']
        # type_counts.to_json(outpath / "column_types.json", orient='records', indent=2)
        type_dct = dict(zip(type_counts["type"], type_counts["count"]))
        with open(outpath / "column_types.json", "w") as f:
            json.dump(type_dct, f, indent=2)
        
        # Table sparseness (percentage of semantic_nulls across all columns)
        sparseness_data = []
        for idx, row in df.iterrows():
            row_count = row['rowCount']
            columns_list = row['columns']
            
            # Skip if columns is NaN or rowCount is invalid
            if isinstance(columns_list, list) and row_count > 0:
                null_percentages = []
                for col in columns_list:
                    null_pct = (col['semantic_nulls'] / row_count * 100)
                    null_percentages.append(null_pct)
                
                if null_percentages:
                    avg_sparseness = sum(null_percentages) / len(null_percentages)
                    sparseness_data.append(avg_sparseness)

        if sparseness_data:  # Only create analysis if we have data
            sparseness_df = pd.DataFrame({'avg_sparseness': sparseness_data})
            sparseness_df['sparseness_bucket'] = pd.cut(
                sparseness_df['avg_sparseness'],
                bins=[0, 1, 5, 10, 25, 50, 100],
                labels=['< 1% sparse', '1-5% sparse', '5-10% sparse', '10-25% sparse', '25-50% sparse', '50%+ sparse'],
                right=False
            )
            
            sparseness_counts = sparseness_df['sparseness_bucket'].value_counts().reset_index()
            sparseness_counts.columns = ['sparseness_bucket', 'count']
            sparseness_counts = sparseness_counts.sort_values('sparseness_bucket')
            # sparseness_counts.to_json(outpath / "table_sparseness.json", orient='records', indent=2)
            sparseness_dct = dict(zip(sparseness_counts["sparseness_bucket"], sparseness_counts["count"]))
            with open(outpath / "table_sparseness.json", "w") as f:
                json.dump(sparseness_dct, f, indent=2)
        else:
            print("No sparseness data available")
        
        print(f"Analysis complete! Results saved to {outpath}")
    
    def _setup_client(self):
        if not self.client:
            self.client = Socrata(self.domain, self.token, timeout=self.timeout)
        elif self.timeout > self.client.timeout:
            self.client.timeout = self.timeout

    def _ensure_meta_dir(self):
        self.base.mkdir(exist_ok=True)
        self.metadatadir.mkdir(exist_ok=True)

    def _ensure_data_dir(self):
        self.base.mkdir(exist_ok=True)
        self.datadir.mkdir(exist_ok=True)

    def _ensure_schema_dir(self):
        self.base.mkdir(exist_ok=True)
        self.schemadir.mkdir(exist_ok=True)

    def _ensure_logger(self):
        if self.log == None:
            self.log = self.setup_logger()
