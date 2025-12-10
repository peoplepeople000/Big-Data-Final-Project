# Big Data Final Project — NYC Open Data Joinability

This repo explores the NYC Open Data (Socrata) catalog and builds a LAZO-based joinability graph across datasets. It downloads raw JSON for each dataset, sketches every column with MinHash + HyperLogLog-style cardinality, estimates Jaccard/containment via LAZO, and exports report-ready CSV/plots (with optional interactive graph).

## Repository Layout
- `domain.py` — Socrata helper (per-domain folders, download JSON, fetch schemas, logging).
- `download_nyc_all.py` — incremental NYC downloader (uses `Domain`), polite rate limiting.
- `build_sketches_all.py` — extract columns from downloaded JSON and build sketches for each column.
- `joinability_pipeline.py` — LAZO pairwise JS/containment over column sketches.
- `join_graph.py` — builds dataset-level graph (also offers pyvis HTML export if needed).
- `run_join_graph.py` — end-to-end: load sketches, compute joinability, export CSV + static charts.
- `run_joinability_example.py` — small demo over a sampled subset of columns.
- `column_extraction.py` — JSON ➜ pandas DataFrames ➜ (dataset, column) → Series dict (string/number cols only).
- `lazo_sketch.py` — ColumnSketch dataclass, MinHash (K=128), lightweight HLL-style cardinality.
- `lazo_estimator.py` — LAZO JS/JC estimation with error-correction heuristic.
- `datasets.ipynb` — early exploration: Socrata API usage, domain crawling, domain filtering.
- `socrata_domains.txt` / `socrata_domains_cities_only.txt` — discovered Socrata portals, filtered to city portals.
- Outputs (generated): `data.cityofnewyork.us/data/*.json`, `nyc_column_sketches.pkl`, `nyc_joinability_pairs.csv`, `reports/*.png`, and (optionally) `nyc_join_graph.html`.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Tokens (optional but recommended to avoid throttling): set `SODAPY_APPTOKEN` or per-domain creds in your shell env before running scripts.

## Workflow
1) **Download NYC datasets (raw JSON)**  
   ```bash
   python3 download_nyc_all.py
   ```  
   - Creates `data.cityofnewyork.us/data/data_<dataset_id>.json`.  
   - Script skips files already present; adjust `max_per_run` to throttle volume; sleeps between requests to be polite.

2) **Build column sketches**  
   ```bash
   python3 build_sketches_all.py
   ```  
   - Loads all JSONs, keeps string/number columns, drops columns with <50 non-null rows, builds MinHash+HLL sketches.  
   - Writes `nyc_column_sketches.pkl`.

3) **Compute joinability + reporting artifacts**  
   - Quick sampled demo (prints top pairs):  
     ```bash
     python3 run_joinability_example.py
     ```  
   - Full pipeline (CSV + charts, with optional HTML):  
     ```bash
     python3 run_join_graph.py
     ```  
     Uses `nyc_column_sketches.pkl`, computes LAZO JS/JC, builds a dataset-level graph, and exports:
       * `nyc_joinability_pairs.csv` — all surviving column pairs with JS/JC metrics.
       * `reports/top_datasets_by_partners.png` — datasets that act as join hubs.
       * `reports/top_dataset_pairs.png` — dataset pairs with the most joinable columns.
       * `reports/containment_distribution.png` — containment distribution across matches.
     If you still want the interactive HTML graph, call `join_graph.export_graph_to_html` inside `run_join_graph.py`.

## LAZO Highlights (used here)
- MinHash signature (K=128) for Jaccard; HLL-style cardinality per column.
- Alpha couples JS with |X|, |Y|: `alpha = (min - js_hat * max) / (1 + js_hat)`.
- Containment estimates derive from alpha; an error-correction heuristic caps containment at theoretical maxima and recomputes JS for consistency.

## Notes
- `datasets.ipynb` contains the original catalog crawl and domain filtering utilities.
- Scripts assume NYC domain folder name `data.cityofnewyork.us` as created by `Domain`.
- Generated files are large; `.gitignore` already excludes bulk data and env artifacts.
