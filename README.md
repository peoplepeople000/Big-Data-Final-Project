# Big Data Final Project — NYC Open Data Joinability

This repo explores the NYC Open Data (Socrata) catalog and builds a LAZO-based joinability graph across datasets. It downloads raw JSON for each dataset, sketches every column with MinHash + HyperLogLog-style cardinality, estimates Jaccard/containment via LAZO, and exports an interactive dataset-level graph.

## Repository Layout
- `domain.py` — Socrata helper (per-domain folders, download JSON, fetch schemas, logging).
- `download_nyc_all.py` — incremental NYC downloader (uses `Domain`), polite rate limiting.
- `build_sketches_all.py` — extract columns from downloaded JSON and build sketches for each column.
- `joinability_pipeline.py` — LAZO pairwise JS/containment over column sketches.
- `join_graph.py` — builds dataset-level graph + pyvis HTML export.
- `run_join_graph.py` — end-to-end: load sketches, compute joinability, emit HTML graph.
- `run_joinability_example.py` — small demo over a sampled subset of columns.
- `column_extraction.py` — JSON ➜ pandas DataFrames ➜ (dataset, column) → Series dict (string/number cols only).
- `lazo_sketch.py` — ColumnSketch dataclass, MinHash (K=128), lightweight HLL-style cardinality.
- `lazo_estimator.py` — LAZO JS/JC estimation with error-correction heuristic.
- `datasets.ipynb` — early exploration: Socrata API usage, domain crawling, domain filtering.
- `socrata_domains.txt` / `socrata_domains_cities_only.txt` — discovered Socrata portals, filtered to city portals.
- Outputs (generated): `data.cityofnewyork.us/data/*.json`, `nyc_column_sketches.pkl`, `nyc_join_graph.html`.

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
   python download_nyc_all.py
   ```  
   - Creates `data.cityofnewyork.us/data/data_<dataset_id>.json`.  
   - Script skips files already present; adjust `max_per_run` to throttle volume; sleeps between requests to be polite.

2) **Build column sketches**  
   ```bash
   python build_sketches_all.py
   ```  
   - Loads all JSONs, keeps string/number columns, drops columns with <50 non-null rows, builds MinHash+HLL sketches.  
   - Writes `nyc_column_sketches.pkl`.

3) **Compute joinability + graph**  
   - Quick sampled demo (prints top pairs):  
     ```bash
     python run_joinability_example.py
     ```  
   - Full pipeline to HTML graph:  
     ```bash
     python run_join_graph.py
     ```  
     Uses `nyc_column_sketches.pkl`, computes LAZO JS/JC, builds dataset-level graph, exports `nyc_join_graph.html` (open in browser).

## LAZO Highlights (used here)
- MinHash signature (K=128) for Jaccard; HLL-style cardinality per column.
- Alpha couples JS with |X|, |Y|: `alpha = (min - js_hat * max) / (1 + js_hat)`.
- Containment estimates derive from alpha; an error-correction heuristic caps containment at theoretical maxima and recomputes JS for consistency.

## Notes
- `datasets.ipynb` contains the original catalog crawl and domain filtering utilities.
- Scripts assume NYC domain folder name `data.cityofnewyork.us` as created by `Domain`.
- Generated files are large; `.gitignore` already excludes bulk data and env artifacts.
