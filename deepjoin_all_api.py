import os
import json
import math
import requests
import numpy as np
import concurrent.futures
from sodapy import Socrata
from tqdm import tqdm

from encoder import embed, cosine
from utils import build_column_text
from sampler import sample_from_api

NYC_DOMAIN = "data.cityofnewyork.us"

MAX_DATASETS = 2000    # SAFE MAX FOR LOCAL 
LIMIT = 50             # sample size per column
TOP_K = 50             # final top joins
DATASET_BATCH = 10     # dataset batch size
CHECKPOINT_FILE = "deepjoin_checkpoint.json"

def get_all_datasets():
    client = Socrata(NYC_DOMAIN, None)
    datasets = client.datasets()
    ids = [d["resource"]["id"] for d in datasets]
    return ids[:MAX_DATASETS] if MAX_DATASETS else ids

def get_columns_for(dataset_id):
    url = f"https://{NYC_DOMAIN}/api/views/{dataset_id}.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        meta = r.json()
        cols = meta.get("columns", [])
        return [c.get("fieldName") for c in cols if c.get("fieldName")]
    except:
        return []

def sample_column(args):
    dataset_id, column = args
    vals = sample_from_api(dataset_id, column, LIMIT)
    return (column, vals)

def sample_dataset_columns(dataset_id):
    cols = get_columns_for(dataset_id)
    if not cols:
        return {}

    samples = {}
    tasks = [(dataset_id, c) for c in cols]

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for col, vals in executor.map(sample_column, tasks):
            samples[col] = vals

    return samples

def embed_dataset(dataset_id, samples):
    embeddings = {}
    for col, vals in samples.items():
        text = build_column_text(f"{dataset_id}.{col}", vals)
        vec = embed(text)
        embeddings[col] = vec
    return embeddings

def save_checkpoint(data):
    safe = {}

    for ds, cols in data.items():
        safe[ds] = {}
        for c, v in cols.items():
            safe[ds][c] = v.tolist()

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(safe, f)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {}

    with open(CHECKPOINT_FILE, "r") as f:
        raw = json.load(f)

    restored = {}
    for ds, cols in raw.items():
        restored[ds] = {}
        for c, v in cols.items():
            restored[ds][c] = np.array(v)

    return restored

def compute_join_scores_safe(dataset_embeddings):
    print("\n[Computing Join Scores — SAFE BLOCK MODE]\n")

    all_cols = []
    for ds, cols in dataset_embeddings.items():
        for col, emb in cols.items():
            all_cols.append((f"{ds}.{col}", emb))

    total = len(all_cols)
    print(f"Total Columns: {total}")
    print(f"Approx Comparisons: {total * (total - 1) // 2:,}")

    results = []

    BLOCK = 500   

    for i in tqdm(range(0, total, BLOCK), desc="Block Rows"):
        blockA = all_cols[i:i + BLOCK]

        for j in range(i, total, BLOCK):
            blockB = all_cols[j:j + BLOCK]

            for name1, e1 in blockA:
                for name2, e2 in blockB:
                    if name1 >= name2:
                        continue
                    score = cosine(e1, e2)
                    if score >= 0.90:
                        results.append((name1, name2, float(score)))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:TOP_K]

def run_deepjoin():
    dataset_ids = get_all_datasets()
    print(f"Total Datasets: {len(dataset_ids)}")

    dataset_embeddings = load_checkpoint()
    completed = set(dataset_embeddings.keys())

    for i in range(0, len(dataset_ids), DATASET_BATCH):
        batch = dataset_ids[i:i + DATASET_BATCH]
        print(f"\nPROCESSING BATCH {i // DATASET_BATCH + 1} ")

        for ds in tqdm(batch, desc="Datasets"):
            if ds in completed:
                continue

            samples = sample_dataset_columns(ds)
            if not samples:
                continue

            embeddings = embed_dataset(ds, samples)
            dataset_embeddings[ds] = embeddings

        save_checkpoint(dataset_embeddings)

    print("\nEMBEDDINGS COMPLETE\n")
    print("STARTING SAFE JOIN\n")

    top_matches = compute_join_scores_safe(dataset_embeddings)

    print("\nTOP 50 JOIN CANDIDATES \n")
    for a, b, score in top_matches:
        print(f"{a} <--> {b} | score={score:.4f}")

if __name__ == "__main__":
    run_deepjoin()
