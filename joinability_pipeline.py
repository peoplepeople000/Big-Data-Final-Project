from __future__ import annotations

import itertools
from typing import Dict, Tuple

import pandas as pd

from lazo_estimator import estimate_js_jc_lazo
from lazo_sketch import ColumnSketch, ColumnSketcher


def build_all_sketches(column_dict: Dict[Tuple[str, str], pd.Series]) -> Dict[Tuple[str, str], ColumnSketch]:
    """
    Build sketches for all columns.
    Returns {(dataset_id, column_name): ColumnSketch}.
    """
    sketcher = ColumnSketcher()
    sketches: Dict[Tuple[str, str], ColumnSketch] = {}
    for (dataset_id, col_name), series in column_dict.items():
        sketches[(dataset_id, col_name)] = sketcher.build_sketch(series, column_name=col_name, dataset_id=dataset_id)
    return sketches


def compute_lazo_joinability(
    sketches: Dict[Tuple[str, str], ColumnSketch],
    js_threshold: float,
    jc_threshold: float,
) -> pd.DataFrame:
    """
    Compute pairwise joinability using LAZO estimates.
    Keeps pairs where JS or either containment exceeds given thresholds.
    """
    rows = []
    items = list(sketches.items())

    for (key_a, sketch_a), (key_b, sketch_b) in itertools.combinations(items, 2):
        js, jc_ab, jc_ba = estimate_js_jc_lazo(sketch_a, sketch_b)

        if (js >= js_threshold) or (jc_ab >= jc_threshold) or (jc_ba >= jc_threshold):
            rows.append(
                {
                    "left_dataset": key_a[0],
                    "left_column": key_a[1],
                    "right_dataset": key_b[0],
                    "right_column": key_b[1],
                    "js": js,
                    "jc_left_in_right": jc_ab,
                    "jc_right_in_left": jc_ba,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["jc_left_in_right", "jc_right_in_left", "js"], ascending=False).reset_index(drop=True)

# The resulting DataFrame can be used to build a join graph:
# nodes = columns; edges = pairs with high JS/JC; weights = JS or containment.
# Later steps could threshold edges, then group columns/datasets into connected components
# to suggest potential joins across NYC Open Data datasets.
