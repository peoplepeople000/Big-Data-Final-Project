from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import pandas as pd
from pyvis.network import Network


def build_dataset_graph(join_df: pd.DataFrame) -> nx.Graph:
    """
    Build a dataset-level undirected graph from column-level joinability results.

    - Nodes: dataset IDs
    - Edge between datasets A and B if at least one column pair is joinable.
    - Edge attributes:
        * num_joinable_columns: how many column pairs connect A and B
        * max_js: maximum JS among column pairs
        * max_jc: maximum JC (in either direction) among column pairs
    """
    G = nx.Graph()

    for _, row in join_df.iterrows():
        ds_left = row["left_dataset"]
        ds_right = row["right_dataset"]
        js = float(row["js"])
        jc_lr = float(row["jc_left_in_right"])
        jc_rl = float(row["jc_right_in_left"])
        max_jc = max(jc_lr, jc_rl)

        if ds_left == ds_right:
            # Skip self-loops at dataset level
            continue

        if not G.has_node(ds_left):
            G.add_node(ds_left)
        if not G.has_node(ds_right):
            G.add_node(ds_right)

        if G.has_edge(ds_left, ds_right):
            # Update existing edge stats
            data = G[ds_left][ds_right]
            data["num_joinable_columns"] += 1
            data["max_js"] = max(data["max_js"], js)
            data["max_jc"] = max(data["max_jc"], max_jc)
        else:
            G.add_edge(
                ds_left,
                ds_right,
                num_joinable_columns=1,
                max_js=js,
                max_jc=max_jc,
            )

    return G


def export_graph_to_html(
    G: nx.Graph,
    output_html: str = "nyc_join_graph.html",
    min_edge_weight: float = 0.0,
) -> None:
    """
    Export the dataset-level graph to an interactive HTML visualization using pyvis.

    - min_edge_weight: minimum max_jc required for an edge to be shown,
      to avoid clutter when there are many weak connections.
    """
    # Important: notebook=False avoids some template issues when running from a plain script.
    net = Network(height="800px", width="100%", notebook=False, directed=False)

    # Layout settings
    net.barnes_hut()

    # Add nodes
    for node in G.nodes():
        net.add_node(node, label=node)

    # Add edges with filtering
    for u, v, data in G.edges(data=True):
        weight = float(data.get("max_jc", 0.0))
        if weight < min_edge_weight:
            continue

        title = (
            f"{u} ↔ {v}<br>"
            f"num_joinable_columns: {data.get('num_joinable_columns', 0)}<br>"
            f"max_js: {data.get('max_js', 0.0):.3f}<br>"
            f"max_jc: {data.get('max_jc', 0.0):.3f}"
        )
        # Use weight to slightly affect edge thickness
        net.add_edge(u, v, value=weight, title=title)

    # Use write_html instead of show to avoid template/render bugs
    net.write_html(output_html, notebook=False)
    print(f"Graph exported to {output_html}")
