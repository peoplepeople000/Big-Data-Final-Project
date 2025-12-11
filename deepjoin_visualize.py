import json
import matplotlib.pyplot as plt
import networkx as nx

# Load DeepJoin edges
with open("join/deepjoin_results.json", "r") as f:
    edges = json.load(f)

# Build column-level similarity graph
G = nx.Graph()

for item in edges:
    c1, c2, score = item["col1"], item["col2"], float(item["score"])
    if score >= 0.9993:   # keep strong matches only
        G.add_edge(c1, c2, weight=score)

# Layout
pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)

plt.figure(figsize=(22, 14))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=80, node_color="skyblue", alpha=0.85)

# Draw edges
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.25, edge_color="gray")

# --- Label only dataset IDs ---
labels = {node: node.split(".")[0] for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_color="black")

plt.title("DeepJoin Column Similarity Graph", fontsize=20)
plt.axis("off")
plt.tight_layout()
plt.show()
