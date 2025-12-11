import json
from collections import Counter
import matplotlib.pyplot as plt

with open("join/deepjoin_results.json", "r") as f:
    data = json.load(f)

# dataset counts
counts = Counter()

for entry in data:
    ds1 = entry["col1"].split(".")[0]
    ds2 = entry["col2"].split(".")[0]
    counts[ds1] += 1
    counts[ds2] += 1

top = counts.most_common(10)
datasets = [x[0] for x in top]
values = [x[1] for x in top]

plt.figure(figsize=(12, 6))
plt.barh(datasets, values, color="skyblue")
plt.title("Top 10 Most Joinable Datasets (DeepJoin)", fontsize=18)
plt.xlabel("Number of Strong Joinable Columns")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
