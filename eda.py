import os
import json
import boto3
import itertools
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# ---------------- CONFIG ---------------- #

BUCKET = "surface-quality-dataset"
PREFIX = "quack_v2_data_user_logs/"
OUTPUT_DIR = "pairwise_eda_report"

print("=== CONFIGURATION ===")
print(f"Bucket      : {BUCKET}")
print(f"Prefix      : {PREFIX}")
print(f"Output dir  : {OUTPUT_DIR}")
print("=====================\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------- #


def fetch_all_comparisons():
    print(">>> [1/10] Fetching comparison_data.json files from S3")

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    rows = []
    file_count = 0
    comparison_count = 0

    for page_idx, page in enumerate(
        paginator.paginate(Bucket=BUCKET, Prefix=PREFIX), start=1
    ):
        print(f"  - Processing S3 page {page_idx}")

        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("comparison_data.json"):
                continue

            file_count += 1
            user_id = obj["Key"].split("/")[-2]

            print(f"    • Reading file #{file_count}: {obj['Key']}")

            body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"]
            data = json.loads(body.read())

            num_comparisons = len(data["comparisons"])
            comparison_count += num_comparisons

            print(f"      → {num_comparisons} comparisons (user={user_id})")

            for c in data["comparisons"]:
                rows.append({
                    "user": user_id,
                    "img1": c["img1"],
                    "img2": c["img2"],
                    "choice": c["choice"],
                    "timestamp": c["timestamp"],
                    "time_taken": c.get("time_taken"),
                    "batch": c.get("batch_number")
                })

    print(f">>> Finished S3 fetch")
    print(f"    Total files read        : {file_count}")
    print(f"    Total comparisons found : {comparison_count}\n")

    return pd.DataFrame(rows)


# ---------------- DATA LOADING ---------------- #

print(">>> [2/10] Loading data into DataFrame")
df = fetch_all_comparisons()
print(f"    DataFrame shape: {df.shape}")
df.to_csv(f"{OUTPUT_DIR}/raw_comparisons.csv", index=False)
print("    Saved raw_comparisons.csv\n")


# ---------------- NORMALIZATION ---------------- #

print(">>> [3/10] Normalizing comparison outcomes")

def normalize(row):
    if row.choice == "left":
        return row.img1, row.img2, 1
    if row.choice == "right":
        return row.img2, row.img1, 1
    return row.img1, row.img2, 0  # draw

df[["winner", "loser", "is_win"]] = df.apply(
    lambda r: pd.Series(normalize(r)), axis=1
)

print("    Normalization complete")
print(f"    Win comparisons : {(df.is_win == 1).sum()}")
print(f"    Draw comparisons: {(df.is_win == 0).sum()}\n")


# ---------------- BASIC STATS ---------------- #

print(">>> [4/10] Computing global summary statistics")

summary = {
    "num_users": df.user.nunique(),
    "num_images": len(set(df.img1) | set(df.img2)),
    "num_comparisons": len(df),
    "draw_rate": float((df.is_win == 0).mean())
}

print(f"    Users       : {summary['num_users']}")
print(f"    Images      : {summary['num_images']}")
print(f"    Comparisons : {summary['num_comparisons']}")
print(f"    Draw rate   : {summary['draw_rate']:.3f}")

with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("    Saved summary.json\n")


# ---------------- IMAGE EXPOSURE ---------------- #

print(">>> [5/10] Computing image exposure statistics")

img_counts = pd.concat([df.img1, df.img2]).value_counts()
img_counts.to_csv(f"{OUTPUT_DIR}/image_exposure.csv")

print(f"    Images with ≥1 comparison: {len(img_counts)}")
print("    Saved image_exposure.csv\n")


# ---------------- PAIR AGREEMENT ---------------- #

print(">>> [6/10] Computing pairwise agreement & entropy")

pair_stats = defaultdict(list)

for (_, g) in df.groupby(["img1", "img2"]):
    votes = Counter(g.choice)
    pair_stats[(g.img1.iloc[0], g.img2.iloc[0])].append(votes)

pair_rows = []
for idx, (pair, votes_list) in enumerate(pair_stats.items(), start=1):
    total = Counter()
    for v in votes_list:
        total.update(v)

    probs = np.array(list(total.values())) / sum(total.values())

    pair_rows.append({
        "img1": pair[0],
        "img2": pair[1],
        "total_votes": sum(total.values()),
        "entropy": entropy(probs),
        "draw_rate": total.get("draw", 0) / sum(total.values())
    })

    if idx % 500 == 0:
        print(f"    Processed {idx} pairs")

pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(f"{OUTPUT_DIR}/pair_agreement.csv", index=False)

print(f"    Total unique pairs: {len(pair_df)}")
print("    Saved pair_agreement.csv\n")


# ---------------- USER AGREEMENT ---------------- #

print(">>> [7/10] Computing user-user agreement matrix")

def user_agreement(u1, u2):
    a = df[df.user == u1]
    b = df[df.user == u2]

    merged = pd.merge(
        a, b,
        on=["img1", "img2"],
        suffixes=("_a", "_b")
    )

    if len(merged) == 0:
        return np.nan

    return np.mean(merged.choice_a == merged.choice_b)

users = df.user.unique()
print(f"    Users: {len(users)}")

ua = pd.DataFrame(
    [[user_agreement(u1, u2) for u2 in users] for u1 in users],
    index=users,
    columns=users
)

ua.to_csv(f"{OUTPUT_DIR}/user_agreement_matrix.csv")
print("    Saved user_agreement_matrix.csv\n")


# ---------------- GRAPH ANALYSIS ---------------- #

print(">>> [8/10] Building comparison graph & SCC analysis")

G = nx.DiGraph()
for _, r in df[df.is_win == 1].iterrows():
    G.add_edge(r.winner, r.loser)

print(f"    Graph nodes: {G.number_of_nodes()}")
print(f"    Graph edges: {G.number_of_edges()}")

sccs = list(nx.strongly_connected_components(G))
scc_sizes = sorted([len(s) for s in sccs if len(s) > 1], reverse=True)

pd.DataFrame({"scc_size": scc_sizes}).to_csv(
    f"{OUTPUT_DIR}/strongly_connected_components.csv", index=False
)

print(f"    SCCs with size >1: {len(scc_sizes)}")
print("    Saved strongly_connected_components.csv\n")


# ---------------- CYCLE DETECTION ---------------- #

# print(">>> [9/10] Detecting cycles (may be slow)")

# cycles = list(nx.simple_cycles(G))
# cycle_df = pd.DataFrame({
#     "cycle_length": [len(c) for c in cycles]
# })

# cycle_df.to_csv(f"{OUTPUT_DIR}/cycles.csv", index=False)

# print(f"    Total cycles found: {len(cycle_df)}")
# print("    Saved cycles.csv")

# print("    Computing cycles per user")

user_cycles = defaultdict(int)
for user, g in df.groupby("user"):
    Gu = nx.DiGraph()
    for _, r in g[g.is_win == 1].iterrows():
        Gu.add_edge(r.winner, r.loser)
    user_cycles[user] = len(list(nx.simple_cycles(Gu)))

pd.Series(user_cycles).to_csv(
    f"{OUTPUT_DIR}/cycles_per_user.csv"
)

print("    Saved cycles_per_user.csv\n")

#cycles3 = [c for c in nx.simple_cycles(G) if len(c) == 3]


cycle_nodes = set().union(*[s for s in sccs if len(s) > 1])
#print(f"    3-cycles found: {len(cycles3)}")

print(f"    Unique nodes in cycles: {len(cycle_nodes)}\n")
scc_sizes = sorted([len(s) for s in sccs], reverse=True)
print(scc_sizes[:5])


# ---------------- DRAW ANALYSIS ---------------- #

print(">>> [10/10] Draw-rate and baseline ranking analysis")

draw_user = df.groupby("user")["is_win"].apply(lambda x: (x == 0).mean())
draw_user.to_csv(f"{OUTPUT_DIR}/draw_rate_per_user.csv")
print("    Saved draw_rate_per_user.csv")

wins = df[df.is_win == 1].groupby("winner").size()
losses = df[df.is_win == 1].groupby("loser").size()

copeland = wins.sub(losses, fill_value=0)
copeland.sort_values(ascending=False).to_csv(
    f"{OUTPUT_DIR}/copeland_scores.csv"
)

print("    Saved copeland_scores.csv\n")


# ---------------- REPORT ---------------- #

print(">>> Writing REPORT.md")

with open(f"{OUTPUT_DIR}/REPORT.md", "w") as f:
    f.write(f"""
# Pairwise Comparison EDA Report

## Dataset Summary
- Users: {summary['num_users']}
- Images: {summary['num_images']}
- Comparisons: {summary['num_comparisons']}
- Draw rate: {summary['draw_rate']:.3f}

## Observations
- Exposure imbalance across images
- Multiple SCCs → rank ambiguity
- Cycles present → non-transitive preferences
- User draw rates and consistency vary significantly
""")

print("\n=== EDA COMPLETE ===")
print(f"All outputs written to: {OUTPUT_DIR}")
