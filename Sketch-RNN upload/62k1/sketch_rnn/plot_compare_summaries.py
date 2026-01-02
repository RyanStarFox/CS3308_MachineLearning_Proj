#!/usr/bin/env python3
"""Read per-category summary_*.csv files in `compare_results/*` and plot a 2x5 grid
of Top-1..Top-5 accuracies, saving to `compare_results/topk_combined.png`.
"""
import os
import sys
import csv
import io
import numpy as np

# optional pandas: use if available, otherwise fall back to csv reader
try:
    import pandas as pd
except Exception:
    pd = None

import matplotlib
# use a non-interactive backend so script works on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARE_DIR = os.path.join(BASE_DIR, 'compare_results')
OUT_PATH = os.path.join(COMPARE_DIR, 'topk_combined.png')

if not os.path.isdir(COMPARE_DIR):
    raise SystemExit(f"compare_results directory not found: {COMPARE_DIR}")

# gather categories (directories)
cats = sorted([d for d in os.listdir(COMPARE_DIR) if os.path.isdir(os.path.join(COMPARE_DIR, d))])
# keep first 10 (two rows x five cols)
cats = cats[:10]

ncols = 5
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
axes_flat = axes.flatten()

for idx in range(nrows * ncols):
    ax = axes_flat[idx]
    if idx < len(cats):
        cat = cats[idx]
        summary_path = os.path.join(COMPARE_DIR, cat, f'summary_{cat}.csv')
        if not os.path.exists(summary_path):
            # fallback: find any summary_*.csv in the directory
            found = None
            for f in os.listdir(os.path.join(COMPARE_DIR, cat)):
                if f.startswith('summary_') and f.endswith('.csv'):
                    found = os.path.join(COMPARE_DIR, cat, f)
                    break
            if found:
                summary_path = found

        ranks = []
        if os.path.exists(summary_path):
            try:
                if pd is not None:
                    df = pd.read_csv(summary_path)
                    # try common column names for rank
                    rank_col = None
                    for c in ['rank_of_correct', 'rank', 'rank_of_match', 'rank_of_correct '] :
                        if c in df.columns:
                            rank_col = c
                            break
                    if rank_col is None:
                        # fallback: choose an integer column with small max value
                        numcols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
                        for c in numcols:
                            try:
                                if df[c].max() <= 20:  # likely a rank
                                    rank_col = c
                                    break
                            except Exception:
                                continue
                    if rank_col is not None:
                        ranks = df[rank_col].astype(int).tolist()
                    else:
                        ranks = []
                else:
                    # simple CSV fallback: look for a column named rank_of_correct or rank
                    with open(summary_path, 'r', encoding='utf-8') as fh:
                        rdr = csv.reader(fh)
                        hdr = next(rdr)
                        # normalize header names
                        hdr_clean = [h.strip() for h in hdr]
                        rank_idx = None
                        for name in ['rank_of_correct', 'rank', 'rank_of_match']:
                            if name in hdr_clean:
                                rank_idx = hdr_clean.index(name)
                                break
                        if rank_idx is not None:
                            for row in rdr:
                                try:
                                    ranks.append(int(row[rank_idx]))
                                except Exception:
                                    continue
            except Exception as e:
                print(f"Failed to read {summary_path}: {e}")
                ranks = []
        else:
            print(f"No summary file for category {cat} (looked for {summary_path})")

        # compute top1..top5
        topk_acc = []
        for k in range(1, 6):
            if len(ranks) > 0:
                topk_acc.append(sum(1 for r in ranks if r <= k) / len(ranks))
            else:
                topk_acc.append(0.0)

        labels = ['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5']
        ax.bar(labels, topk_acc, color='C0')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy' if idx % ncols == 0 else '')
        ax.set_title(cat)
        # annotate
        for i, v in enumerate(topk_acc):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    else:
        ax.axis('off')

plt.tight_layout()
# ensure output dir exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=200)
print(f"Saved combined Top-K figure to: {OUT_PATH}")

# also show when run in interactive environment
try:
    plt.show()
except Exception:
    pass
