#!/usr/bin/env python3
"""
Compare samples in `sampling_ede/<class>` with candidate `_z.npy` files
from `test_z/<class>`.

For each sample in `sampling_ede/<class>` the script:
- selects 19 random `_z.npy` files from `test_z/<class>` (deterministic with seed)
- tries to find the `_z.npy` file whose filename contains the sample index and adds
  it as the correct answer (if found)
- computes cosine similarity between the sampling sample and each candidate
- sorts candidates by similarity (desc) and writes per-sample CSV with rank,
  candidate filename and similarity
- writes a `summary_<class>.csv` mapping each sampling file to the rank of the
  correct candidate (or -1 if not found)

Usage: python compare_sampling_ede_to_testz.py --class airplane
"""

import os
import argparse
import numpy as np
import random
import re
import csv
from pathlib import Path
from math import sqrt


def list_npy_files_recursive(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for fn in filenames:
            if fn.lower().endswith('.npy'):
                files.append(os.path.join(root, fn))
    return sorted(files)


def flatten_array(a):
    arr = np.asarray(a)
    return arr.ravel()


def cosine_similarity(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = flatten_array(a)
    b = flatten_array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_index_in_name(name):
    # return first integer sequence found in basename, or None
    m = re.search(r'(\d+)', name)
    return m.group(1) if m else None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Compare sampling_ede samples to test_z candidates')
    parser.add_argument('--class', dest='class_name', required=True,
                        help='Category name (subfolder name inside sampling_ede and test_z)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for candidate selection')
    parser.add_argument('--candidates', type=int, default=19, help='Number of random candidates to pick from test_z')
    parser.add_argument('--outdir', default=None, help='Output directory (default: compare_results/<class>)')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    sampling_dir = script_dir / 'sampling_ede' / args.class_name
    testz_dir = script_dir / 'test_z' / args.class_name

    if not sampling_dir.exists():
        raise SystemExit(f"sampling_ede class folder not found: {sampling_dir}")
    if not testz_dir.exists():
        raise SystemExit(f"test_z class folder not found: {testz_dir}")

    sampling_files = list_npy_files_recursive(str(sampling_dir))
    testz_files_all = list_npy_files_recursive(str(testz_dir))

    # filter test_z files that end with _z.npy (or contain '_z' before extension)
    testz_z_files = [p for p in testz_files_all if re.search(r'_z(?=\.npy$)', os.path.basename(p))]
    if len(testz_z_files) == 0:
        print('Warning: no `_z.npy` files found in', testz_dir)
        testz_z_files = testz_files_all

    outdir = Path(args.outdir) if args.outdir else (script_dir / 'compare_results' / args.class_name)
    ensure_dir(outdir)

    random.seed(args.seed)

    summary_rows = []

    for samp_path in sampling_files:
        samp_name = os.path.basename(samp_path)
        samp_index = find_index_in_name(samp_name)

        # choose random candidates from testz_z_files
        pool = list(testz_z_files)
        # Prefer to exclude the matched correct candidate if present in pool when selecting randoms
        matched_correct = None
        if samp_index is not None:
            for p in pool:
                # extract integer tokens from candidate basename and require exact equality
                tokens = re.findall(r"(\d+)", os.path.basename(p))
                if any(t == samp_index for t in tokens):
                    matched_correct = p
                    break

        # build random selection excluding matched_correct
        pool_for_pick = [p for p in pool if p != matched_correct]
        pick_count = min(args.candidates, len(pool_for_pick))
        random_candidates = random.sample(pool_for_pick, pick_count) if pick_count > 0 else []

        candidates = list(random_candidates)
        if matched_correct:
            candidates.append(matched_correct)

        # If we didn't find a matched correct, we still ensure we have at least one extra candidate
        if not matched_correct and len(candidates) == 0 and len(pool) > 0:
            # fallback: pick one
            candidates.append(random.choice(pool))

        # load sample
        try:
            samp_arr = np.load(samp_path)
        except Exception as e:
            print(f'Failed to load {samp_path}: {e}')
            continue

        results = []
        for cand in candidates:
            try:
                cand_arr = np.load(cand)
            except Exception as e:
                print(f'Failed to load candidate {cand}: {e}')
                sim = -1.0
            else:
                sim = cosine_similarity(samp_arr, cand_arr)
            results.append((cand, sim))

        # sort by similarity descending
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

        # write per-sample CSV: rank, candidate_filename (relative to test_z class dir), similarity
        samp_out_csv = outdir / f"{samp_name}_similarities.csv"
        with open(samp_out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'candidate_relpath', 'similarity'])
            for rank, (candpath, sim) in enumerate(results_sorted, start=1):
                rel = os.path.relpath(candpath, start=testz_dir)
                writer.writerow([rank, rel, f"{sim:.6f}"])

        # determine rank of matched_correct
        rank_of_correct = -1
        if matched_correct:
            for i, (candpath, sim) in enumerate(results_sorted, start=1):
                if os.path.normpath(candpath) == os.path.normpath(matched_correct):
                    rank_of_correct = i
                    break

        summary_rows.append({'sample': samp_name, 'matched_correct': os.path.relpath(matched_correct, start=testz_dir) if matched_correct else '', 'rank': rank_of_correct, 'total_candidates': len(results_sorted)})

    # write summary CSV
    summary_csv = outdir / f"summary_{args.class_name}.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'matched_correct_relpath', 'rank_of_correct', 'total_candidates'])
        for r in summary_rows:
            writer.writerow([r['sample'], r['matched_correct'], r['rank'], r['total_candidates']])

    print('Done. Results in', outdir)


if __name__ == '__main__':
    main()
