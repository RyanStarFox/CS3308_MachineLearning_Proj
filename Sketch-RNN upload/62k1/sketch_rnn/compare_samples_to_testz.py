#!/usr/bin/env python3
"""
compare_samples_to_testz.py

Usage:
  python compare_samples_to_testz.py --category airplane

This script:
 - For a given category name, iterates over all sample folders under
   `sampling_dir_2/<category>`.
 - Loads `sample_pred_cond_s3.npz` (expects an array under any key, typically 'strokes').
 - Loads all `_z.npy` files under `test_z/<category>` (files named like '<cat>_test_<idx>_z.npy').
 - Computes cosine similarity between the flattened sample array and each test z-vector.
   If lengths differ, both vectors are truncated to the minimum length before computing similarity.
 - Writes a per-sample sorted similarity file `similarity_sorted.txt` in the sample folder
   with `rank, test_index, similarity` and
   writes a category summary `category_summary.csv` in `sampling_dir_2/<category>` listing
   each sample folder, its `original_index`, the rank of that original index among similarities,
   and the similarity value.

Notes:
 - This script attempts to be robust to minor shape mismatches by truncating to the
   minimum matching length. If you prefer different behavior (padding or projecting),
   modify the `compute_cosine` function accordingly.

"""
import os
import sys
import argparse
import numpy as np
import re
from math import sqrt

# import the required libraries
import numpy as np
import time
import random
import pickle as cPickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import xrange

# libraries required for visualisation:
from IPython.display import SVG, display
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

import svgwrite # conda install -c omnia svgwrite=1.1.6

from magenta.models.sketch_rnn.sketch_rnn_train import reset_graph, load_dataset, load_checkpoint
from magenta.models.sketch_rnn import model as sketch_rnn_model
from magenta.models.sketch_rnn.model import Model
from magenta.models.sketch_rnn import utils


def load_env_compatible(data_dir, model_dir):
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    data = json.load(f)
  fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
  for fix in fix_list:
    data[fix] = (data[fix] == 1)
  model_params.parse_json(json.dumps(data))
  return load_dataset(data_dir, model_params, inference_mode=True)

def to_big_strokes(stroke, max_len=250):
  """Converts from stroke-3 to stroke-5 format and pads to given length."""
  # (But does not insert special start token).

  result = np.zeros((max_len, 5), dtype=float)
  l = len(stroke)
  assert l <= max_len
  result[0:l, 0:2] = stroke[:, 0:2]
  result[0:l, 3] = stroke[:, 2]
  result[0:l, 2] = 1 - result[0:l, 3]
  result[l:, 4] = 1
  return result

# quiet encoder that does not call draw_strokes
def encode_quiet(input_strokes):
    strokes = to_big_strokes(input_strokes, max_seq_len).tolist()
    strokes.insert(0, [0, 0, 1, 0, 0])
    seq_len = [len(input_strokes)]
    return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

data_dir = './QuickDraw_generation'
models_root_dir = './logs'   
model_dir = './logs'

[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env_compatible(data_dir, model_dir)

# construct the sketch-rnn model here:
reset_graph()
model = Model(hps_model)
eval_model = Model(eval_hps_model, reuse=True)
sample_model = Model(sample_hps_model, reuse=True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

with open(models_root_dir + "/model_config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
max_seq_len = config.get('max_seq_len', 250)

def compute_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between 1D arrays a and b.
    If shapes differ, truncate both to the minimum length.
    Returns a float in [-1,1]."""
    if a.ndim != 1:
        a = a.ravel()
    if b.ndim != 1:
        b = b.ravel()
    min_len = min(a.shape[0], b.shape[0])
    if min_len == 0:
        return 0.0
    if min_len != a.shape[0] or min_len != b.shape[0]:
        a = a[:min_len]
        b = b[:min_len]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_npz_array(path: str) -> np.ndarray:
    """Load any array from an .npz file. If multiple arrays present, return the first."""
    data = np.load(path, allow_pickle=True)
    try:
        # Prefer a key named 'strokes' if present
        if 'strokes' in data:
            arr = data['strokes']
        else:
            # take first item
            arr = data[data.files[0]]
    finally:
        data.close()
    return arr


def find_test_z_vectors(test_z_dir: str, category: str):
    """Return list of tuples (test_index:int, vector:np.ndarray, filename).
    Looks for files like '<category>_test_<idx>_z.npy' in `test_z_dir/<category>`.
    """
    cat_dir = os.path.join(test_z_dir, category)
    if not os.path.isdir(cat_dir):
        raise FileNotFoundError(f"test_z category folder not found: {cat_dir}")
    files = os.listdir(cat_dir)
    pattern = re.compile(rf"{re.escape(category)}_test_(\d+)_z\.npy$")
    results = []
    for fn in files:
        m = pattern.match(fn)
        if not m:
            continue
        idx = int(m.group(1))
        path = os.path.join(cat_dir, fn)
        try:
            vec = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
            continue
        results.append((idx, vec, path))
    results.sort(key=lambda t: t[0])
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare sampling_dir_2 samples to test_z vectors.')
    parser.add_argument('--category', '-c', required=True, help='Category name (directory name)')
    parser.add_argument('--sampling-dir', default='sampling_dir_2', help='Path to sampling_dir_2')
    parser.add_argument('--test-z-dir', default='test_z', help='Path to test_z')
    parser.add_argument('--out-name', default='similarity_sorted.txt', help='Per-sample output filename')
    args = parser.parse_args()

    sampling_dir = args.sampling_dir
    test_z_dir = args.test_z_dir
    category = args.category

    cat_sample_dir = os.path.join(sampling_dir, category)
    if not os.path.isdir(cat_sample_dir):
        print(f"Error: category folder not found in sampling_dir_2: {cat_sample_dir}")
        sys.exit(1)

    print(f"Loading test z-vectors for category '{category}' from '{test_z_dir}'...")
    test_vectors = find_test_z_vectors(test_z_dir, category)
    if len(test_vectors) == 0:
        print(f"No test z-vectors found for category {category} in {test_z_dir}/{category}")
        sys.exit(1)

    # Build map from test index to vector for quick lookup
    test_index_to_vec = {t[0]: t[1] for t in test_vectors}

    # Prepare category summary
    summary_rows = []  # list of (sample_folder, original_index, rank_of_original, similarity)

    # iterate sample folders (numerical or otherwise)
    sample_folders = sorted([d for d in os.listdir(cat_sample_dir) if os.path.isdir(os.path.join(cat_sample_dir, d))], key=lambda x: int(x) if x.isdigit() else x)

    for sample_folder in sample_folders:
        sample_path = os.path.join(cat_sample_dir, sample_folder)
        s3_path = os.path.join(sample_path, 'sample_pred_cond_s3.npz')
        if not os.path.exists(s3_path):
            print(f"Skipping {sample_folder}: missing {s3_path}")
            continue
        try:
            sample_arr = load_npz_array(s3_path)
        except Exception as e:
            print(f"Failed to load {s3_path}: {e}")
            continue
        
        sample_arr = encode_quiet(sample_arr)
        
        # Flatten sample array to 1D vector
        sample_vec = np.asarray(sample_arr).ravel()

        similarities = []  # list of (test_index, similarity)
        for test_idx, test_vec, test_path in test_vectors:
            sim = compute_cosine(sample_vec, np.asarray(test_vec).ravel())
            similarities.append((test_idx, sim))

        # sort descending by similarity
        similarities.sort(key=lambda t: t[1], reverse=True)

        # write per-sample sorted similarity file
        out_file = os.path.join(sample_path, args.out_name)
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('# rank\ttest_index\tsimilarity\n')
            for rank, (test_idx, sim) in enumerate(similarities, start=1):
                f.write(f"{rank}\t{test_idx}\t{sim:.6f}\n")

        # determine original_index if available
        orig_index = None
        orig_txt = os.path.join(sample_path, 'original_index.txt')
        if os.path.exists(orig_txt):
            try:
                with open(orig_txt, 'r', encoding='utf-8') as f:
                    s = f.read().strip()
                    orig_index = int(s)
            except Exception:
                orig_index = None

        # find rank for original_index
        orig_rank = None
        orig_similarity = None
        if orig_index is not None:
            for rank, (test_idx, sim) in enumerate(similarities, start=1):
                if test_idx == orig_index:
                    orig_rank = rank
                    orig_similarity = sim
                    break

        summary_rows.append((sample_folder, orig_index if orig_index is not None else '', orig_rank if orig_rank is not None else '', f"{orig_similarity:.6f}" if orig_similarity is not None else ''))

        print(f"Processed sample {sample_folder}: original_index={orig_index}, rank={orig_rank}, similarity={orig_similarity}")

    # save category summary CSV
    summary_path = os.path.join(cat_sample_dir, 'category_summary.csv')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('sample_folder,original_index,original_rank,original_similarity\n')
        for row in summary_rows:
            f.write(','.join([str(x) for x in row]) + '\n')

    print('Done. Per-sample similarity files written and category summary at', summary_path)


if __name__ == '__main__':
    main()
