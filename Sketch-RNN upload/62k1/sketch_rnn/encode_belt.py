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

# import our command line tools
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

# quiet encoder that does not call draw_strokes
def encode_quiet(input_strokes):
    strokes = to_big_strokes(input_strokes, max_seq_len).tolist()
    strokes.insert(0, [0, 0, 1, 0, 0])
    seq_len = [len(input_strokes)]
    return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

def detect_stroke_format(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return 'unknown'
    ncols = arr.shape[1]
    if ncols == 3:
        return 'stroke3'
    if ncols == 5:
        return 'stroke5'
    # heuristic: if last 3 columns look like a one-hot/probability vector -> stroke5
    if ncols >= 3:
        last3 = arr[:, -3:]
        sums = last3.sum(axis=1)
        if np.allclose(sums, 1.0, atol=1e-6):
            return 'stroke5'
    return 'unknown'

def to_stroke5(strokes, max_len=None):
    """Convert stroke3-like array (N x 3) to stroke5 (big strokes).
    This uses `to_big_strokes` which expects "normal" stroke format (dx, dy, pen_state) and
    returns a 5-dim array (dx, dy, p1, p2, p3).
    """
    strokes = np.asarray(strokes)
    # If already stroke5, return copy
    if detect_stroke_format(strokes) == 'stroke5':
        return strokes.copy()
    # If shape (N,3) possibly with absolute coords, assume it's normal strokes accepted by to_big_strokes
    # to_big_strokes signature: to_big_strokes(stroke, max_len=...)
    if max_len is None:
        # let the util choose default (some implementations require max_len)
        return to_big_strokes(strokes)
    else:
        return to_big_strokes(strokes, max_len)


def to_stroke3(strokes):
    """Convert stroke5 (big strokes) to stroke3 (normal strokes).
    Uses `to_normal_strokes` which expects big stroke and returns normal strokes.
    """
    strokes = np.asarray(strokes)
    if detect_stroke_format(strokes) == 'stroke3':
        return strokes.copy()
    return to_normal_strokes(strokes)

def load_env_compatible(data_dir, model_dir):
  """Loads environment for inference mode, used in jupyter notebook."""
  # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
  # to work with depreciated tf.HParams functionality
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    data = json.load(f)
  fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
  for fix in fix_list:
    data[fix] = (data[fix] == 1)
  model_params.parse_json(json.dumps(data))
  return load_dataset(data_dir, model_params, inference_mode=True)

data_dir = './QuickDraw_generation'
models_root_dir = './logs'   
model_dir = './logs'

with open(models_root_dir + "/model_config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
max_seq_len = config.get('max_seq_len', 250)

[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env_compatible(data_dir, model_dir)
# construct the sketch-rnn model here:
reset_graph()
model = Model(hps_model)
eval_model = Model(eval_hps_model, reuse=True)
sample_model = Model(sample_hps_model, reuse=True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# Batch-encode QuickDraw datasets to z and save into ./test_z
out_dir = './test_z/belt'
os.makedirs(out_dir, exist_ok=True)
qd_dir = './qdcopy/belt'
files = [f for f in os.listdir(qd_dir) if f.endswith('.npz') or f.endswith('.npy')]
print('Found QuickDraw files:', files)

count = 0
def save_z_and_raw(out_dir, name, z_vec, raw_strokes):
    """Save z vector and raw strokes with matching name prefixes.
    z saved as '<name>_z.npy', raw as '<name>_raw.npy'.
    """
    z_path = os.path.join(out_dir, f"{name}_z.npy")
    raw_path = os.path.join(out_dir, f"{name}_raw.npy")
    try:
        np.save(z_path, z_vec)
    except Exception as e:
        print('  Error saving z to', z_path, e)
    try:
        np.save(raw_path, raw_strokes)
    except Exception as e:
        print('  Error saving raw strokes to', raw_path, e)

for fname in files:
    path = os.path.join(qd_dir, fname)
    base = os.path.splitext(fname)[0]
    print('Processing', fname)
    if fname.endswith('.npz'):
        npz = np.load(path, allow_pickle=True, encoding='latin1')
        for k in npz.files:
            item = npz[k]
            # case: object-dtype array (list of samples)
            if isinstance(item, np.ndarray) and item.dtype == object:
                for i, s in enumerate(item):
                    s_arr = np.asarray(s)
                    fmt = detect_stroke_format(s_arr)
                    if fmt == 'stroke5':
                        s3 = to_stroke3(s_arr)
                    else:
                        s3 = s_arr
                    try:
                        z = encode_quiet(s3)
                        save_z_and_raw(out_dir, f'{base}_{k}_{i}', z, s3)
                        count += 1
                    except Exception as e:
                        print('  Error encoding', base, k, i, e)
            # case: single 2D strokes array per key
            elif isinstance(item, np.ndarray) and item.ndim == 2 and item.shape[1] in (3,5):
                s_arr = item
                fmt = detect_stroke_format(s_arr)
                s3 = to_stroke3(s_arr) if fmt == 'stroke5' else s_arr
                try:
                    z = encode_quiet(s3)
                    save_z_and_raw(out_dir, f'{base}_{k}', z, s3)
                    count += 1
                except Exception as e:
                    print('  Error encoding', base, k, e)
            else:
                # try nested iteration fallback
                try:
                    for i, s in enumerate(item):
                        s_arr = np.asarray(s)
                        if isinstance(s_arr, np.ndarray) and s_arr.ndim == 2:
                            fmt = detect_stroke_format(s_arr)
                            s3 = to_stroke3(s_arr) if fmt == 'stroke5' else s_arr
                            try:
                                        z = encode_quiet(s3)
                                        save_z_and_raw(out_dir, f'{base}_{k}_{i}', z, s3)
                                        count += 1
                            except Exception as e:
                                print('  Error encoding nested', base, k, i, e)
                except Exception as e:
                    print('  Skipping', base, k, 'unhandled format:', e)
    elif fname.endswith('.npy'):
        arr = np.load(path, allow_pickle=True, encoding='latin1')
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            for i, s in enumerate(arr):
                s_arr = np.asarray(s)
                fmt = detect_stroke_format(s_arr)
                s3 = to_stroke3(s_arr) if fmt == 'stroke5' else s_arr
                try:
                    z = encode_quiet(s3)
                    save_z_and_raw(out_dir, f'{base}_{i}', z, s3)
                    count += 1
                except Exception as e:
                    print('  Error encoding', base, i, e)
        elif isinstance(arr, np.ndarray) and arr.ndim == 2:
            s_arr = arr
            fmt = detect_stroke_format(s_arr)
            s3 = to_stroke3(s_arr) if fmt == 'stroke5' else s_arr
            try:
                z = encode_quiet(s3)
                save_z_and_raw(out_dir, f'{base}', z, s3)
                count += 1
            except Exception as e:
                print('  Error encoding', base, e)

print('Finished. Saved', count, 'z vectors to', out_dir)

