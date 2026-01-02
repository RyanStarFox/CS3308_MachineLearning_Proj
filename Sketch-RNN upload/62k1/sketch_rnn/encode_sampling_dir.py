#!/usr/bin/env python
"""Encode sample_pred_cond_s3.npz files under sampling_dir into z npy files.

Saves outputs to `sampling_ede/<category>/<sample>_z.npy`.
"""
import os
import json
import numpy as np
import tensorflow.compat.v1 as tf

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


def detect_stroke_format(arr):
  arr = np.asarray(arr)
  if arr.ndim != 2:
    return 'unknown'
  ncols = arr.shape[1]
  if ncols == 3:
    return 'stroke3'
  if ncols == 5:
    return 'stroke5'
  if ncols >= 3:
    last3 = arr[:, -3:]
    sums = last3.sum(axis=1)
    if np.allclose(sums, 1.0, atol=1e-6):
      return 'stroke5'
  return 'unknown'


def to_stroke3(strokes, max_len=None):
  strokes = np.asarray(strokes)
  if detect_stroke_format(strokes) == 'stroke3':
    return strokes.copy()
  # expects stroke5 -> convert using utils.to_normal_strokes
  return utils.to_normal_strokes(strokes)


def encode_quiet(sess, eval_model, input_strokes, max_seq_len):
  # convert and pad using utils.to_big_strokes
  strokes = utils.to_big_strokes(input_strokes, max_seq_len).tolist()
  strokes.insert(0, [0, 0, 1, 0, 0])
  seq_len = [len(input_strokes)]
  # use deterministic mean as encoding
  return sess.run(eval_model.mean, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]


def process_sampling_dir(sampling_dir='./sampling_dir_2', out_root='./sampling_ede'):
  data_dir = './QuickDraw_generation'
  model_dir = './logs'

  # read model config to get max_seq_len
  with open(os.path.join(model_dir, 'model_config.json'), 'r', encoding='utf-8') as f:
    config = json.load(f)
  max_seq_len = config.get('max_seq_len', 250)

  # load env and hparams
  train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model = load_env_compatible(data_dir, model_dir)

  # build models
  reset_graph()
  model = Model(hps_model)
  eval_model = Model(eval_hps_model, reuse=True)
  sample_model = Model(sample_hps_model, reuse=True)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  # restore weights
  load_checkpoint(sess, model_dir)

  # iterate categories
  categories = [d for d in os.listdir(sampling_dir) if os.path.isdir(os.path.join(sampling_dir, d))]
  total = 0
  for cat in categories:
    cat_in = os.path.join(sampling_dir, cat)
    cat_out = os.path.join(out_root, cat)
    os.makedirs(cat_out, exist_ok=True)
    # each sample in category is in a numbered subdir
    subdirs = sorted([d for d in os.listdir(cat_in) if os.path.isdir(os.path.join(cat_in, d))], key=lambda x: int(x) if x.isdigit() else x)
    for sd in subdirs:
      sub_in = os.path.join(cat_in, sd)
      npz_path = os.path.join(sub_in, 'sample_pred_cond_s3.npz')
      if not os.path.exists(npz_path):
        print('Missing', npz_path)
        continue
      try:
        npz = np.load(npz_path, allow_pickle=True)
      except Exception as e:
        print('Failed to load', npz_path, e)
        continue

      # try to find a 2D stroke array inside
      stroke = None
      if hasattr(npz, 'files') and len(npz.files) >= 1:
        for k in npz.files:
          item = npz[k]
          if isinstance(item, np.ndarray) and item.ndim == 2 and item.shape[1] in (3,5):
            stroke = item
            break
          # object-dtype sequences: take first element
          if isinstance(item, np.ndarray) and item.dtype == object:
            for cand in item:
              arr = np.asarray(cand)
              if arr.ndim == 2 and arr.shape[1] in (3,5):
                stroke = arr
                break
            if stroke is not None:
              break
      else:
        # fallback: try np.load returned array
        arr = np.asarray(npz)
        if arr.ndim == 2 and arr.shape[1] in (3,5):
          stroke = arr

      if stroke is None:
        print('No 2D stroke array found in', npz_path)
        continue

      fmt = detect_stroke_format(stroke)
      s3 = stroke if fmt == 'stroke3' else to_stroke3(stroke, max_seq_len)
      try:
        z = encode_quiet(sess, eval_model, s3, max_seq_len)
        out_name = os.path.join(cat_out, f'{sd}_z.npy')
        np.save(out_name, z)
        total += 1
      except Exception as e:
        print('Error encoding', npz_path, e)

  print('Finished encoding. Saved', total, 'z vectors to', out_root)


if __name__ == '__main__':
  process_sampling_dir(sampling_dir='./sampling_dir_2', out_root='./sampling_ede')
