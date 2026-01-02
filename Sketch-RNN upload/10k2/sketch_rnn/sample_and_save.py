#!/usr/bin/env python3
"""Sample from QuickDraw .npz test sets, encode/decode and save outputs.

Saves for each category (20 samples):
  sample_gt.svg
  sample_gt.npy
  sample_pred_cond.svg
  sample_pred_cond_s3.npz
  sample_pred_cond_s5.npz
  sample_pred_cond_custom.npy
  sample_pred_cond_tag.txt

This script uses the local sketch-rnn implementation in this folder.
"""
import os
import sys
import json
import random
import numpy as np
import tensorflow.compat.v1 as tf

# Add repo path to import local modules
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from sketch_rnn_train import load_dataset, load_model, reset_graph, load_checkpoint
from model import Model, sample as model_sample
from utils import to_big_strokes, to_normal_strokes, get_bounds
import sketch_rnn_train

try:
    from IPython.display import SVG, display
except Exception:
    # fallback no-op display for non-notebook environments
    def display(x):
        return None

import svgwrite


def draw_strokes(data, factor=0.2, svg_filename='/tmp/sketch_rnn/svg/sample.svg'):
    os.makedirs(os.path.dirname(svg_filename), exist_ok=True)
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    try:
        display(SVG(dwg.tostring()))
    except Exception:
        pass


def stroke5_to_custom(stroke5):
    """
    Convert stroke-5 format to custom (N,4) format:
    x: absolute x
    y: absolute y
    dx: relative x
    dy: relative y
    Only keep movements when the pen is down (p1 == 1).
    """
    dx = stroke5[:, 0]
    dy = stroke5[:, 1]
    x = np.cumsum(dx)
    y = np.cumsum(dy)

    mask = stroke5[:, 2] == 1

    return np.stack([x[mask], y[mask], dx[mask], dy[mask]], axis=1)


def main():
    data_dir = './QuickDraw_generation'
    model_dir = './logs'
    sampling_dir = './sampling_dir'
    os.makedirs(sampling_dir, exist_ok=True)

    # Load model hyperparams from model_config.json
    with open(os.path.join(model_dir, 'model_config.json'), 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # Prepare hparams model object
    import sketch_rnn_train as srt
    from model import get_default_hparams as _get_def
    from model import copy_hparams
    import model as sketch_rnn_model_module

    # Build model params from config
    hps = sketch_rnn_model_module.get_default_hparams()
    # apply config
    # convert booleans if needed
    for k, v in cfg.items():
        if k in hps.values().keys() if hasattr(hps, 'values') else []:
            pass
    # simple parse via JSON string to keep compatibility
    try:
        hps.parse_json(json.dumps(cfg))
    except Exception:
        # fallback: set attributes directly
        for kk, vv in cfg.items():
            setattr(hps, kk, vv)

    # Load dataset once to get normalizing scale factor
    print('Loading dataset to compute normalizing scale...')
    # reuse load_dataset from sketch_rnn_train
    train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model = sketch_rnn_train.load_dataset(data_dir, hps, inference_mode=True)
    normalizing_scale = train_set.scale_factor
    print('normalizing_scale:', normalizing_scale)

    # Build models
    sketch_rnn_train.reset_graph()
    model = Model(hps_model)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sketch_rnn_train.load_checkpoint(sess, model_dir)

    # replicate encode/decode functions (unchanged logic as notebook)
    def encode(input_strokes):
        strokes = to_big_strokes(input_strokes, hps_model.max_seq_len).tolist()
        strokes.insert(0, [0, 0, 1, 0, 0])
        seq_len = [len(input_strokes)]
        draw_strokes(to_normal_strokes(np.array(strokes)))
        return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

    def decode(z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
        z = None
        if z_input is not None:
            z = [z_input]
        sample_strokes_s5, m = model_sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
        strokes_s3 = to_normal_strokes(sample_strokes_s5)
        if draw_mode:
            draw_strokes(strokes_s3, factor)
        return sample_strokes_s5, strokes_s3

    # iterate categories from config data_set list if present
    categories = cfg.get('data_set') if isinstance(cfg.get('data_set'), list) else sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

    for cat in categories:
        cat_name = os.path.splitext(os.path.basename(cat))[0]
        print('Processing category', cat_name)
        cat_dir = os.path.join(sampling_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        npz_path = os.path.join(data_dir, cat)
        if not os.path.exists(npz_path):
            print('Warning: missing', npz_path)
            continue

        data = np.load(npz_path, allow_pickle=True, encoding='latin1')
        test_arr = data['test']
        N = min(20, len(test_arr))
        idxs = list(range(len(test_arr)))
        random.shuffle(idxs)
        chosen = idxs[:N]

        for i, idx in enumerate(chosen):
            sample = test_arr[idx]
            sample_fname_dir = os.path.join(cat_dir, str(i))
            os.makedirs(sample_fname_dir, exist_ok=True)

            # save original raw sample
            np.save(os.path.join(sample_fname_dir, 'sample_gt.npy'), sample)
            # save GT svg (draw original strokes)
            try:
                draw_strokes(sample, svg_filename=os.path.join(sample_fname_dir, 'sample_gt.svg'))
            except Exception as e:
                print('draw error', e)

            # normalize sample as DataLoader would
            sample_norm = np.copy(sample).astype(np.float32)
            sample_norm[:, 0:2] /= normalizing_scale

            # encode -> z
            z = encode(sample_norm)

            # decode using z (we save both s5 and s3)
            sample_s5, sample_s3 = decode(z, draw_mode=False)

            # Save s5 and s3
            np.savez_compressed(os.path.join(sample_fname_dir, 'sample_pred_cond_s5.npz'), strokes=sample_s5)
            np.savez_compressed(os.path.join(sample_fname_dir, 'sample_pred_cond_s3.npz'), strokes=sample_s3)

            # Save custom conversion
            custom = stroke5_to_custom(sample_s5)
            np.save(os.path.join(sample_fname_dir, 'sample_pred_cond_custom.npy'), custom)

            # Save tag file with category name
            with open(os.path.join(sample_fname_dir, 'sample_pred_cond_tag.txt'), 'w', encoding='utf-8') as f:
                f.write(cat_name)

            # Save predicted svg
            try:
                draw_strokes(sample_s3, svg_filename=os.path.join(sample_fname_dir, 'sample_pred_cond.svg'))
            except Exception as e:
                print('draw_pred error', e)

        data.close()

    print('Done. Samples saved under', sampling_dir)


if __name__ == '__main__':
    main()
