#!/usr/bin/env python3
"""Utility: detect stroke format (.npy) and convert between stroke3 and stroke5.

Place this file in the same folder as `utils.py` so it can import `to_big_strokes` and
`to_normal_strokes` as `from utils import ...`.

Examples:
  python convert_strokes.py detect path/to/file.npy
  python convert_strokes.py convert path/to/file.npy --to stroke5 --out out.npy
  python convert_strokes.py batch_convert path/dir --to stroke3
"""
import os
import argparse
import numpy as np

try:
    # when running inside this repo directory
    from utils import to_big_strokes, to_normal_strokes
except Exception:
    # fallback to magenta package if available
    try:
        from magenta.models.sketch_rnn.utils import to_big_strokes, to_normal_strokes
    except Exception:
        raise ImportError('Could not import to_big_strokes/to_normal_strokes from local utils or magenta package')


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


def load_npy(path):
    return np.load(path, allow_pickle=True)


def save_npy(path, arr):
    np.save(path, np.asarray(arr))


def cmd_detect(args):
    arr = load_npy(args.file)
    fmt = detect_stroke_format(arr)
    print(f"Detected format: {fmt}")


def cmd_convert(args):
    arr = load_npy(args.file)
    src = detect_stroke_format(arr)
    print(f"Source format: {src}")
    if args.to == 'stroke5':
        out = to_stroke5(arr, max_len=args.max_len)
    elif args.to == 'stroke3':
        out = to_stroke3(arr)
    else:
        raise ValueError('Unknown target format: ' + args.to)
    out_path = args.out or (os.path.splitext(args.file)[0] + f'.{args.to}.npy')
    save_npy(out_path, out)
    print(f"Saved converted file to: {out_path}")


def cmd_batch_convert(args):
    files = []
    for root, _, filenames in os.walk(args.path):
        for fn in filenames:
            if fn.endswith('.npy'):
                files.append(os.path.join(root, fn))
    print(f"Found {len(files)} .npy files under {args.path}")
    out_dir = args.out_dir or os.path.join(args.path, 'converted_'+args.to)
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        try:
            arr = load_npy(f)
            if args.to == 'stroke5':
                out = to_stroke5(arr, max_len=args.max_len)
            else:
                out = to_stroke3(arr)
            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(out_dir, base + f'.{args.to}.npy')
            save_npy(out_path, out)
        except Exception as e:
            print(f"Failed {f}: {e}")
    print('Batch conversion complete.')


def make_parser():
    p = argparse.ArgumentParser(description='Detect/convert sketch-rnn stroke formats')
    sub = p.add_subparsers(dest='cmd')

    d = sub.add_parser('detect')
    d.add_argument('file', help='.npy file to inspect')

    c = sub.add_parser('convert')
    c.add_argument('file', help='.npy file to convert')
    c.add_argument('--to', choices=['stroke3', 'stroke5'], required=True)
    c.add_argument('--out', help='output .npy path')
    c.add_argument('--max_len', type=int, default=None, help='optional max_len for to_big_strokes')

    b = sub.add_parser('batch_convert')
    b.add_argument('path', help='directory to search for .npy files')
    b.add_argument('--to', choices=['stroke3', 'stroke5'], required=True)
    b.add_argument('--out_dir', help='directory to write converted files')
    b.add_argument('--max_len', type=int, default=None)

    return p


def main():
    p = make_parser()
    args = p.parse_args()
    if args.cmd == 'detect':
        cmd_detect(args)
    elif args.cmd == 'convert':
        cmd_convert(args)
    elif args.cmd == 'batch_convert':
        cmd_batch_convert(args)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
