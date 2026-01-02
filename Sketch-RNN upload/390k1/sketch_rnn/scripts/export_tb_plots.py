#!/usr/bin/env python3
"""
Export TensorBoard scalar plots from event files to PNG and CSV.

Usage:
  python3 scripts/export_tb_plots.py --logdir logs --outdir logs/tb_plots

The script uses `tensorboard`'s EventAccumulator and `matplotlib` to
draw and save plots. If packages are missing, install with:
  pip install tensorboard matplotlib
"""
import os
import argparse
import math

def sanitize(name):
    return name.replace('/', '_').replace(' ', '_')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='logs', help='TensorBoard logs directory')
    parser.add_argument('--outdir', default='logs/tb_plots', help='Output directory for PNG/CSV')
    parser.add_argument('--max-points', type=int, default=2000, help='Max points to plot (downsample if larger)')
    args = parser.parse_args()

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        print('ERROR: could not import EventAccumulator from tensorboard:', e)
        print('Try: pip install tensorboard')
        raise

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('ERROR: matplotlib not available:', e)
        print('Try: pip install matplotlib')
        raise

    logdir = args.logdir
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Find event files (recursively)
    event_files = []
    for root, _, files in os.walk(logdir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, f))

    if not event_files:
        print('No event files found in', logdir)
        return

    print('Found', len(event_files), 'event files. Loading...')

    # Load (if multiple files from different runs, we'll merge by tag)
    ea = EventAccumulator(logdir)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if not tags:
        print('No scalar tags found in event files.')
        return

    print('Scalar tags:', ', '.join(tags))

    for tag in tags:
        try:
            events = ea.Scalars(tag)
        except Exception as e:
            print('Could not read tag', tag, e)
            continue

        steps = [e.step for e in events]
        vals = [e.value for e in events]

        # downsample if too many points
        n = len(steps)
        if n == 0:
            continue
        stride = 1
        if n > args.max_points:
            stride = max(1, math.ceil(n / args.max_points))

        steps_ds = steps[::stride]
        vals_ds = vals[::stride]

        base = sanitize(tag)
        png_path = os.path.join(outdir, base + '.png')
        csv_path = os.path.join(outdir, base + '.csv')

        # save csv
        try:
            with open(csv_path, 'w') as f:
                f.write('step,value\n')
                for s, v in zip(steps, vals):
                    f.write(f'{s},{v}\n')
        except Exception as e:
            print('Failed to write CSV for', tag, e)

        # plot
        try:
            plt.figure(figsize=(8,4))
            plt.plot(steps_ds, vals_ds, '-o', markersize=3)
            plt.title(tag)
            plt.xlabel('step')
            plt.ylabel('value')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()
            print('Saved', png_path)
        except Exception as e:
            print('Failed to plot tag', tag, e)

    print('All done. Plots saved to', outdir)

if __name__ == '__main__':
    main()
