from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import cairosvg
import tempfile

import model as sketch_rnn_model
from sketch_pix2seq_train import reset_graph, load_checkpoint
from sketch_pix2seq_sampling import encode, decode, load_env_compatible, draw_strokes, make_grid_svg
from utils import slerp


def load_image(image_path, img_h=48, img_w=48):
    """Load a single PNG image and prepare it for the model."""
    image = Image.open(image_path).convert('L')
    image = image.resize((img_w, img_h), Image.LANCZOS)
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=2)
    # Create batch of size 1
    img_batch = np.expand_dims(image, axis=0)
    return img_batch


def svg_to_png(svg_path, png_path, width=256, height=256):
    """Convert SVG to PNG using cairosvg."""
    # Read SVG file
    with open(svg_path, 'rb') as f:
        dwg_string = f.read()
    
    # Assume square SVG for simplicity
    svg_w, svg_h = 300, 300  # Default SVG size from draw_strokes
    png_w, png_h = width, height
    x_scale = png_w / svg_w
    y_scale = png_h / svg_h

    if x_scale > y_scale:
        cairosvg.svg2png(bytestring=dwg_string, write_to=png_path, output_height=png_h)
    else:
        cairosvg.svg2png(bytestring=dwg_string, write_to=png_path, output_width=png_w)


def vary_temperature(image_path, data_dir, model_dir, output_dir, 
                     num_steps=21, temp_min=0.1, temp_max=2.1):
    """
    Generate sketches from a single input image with varying temperatures.
    
    Args:
        image_path: Path to input PNG image
        data_dir: Directory containing the dataset
        model_dir: Directory containing the trained model
        output_dir: Directory to save results
        num_steps: Number of temperature steps (default: 21)
        temp_min: Minimum temperature (default: 0.1)
        temp_max: Maximum temperature (default: 2.1)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment and model
    print("Loading model...")
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = \
        load_env_compatible(data_dir, model_dir)
    
    # Build models
    reset_graph()
    model = sketch_rnn_model.Model(hps_model)
    eval_model = sketch_rnn_model.Model(eval_hps_model, reuse=True)
    sample_model = sketch_rnn_model.Model(sample_hps_model, reuse=True)
    
    # Create session and load checkpoint
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    load_checkpoint(sess, model_dir)
    
    # Load and encode input image
    print(f"Loading image: {image_path}")
    img = load_image(image_path, eval_model.hps.img_H, eval_model.hps.img_W)
    z = encode(img, sess, eval_model)
    print(f"Encoded z shape: {z.shape}")
    
    # Generate temperature list
    print(f"Generating {num_steps} samples with temperatures from {temp_min} to {temp_max}...")
    temperatures = np.linspace(temp_min, temp_max, num_steps)
    
    # Decode and save each sample with different temperatures
    print("Decoding and saving images with varying temperatures...")
    reconstructions = []
    for i in range(num_steps):
        temp = temperatures[i]
        print(f"Generating image {i+1}/{num_steps} with temperature {temp:.2f}...")
        strokes_out_3, strokes_out_5 = decode(
            sess,
            sample_model,
            eval_model.hps.max_seq_len,
            z,
            temperature=temp
        )
        
        # Save individual SVG
        svg_path = os.path.join(output_dir, f'temperature_{temp:.2f}_{i:03d}.svg')
        draw_strokes(strokes_out_3, svg_path)
        
        # Convert SVG to PNG
        png_path = os.path.join(output_dir, f'temperature_{temp:.2f}_{i:03d}.png')
        try:
            svg_to_png(svg_path, png_path)
            print(f"Saved: {png_path}")
        except Exception as e:
            print(f"Warning: Could not convert SVG to PNG: {e}")
        
        # Add to grid (row 0, column i)
        reconstructions.append([strokes_out_3, [0, i]])
    
    # Create and save grid visualization
    print("Creating grid visualization...")
    stroke_grid = make_grid_svg(reconstructions)
    grid_svg_path = os.path.join(output_dir, 'temperature_grid.svg')
    draw_strokes(stroke_grid, grid_svg_path)
    
    # Convert grid to PNG
    grid_png_path = os.path.join(output_dir, 'temperature_grid.png')
    try:
        # Adjust width based on number of steps
        grid_width = num_steps * 128
        svg_to_png(grid_svg_path, grid_png_path, width=grid_width, height=256)
        print(f"Saved grid: {grid_png_path}")
    except Exception as e:
        print(f"Warning: Could not convert grid SVG to PNG: {e}")
    
    print(f"\nTemperature variation complete! Results saved to: {output_dir}")
    print(f"Temperature range: {temp_min} to {temp_max}")
    sess.close()


def main():
    parser = argparse.ArgumentParser(description='Generate sketches with varying temperatures from a single image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input PNG image')
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='outputs/snapshot',
                        help='Directory containing the trained model')
    parser.add_argument('--output_dir', type=str, default='outputs/temperature',
                        help='Directory to save results')
    parser.add_argument('--num_steps', type=int, default=21,
                        help='Number of temperature steps (default: 21)')
    parser.add_argument('--temp_min', type=float, default=0.1,
                        help='Minimum temperature (default: 0.1)')
    parser.add_argument('--temp_max', type=float, default=2.1,
                        help='Maximum temperature (default: 2.1)')
    
    args = parser.parse_args()
    
    vary_temperature(
        args.image,
        args.data_dir,
        args.model_dir,
        args.output_dir,
        args.num_steps,
        args.temp_min,
        args.temp_max
    )


if __name__ == '__main__':
    main()