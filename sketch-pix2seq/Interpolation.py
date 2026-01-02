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


def interpolate_images(image_path1, image_path2, data_dir, model_dir, output_dir, 
                       num_steps=10, temperature=0.1):
    """
    Perform spherical interpolation between two input images.
    
    Args:
        image_path1: Path to first PNG image
        image_path2: Path to second PNG image
        data_dir: Directory containing the dataset
        model_dir: Directory containing the trained model
        output_dir: Directory to save interpolation results
        num_steps: Number of interpolation steps (default: 10)
        temperature: Sampling temperature (default: 0.1)
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
    
    # Load and encode input images
    print(f"Loading image 1: {image_path1}")
    img1 = load_image(image_path1, eval_model.hps.img_H, eval_model.hps.img_W)
    z0 = encode(img1, sess, eval_model)
    print(f"Encoded z0 shape: {z0.shape}")
    
    print(f"Loading image 2: {image_path2}")
    img2 = load_image(image_path2, eval_model.hps.img_H, eval_model.hps.img_W)
    z1 = encode(img2, sess, eval_model)
    print(f"Encoded z1 shape: {z1.shape}")
    
    # Perform spherical interpolation
    print(f"Performing spherical interpolation with {num_steps} steps...")
    z_list = []
    for t in np.linspace(0, 1, num_steps):
        z_interp = slerp(z0, z1, t)
        z_list.append(z_interp)
    
    # Decode and save each interpolated latent vector
    print("Decoding and saving interpolated images...")
    reconstructions = []
    for i in range(num_steps):
        print(f"Generating image {i+1}/{num_steps}...")
        strokes_out_3, strokes_out_5 = decode(
            sess,
            sample_model,
            eval_model.hps.max_seq_len,
            z_list[i],
            temperature=temperature
        )
        
        # Save individual SVG
        svg_path = os.path.join(output_dir, f'interpolation_{i:03d}.svg')
        draw_strokes(strokes_out_3, svg_path)
        
        # Convert SVG to PNG
        png_path = os.path.join(output_dir, f'interpolation_{i:03d}.png')
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
    grid_svg_path = os.path.join(output_dir, 'interpolation_grid.svg')
    draw_strokes(stroke_grid, grid_svg_path)
    
    # Convert grid to PNG
    grid_png_path = os.path.join(output_dir, 'interpolation_grid.png')
    try:
        svg_to_png(grid_svg_path, grid_png_path, width=1024, height=256)
        print(f"Saved grid: {grid_png_path}")
    except Exception as e:
        print(f"Warning: Could not convert grid SVG to PNG: {e}")
    
    print(f"\nInterpolation complete! Results saved to: {output_dir}")
    sess.close()


def main():
    parser = argparse.ArgumentParser(description='Interpolate between two sketch images')
    parser.add_argument('--image1', type=str, required=True,
                        help='Path to first PNG image')
    parser.add_argument('--image2', type=str, required=True,
                        help='Path to second PNG image')
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='outputs/snapshot',
                        help='Directory containing the trained model')
    parser.add_argument('--output_dir', type=str, default='outputs/interpolation',
                        help='Directory to save interpolation results')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of interpolation steps')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    
    args = parser.parse_args()
    
    interpolate_images(
        args.image1,
        args.image2,
        args.data_dir,
        args.model_dir,
        args.output_dir,
        args.num_steps,
        args.temperature
    )


if __name__ == '__main__':
    main()