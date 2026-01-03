#!/bin/bash
# Example script to run interpolation between two images

# Example 1: Interpolate between 
python Interpolation.py \
    --image1 Pictures/z0_i5.png \
    --image2 Pictures/z1_i5.png \
    --data_dir datasets \
    --model_dir outputs/snapshot \
    --output_dir outputs/interpolation/angel_cake \
    --num_steps 10 \
    --temperature 0.1

# Example 2: Interpolate between 
python Interpolation.py \
    --image1 Pictures/z0_i6.png \
    --image2 Pictures/z1_i6.png \
    --data_dir datasets \
    --model_dir outputs/snapshot \
    --output_dir outputs/interpolation/airplane_bus \
    --num_steps 10 \
    --temperature 0.1
