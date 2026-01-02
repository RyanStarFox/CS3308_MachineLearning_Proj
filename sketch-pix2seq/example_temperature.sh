#!/bin/bash

# Example 1:
python change_temperature.py \
    --image Pictures/t0_r.png \
    --data_dir datasets \
    --model_dir outputs/snapshot \
    --output_dir outputs/temperature/angel \
    --num_steps 10 \
    --temp_min 0.1 \
    --temp_max 1.9

python change_temperature.py \
    --image Pictures/t8_r.png \
    --data_dir datasets \
    --model_dir outputs/snapshot \
    --output_dir outputs/temperature/cake \
    --num_steps 10 \
    --temp_min 0.1 \
    --temp_max 1.9