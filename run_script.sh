#!/bin/bash

# Define the Python script to run
SCRIPT_NAME="./sample_condition.py"

echo "Starting model run..."

# Run the python script with the specified arguments
python $SCRIPT_NAME \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/spad_config.yaml \
    --gpu 0

echo "Run complete."