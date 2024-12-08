#!/bin/bash

# Clear the terminal
clear

# Create the directory for the UAV work
mkdir -p hockey

# Export the CUDA home directory
export CUDA_HOME=/usr/local/cuda

# Set the runtime variables
WORK_DIR="hockey"
RUNTIME_CONFIG_FILE="configs/co_dino_hockey/co_dino_5scale_swin_large_16e_o365tococo.py"

# Get the number of available GPUs using nvidia-smi
RUNTIME_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "Detected $RUNTIME_GPUS GPUs."

# Check if number of GPUs is greater than 1
if [ "$RUNTIME_GPUS" -gt 1 ]; then
    # Run the main Python training distributed script
    echo "Starting distributed training..."
    if ! python3 -m torch.distributed.launch --nproc_per_node=$RUNTIME_GPUS --master_port=29300 \
        $(dirname "$0")/train.py "$RUNTIME_CONFIG_FILE" --launcher pytorch --work-dir "$SAFETY_SET"; then
      echo "Error: Distributed Python script execution failed."
      exit 1
    fi
else
    # Run the main Python training script
    echo "Starting training..."
    if ! python3 train.py "$RUNTIME_CONFIG_FILE" --work-dir "$SAFETY_SET"; then
      echo "Error: Python script execution failed."
      exit 1
    fi
fi