#!/bin/bash

# Create the directory for the UAV work
mkdir -p hockey

# Export the CUDA home directory
export CUDA_HOME=/usr/local/cuda

# Always train on one GPU with the specific config
echo "Starting training..."
if ! python3 train.py configs/co_dino_hockey/co_dino_5scale_swin_large_16e_o365tococo.py --work-dir "hockey"; then
  echo "Error: Python script execution failed."
  kill $PID1
  exit 1
fi

# Ensure that the sync process is terminated after the script completes:
kill $PID1
