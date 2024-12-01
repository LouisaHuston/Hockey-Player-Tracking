#!/bin/bash

# Define the working directory
HOCKEY_WORK_DIR="hockey_work_dir"

# Create the directory for the UAV work
mkdir -p $HOCKEY_WORK_DIR

# Export the CUDA home directory
export CUDA_HOME=/usr/local/cuda

# Always train on one GPU with the specific config
echo "Starting training..."
if ! python3 train.py configs/co_dino_hockey/co_dino_5scale_r50_1x.py --work-dir "$HOCKEY_WORK_DIR"; then
  echo "Error: Python script execution failed."
  kill $PID1
  exit 1
fi

# Ensure that the sync process is terminated after the script completes:
kill $PID1
