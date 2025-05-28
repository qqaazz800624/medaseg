#!/bin/bash

# Set default CUDA device to use
DEFAULT_CUDA_DEVICE="1"

# Allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${1:-$DEFAULT_CUDA_DEVICE}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

python -m manafaln.apps.train \
       --config configs/config_train_hypo_mri_nnUNet.yaml \
       --seed 42 $@