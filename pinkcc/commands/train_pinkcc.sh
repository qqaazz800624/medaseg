#!/bin/bash

# Set default CUDA device to use
DEFAULT_CUDA_DEVICE="0"

# Allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${1:-$DEFAULT_CUDA_DEVICE}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

python -m manafaln.apps.train \
       --config configs/config_finetune_pinkcc_segresnet.yaml \
       --seed 42 $@