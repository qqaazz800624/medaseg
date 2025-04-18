#!/bin/bash

# Set default CUDA device to use
DEFAULT_CUDA_DEVICE="2"

# Allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${1:-$DEFAULT_CUDA_DEVICE}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

python -m manafaln.apps.test \
    -c ${1:-"configs/config_finetune_pinkcc_segresnet.yaml"} \
    -f PINKCC/49cfo62r/checkpoints/best_model.ckpt \
    