#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m manafaln.apps.train --config config/config_train_t1_prect.yaml --seed 42 $@
