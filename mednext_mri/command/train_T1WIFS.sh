#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python -m manafaln.apps.train --config config/config_train_t1.yaml --seed 42 $@
