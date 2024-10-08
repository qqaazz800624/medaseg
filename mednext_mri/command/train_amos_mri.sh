#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python -m manafaln.apps.train --config config/config_train_amos_mri.yaml --seed 42 $@
