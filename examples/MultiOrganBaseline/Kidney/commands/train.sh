#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD/custom:$PYTHONPATH

python -m manafaln.apps.train -c config/config_train.json
