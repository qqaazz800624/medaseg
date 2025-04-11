#!/bin/bash

python -m manafaln.apps.predict \
    -c ${1:-"config/config_train_pinkcc_nnUNet.yaml"} \
    -f PINKCC/version_0/checkpoints/best_model.ckpt 