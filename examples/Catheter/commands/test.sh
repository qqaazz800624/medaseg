#!/bin/bash

python -m manafaln.apps.validate \
    -c config/train.yaml \
    -f lightning_logs/version_$1/checkpoints/best_model.ckpt \
    -d config/test_data.yaml \
    $@
