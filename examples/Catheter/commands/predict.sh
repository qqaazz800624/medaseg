#!/bin/bash

python -m manafaln.apps.predict \
    -c config/train.yaml \
    -f lightning_logs/version_$1/checkpoints/best_model.ckpt \
    $@
