#!/bin/bash

python -m manafaln.apps.validate \
    -c config/train.yaml \
    -w lightning_logs/version_$1/hparams.yaml \
    -d lightning_logs/version_$1/hparams.yaml \
    -f lightning_logs/version_$1/checkpoints/best_model.ckpt \
    $@
