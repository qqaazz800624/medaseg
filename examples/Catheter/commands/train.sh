#!/bin/bash

python -m manafaln.apps.train \
    -s 42 \
    -c ${1:-"config/train.yaml"}
