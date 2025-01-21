#!/bin/bash

python -W ignore::UserWarning: \
  -m manafaln.apps.train \
  --config config/config_train.yaml
