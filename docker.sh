#!/bin/bash

WORKDIR=$PWD

docker run \
  -it --rm \
  -u 1083:1091 \
  --ipc=host \
  --net=host \
  --gpus=all \
  -v $WORKDIR/apps:/opt/monailabel/apps \
  -v $WORKDIR/datasets:/opt/monailabel/datasets \
  projectmonai/monailabel:latest \
  monailabel start_server --app apps/radiology \
  --studies datasets/Task09_Spleen/imagesTr \
  --conf models segmentation_spleen \
  --conf skip_scoring false \
  --conf skip_strategies false \
  --conf epistemic_enabled true
#  /bin/bash
