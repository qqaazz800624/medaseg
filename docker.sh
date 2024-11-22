#!/bin/bash

WORKDIR=$PWD

docker run \
  -it --rm \
  --ipc=host \
  --net=host \
  -v $WORKDIR/apps:/opt/monailabel/apps \
  -v $WORKDIR/datasets:/opt/monailabel/datasets \
  projectmonai/monailabel:latest \
  /bin/bash
