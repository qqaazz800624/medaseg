#!/bin/bash

HOST=$(curl -s https://ifconfig.co)
LOGDIR=lightning_logs

tensorboard --host $HOST --logdir $LOGDIR --port 8013 $@
