#!/bin/bash

HOST=$(curl -s https://ifconfig.co)
LOGDIR=models

tensorboard --host $HOST --logdir $LOGDIR $@ 
