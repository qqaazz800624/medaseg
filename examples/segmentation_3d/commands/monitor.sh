#!/bin/bash

HOST=$(curl https://ifconfig.co)
PORT=8010

tensorboard --logdir models --host $HOST --port $PORT
