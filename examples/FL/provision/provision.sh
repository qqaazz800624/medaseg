#!/bin/bash

# Run provision
mkdir -p build
provision -p project.yml -w $PWD/build
