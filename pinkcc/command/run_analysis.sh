#!/bin/bash

python scripts/run_analysis.py \
    --datalist /neodata/open_dataset/PINKCC/datalist.json \
    --dataroot /neodata/open_dataset/PINKCC \
    --data_split_keys fold_0 fold_1 fold_2 fold_3 \
    --output scripts/analysis.json \
