#!/bin/bash

python scripts/run_analysis.py \
    --datalist /neodata/hsu/hypo/Final_dataset/HYPO_NTUH_dataset/datalist.json \
    --dataroot /neodata/hsu/hypo/Final_dataset/HYPO_NTUH_dataset \
    --data_split_keys fold_0 fold_1 fold_2 fold_3 fold_4 fold_5 fold_6 fold_7 fold_8 \
    --output scripts/analysis.json \
