#!/bin/bash

python scripts/run_analysis_new.py \
    --data-list-keys fold_0 \
    --data-list-keys fold_1 \
    --data-list-keys fold_2 \
    --data-list-keys fold_3 \
    --data-list-keys fold_4 \
    --data-list-keys fold_5 \
    --data-list-keys fold_6 \
    --data-list-keys fold_7 \
    --data-list-keys fold_8 \
    --data-list-keys fold_9 \
    --output-file scripts/analysis.json \
    /neodata/hsu/hypo/Final_dataset/HYPO_NTUH_dataset_affine_matched \
    /neodata/hsu/hypo/Final_dataset/HYPO_NTUH_dataset_affine_matched/datalist.json

