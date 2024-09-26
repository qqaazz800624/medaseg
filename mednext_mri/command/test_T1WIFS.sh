#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python -m manafaln.apps.predict --config config/config_infer_t1.yaml --ckpt MRI/21xk6psn/checkpoints/best_model.ckpt $@
python -m manafaln.apps.predict --config config/config_infer_t1_ext.yaml --ckpt MRI/21xk6psn/checkpoints/best_model.ckpt $@