Usage:

 - Training
  - Use `commands/train.sh` for single GPU training
  - Use `commands/train_mgpu.sh` for two GPU training

 - Validation
  - Run `python -m manafaln.apps.validate --config config/config_train.json --ckpt $CKPT_PATH` for validation.

 - Testing
  - Run `python -m manafaln.apps.validate --config config/config_test.json --ckpt $CKPT_PATH` for testing.

 - Inference
  - Run `python -m manafaln.apps.inference --config config/config_test.json --ckpt $CKPT_PATH` to run inference of the whole dataset.

