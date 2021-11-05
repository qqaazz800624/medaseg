import json
from typing import Dict
from argparse import ArgumentParser

from pytorch_lightning import Trainer

def parse_trainer_args():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Training config file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        help="Initial training learning rate"
    )

    return parser.parse_args()

def load_training_config(config_file: str) -> Dict:
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def configure_training_args(args, config):
    trainer  = config["trainer"]
    data     = config["data"]
    workflow = config["workflow"]

    # Overwrite data settings
    if hasattr(args, "batch_size"):
        data["training"]["dataloader"]["args"]["batch_size"] = args.batch_size

    # Overwrite workflow settings
    lr = getattr(args, "learning_rate", None)
    lr = lr or workflow["settings"].get("learning_rate", None)
    if lr is not None:
        workflow["components"]["optimizer"]["args"]["lr"] = lr

    # Overwrite trainer setting with args
    skips = ["config", "batch_size", "learning_rate"]
    for key in args.__dict__.keys():
        if not key in skips:
            trainer["settings"][key] = getattr(args, key)

    return trainer, data, workflow
