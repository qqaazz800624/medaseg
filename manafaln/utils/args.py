import json
from typing import Dict
from argparse import ArgumentParser

from pytorch_lightning import Trainer

def parse_trainer_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Training config file"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        required=False,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        required=False,
        help="Initial training leraning rate"
    )

    # Ignore lightning trainer args first
    args, _ = parser.parse_known_args()
    return args

def load_training_config(config_file: str) -> Dict:
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def configure_training(args, config):
    trainer  = config["trainer"]
    data     = config["data"]
    workflow = config["workflow"]

    # Overwrite data settings
    if batch_size := getattr(args, "batch_size", None):
        data["training"]["dataloader"]["args"]["batch_size"] = batch_size

    # Overwrite workflow settings
    lr = getattr(args, "learning_rate", None)
    lr = lr or workflow["settings"].get("learning_rate", None)
    if lr is not None:
        workflow["components"]["optimizer"]["args"]["lr"] = lr

    # Overwrite trainer setting with args
    parser = ArgumentParser(conflict_handler="resolve")
    parser = Trainer.add_argparse_args(parser)

    # Overwrite default values with config values
    for key, value in trainer["settings"].items():
        parser.add_argument(f"--{key}", default=value)

    trainer_args, _ = parser.parse_known_args()
    trainer["settings"] = trainer_args.__dict__

    return trainer, data, workflow

