import json
from typing import Dict
from argparse import ArgumentParser

from pytorch_lightning import Trainer

def parse_train_args():
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

def parse_validate_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Training config file for restoring checkpoint"
    )

    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="(Optional) Use this option to override data in training config"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint"
    )

    args, _ = parser.parse_known_args()
    return args

def load_validate_config(config_train: str, config_data: str = None) -> Dict:
    with open(config_train) as fc:
        config = json.load(fc)

    if config_data:
        with open(config_data) as fd:
            data = json.load(fd)
        # Validate the format of data config
        # can be an independent json file or other training config
        keys = data.keys()
        if "name" in keys or "path" in keys:
            config["data"] = data
        elif "data" in keys:
            config["data"] = data["data"]
        else:
            raise ValueError("Data config must contains data section or data module name")

    return config

def configure_validation(args, config):
    trainer  = config["trainer"]
    data     = config["data"]
    workflow = config["workflow"]

    # Overwrite trainer setting with args
    parser = ArgumentParser(conflict_handler="resolve")
    parser = Trainer.add_argparse_args(parser)

    # Overwrite default values with config values
    for key, value in trainer["settings"].items():
        parser.add_argument(f"--{key}", default=value)

    trainer_args, _ = parser.parse_known_args()
    trainer["settings"] = trainer_args.__dict__

    return trainer, data, workflow

