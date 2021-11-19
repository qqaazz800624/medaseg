import os
import sys
import json
from abc import ABC, abstractmethod
from typing import Dict
from argparse import ArgumentParser

from pytorch_lightning import Trainer

class Configurator(ABC):
    def __init__(self):
        self.app_parser = ArgumentParser()

    @abstractmethod
    def preprocess_args(self, args) -> None:
        raise NotImplementedError

    @abstractmethod
    def configure_data(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def configure_trainer(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def configure_workflow(self) -> Dict:
        raise NotImplementedError

    def configure(self) -> None:
        # Use help will quit app, so add trainer help here
        if "--help" in sys.argv or "-h" in sys.argv:
            self.app_parser = Trainer.add_argparse_args(self.app_parser)
            args = self.app_parser.parse_args()

        # Parse app arguments first
        args, _ = self.app_parser.parse_known_args()
        # Do whatever you like
        self.preprocess_args(args)

        self.config_data     = self.configure_data()
        self.config_trainer  = self.configure_trainer()
        self.config_workflow = self.configure_workflow()

    def get_data_config(self) -> Dict:
        return self.config_data

    def get_trainer_config(self) -> Dict:
        return self.config_trainer

    def get_workflow_config(self) -> Dict:
        return self.config_workflow

class TrainConfigurator(Configurator):
    def __init__(self):
        super().__init__()

        self.app_parser.add_argument(
            "--config",
            "-c",
            type=str,
            help="Training config file"
        )
        self.app_parser.add_argument(
            "--batch_size",
            "-b",
            type=int,
            required=False,
            help="Training batch size"
        )
        self.app_parser.add_argument(
            "--learning_rate",
            "-lr",
            type=float,
            required=False,
            help="Initial training leraning rate"
        )

    def preprocess_args(self, args) -> None:
        with open(args.config) as f:
            config = json.load(f)

        self.args = args
        self.config = config

    def configure_data(self) -> None:
        data = dict(self.config["data"])
        if batch_size := getattr(self.args, "batch_size", None):
            data["training"]["dataloader"]["args"]["batch_size"] = batch_size
        return data

    def configure_trainer(self) -> None:
        trainer = dict(self.config["trainer"])
        # Overwrite trainer setting with args
        parser = ArgumentParser(conflict_handler="resolve")
        parser = Trainer.add_argparse_args(parser)

        # Overwrite default values with config values
        for key, value in trainer["settings"].items():
            parser.add_argument(f"--{key}", default=value)

        trainer_args, _ = parser.parse_known_args()
        trainer["settings"] = trainer_args.__dict__

        return trainer

    def configure_workflow(self) -> None:
        workflow = dict(self.config["workflow"])
        lr = getattr(self.args, "learning_rate", None)
        lr = lr or workflow["settings"].get("learning_rate", None)
        if lr is not None:
            workflow["components"]["optimizer"]["args"]["lr"] = lr
        return workflow

class InferenceConfigurator(Configurator):
    def __init__(self):
        super().__init__()

        self.app_parser.add_argument(
            "--config",
            "-c",
            type=str,
            help="Training config file for restoring checkpoint"
        )
        self.app_parser.add_argument(
            "--data",
            "-d",
            type=str,
            default=None,
            help="(Optional) data config file"
        )
        self.app_parser.add_argument(
            "--ckpt",
            type=str,
            help="Path to the model checkpoint"
        )

    def preprocess_args(self, args) -> None:
        with open(args.config) as fc:
            config = json.load(fc)

        if config_data := getattr(args, "data", None):
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

        self.args   = args
        self.config = config

    def configure_data(self) -> None:
        return dict(self.config["data"])

    def configure_trainer(self) -> None:
        trainer = dict(self.config["trainer"])

        # Overwrite trainer setting with args
        parser = ArgumentParser(conflict_handler="resolve")
        parser = Trainer.add_argparse_args(parser)

        # Overwrite default values with config values
        for key, value in trainer["settings"].items():
            parser.add_argument(f"--{key}", default=value)

        trainer_args, _ = parser.parse_known_args()
        trainer["settings"] = trainer_args.__dict__

        return trainer

    def configure_workflow(self) -> None:
        return dict(self.config["workflow"])

    def get_ckpt_path(self):
        return self.args.ckpt

