import logging
from abc import ABC, abstractmethod 
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict

from ruamel.yaml import YAML

def load_config(config_path: str) -> Dict:
    loader = YAML()
    with open(config_path) as f:
        config = loader.load(f)
    return config

class Configurator(ABC):
    def __init__(self, app_name=None, description=None):
        self.app_parser = ArgumentParser(
            prog=app_name,
            description=description
        )

        self.config_trainer  = None
        self.config_data     = None
        self.config_workflow = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def process_args(self, args) -> None:
        self.args = args

    @abstractmethod
    def configure_trainer(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def configure_data(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def configure_workflow(self) -> Dict:
        raise NotImplementedError

    def validate_config(self) -> None:
        if self.config_trainer is None:
            raise RuntimeError("Unable to configure trainer.")
        if self.config_data is None:
            raise RuntimeError("Unable to configure data.")
        if self.config_workflow is None:
            raise RuntimeError("Unable to configure workflow.")

    def configure(self) -> None:
        # Ignore unknown args
        args, _ = self.app_parser.parse_known_args()

        # Do whatever you like
        self.process_args(args)

        # Generate target configurations
        self.config_trainer  = self.configure_trainer()
        self.config_data     = self.configure_data()
        self.config_workflow = self.configure_workflow()

        # Check if configurations are exist
        self.validate_config()

    def get_trainer_config(self) -> Dict:
        return self.config_trainer

    def get_data_config(self) -> Dict:
        return self.config_data

    def get_workflow_config(self) -> Dict:
        return self.config_workflow

class DefaultConfigurator(Configurator):
    def __init__(self, app_name=None, description=None):
        super().__init__(app_name=app_name, description=description)

        self.app_parser.add_argument(
            "--config", "-c", type=str, default=None, help="Path to config file."
        )
        self.app_parser.add_argument(
            "--trainer", "-t", type=str, default=None, help="Path to trainer config file."
        )
        self.app_parser.add_argument(
            "--data", "-d", type=str, default=None, help="Path to data config file."
        )
        self.app_parser.add_argument(
            "--workflow", "-w", type=str, default=None, help="Path to workflow config file."
        )
        self.app_parser.add_argument(
            "--ckpt", "-f", type=str, default=None, help="Path to checkpoint file."
        )

    def process_args(self, args) -> None:
        if args.config:
            config = load_config(args.config)
            self.logger.info(f"Load global configuration from {args.config}.")
        elif None in [args.trainer, args.data, args.workflow]:
            # At least one configuration is missing
            raise ValueError(
                "Must provide a complete config or all three trainer, data and workflow configs."
            )
        else:
            # Config details from different config files
            config = OrderedDict()

        # Override the configuration by component config file
        for f, c in zip([args.trainer, args.data, args.workflow], ["trainer", "data", "workflow"]):
            if f:
                config_f = load_config(f)
                config[c] = config_f[c]
                self.logger.info(f"Load {c} configuration from {f}.")

        self.raw_config = config
        self.ckpt_path = args.ckpt

    def configure_trainer(self) -> Dict:
        return self.raw_config["trainer"]

    def configure_data(self) -> Dict:
        return self.raw_config["data"]

    def configure_workflow(self) -> Dict:
        return self.raw_config["workflow"]

    def validate_config(self) -> None:
        super().validate_config()

    def get_ckpt_path(self) -> str:
        return self.ckpt_path

