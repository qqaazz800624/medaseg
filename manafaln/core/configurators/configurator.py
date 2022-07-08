import logging
from abc import ABC, abstractmethod 
from argparse import ArgumentParser
from typing import Dict

from ruamel.yaml import YAML

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

def load_config(config_path: str) -> Dict:
    loader = YAML()
    with open(config_path) as f:
        config = loader.load(f)
    return config

