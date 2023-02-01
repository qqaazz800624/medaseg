from typing import Dict

from manafaln.core.builders import DataLoaderBuilder
from manafaln.data.decathlon import DecathlonDataModule


class SemiSupervisedDataModule(DecathlonDataModule):
    def get_train_datasets(self):
        datasets = getattr(self, "train_datasets", None)
        if datasets is None:
            config = self.hparams.data["training"]
            self.train_datasets = {}

            labeled_config = config["labeled"]
            labeled_dataset = self.build_dataset(labeled_config)
            self.train_datasets["labeled"] = labeled_dataset

            unlabeled_config = config["unlabeled"]
            unlabeled_dataset = self.build_dataset(unlabeled_config)
            self.train_datasets["unlabeled"] = unlabeled_dataset

        return self.train_datasets

    def train_dataloader(self) -> Dict:
        # Create data loader builder
        builder = DataLoaderBuilder()

        # Get dataset
        datasets = self.get_train_datasets()

        # Build data loaders
        configs = self.hparams.data["training"]

        loaders = {}

        labeled_dataset = datasets["labeled"]
        labeled_loader_config = configs["labeled"]["dataloader"]
        labeled_dataloader = builder(labeled_loader_config, labeled_dataset)
        loaders["labeled"] = labeled_dataloader

        unlabeled_dataset = datasets["unlabeled"]
        unlabeled_loader_config = configs["unlabeled"]["dataloader"]
        unlabeled_dataloader = builder(unlabeled_loader_config, unlabeled_dataset)
        loaders["unlabeled"] = unlabeled_dataloader

        return loaders
