import os
from typing import Dict, List
from shutil import copytree

import torch
import monai
from monai.data.decathlon_datalist import load_decathlon_datalist
from monai.transforms import Compose
from pytorch_lightning import LightningDataModule

from manafaln.core.builders import (
    DatasetBuilder,
    DataLoaderBuilder,
    TransformBuilder
)
from manafaln.core.transforms import build_transforms

class DecathlonDataModule(LightningDataModule):
    """
    Data module for loading and preparing datasets for the Decathlon challenge.

    Args:
        config (Dict): Configuration dictionary for the data module.

    Attributes:
        data_list (str): Path to the data list file.
        data_root (str): Root directory of the data.
        is_segmentation (bool): Flag indicating whether the task is segmentation or not.
        use_shm_cache (bool): Flag indicating whether to use shared memory cache or not.
        shm_cache_path (str): Path to the shared memory cache.
        ori_data_root (str): Original data root directory.

    """

    def __init__(self, config: Dict):
        super().__init__()

        # Data module do not load from checkpoints, but saving these
        # informations is useful for managing your experiments
        self.save_hyperparameters({"data": config})

        settings = config["settings"]

        # Must get configurations
        self.data_list       = settings["data_list"]
        self.data_root       = settings.get("data_root", None)
        self.is_segmentation = settings.get("is_segmentation", True)

        # Optional configurations
        # Use SHM if you have large size ram disk
        self.use_shm_cache = settings.get("use_shm_cache", False)
        self.shm_cache_path = settings.get("shm_cache_path", ".")

        if self.use_shm_cache:
            self.ori_data_root = self.data_root
            self.data_root = os.path.join(
                self.shm_cache_path,
                os.path.basename(self.data_root)
            )

    def prepare_data(self):
        """
        Prepares the data for training by copying the data to shared memory cache if necessary.
        """
        if (self.use_shm_cache) and (not os.path.exists(self.data_root)):
            # Copy the whole directory to SHM
            copytree(self.ori_data_root, self.data_root)

    def build_dataset(self, config: dict):
        """
        Builds the dataset for a given configuration.

        Args:
            config (dict): Configuration dictionary for the dataset.

        Returns:
            dataset: The built dataset.

        """
        if isinstance(config["data_list_key"], str):
            files = load_decathlon_datalist(
                data_list_file_path=self.data_list,
                is_segmentation=self.is_segmentation,
                data_list_key=config["data_list_key"],
                base_dir=self.data_root
            )
        else:
            files = []
            for key in config["data_list_key"]:
                subset = load_decathlon_datalist(
                    data_list_file_path=self.data_list,
                    is_segmentation=self.is_segmentation,
                    data_list_key=key,
                    base_dir=self.data_root
                )
                files = files + subset

        builder = DatasetBuilder()

        transforms = build_transforms(config["transforms"])
        dataset = builder(config["dataset"], [files], { "transform": transforms })

        return dataset

    def get_train_dataset(self):
        """
        Returns the training dataset.

        Returns:
            dataset: The training dataset.

        """
        dataset = getattr(self, "train_dataset", None)
        if dataset is None:
            config = self.hparams.data["training"]
            self.train_dataset = self.build_dataset(config)
        return self.train_dataset

    def get_val_dataset(self):
        """
        Returns the validation dataset.

        Returns:
            dataset: The validation dataset.

        """
        dataset = getattr(self, "val_dataset", None)
        if dataset is None:
            config = self.hparams.data["validation"]
            self.val_dataset = self.build_dataset(config)
        return self.val_dataset

    def get_test_dataset(self):
        """
        Returns the test dataset.

        Returns:
            dataset: The test dataset.

        """
        dataset = getattr(self, "test_dataset", None)
        if dataset is None:
            config = self.hparams.data["test"]
            self.test_dataset = self.build_dataset(config)
        return self.test_dataset

    def get_predict_dataset(self):
        """
        Returns the predict dataset.

        Returns:
            dataset: The predict dataset.

        """
        dataset = getattr(self, "predict_dataset", None)
        if dataset is None:
            config = self.hparams.data["predict"]
            self.predict_dataset = self.build_dataset(config)
        return self.predict_dataset

    def build_loader(self, phase: str):
        """
        Builds the data loader for a given phase.

        Args:
            phase (str): The phase of the data loader.

        Returns:
            loader: The built data loader.

        Raises:
            ValueError: If the phase is not allowed for the data module.

        """
        phase_to_dataset = {
            "training": self.get_train_dataset,
            "validation": self.get_val_dataset,
            "test": self.get_test_dataset,
            "predict": self.get_predict_dataset
        }

        if not phase in phase_to_dataset.keys():
            raise ValueError(f"{phase} split is not allowed for data module")

        # Create data loader builder
        builder = DataLoaderBuilder()

        # Get dataset
        dataset = phase_to_dataset[phase]()

        # Build data loader
        config = self.hparams.data[phase]
        loader = builder(config["dataloader"], dataset)

        return loader

    def train_dataloader(self):
        """
        Returns the data loader for training.

        Returns:
            loader: The data loader for training.

        """
        return self.build_loader(phase="training")

    def val_dataloader(self):
        """
        Returns the data loader for validation.

        Returns:
            loader: The data loader for validation.

        """
        return self.build_loader(phase="validation")

    def test_dataloader(self):
        """
        Returns the data loader for testing.

        Returns:
            loader: The data loader for testing.

        """
        return self.build_loader(phase="test")

    def predict_dataloader(self):
        """
        Returns the data loader for prediction.

        Returns:
            loader: The data loader for prediction.

        """
        return self.build_loader(phase="predict")
