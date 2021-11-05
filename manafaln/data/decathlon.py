import os
from typing import Dict
from shutil import copytree

import torch
import monai
from monai.data.decathlon_datalist import load_decathlon_datalist
from pytorch_lightning import LightningDataModule

from manafaln.common.constants import ComponentType
from manafaln.utils.builders import (
    instantiate,
    build_transforms
)

class DecathlonDataModule(LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()

        # Data module do not load from checkpoints, but saving these
        # informations is useful for managing your experiments
        self.save_hyperparameters({"data": config})

        settings = config["settings"]

        # Must get configurations
        self.data_root       = settings["data_root"]
        self.data_list       = settings["data_list"]
        self.is_segmentation = settings["is_segmentation"]

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
        if (self.use_shm_cache) and (not os.path.exists(self.data_root)):
            # Copy the whole directory to SHM
            copytree(self.ori_data_root, self.data_root)

    def build_loader(self, phase: str):
        print(phase)
        if not phase in ["training", "validation", "test"]:
            raise ValueError(f"{phase} split is not allowed for data module")
        print(phase)

        config = self.hparams.data[phase]

        files = load_decathlon_datalist(
            data_list_file_path=self.data_list,
            is_segmentation=self.is_segmentation,
            data_list_key=config["data_list_key"],
            base_dir=self.data_root
        )
        transforms = build_transforms(config["transforms"])

        dataset = instantiate(
            name=config["dataset"]["name"],
            path=config["dataset"].get("path", None),
            component_type=ComponentType.DATASET,
            data=files,
            transform=transforms,
            **config["dataset"].get("args", {})
        )

        loader = instantiate(
            name=config["dataloader"]["name"],
            path=config["dataloader"].get("path", None),
            component_type=ComponentType.DATALOADER,
            dataset=dataset,
            **config["dataloader"].get("args", {})
        )

        return loader

    def train_dataloader(self):
        return self.build_loader(phase="training")

    def val_dataloader(self):
        return self.build_loader(phase="validation")

    def test_dataloader(self):
        return self.build_loader(phase="test")

