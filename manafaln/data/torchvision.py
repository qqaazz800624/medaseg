import json
from typing import Any, Callable, Dict, Hashable, Sequence, Tuple, Optional
from logging import getLogger

import torch
import monai
from monai.data import Dataset, DataLoader
from torch.utils.data import Subset
from lightning import LightningDataModule
from manafaln.core.builders import DatasetBuilder, DataLoaderBuilder
from manafaln.core.transforms import build_transforms


class TorchVisionDataset(Dataset):
    def __init__(
        self,
        name: str,
        path: str = "torchvision.datasets",
        keys: Sequence[str] = ["image", "label"],
        transform: Optional[Callable] = None,
        **kwargs
    ):
        self.logger = getLogger(self.__class__.__name__)

        if kwargs.pop("target_transform", None) is not None:
            self.logger.warn(
                "Remove unsupported argument `target_transform`."
            )

        builder = DatasetBuilder()
        self.dataset = builder(
            config={
                "name": name,
                "path": path
            },
            args=[],
            kwargs=kwargs
        )

        self.keys = keys
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[Hashable, Any]:
        data = self.dataset[index]

        if len(self.keys) != len(data):
            raise AssertionError(
                f"The number of keys({len(self.keys)}) does not match the number of data({len(data)})."
            )

        out: Dict[Hashable, Any] = dict(zip(self.keys, data))
        if self.transform is not None:
            out = self.transform(out)
        return out

class TorchVisionDataModule(LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters({"data": config})

        dataset = config["dataset"]
        settings = config["settings"]

        self.dataset = TorchVisionDataset(
            name=dataset["name"],
            path=dataset.get("path", "torchvision.datasets"),
            keys=dataset.get("keys", ["image", "label"]),
            **dataset.get("args", {})
        )
        self.data_list = settings["data_list"]
        self.data_split = settings["data_split"]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    @staticmethod
    def get_split_indices(data_list: str, split_key: str) -> Sequence[int]:
        with open(data_list) as f:
            data_list = json.load(f)
        return data_list[split_key]

    def build_dataset(self, config: Dict, split: str) -> Dataset:
        indices = self.get_split_indices(
            self.data_list,
            self.data_split[split]
        )

        builder = DatasetBuilder()

        transforms = build_transforms(config["transforms"])
        subset = Subset(self.dataset, indices)
        dataset = builder(config["dataset"], [subset], {"transform": transforms})

        return dataset

    def get_train_dataset(self) -> Dataset:
        if self.train_dataset is None:
            config = self.hparams.data["training"]
            self.train_dataset = self.build_dataset(config, "training")
        return self.train_dataset

    def get_val_dataset(self) -> Dataset:
        if self.val_dataset is None:
            config = self.hparams.data["validation"]
            self.val_dataset = self.build_dataset(config, "validation")
        return self.val_dataset

    def get_test_dataset(self) -> Dataset:
        if self.test_dataset is None:
            config = self.hparams.data["test"]
            self.test_dataset = self.build_dataset(config, "test")
        return self.test_dataset

    def get_predict_dataset(self) -> Dataset:
        if self.predict_dataset is None:
            config = self.hparams.data["predict"]
            self.predict_dataset = self.build_dataset(config, "predict")
        return self.predict_dataset

    def train_dataloader(self) -> DataLoader:
        builder = DataLoaderBuilder()

        config = self.hparams.data["training"]
        dataset = self.get_train_dataset()

        return builder(config["dataloader"], dataset)

    def val_dataloader(self) -> DataLoader:
        builder = DataLoaderBuilder()

        config = self.hparams.data["validation"]
        dataset = self.get_val_dataset()

        return builder(config["dataloader"], dataset)

    def test_dataloader(self) -> DataLoader:
        builder = DataLoaderBuilder()

        config = self.hparams.data["test"]
        dataset = self.get_test_dataset()

        return builder(config["dataloader"], dataset)

    def predict_dataloader(self) -> DataLoader:
        builder = DataLoaderBuilder()

        config = self.hparams.data["predict"]
        dataset = self.get_predict_dataset()

        return builder(config["dataloader"], dataset)

