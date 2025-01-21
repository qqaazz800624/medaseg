import os
from typing import Any, Callable, Sequence, Tuple
import logging

import torch
import torch.nn as nn
from torch.optim import Optimizer
from monai.data import partition_dataset, Dataset, DataLoader
from monai.inferers import Inferer
from monai.handlers import LrScheduleHandler
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

from manafaln.core.builders import (
    ModelBuilder,
    OptimizerBuilder,
    SchedulerBuilder,
    LossBuilder,
    TransformBuilder,
    InfererBuilder,
    DatasetBuilder,
    DataLoaderBuilder
)

logger = logging.getLogger(__name__)


class ManafalnTrainTask(BasicTrainTask):
    def __init__(
        self,
        config: dict,
        labels: Sequence[str],
        model_dir: os.PathLike[str],
        load_path: os.PathLike[str],
        publish_path: os.PathLike[str],
        model_dict_key: str,
        **kwargs
    ) -> None:
        builder = ModelBuilder()
        self._network = builder(config["model"])

        super().__init__(
            model_dir,
            description=config["description"],
            amp=True,
            load_path=load_path,
            load_dict=None,
            publish_path=publish_path,
            stats_path=None,
            train_save_interval=config["train_save_interval"],
            val_interval=config["val_interval"],
            n_saved=config["n_saved"],
            final_filename="last.ckpt",
            key_metric_filename="best_model.ckpt",
            model_dict_key=model_dict_key,
            find_unused_parameters=config["find_unused_parameters"],
            load_strict=config["load_strict"],
            labels=labels,
            disable_meta_tracking=False,
        )

        # Update config
        self._config.update(config["config"])
        self._manafaln_config = config

    def network(self, context: Context) -> nn.Module:
        return self._network

    def optimizer(self, context: Context) -> Optimizer:
        builder = OptimizerBuilder()
        optimizer = builder(
            self._manafaln_config["optimizer"],
            context.network.parameters()
        )
        return optimizer

    def loss_function(self, context: Context) -> Callable:
        builder = LossBuilder()
        return builder(self._manafaln_config["loss"])

    def lr_scheduler_handler(self, context: Context) -> LrScheduleHandler:
        builder = SchedulerBuilder()
        lr_scheduler = builder(
            self._manafaln_config["scheduler"],
            context.optimizer
        )
        return LrScheduleHandler(lr_scheduler, print_lr=True)

    def _dataset(
        self,
        context: Context,
        datalist: Sequence[Any],
        is_train: bool
    ) -> Tuple[Dataset, Sequence[Any]]:
        if context.multi_gpu:
            world_size = torch.distributed.get_world_size()
            if len(datalist) // world_size:
                splits = partition_dataset(
                    data=datalist,
                    num_partitions=world_size,
                    even_divisible=True
                )
                datalist = splits[context.local_rank]

        if is_train:
            transforms = self._validate_transforms(
                self.train_pre_transforms(context),
                "Training", "pre"
            )
        else:
            transforms = self._validate_transforms(
                self.val_pre_transforms(context),
                "Validation", "pre"
            )

        builder = DatasetBuilder()
        dataset = builder(
            self._manafaln_config["dataset"],
            [datalist],
            {"transform": transforms}
        )
        return dataset, datalist

    def _dataloader(
        self,
        context: Context,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool = False
    ) -> DataLoader:
        builder = DataLoaderBuilder()

        # We are going to patch the settings, make a copy for it
        config = dict(self._manafaln_config["dataloader"])
        config["args"] = config.get("args", {})
        config["args"]["batch_size"] = batch_size
        config["args"]["num_workers"] = num_workers
        config["args"]["shuffle"] = shuffle

        return builder(config, dataset)

    def train_pre_transforms(self, context: Context) -> Sequence[Callable]:
        config = self._manafaln_config["training"]["pre_transforms"]
        builder = TransformBuilder()
        return [builder(t) for t in config]

    def train_post_transforms(self, context: Context) -> Sequence[Callable]:
        config = self._manafaln_config["training"]["post_transforms"]
        builder = TransformBuilder()
        return [builder(t) for t in config]

    def val_pre_transforms(self, context: Context) -> Sequence[Callable]:
        config = self._manafaln_config["validation"]["pre_transforms"]
        builder = TransformBuilder()
        return [builder(t) for t in config]

    def val_post_transforms(self, context: Context) -> Sequence[Callable]:
        config = self._manafaln_config["validation"]["post_transforms"]
        builder = TransformBuilder()
        return [builder(t) for t in config]

    def train_inferer(self, context: Context) -> Inferer:
        builder = InfererBuilder()
        return builder(self._manafaln_config["inferers"]["train_inferer"])

    def val_inferer(self, context: Context) -> Inferer:
        builder = InfererBuilder()
        return builder(self._manafaln_config["inferers"]["val_inferer"])

