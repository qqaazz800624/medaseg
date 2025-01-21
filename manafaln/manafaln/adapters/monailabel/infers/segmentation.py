import os
import copy
import logging
from typing import Callable, Literal, Sequence

import torch
import torch.nn as nn
from monai.inferers import Inferer
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored

from manafaln.core.builders import (
    InfererBuilder,
    ModelBuilder,
    TransformBuilder
)

logger = logging.getLogger(__name__)


class ManafalnInferTask(BasicInferTask):
    def __init__(
        self,
        config: dict,
        model_path: Sequence[os.PathLike[str]],
        labels: Sequence[str],
        model_type: Literal["checkpoint", "torchscript"] = "checkpoint",
        model_state_dict: str = "state_dict",
        **kwargs
    ):
        network = None
        if model_type == "checkpoint":
            model_builder = ModelBuilder()
            network = model_builder(config["model"])

        super().__init__(
            path=model_path,
            network=network,
            type=InferType.SEGMENTATION,
            labels=labels,
            dimension=config["dimension"],
            description=config["description"],
            model_state_dict=model_state_dict,
            input_key="image",
            output_label_key="pred",
            roi_size=config["roi_size"],
            load_strict=config["load_strict"],
            **kwargs
        )

        # BasicInferTask uses _config to save configuration options,
        # to avoid conflicts, here we use _manafaln_config.
        self._manafaln_config = config

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        builder = TransformBuilder()
        return [builder(t) for t in self._manafaln_config["pre_transforms"]]

    def inferer(self, data=None) -> Inferer:
        builder = InfererBuilder()
        return builder(self._manafaln_config["inferer"])

    def inverse_transforms(self, data=None) -> Sequence[Callable]:
        builder = TransformBuilder()
        return [builder(t) for t in self._manafaln_config["inv_transforms"]]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        builder = TransformBuilder()
        return [builder(t) for t in self._manafaln_config["post_transforms"]]

