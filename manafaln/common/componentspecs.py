from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
if torch.__version__ > "2.0.0":
    from torch.optim.lr_scheduler import LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics import Metric as MetricV2
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule
)
from pytorch_lightning.loggers.logger import Logger
from monai.metrics import Metric
from monai.transforms import Transform

from manafaln.common.constants import ComponentType
from manafaln.common.libspecs import (
    LibSpecNative,
    LibSpecMONAI,
    LibSpecPyTorch,
    LibSpecPyTorchLightning,
    LibSpecTorchMetrics,
    LibSpecTorchVision
)

# Aliases for libraries
_Native       = LibSpecNative
_MONAI        = LibSpecMONAI
_PyTorch      = LibSpecPyTorch
_Lightning    = LibSpecPyTorchLightning
_TorchMetrics = LibSpecTorchMetrics
_TorchVision  = LibSpecTorchVision

class ComponetSpecMeta(type):
    def __new__(mcls, name, base, attrs):
        if not "TYPE" in attrs.keys():
            raise AttributeError(f"TYPE not defined in ComponentSpec {name}")
        if not "PROVIDERS" in attrs.keys():
            raise AttributeError(f"PROVIDERS not defined in ComponentSpec {name}")
        if not "INSTANCE_TYPE" in attrs.keys():
            raise AttributeError(f"COMPONENT_TYPE not defined in ComponentSpec {name}")
        return super().__new__(mcls, name, base, attrs)

class ComponentSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.UNKNOWN
    PROVIDERS = []
    INSTANCE_TYPE = Any

class ModelSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.MODEL
    PROVIDERS = [_Native, _MONAI, _TorchVision]
    INSTANCE_TYPE = nn.Module

class LossSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.LOSS
    PROVIDERS = [_Native, _MONAI, _PyTorch]
    INSTANCE_TYPE = Callable

class InfererSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.INFERER
    PROVIDERS = [_MONAI]
    INSTANCE_TYPE = Callable

class OptimizerSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.OPTIMIZER
    PROVIDERS = [_PyTorch, _MONAI]
    INSTANCE_TYPE = Optimizer

class SchedulerSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.SCHEDULER
    PROVIDERS = [_MONAI, _PyTorch]
    INSTANCE_TYPE = LRScheduler

class MetricSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.METRIC
    PROVIDERS = [_Native, _MONAI]
    INSTANCE_TYPE = Metric

class DatasetSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.DATASET
    PROVIDERS = [_MONAI, _TorchVision]
    INSTANCE_TYPE = Dataset

class DataLoaderSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.DATALOADER
    PROVIDERS = [_MONAI]
    INSTANCE_TYPE = DataLoader

class TransformSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.TRANSFORM
    PROVIDERS = [_Native, _MONAI]
    INSTANCE_TYPE = Transform

class DataModuleSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.DATAMODULE
    PROVIDERS = [_Native]
    INSTANCE_TYPE = LightningDataModule

class WorkflowSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.WORKFLOW
    PROVIDERS = [_Native]
    INSTANCE_TYPE = LightningModule

class CallbackSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.CALLBACK
    PROVIDERS = [_Native, _Lightning]
    INSTANCE_TYPE = Callback

class LoggerSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.LOGGER
    PROVIDERS = [_Native, _Lightning]
    INSTANCE_TYPE = Logger

class MetricV2Spec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.METRICV2
    PROVIDERS = [_Native, _TorchMetrics]
    INSTANCE_TYPE = MetricV2

class SamplerSpec(metaclass=ComponetSpecMeta):
    TYPE = ComponentType.SAMPLER
    PROVIDERS = [_Native, _PyTorch]
    INSTANCE_TYPE = Sampler

ComponentSpecs = {
    "UNKNOWN": ComponentSpec,
    "MODEL": ModelSpec,
    "LOSS": LossSpec,
    "INFERER": InfererSpec,
    "OPTIMIZER": OptimizerSpec,
    "SCHEDULER": SchedulerSpec,
    "METRIC": MetricSpec,
    "DATASET": DatasetSpec,
    "DATALOADER": DataLoaderSpec,
    "TRANSFORM": TransformSpec,
    "DATAMODULE": DataModuleSpec,
    "WORKFLOW": WorkflowSpec,
    "CALLBACK": CallbackSpec,
    "LOGGER": LoggerSpec,
    "SAMPLER": SamplerSpec,
    "METRICV2": MetricV2Spec
}
