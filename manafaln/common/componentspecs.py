from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer

if torch.__version__ >= "2.0.0":
    from torch.optim.lr_scheduler import LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics import Metric as MetricV2
from lightning import Callback, LightningDataModule, LightningModule
from lightning.pytorch.loggers import Logger
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
    """
    Base class for component specifications.

    Attributes:
        TYPE (ComponentType): The type of the component.
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance.
    """
    TYPE = ComponentType.UNKNOWN
    PROVIDERS = []
    INSTANCE_TYPE = Any

class ModelSpec(metaclass=ComponetSpecMeta):
    """
    Specification for model components.

    Attributes:
        TYPE (ComponentType): The type of the component (MODEL).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (nn.Module).
    """
    TYPE = ComponentType.MODEL
    PROVIDERS = [_Native, _MONAI, _TorchVision]
    INSTANCE_TYPE = nn.Module

class LossSpec(metaclass=ComponetSpecMeta):
    """
    Specification for loss components.

    Attributes:
        TYPE (ComponentType): The type of the component (LOSS).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Callable).
    """
    TYPE = ComponentType.LOSS
    PROVIDERS = [_Native, _MONAI, _PyTorch]
    INSTANCE_TYPE = Callable

class InfererSpec(metaclass=ComponetSpecMeta):
    """
    Specification for inferer components.

    Attributes:
        TYPE (ComponentType): The type of the component (INFERER).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Callable).
    """
    TYPE = ComponentType.INFERER
    PROVIDERS = [_Native, _MONAI]
    INSTANCE_TYPE = Callable

class OptimizerSpec(metaclass=ComponetSpecMeta):
    """
    Specification for optimizer components.

    Attributes:
        TYPE (ComponentType): The type of the component (OPTIMIZER).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Optimizer).
    """
    TYPE = ComponentType.OPTIMIZER
    PROVIDERS = [_PyTorch, _MONAI]
    INSTANCE_TYPE = Optimizer

class SchedulerSpec(metaclass=ComponetSpecMeta):
    """
    Specification for scheduler components.

    Attributes:
        TYPE (ComponentType): The type of the component (SCHEDULER).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (LRScheduler).
    """
    TYPE = ComponentType.SCHEDULER
    PROVIDERS = [_MONAI, _PyTorch]
    INSTANCE_TYPE = LRScheduler

class MetricSpec(metaclass=ComponetSpecMeta):
    """
    Specification for metric components.

    Attributes:
        TYPE (ComponentType): The type of the component (METRIC).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Metric).
    """
    TYPE = ComponentType.METRIC
    PROVIDERS = [_Native, _MONAI]
    INSTANCE_TYPE = Metric

class DatasetSpec(metaclass=ComponetSpecMeta):
    """
    Specification for dataset components.

    Attributes:
        TYPE (ComponentType): The type of the component (DATASET).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Dataset).
    """
    TYPE = ComponentType.DATASET
    PROVIDERS = [_MONAI, _TorchVision]
    INSTANCE_TYPE = Dataset

class DataLoaderSpec(metaclass=ComponetSpecMeta):
    """
    Specification for data loader components.

    Attributes:
        TYPE (ComponentType): The type of the component (DATALOADER).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (DataLoader).
    """
    TYPE = ComponentType.DATALOADER
    PROVIDERS = [_MONAI]
    INSTANCE_TYPE = DataLoader

class TransformSpec(metaclass=ComponetSpecMeta):
    """
    Specification for transform components.

    Attributes:
        TYPE (ComponentType): The type of the component (TRANSFORM).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Transform).
    """
    TYPE = ComponentType.TRANSFORM
    PROVIDERS = [_Native, _MONAI]
    INSTANCE_TYPE = Transform

class DataModuleSpec(metaclass=ComponetSpecMeta):
    """
    Specification for data module components.

    Attributes:
        TYPE (ComponentType): The type of the component (DATAMODULE).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (LightningDataModule).
    """
    TYPE = ComponentType.DATAMODULE
    PROVIDERS = [_Native]
    INSTANCE_TYPE = LightningDataModule

class WorkflowSpec(metaclass=ComponetSpecMeta):
    """
    Specification for workflow components.

    Attributes:
        TYPE (ComponentType): The type of the component (WORKFLOW).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (LightningModule).
    """
    TYPE = ComponentType.WORKFLOW
    PROVIDERS = [_Native]
    INSTANCE_TYPE = LightningModule

class CallbackSpec(metaclass=ComponetSpecMeta):
    """
    Specification for callback components.

    Attributes:
        TYPE (ComponentType): The type of the component (CALLBACK).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Callback).
    """
    TYPE = ComponentType.CALLBACK
    PROVIDERS = [_Native, _Lightning]
    INSTANCE_TYPE = Callback

class LoggerSpec(metaclass=ComponetSpecMeta):
    """
    Specification for logger components.

    Attributes:
        TYPE (ComponentType): The type of the component (LOGGER).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Logger).
    """
    TYPE = ComponentType.LOGGER
    PROVIDERS = [_Native, _Lightning]
    INSTANCE_TYPE = Logger

class MetricV2Spec(metaclass=ComponetSpecMeta):
    """
    Specification for MetricV2 components.

    Attributes:
        TYPE (ComponentType): The type of the component (METRICV2).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (MetricV2).
    """
    TYPE = ComponentType.METRICV2
    PROVIDERS = [_Native, _TorchMetrics]
    INSTANCE_TYPE = MetricV2

class SamplerSpec(metaclass=ComponetSpecMeta):
    """
    Specification for sampler components.

    Attributes:
        TYPE (ComponentType): The type of the component (SAMPLER).
        PROVIDERS (List[LibSpec]): The list of libraries that provide the component.
        INSTANCE_TYPE (type): The type of the component instance (Sampler).
    """
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
