from typing import Union, List, Dict, Callable
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
from monai.metrics import Metric
from monai.transforms import Compose
from pytorch_lightning import (
    LightningModule,
    LightningDataModule
)
from pytorch_lightning.callbacks import Callback

from manafaln.common.constants import ComponentType, ComponentPaths

def get_default_path(component_type: Union[ComponentType, str]) -> str:
    if isinstance(component_type, ComponentType):
        mapping = {
            ComponentType.MODEL:      ComponentPaths.DEFAULT_MODEL_PATH,
            ComponentType.LOSS:       ComponentPaths.DEFAULT_LOSS_PATH,
            ComponentType.INFERER:    ComponentPaths.DEFAULT_INFERER_PATH,
            ComponentType.OPTIMIZER:  ComponentPaths.DEFAULT_OPTIMIZER_PATH,
            ComponentType.SCHEDULER:  ComponentPaths.DEFAULT_SCHEDULER_PATH,
            ComponentType.METRIC:     ComponentPaths.DEFAULT_METRIC_PATH,
            ComponentType.DATASET:    ComponentPaths.DEFAULT_DATASET_PATH,
            ComponentType.DATALOADER: ComponentPaths.DEFAULT_DATALOADER_PATH,
            ComponentType.TRANSFORM:  ComponentPaths.DEFAULT_TRANSFORM_PATH,
            ComponentType.WORKFLOW:   ComponentPaths.DEFAULT_WORKFLOW_PATH,
            ComponentType.DATAMODULE: ComponentPaths.DEFAULT_DATAMODULE_PATH,
            ComponentType.CALLBACK:   ComponentPaths.DEFAULT_CALLBACK_PATH
        }
    else:
        component_type = component_type.lower()
        mapping = {
            "model":      ComponentPaths.DEFAULT_MODEL_PATH,
            "loss":       ComponentPaths.DEFAULT_LOSS_PATH,
            "inferer":    ComponentPaths.DEFAULT_INFERER_PATH,
            "optimizer":  ComponentPaths.DEFAULT_OPTIMIZER_PATH,
            "scheduler":  ComponentPaths.DEFAULT_SCHEDULER_PATH,
            "metric":     ComponentPaths.DEFAULT_METRIC_PATH,
            "dataset":    ComponentPaths.DEFAULT_DATASET_PATH,
            "dataloader": ComponentPaths.DEFAULT_DATALOADER_PATH,
            "transform":  ComponentPaths.DEFAULT_TRANSFORM_PATH,
            "workflow":   ComponentPaths.DEFAULT_WORKFLOW_PATH,
            "datamodule": ComponentPaths.DEFAULT_DATAMODULE_PATH,
            "callback":   ComponentPaths.DEFAULT_CALLBACK_PATH
        }
    return mapping.get(component_type, "manafaln")

class InstanceQuery:
    def __init__(
        self,
        name: str,
        path: str,
        component_type: Union[ComponentType, str] = ComponentType.UNKNOWN
    ):
        self.name = name
        self.path = path
        self.component_type = component_type

    def get_name(self) -> str:
        return self.name

    def get_path(self) -> str:
        return self.path

    def get_type(self) -> ComponentType:
        return self.component_type

def get_class(
        name: str,
        path: str,
        component_type: Union[ComponentType, str] = ComponentType.UNKNOWN
    ):
    if path is None:
        path = get_default_path(component_type)
    M = import_module(path)
    return getattr(M, name)

def instance_builder(instance_query: InstanceQuery, /, **kwargs):
    C = get_class(
        name=instance_query.get_name(),
        path=instance_query.get_path(),
        component_type=instance_query.get_type()
    )
    return C(**kwargs)

def instantiate(
        name: str,
        path: str = None,
        component_type: Union[ComponentType, str] = ComponentType.UNKNOWN,
        **kwargs
    ):
    C = get_class(name=name, path=path, component_type=component_type)
    return C(**kwargs)

def build_model(config: Dict) -> nn.Module:
    query = InstanceQuery(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.MODEL
    )
    model = instance_builder(query, **config.get("args", {}))

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{type(model)} is not a PyTorch module")
    return model

def build_loss_fn(config: Dict) -> Callable:
    query = InstanceQuery(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.LOSS
    )
    loss_fn = instance_builder(query, **config.get("args", {}))

    if not isinstance(loss_fn, Callable):
        raise TypeError(f"{type(loss_fn)} is not Callable type")
    return loss_fn

def build_inferer(config: Dict) -> Callable:
    query = InstanceQuery(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.INFERER
    )
    inferer = instance_builder(query, **config.get("args", {}))

    if not isinstance(inferer, Callable):
        raise TypeError(f"{type(inferer)} is not Callable type")
    return inferer

def build_optimizer(config: Dict, model_params: List) -> optim.Optimizer:
    query = InstanceQuery(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.OPTIMIZER
    )
    opt = instance_builder(
        query,
        params=model_params,
        **config.get("args", {})
    )

    if not isinstance(opt, optim.Optimizer):
        raise TypeError(f"{type(opt)} is not a troch Optimizer")
    return opt

def build_scheduler(config: Dict, opt: optim.Optimizer):
    query = InstanceQuery(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.SCHEDULER
    )
    sch = instance_builder(
        query,
        optimizer=opt,
        **config.get("args", {})
    )
    return sch

def build_metric(config: Dict) -> Metric:
    query = InstanceQuery(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.METRIC
    )
    metric = instance_builder(query, **config.get("args", {}))

    if not isinstance(metric, Metric):
        raise TypeError(f"{type(metric)} is not a MONAI Metric")
    return metric

def build_transforms(trans_configs: List[Dict]) -> Callable:
    transforms = []
    for config in trans_configs:
        query = InstanceQuery(
            name=config["name"],
            path=config.get("path", None),
            component_type=ComponentType.TRANSFORM
        )
        trans = instance_builder(query, **config.get("args", {}))
        if not isinstance(trans, Callable):
            raise TypeError(f"{type(trans)} is not Callable type")
        transforms.append(trans)
    return Compose(transforms)

def build_workflow(config: Dict) -> LightningModule:
    return instantiate(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.WORKFLOW,
        config=config
    )

def build_data_module(config: Dict) -> LightningDataModule:
    return instantiate(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.DATAMODULE,
        config=config
    )

def build_callback(config: Dict) -> Callback:
    callback = instantiate(
        name=config["name"],
        path=config.get("path", None),
        component_type=ComponentType.CALLBACK,
        **config.get("args", {})
    )

    if not isinstance(callback, Callback):
        raise TypeError(f"{type(callback)} is not a valid Callback type")
    return callback
