from typing import Union, List, Dict, Callable
from importlib import import_module

from monai.transforms import Compose
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
            ComponentType.TRANSFORM:  ComponentPaths.DEFAULT_TRANSFORM_PATH
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
            "transform":  ComponentPaths.DEFAULT_TRANSFORM_PATH
        }
    return mapping.get(component_type, "manafaln")

def instantiate(
        name: str,
        path: str = None,
        component_type: Union[ComponentType, str] = ComponentType.UNKNOWN,
        **kwargs
    ):
    if path is None:
        path = get_default_path(component_type)

    Module = import_module(path)
    Class  = getattr(Module, name)

    return Class(**kwargs)

def build_transforms(trans_configs: List[Dict]) -> Callable:
    transforms = []
    for config in trans_configs:
        trans = instantiate(
            name=config["name"],
            path=config.get("path", None),
            component_type=ComponentType.TRANSFORM,
            **config.get("args", {})
        )
        transforms.append(trans)
    return Compose(transforms)
