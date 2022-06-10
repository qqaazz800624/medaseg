import abc
import inspect
import importlib
from typing import Dict, List

import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset
from manafaln.common.constants import ComponentType
from manafaln.common.componentspecs import ComponentSpecs

class ComponentBuilder(object):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.UNKNOWN,
        check_instance: bool = True
    ):
        self.component_type = ComponentType(component_type)
        self.check_instance = check_instance

    def _build_instance(self, spec, name, path, args, kwargs):
        out = None
        if path is not None:
            M = importlib.import_module(path)
            C = getattr(M, name)
            out = C(*args, **kwargs)
        else:
            for provider in spec.PROVIDERS:
                path = getattr(provider, self.component_type.name)
                try:
                    M = importlib.import_module(path)
                    C = getattr(M, name)
                    out = C(*args, **kwargs)
                except:
                    continue
                break
        return out

    def _check_instance(self, spec, instance):
        if not isinstance(instance, spec.INSTANCE_TYPE):
            raise TypeError(
                f"Builder expect {spec.INSTANCE_TYPE} but got {type(instance)} instead"
            )

    def __call__(self, config: Dict):
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = config.get("args", {}) # Actually kwargs

        out = self._build_instance(spec, name, path, [], args)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out

class ModelBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.MODEL,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

class LossBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.LOSS,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

class InfererBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.INFERER,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

class OptimizerBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.OPTIMIZER,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, params: Dict[str, torch.Tensor]):
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = [params]
        kwargs = config.get("args", {})

        out = self._build_instance(spec, name, path, args, kwargs)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out

class SchedulerBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.SCHEDULER,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, opt: Optimizer):
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = [opt]
        kwargs = config.get("args", {})

        out = self._build_instance(spec, name, path, args, kwargs)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out

class MetricBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.METRIC,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

class DatasetBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.DATASET,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, args: List, kwargs: Dict):
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = args
        kwargs = kwargs

        out = self._build_instance(spec, name, path, args, kwargs)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out

class DataLoaderBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.DATALOADER,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, dataset: Dataset):
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = [dataset]
        kwargs = config.get("args", {})

        out = self._build_instance(spec, name, path, args, kwargs)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out

class TransformBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.TRANSFORM,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

class DataModuleBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.DATAMODULE,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

class WorkflowBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.WORKFLOW,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

class CallbackBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.CALLBACK,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

