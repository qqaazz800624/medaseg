import abc
import inspect
import importlib
import logging
from typing import Any, Callable, Dict, List, Optional

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
        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_instance(self, spec, name, path, args, kwargs):
        out = None
        if path is None:
            for provider in spec.PROVIDERS:
                path = getattr(provider, self.component_type.name)
                try:
                    M = importlib.import_module(path)
                    C = getattr(M, name)
                    out = C(*args, **kwargs)
                except ModuleNotFoundError as e:
                    self.logger.warning("Module {path} not found")
                    continue
                except AttributeError:
                    self.logger.debug("Unable to find {name} in {path}")
                    continue
                break
        else:
            M = importlib.import_module(path)
            C = getattr(M, name)
            out = C(*args, **kwargs)
        return out

    def _check_instance(self, spec, instance):
        if spec.INSTANCE_TYPE == Any:
            return
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
            raise RuntimeError(f"Failed to build component: {name}.")

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

        # Build _kwargs from config and kwargs
        _kwargs = config.get("args", {})
        for k in kwargs:
            if k in _kwargs:
                self.logger.warning("Overwrite kwargs from {_kwargs[k]} to {kwargs[k]}")
            _kwargs[k] = kwargs[k]

        out = self._build_instance(spec, name, path, args, _kwargs)
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

    def __call__(self, config: Dict):
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = { "config": config }

        out = self._build_instance(spec, name, path, [], args)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out

class WorkflowBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.WORKFLOW,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, ckpt: Optional[str] = None):
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = { "config": config }

        if ckpt is None:
            out = self._build_instance(spec, name, path, [], args)
            if out is None:
                raise RuntimeError(f"Could not found {name} in supported libraries.")

            if self.check_instance:
                self._check_instance(spec, out)

            return out
        else:
            out = None
            if path is None:
                for provider in spec.PROVIDERS:
                    path = getattr(provider, self.component_type.name)
                    try:
                        M = importlib.import_module(path)
                        C = getattr(M, name)
                    except ModuleNotFoundError as e:
                        self.logger.warning("Module {path} not found")
                        continue
                    except AttributeError:
                        self.logger.debug("Unable to find {name} in {path}")
                        continue
                    break
            else:
                M = importlib.import_module(path)
                C = getattr(M, name)
            out = C.load_from_checkpoint(ckpt, config=config, strict=False)

            if out is None:
                raise RuntimeError(f"Unable to restore ckpt {ckpt} to workflow {name}")

            if self.check_instance:
                self._check_instance(spec, out)

            return out

    def restore_from_checkpoint(
        self,
        ckpt: str,
        config: Optional[Dict] = None
    ):
        checkpoint = torch.load(ckpt)
        if config is None:
            # Try to get config from checkpoint file
            config = checkpoint["hyper_parameters"]["workflow"]

        # TODO: Support pre-trained weight only ckpt & var name mapping

        return self.__call__(config, ckpt)


class CallbackBuilder(ComponentBuilder):
    def __init__(
        self,
        component_type: ComponentType = ComponentType.CALLBACK,
        check_instance: bool = True
    ):
        super().__init__(component_type, check_instance)

