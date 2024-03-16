import abc
import inspect
import importlib
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Sized

import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, Sampler
from manafaln.common.constants import ComponentType
from manafaln.common.componentspecs import ComponentSpecs


class ComponentBuilder(object):
    """A base class for building components."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.UNKNOWN,
        check_instance: bool = True
    ):
        """
        Initializes the ComponentBuilder.

        Args:
            component_type (ComponentType): The type of the component.
            check_instance (bool): Whether to check if the constructed
                object is an instance of the expected class.
        """
        self.component_type = ComponentType(component_type)
        self.check_instance = check_instance
        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_instance(self, spec, name, path, args, kwargs):
        """
        Builds an instance of the component.

        Args:
            spec: The component specification.
            name (str): The name of the component.
            path (str): The path to the module containing the component.
            args: Positional arguments to be passed to the component constructor.
            kwargs: Keyword arguments to be passed to the component constructor.

        Returns:
            The instance of the component.
        """
        out = None
        if path is None:
            for provider in spec.PROVIDERS:
                path = getattr(provider, self.component_type.name)
                try:
                    M = importlib.import_module(path)
                    C = getattr(M, name)
                    out = C(*args, **kwargs)
                except ModuleNotFoundError as e:
                    import traceback as tb
                    self.logger.warning(tb.format_exc())
                    self.logger.warning(f"Unable to import module {path}.")
                    continue
                except AttributeError:
                    self.logger.debug(f"Component {name} not found in module {path}.")
                    continue
                break
        else:
            M = importlib.import_module(path)
            C = getattr(M, name)
            out = C(*args, **kwargs)
        return out

    def _check_instance(self, spec, instance):
        """
        Checks if the instance is an instance of the expected class.

        Args:
            spec: The component specification.
            instance: The instance to be checked.

        Raises:
            TypeError: If the instance is not an instance of the expected class.
        """
        if spec.INSTANCE_TYPE == Any:
            return
        if not isinstance(instance, spec.INSTANCE_TYPE):
            raise TypeError(
                f"Builder expect {spec.INSTANCE_TYPE} but got {type(instance)} instead"
            )

    def __call__(self, config: Dict):
        """
        Builds the component based on the given configuration.

        Args:
            config (Dict): The configuration for building the component.

        Returns:
            The built component.
        """
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
    """A class for building models."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.MODEL,
        check_instance: bool = True
    ):
        """
        Initializes the ModelBuilder.

        Args:
            component_type (ComponentType): The type of the model component.
            check_instance (bool): Whether to check if the constructed
                model is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)


class LossBuilder(ComponentBuilder):
    """A class for building loss functions."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.LOSS,
        check_instance: bool = True
    ):
        """
        Initializes the LossBuilder.

        Args:
            component_type (ComponentType): The type of the loss component.
            check_instance (bool): Whether to check if the constructed
                loss function is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)


class InfererBuilder(ComponentBuilder):
    """A class for building inferers."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.INFERER,
        check_instance: bool = True
    ):
        """
        Initializes the InfererBuilder.

        Args:
            component_type (ComponentType): The type of the inferer component.
            check_instance (bool): Whether to check if the constructed
                inferer is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)


class OptimizerBuilder(ComponentBuilder):
    """A class for building optimizers."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.OPTIMIZER,
        check_instance: bool = True
    ):
        """
        Initializes the OptimizerBuilder.

        Args:
            component_type (ComponentType): The type of the optimizer component.
            check_instance (bool): Whether to check if the constructed
                optimizer is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, params: Dict[str, torch.Tensor]):
        """
        Builds the optimizer based on the given configuration and parameters.

        Args:
            config (Dict): The configuration for building the optimizer.
            params (Dict[str, torch.Tensor]): The parameters to be optimized.

        Returns:
            The built optimizer.
        """
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
    """A class for building schedulers."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.SCHEDULER,
        check_instance: bool = True
    ):
        """
        Initializes the SchedulerBuilder.

        Args:
            component_type (ComponentType): The type of the scheduler component.
            check_instance (bool): Whether to check if the constructed
                scheduler is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, opt: Optimizer):
        """
        Builds the scheduler based on the given configuration and optimizer.

        Args:
            config (Dict): The configuration for building the scheduler.
            opt (Optimizer): The optimizer to be scheduled.

        Returns:
            The built scheduler.
        """
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
    """A class for building metrics."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.METRIC,
        check_instance: bool = True
    ):
        """
        Initializes the MetricBuilder.

        Args:
            component_type (ComponentType): The type of the metric component.
            check_instance (bool): Whether to check if the constructed
                metric is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)


class DatasetBuilder(ComponentBuilder):
    """A class for building datasets."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.DATASET,
        check_instance: bool = True
    ):
        """
        Initializes the DatasetBuilder.

        Args:
            component_type (ComponentType): The type of the dataset component.
            check_instance (bool): Whether to check if the constructed
                dataset is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, args: List, kwargs: Dict):
        """
        Builds the dataset based on the given configuration, args, and kwargs.

        Args:
            config (Dict): The configuration for building the dataset.
            args (List): Positional arguments to be passed to the dataset constructor.
            kwargs (Dict): Keyword arguments to be passed to the dataset constructor.

        Returns:
            The built dataset.
        """
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = args

        # Build _kwargs from config and kwargs
        _kwargs = config.get("args", {})
        for k in kwargs:
            if k in _kwargs:
                self.logger.warning(f"Overwrite kwargs from {_kwargs[k]} to {kwargs[k]}")
            _kwargs[k] = kwargs[k]

        out = self._build_instance(spec, name, path, args, _kwargs)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out


class DataLoaderBuilder(ComponentBuilder):
    """A class for building data loaders."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.DATALOADER,
        check_instance: bool = True
    ):
        """
        Initializes the DataLoaderBuilder.

        Args:
            component_type (ComponentType): The type of the data loader component.
            check_instance (bool): Whether to check if the constructed
                data loader is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, dataset: Dataset):
        """
        Builds the data loader based on the given configuration and dataset.

        Args:
            config (Dict): The configuration for building the data loader.
            dataset (Dataset): The dataset to be loaded.

        Returns:
            The built data loader.
        """
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = [dataset]
        kwargs = config.get("args", {})

        sampler_config = kwargs.get("sampler", None)
        if sampler_config is not None:
            sampler_builder = SamplerBuilder(check_instance=self.check_instance)
            data_source = dataset.data
            sampler: Sampler = sampler_builder(sampler_config, data_source)
            kwargs["sampler"] = sampler

        collate_fn_config = kwargs.get("collate_fn", None)
        if collate_fn_config is not None:
            M = importlib.import_module(collate_fn_config["path"])
            collate_fn = getattr(M, collate_fn_config["name"])
            kwargs["collate_fn"] = collate_fn

        out = self._build_instance(spec, name, path, args, kwargs)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out


class TransformBuilder(ComponentBuilder):
    """A class for building data transforms."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.TRANSFORM,
        check_instance: bool = True
    ):
        """
        Initializes the TransformBuilder.

        Args:
            component_type (ComponentType): The type of the transform component.
            check_instance (bool): Whether to check if the constructed
                transform is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)


class DataModuleBuilder(ComponentBuilder):
    """A class for building data modules."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.DATAMODULE,
        check_instance: bool = True
    ):
        """
        Initializes the DataModuleBuilder.

        Args:
            component_type (ComponentType): The type of the data module component.
            check_instance (bool): Whether to check if the constructed
                data module is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict):
        """
        Builds the data module based on the given configuration.

        Args:
            config (Dict): The configuration for building the data module.

        Returns:
            The built data module.
        """
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
    """A class for building workflows."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.WORKFLOW,
        check_instance: bool = True
    ):
        """
        Initializes the WorkflowBuilder.

        Args:
            component_type (ComponentType): The type of the workflow component.
            check_instance (bool): Whether to check if the constructed
                workflow is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, ckpt: Optional[str] = None):
        """
        Builds the workflow based on the given configuration and checkpoint.

        Args:
            config (Dict): The configuration for building the workflow.
            ckpt (Optional[str]): The path to the checkpoint to restore.

        Returns:
            The built workflow.
        """
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
                        self.logger.warning(f"Module {path} not found")
                        continue
                    except AttributeError:
                        self.logger.debug(f"Unable to find {name} in {path}")
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
        """
        Restores the workflow from a checkpoint.

        Args:
            ckpt (str): The path to the checkpoint.
            config (Optional[Dict]): The configuration for building the workflow.

        Returns:
            The restored workflow.
        """
        checkpoint = torch.load(ckpt)
        if config is None:
            # Try to get config from checkpoint file
            config = checkpoint["hyper_parameters"]["workflow"]

        # TODO: Support pre-trained weight only ckpt & var name mapping

        return self.__call__(config, ckpt)


class CallbackBuilder(ComponentBuilder):
    """A class for building callbacks."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.CALLBACK,
        check_instance: bool = True
    ):
        """
        Initializes the CallbackBuilder.

        Args:
            component_type (ComponentType): The type of the callback component.
            check_instance (bool): Whether to check if the constructed
                callback is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)


class LoggerBuilder(ComponentBuilder):
    """A class for building loggers."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.LOGGER,
        check_instance: bool = True
    ):
        """
        Initializes the LoggerBuilder.

        Args:
            component_type (ComponentType): The type of the logger component.
            check_instance (bool): Whether to check if the constructed
                logger is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)


class MetricV2Builder(ComponentBuilder):
    """A class for building metric_v2."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.METRICV2,
        check_instance: bool = True
    ):
        """
        Initializes the MetricV2Builder.

        Args:
            component_type (ComponentType): The type of the metric_v2 component.
            check_instance (bool): Whether to check if the constructed
                metric_v2 is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)


class SamplerBuilder(ComponentBuilder):
    """A class for building samplers."""

    def __init__(
        self,
        component_type: ComponentType = ComponentType.SAMPLER,
        check_instance: bool = True
    ):
        """
        Initializes the SamplerBuilder.

        Args:
            component_type (ComponentType): The type of the sampler component.
            check_instance (bool): Whether to check if the constructed
                sampler is an instance of the expected class.
        """
        super().__init__(component_type, check_instance)

    def __call__(self, config: Dict, data_source: Sized) -> Sampler:
        """
        Builds the sampler based on the given configuration and data source.

        Args:
            config (Dict): The configuration for building the sampler.
            data_source (Sized): The data source to be sampled.

        Returns:
            The built sampler.
        """
        name = config["name"]
        path = config.get("path", None)
        spec = ComponentSpecs[self.component_type.name]
        args = [data_source]
        kwargs = config.get("args", {})

        out = self._build_instance(spec, name, path, args, kwargs)
        if out is None:
            raise RuntimeError(f"Could not found {name} in supported libraries.")

        if self.check_instance:
            self._check_instance(spec, out)

        return out
