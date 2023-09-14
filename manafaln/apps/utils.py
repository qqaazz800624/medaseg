from typing import Any, Dict, List, Optional, Union
from manafaln.core.builders import (
    DataModuleBuilder,
    WorkflowBuilder,
    CallbackBuilder,
    LoggerBuilder
)
from manafaln.utils.misc import ensure_list

def build_data_module(config: Dict, check_instance: bool = True) -> Any:
    """
    Builds a data module using the provided configuration.

    Args:
        config (Dict): The configuration for building the data module.
        check_instance (bool, optional): Whether to check if the built data module is an instance of `DataModule`. Defaults to True.

    Returns:
        Any: The built data module.
    """
    builder = DataModuleBuilder(check_instance=check_instance)
    return builder(config)

def build_workflow(config: Dict, ckpt: Optional[str] = None, check_instance: bool = True) -> Any:
    """
    Builds a workflow using the provided configuration.

    Args:
        config (Dict): The configuration for building the workflow.
        ckpt (Optional[str], optional): The checkpoint to load the workflow from. Defaults to None.
        check_instance (bool, optional): Whether to check if the built workflow is an instance of `Workflow`. Defaults to True.

    Returns:
        Any: The built workflow.
    """
    builder = WorkflowBuilder(check_instance=check_instance)
    return builder(config, ckpt=ckpt)

def build_callbacks(config_callbacks: List[Dict], check_instance: bool = True) -> List[Any]:
    """
    Builds a list of callbacks using the provided configuration.

    Args:
        config_callbacks (List[Dict]): The configuration for building the callbacks.
        check_instance (bool, optional): Whether to check if the built callbacks are instances of `Callback`. Defaults to True.

    Returns:
        List[Any]: The built callbacks.
    """
    builder = CallbackBuilder(check_instance=check_instance)
    callbacks = [builder(c) for c in config_callbacks]
    return callbacks

def build_logger(config_logger: Optional[Union[bool, Dict, List[Dict]]], check_instance: bool = True) -> Union[bool, List[Any]]:
    """
    Builds a logger using the provided configuration.

    Args:
        config_logger (Optional[Union[bool, Dict, List[Dict]]]): The configuration for building the logger.
        check_instance (bool, optional): Whether to check if the built logger is an instance of `Logger`. Defaults to True.

    Returns:
        Union[bool, List[Any]]: The built logger. If `config_logger` is a boolean or None, it is returned as is. Otherwise, a list of built loggers is returned.
    """
    if isinstance(config_logger, bool) or config_logger is None:
        return config_logger
    builder = LoggerBuilder(check_instance=check_instance)
    config_logger = ensure_list(config_logger)
    return [builder(c) for c in config_logger]
