from typing import Dict, List, Optional, Union
from manafaln.core.builders import (
    DataModuleBuilder,
    WorkflowBuilder,
    CallbackBuilder,
    LoggerBuilder
)
from manafaln.utils.misc import ensure_list

def build_data_module(config: Dict, check_instance: bool = True):
    builder = DataModuleBuilder(check_instance=check_instance)
    return builder(config)

def build_workflow(config: Dict, ckpt: Optional[str] = None, check_instance: bool = True):
    builder = WorkflowBuilder(check_instance=check_instance)
    return builder(config, ckpt=ckpt)

def build_callbacks(config_callbacks: List[Dict], check_instance: bool = True):
    builder = CallbackBuilder(check_instance=check_instance)
    callbacks = [builder(c) for c in config_callbacks]
    return callbacks

def build_logger(config_logger: Optional[Union[bool, Dict, List[Dict]]], check_instance: bool = True):
    if isinstance(config_logger, bool) or config_logger is None:
        return config_logger
    builder = LoggerBuilder(check_instance=check_instance)
    config_logger = ensure_list(config_logger)
    return [builder(c) for c in config_logger]
