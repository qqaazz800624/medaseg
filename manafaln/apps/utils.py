from typing import Dict, List, Optional
from manafaln.core.builders import (
    DataModuleBuilder,
    WorkflowBuilder,
    CallbackBuilder
)

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

