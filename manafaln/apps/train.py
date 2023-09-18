from typing import Dict, Optional

from lightning import Trainer
from monai.utils import set_determinism

from manafaln.apps.utils import (
    build_callbacks,
    build_data_module,
    build_logger,
    build_workflow,
)
from manafaln.core.configurators import TrainConfigurator


def run(
    config_train: Dict,
    config_data: Dict,
    config_workflow: Dict,
    seed: Optional[int] = None,
    ckpt_path: Optional[str] = None,
):
    """
    Run the training workflow.

    Args:
        config_train (Dict): The configuration for training.
        config_data (Dict): The configuration for data.
        config_workflow (Dict): The configuration for the workflow.
        seed (Optional[int], optional): The random seed for deterministic training. Defaults to None.
        ckpt_path (Optional[str], optional): The path to the checkpoint file. Defaults to None.
    """
    # Set seed for deterministic
    if seed is not None:
        # Don't touch algorithm settings here
        set_determinism(seed=seed)
        # If deterministic is not set, then set deterministic
        # If deterministic is set, then don't touch it
        if "deterministic" not in config_train["settings"].keys():
            config_train["settings"]["deterministic"] = "warn"

    # Configure data first
    data = build_data_module(config_data)

    # Configure workflow
    workflow = build_workflow(config_workflow, ckpt=None)

    # Create logger, defaults to TensorBoardLogger with ruamel saver
    logger = config_train.get("logger", {"name": "TensorBoardLogger"})
    logger = build_logger(logger)

    # Create callbacks
    callbacks = config_train.get("callbacks", [])
    callbacks = build_callbacks(callbacks)

    # Create trainer
    trainer = Trainer(callbacks=callbacks, logger=logger, **config_train["settings"])

    if config_train["settings"].get("auto_lr_find", False):
        trainer.tune()

    # Start training
    trainer.fit(workflow, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # Use configurator for argument parsing & loading config files
    c = TrainConfigurator()
    c.configure()

    # Get config results
    data = c.get_data_config()
    train = c.get_trainer_config()
    workflow = c.get_workflow_config()

    # Get random seed for deterministic training
    seed = c.get_random_seed()

    ckpt_path = c.get_ckpt_path()

    # Run
    run(
        config_train=train,
        config_data=data,
        config_workflow=workflow,
        seed=seed,
        ckpt_path=ckpt_path,
    )
