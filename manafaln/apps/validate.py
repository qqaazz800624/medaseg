from typing import Dict, List

from torch import Tensor
from pytorch_lightning import Trainer

from manafaln.core.configurators import ValidateConfigurator
from manafaln.apps.utils import (
    build_data_module,
    build_workflow,
    build_callbacks
)

def run(config_train, config_data, config_workflow, ckpt) -> List[Dict[str, float]]:
    """
    Run the validation process.

    Args:
        config_train (dict): Configuration for the trainer.
        config_data (dict): Configuration for the data module.
        config_workflow (dict): Configuration for the workflow.
        ckpt (str): Path to the checkpoint file.

    Returns:
        List[Dict[str, float]]: Computed metrics of different dataloaders.
    """
    # Configure data module (only val_dataloader will be used)
    data = build_data_module(config_data)

    # Restore workflow
    workflow = build_workflow(config_workflow)

    # NO LOGGING FOR VALIDATION
    config_train["settings"]["logger"] = False
    config_train["settings"]["enable_checkpointing"] = False

    # Create callbacks
    blacklist = ["ModelCheckpoint"]
    callbacks = config_train.get("callbacks", [])
    callbacks = [c for c in callbacks if c["name"] not in blacklist]
    callbacks = build_callbacks(callbacks)

    # Build trainer for validation
    trainer = Trainer(
        callbacks=callbacks,
        **config_train["settings"]
    )

    # Start validation
    metrics = trainer.validate(workflow, data.val_dataloader(), ckpt_path=ckpt)

    # Return metrics
    return metrics

if __name__ == "__main__":
    c = ValidateConfigurator()
    c.configure()

    data     = c.get_data_config()
    train    = c.get_trainer_config()
    workflow = c.get_workflow_config()

    # Extra information for validation only
    ckpt_path = c.get_ckpt_path()

    # Run
    run(train, data, workflow, ckpt_path)
