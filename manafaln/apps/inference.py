from pytorch_lightning import Trainer

from manafaln.core.configurators import InferenceConfigurator
from manafaln.apps.utils import (
    build_data_module,
    build_workflow,
    build_callbacks
)

def run(config_train, config_data, config_workflow, ckpt):
    """
    Runs the inference process using the provided configurations.

    Args:
        config_train (dict): The configuration for training.
        config_data (dict): The configuration for data.
        config_workflow (dict): The configuration for workflow.
        ckpt (str): The path to the checkpoint file.

    Returns:
        None
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

    # Start inference
    trainer.test(workflow, data.test_dataloader(), ckpt_path=ckpt)

if __name__ == "__main__":
    c = InferenceConfigurator()
    c.configure()

    data     = c.get_data_config()
    train    = c.get_trainer_config()
    workflow = c.get_workflow_config()

    # Extra information for validation only
    ckpt_path = c.get_ckpt_path()

    # Run
    run(train, data, workflow, ckpt_path)

