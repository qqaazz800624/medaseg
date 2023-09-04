from pytorch_lightning import Trainer

from manafaln.core.configurators import InferenceConfigurator
from manafaln.apps.utils import (
    build_data_module,
    build_workflow,
    build_callbacks
)

def run(config_train, config_data, config_workflow, ckpt):
    """
    Runs the inference process.

    Args:
        config_train (dict): Configuration for the trainer.
        config_data (dict): Configuration for the data module.
        config_workflow (dict): Configuration for the workflow.
        ckpt (str): Path to the checkpoint file.

    Returns:
        None
    """
    # Configure data module (only predict_dataloader will be used)
    data = build_data_module(config_data)

    # Restore workflow
    workflow = build_workflow(config_workflow, ckpt=ckpt)

    # NO LOGGING FOR INFERENCE
    config_train["settings"]["logger"] = False
    config_train["settings"]["enable_checkpointing"] = False

    # Create callbacks
    blacklist = ["ModelCheckpoint"]
    callbacks = config_train.get("callbacks", [])
    callbacks = [c for c in callbacks if c["name"] not in blacklist]
    callbacks = build_callbacks(callbacks)

    # Build trainer for inference
    trainer = Trainer(
        callbacks=callbacks,
        **config_train["settings"]
    )

    # Start inference
    trainer.predict(workflow, data.predict_dataloader())

if __name__ == "__main__":
    c = InferenceConfigurator()
    c.configure()

    data     = c.get_data_config()
    train    = c.get_trainer_config()
    workflow = c.get_workflow_config()

    # Extra information for inference only
    ckpt_path = c.get_ckpt_path()

    # Run
    run(train, data, workflow, ckpt_path)

