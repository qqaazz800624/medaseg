from pytorch_lightning import Trainer

from manafaln.core.configurators import TrainConfigurator
from manafaln.apps.utils import (
    build_data_module,
    build_workflow,
    build_callbacks
)

def run(config_train, config_data, config_workflow):
    # Configure data first
    data = build_data_module(config_data)

    # Configure workflow
    workflow = build_workflow(config_workflow, ckpt=None)

    # Create callbacks
    callbacks = config_train.get("callbacks", [])
    callbacks = build_callbacks(callbacks)

    # Create trainer
    trainer = Trainer(
        callbacks=callbacks,
        **config_train["settings"]
    )

    if config_train["settings"].get("auto_lr_find", False):
        trainer.tune()

    # Start training
    trainer.fit(workflow, data)

if __name__ == "__main__":
    # Use configurator for argument parsing & loading config files
    c = TrainConfigurator()
    c.configure()

    # Get config results
    data     = c.get_data_config()
    train    = c.get_trainer_config()
    workflow = c.get_workflow_config()

    # Run
    run(config_train=train, config_data=data, config_workflow=workflow)

