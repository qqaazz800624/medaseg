from pytorch_lightning import Trainer

from manafaln.utils.args import (
    parse_trainer_args,
    load_training_config,
    configure_training_args
)
from manafaln.utils.builders import (
    build_callback,
    build_workflow,
    build_data_module
)

def run(config_train, config_data, config_workflow):
    # Configure data first
    data = build_data_module(config_data)

    # Configure workflow
    workflow = build_workflow(config_workflow)

    # Create callbacks
    callbacks = config_train.get("callbacks", [])
    callbacks = [build_callback(c) for c in callbacks]

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
    args   = parse_trainer_args()
    config = load_training_config(args.config)

    # Integrate the settings in args & config file
    # When there is a setting conflict between args and config, the value set
    # in the config file will be ignored, and the settings in data and
    # workflow also override the values in components as default
    train, data, workflow = configure_training(args=args, config=config)

    # Run
    run(config_train=train, config_data=data, config_workflow=workflow)

