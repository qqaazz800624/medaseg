from pytorch_lightning import Trainer

from manafaln.utils.args import (
    parse_validate_args,
    load_validate_config,
    configure_validation
)
from manafaln.utils.builders import build_data_module
from manafaln.utils.checkpoint import restore_from_checkpoint

def run(config_train, config_data, config_workflow, ckpt):
    # Configure data module (only val_dataloader will be used)
    data = build_data_module(config_data)

    # Restore workflow
    workflow = restore_from_checkpoint(ckpt, config=config_workflow)

    # Build trainer for validation
    trainer = Trainer(**config_train["settings"])

    # Start validation
    metrics = trainer.validate(workflow, data.val_dataloader())

    print(metrics)

if __name__ == "__main__":
    # Load all settings
    args   = parse_validate_args()
    config = load_validate_config(args.config, args.data)

    # Organize all settings
    train, data, workflow = configure_validation(args, config)

    # Run
    run(train, data, workflow, args.ckpt)
