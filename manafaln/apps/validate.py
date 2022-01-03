from pytorch_lightning import Trainer

from manafaln.utils.args import InferenceConfigurator
from manafaln.utils.builders import build_data_module
from manafaln.utils.checkpoint import restore_from_checkpoint

def run(config_train, config_data, config_workflow, ckpt):
    # Configure data module (only val_dataloader will be used)
    data = build_data_module(config_data)

    # Restore workflow
    workflow = restore_from_checkpoint(ckpt, config=config_workflow)

    # NO LOGGING FOR VALIDATION
    config_train["settings"]["logger"] = False
    config_train["settings"]["checkpoint_callback"] = False

    # Build trainer for validation
    trainer = Trainer(**config_train["settings"])

    # Start validation
    metrics = trainer.validate(workflow, data.val_dataloader())

    print(metrics)

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
