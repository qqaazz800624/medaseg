import json
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint
)

from manafaln.utils import build_workflow, build_data_module

def train(args):
    # Load training config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Configure data first
    data = build_data_module(config["data"])

    # Configure workflow
    workflow = build_workflow(config["workflow"])

    # Process trainer configurations
    # Prefer arguments over config file

    # Create callbacks
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint()
    ]

    # Create trainer
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks
    )

    # Start training
    trainer.fit(workflow, data)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--config", "-c", type=str, help="Training config file")

    args = parser.parse_args()

    train(args)
