import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Sequence

import wandb
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from monai.utils import set_determinism
from ruamel.yaml import YAML

from manafaln.apps.utils import (
    build_callbacks,
    build_data_module,
    build_logger,
    build_workflow
)
from manafaln.utils.yaml import yaml_update_anchors


class SweepRunner:
    def __init__(
        self,
        template: str,
        anchor_keys: Sequence,
        project: str,
        group: Optional[str] = None,
        entity: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:
        self.template = template
        self.anchor_keys = anchor_keys
        self.project = project
        self.group = group
        self.entity = entity
        self.seed = seed

        self.yaml_loader = YAML()

    def execute(self):
        wandb.init(project=self.project, group=self.group, entity=self.entity)

        # Create config file for next training
        anchors = {}
        for key in self.anchor_keys:
            anchors[key] = getattr(wandb.config, key)
        config_file = os.path.join(wandb.run.dir, "config_sweep.yaml")
        yaml_update_anchors(self.template, anchors, config_file)

        with open(config_file) as f:
            config = self.yaml_loader.load(f)

        # Set seed if necessary
        if self.seed is not None:
            set_determinism(self.seed, use_deterministic_algorithms=True)

        # Build data module
        data = build_data_module(config["data"])

        # Build workflow
        workflow = build_workflow(config["workflow"])

        # Override logger settings from default config
        logger = WandbLogger(
            project=self.project,
            id=wandb.run.id
        )

        # Build trainer callbacks
        callbacks = config["trainer"].get("callbacks", [])
        callbacks = build_callbacks(callbacks)

        # Build Trainer
        trainer = Trainer(
            callbacks=callbacks,
            logger=logger,
            **config["trainer"]["settings"]
        )

        # Run training
        trainer.fit(workflow, data)

        # Run test
        trainer.test(workflow, ckpt_path="best", datamodule=data)

        # Copy config file to log_dir for convenience
        # if trainer.logger.log_dir is not None:
        #     shutil.copy(
        #         config_file,
        #         os.path.join(trainer.logger.log_dir, "config_sweep.yaml")
        #     )

        # Cleanup
        wandb.finish()

if __name__ == "__main__":
    wandb.login()

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Sweep config YAML.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sweep.")
    parser.add_argument("--count", type=int, default=None, help="Number of trails of sweep.")
    args = parser.parse_args()

    yaml = YAML()
    with open(args.config) as f:
        config = yaml.load(f)

    # Template for training
    template = config["template"]

    # Load W&B project configuration
    wandb_config = config["wandb"]
    wandb_config.setdefault("group", None)
    wandb_config.setdefault("entity", None)

    # Build Sweep runner
    anchor_keys = config["configuration"]["parameters"].keys()
    runner = SweepRunner(
        template,
        anchor_keys,
        project=wandb_config["project"],
        group=wandb_config["group"],
        seed=args.seed
    )

    # Start sweep
    sweep_id = wandb.sweep(
        sweep=config["configuration"],
        project=wandb_config["project"],
        entity=wandb_config["entity"]
    )
    wandb.agent(sweep_id, function=runner.execute, count=args.count)

