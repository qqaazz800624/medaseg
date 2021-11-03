import torch
import monai
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only

from manafaln.utils import (
    build_model,
    build_loss_fn,
    build_inferer,
    build_optimizer,
    build_scheduler,
    build_transforms
)

class SupervisedSegmentation(LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        # Check hyperparameters for conflict

        # Save all hyperparameters
        self.save_hyperparameters(config)

        # Get configurations for all components
        components = self.hparams.components

        self.model   = build_model(components["model"])
        self.loss_fn = build_loss_fn(components["loss"])
        self.inferer = build_inferer(components["inferer"])

        self.post_transforms = {}
        for phase in ["training", "validation", "test"]:
            self.post_transforms[phase] = build_transforms(
                components["post_transforms"].get(phase, [])
            )

        # TODO: Add training/validation metrics here
        # train_metrics = []
        # valid_metrics = []

    def forward(self, data):
        return self.inferer(data, self.model)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        # Apply model & compute loss
        batch["preds"] = self.model(image)
        loss = self.loss_fn(batch["preds"], batch["label"])

        # Apply post transforms & compute metrics
        batch = self.post_transforms["training"](batch)

        # Log training step metrics
        self.log_dict({
            "train_loss": loss
        })

        return loss

    @rank_zero_only
    def summarize_validation(self, metrics):
        pass

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        # Run inference
        batch["preds"] = self.forward(image)

        # Apply post transforms
        batch = self.post_transforms["validation"](batch)

        # Compute metrics here

        return 0

    def validation_epoch_end(self, validation_step_outputs):
        return None

    def test_step(self, batch, batch_idx):
        # No label for test
        image = batch["image"]

        # Run inference
        batch["preds"] = self.forward(image)

        # Apply post transforms
        batch = self.post_transforms["test"](batch)

        # TODO: Save results

        return {}

    def configure_optimizers(self):
        # Extract optimizer & scheduler configurations
        opt_config = self.hparams.components["optimizer"]
        sch_config = self.hparams.components.get("scheduler", None)

        opt = {
            "optimizer": build_optimizer(opt_config, self.model.parameters())
        }

        if not sch_config is None:
            # Get or set default scheduler mode
            interval = self.hparams.settings.get("interval", "epoch")
            frequency = self.hparams.settings.get("frequency", 1)

            opt["lr_scheduler"] = {
                "scheduler": build_scheduler(opt_config, opt),
                "interval": interval,
                "frequency": frequency
            }

        return opt

