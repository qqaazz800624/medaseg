import torch
import monai
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only

from manafaln.utils.metrics import MetricCollection
from manafaln.utils.builders import (
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

        # Save all hyperparameters
        self.save_hyperparameters({ "workflow": config })

        # Get configurations for all components
        components = self.hparams.workflow["components"]

        self.model   = build_model(components["model"])
        self.loss_fn = build_loss_fn(components["loss"])
        self.inferer = build_inferer(components["inferer"])

        self.post_transforms = {}
        for phase in ["training", "validation", "test"]:
            self.post_transforms[phase] = build_transforms(
                components["post_transforms"].get(phase, [])
            )

        self.train_metrics = MetricCollection(components["training_metrics"])
        self.valid_metrics = MetricCollection(components["validation_metrics"])

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

        # Add result to metric
        self.train_metrics.apply(batch)

        # Log current loss value
        self.log_dict({ "train_loss": loss })

        return loss

    def training_epoch_end(self, outputs):
        m = self.train_metrics.aggregate()
        self.log_dict(m)

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        # Run inference
        batch["preds"] = self.forward(image)

        # Apply post transforms
        batch = self.post_transforms["validation"](batch)

        # Compute metrics here
        self.valid_metrics.apply(batch)

        return None

    def validation_epoch_end(self, validation_step_outputs):
        m = self.valid_metrics.aggregate()
        self.log_dict(m)

    def test_step(self, batch, batch_idx):
        # No label for test
        image = batch["image"]

        # Run inference
        batch["preds"] = self.forward(image)

        # Apply post transforms
        batch = self.post_transforms["test"](batch)

        # Nothing to output for pure inference
        return None

    def configure_optimizers(self):
        # Extract optimizer & scheduler configurations
        workflow = self.hparams.workflow
        opt_config = workflow["components"]["optimizer"]
        sch_config = workflow["components"].get("scheduler", None)

        opt = {
            "optimizer": build_optimizer(opt_config, self.model.parameters())
        }

        if not sch_config is None:
            # Get or set default scheduler mode
            if sch_opts := workflow["settings"].get("scheduler", None):
                interval = sch_opts.get("interval", "epoch")
                frequency = sch_opts.get("frequency", 1)
            else:
                interval = "epoch"
                frequency = 1

            opt["lr_scheduler"] = {
                "scheduler": build_scheduler(sch_config, opt["optimizer"]),
                "interval": interval,
                "frequency": frequency
            }

        return opt

