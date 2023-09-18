from copy import deepcopy

import torch
import monai
from monai.transforms import Decollated
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only

from manafaln.core.metric import MetricCollection
from manafaln.core.builders import (
    ModelBuilder,
    LossBuilder,
    InfererBuilder,
    OptimizerBuilder,
    SchedulerBuilder
)
from manafaln.core.transforms import build_transforms

def configure_batch_decollate(settings, phase, keys):
    """
    Configure batch decollate based on the given settings, phase, and keys.

    Args:
        settings (dict): The settings dictionary.
        phase (str): The phase of the batch.
        keys (list): The list of keys for decollation.

    Returns:
        Decollated or None: The decollated object or None if decollation is not needed.
    """
    decollate = settings.get("decollate", False)
    decollate_phases = settings.get("decollate_phases", [])
    if decollate and phase in decollate_phases:
        return Decollated(keys=keys, allow_missing_keys=True)
    return None

class SupervisedLearning(LightningModule):
    """
    A PyTorch Lightning module for supervised learning.

    Args:
        config (dict): The configuration dictionary.

    Attributes:
        model: The model for supervised learning.
        loss_fn: The loss function for supervised learning.
        inferer: The inferer for supervised learning.
        train_decollate: The decollated object for training phase.
        valid_decollate: The decollated object for validation phase.
        test_decollate: The decollated object for test phase.
        post_transforms (dict): The post transforms for each phase.
        train_metrics: The metric collection for training phase.
        valid_metrics: The metric collection for validation phase.
    """
    def __init__(self, config: dict):
        super().__init__()

        # Save all hyperparameters
        self.save_hyperparameters({ "workflow": config })

        # Get configurations for all components
        components = self.hparams.workflow["components"]

        self.model   = self.build_model(components["model"])
        self.loss_fn = self.build_loss_fn(components["loss"])
        self.inferer = self.build_inferer(components["inferer"])

        # Configure batch decollate
        settings = config.get("settings", {})
        self.train_decollate = self.configure_batch_decollate(
            settings, "training"
        )
        self.valid_decollate = self.configure_batch_decollate(
            settings, "validation"
        )
        self.test_decollate = self.configure_batch_decollate(
            settings, "test"
        )

        self.post_transforms = {}
        for phase in ["training", "validation", "test"]:
            self.post_transforms[phase] = build_transforms(
                components["post_transforms"].get(phase, [])
            )

        self.train_metrics = MetricCollection(components["training_metrics"])
        self.valid_metrics = MetricCollection(components["validation_metrics"])

    def build_model(self, config):
        """
        Build the model based on the given configuration.

        Args:
            config: The configuration for building the model.

        Returns:
            The built model.
        """
        builder = ModelBuilder()
        return builder(config)

    def build_loss_fn(self, config):
        """
        Build the loss function based on the given configuration.

        Args:
            config: The configuration for building the loss function.

        Returns:
            The built loss function.
        """
        builder = LossBuilder()
        return builder(config)

    def build_inferer(self, config):
        """
        Build the inferer based on the given configuration.

        Args:
            config: The configuration for building the inferer.

        Returns:
            The built inferer.
        """
        builder = InfererBuilder()
        return builder(config)

    def configure_batch_decollate(self, config, phase):
        """
        Configure batch decollate based on the given configuration and phase.

        Args:
            config: The configuration for batch decollate.
            phase: The phase of the batch.

        Returns:
            Decollated or None: The decollated object or None if decollation is not needed.
        """
        if phase == "training" or phase == "validation":
            keys = [
                "image", "image_meta_dict", "image_transforms",
                "label", "label_meta_dict", "label_transforms",
                "preds"
            ]
        elif phase == "test":
            keys = ["image", "image_meta_dict", "image_transforms", "preds"]
        else:
            raise RuntimeError("Cannot configure decollate for unknow phase.")

        if config.get("decollate", False):
            if phase in config.get("decollate_phases", []):
                return Decollated(keys=keys, allow_missing_keys=True)
        return None

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data: The input data.

        Returns:
            The output of the model.
        """
        return self.inferer(data, self.model)

    def training_step(self, batch, batch_idx):
        """
        Training step of the module.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            The loss value.
        """
        image = batch["image"]
        label = batch["label"]

        # Apply model & compute loss
        batch["preds"] = self.model(image)
        loss = self.loss_fn(batch["preds"], batch["label"])

        if self.train_decollate is not None:
            # Decolloate batch before post transform
            for item in self.train_decollate(batch):
                # Apply post transform on single item
                item = self.post_transforms["training"](item)
                # Compute metric for single item
                self.train_metrics.apply(item)
        else:
            # Apply post transforms on the whole batch
            batch = self.post_transforms["training"](batch)
            # Add result of whole batch to metric
            self.train_metrics.apply(batch)

        # Log current loss value
        self.log_dict({ "train_loss": loss }, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        """
        Training epoch end hook.

        """
        m = self.train_metrics.aggregate()
        self.log_dict(m, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the module.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            The metrics of the batch.
        """
        batch = deepcopy(batch)

        image = batch["image"]
        label = batch["label"]

        # Run inference
        batch["preds"] = self.forward(image)

        # Post transform & compute metrics
        metrics = []
        if self.valid_decollate is not None:
            for item in self.valid_decollate(batch):
                # Apply post transforms first
                item = self.post_transforms["validation"](item)
                # Calculate iteration metrics
                m = self.valid_metrics.apply(item)
                # Save meta data and results
                # m["preds"] = item["preds"]
                m["image_meta_dict"] = item.get("image_meta_dict", {})
                m["label_meta_dict"] = item.get("label_meta_dict", {})
                metrics.append(m)
        else:
            batch = self.post_transforms["validation"](batch)
            m = self.valid_metrics.apply(batch)
            # m["preds"] = batch["preds"]
            m["image_meta_dict"] = batch.get("image_meta_dict")
            m["label_meta_dict"] = batch.get("label_meta_dict")
            metrics.append(m)

        # Output metrics and meta data of this batch
        return metrics

    def on_validation_epoch_end(self):
        """
        Validation epoch end hook.

        """
        m = self.valid_metrics.aggregate()
        self.log_dict(m, sync_dist=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        """
        Test step of the module.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
        """
        # No label for test
        image = batch["image"]

        # Run inference
        batch["preds"] = self.forward(image)

        if self.test_decollate is not None:
            for item in self.test_decollate(batch):
                item = self.post_transforms["test"](item)
        else:
            batch = self.post_transforms["test"](batch)

        # Nothing to output for pure inference
        return None

    def get_optimize_parameters(self):
        """
        Get the parameters for optimization.

        Returns:
            The parameters for optimization.
        """
        return self.model.parameters()

    def configure_optimizers(self):
        """
        Configure the optimizers.

        Returns:
            dict: The optimizer configuration.
        """
        # Extract optimizer & scheduler configurations
        workflow = self.hparams.workflow
        opt_config = workflow["components"]["optimizer"]
        sch_config = workflow["components"].get("scheduler", None)

        opt_builder = OptimizerBuilder()
        sch_builder = SchedulerBuilder()

        opt = {
            "optimizer": opt_builder(
                opt_config,
                params=self.get_optimize_parameters()
            )
        }

        if not sch_config is None:
            # Get or set default scheduler mode
            sch_opts = workflow["settings"].get("scheduler", None)
            if sch_opts is not None:
                interval = sch_opts.get("interval", "epoch")
                frequency = sch_opts.get("frequency", 1)
            else:
                interval = "epoch"
                frequency = 1

            opt["lr_scheduler"] = {
                "scheduler": sch_builder(sch_config, opt=opt["optimizer"]),
                "interval": interval,
                "frequency": frequency
            }

        return opt

