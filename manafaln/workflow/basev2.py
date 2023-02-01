from monai.transforms import Decollated
from monai.utils import ensure_tuple
from pytorch_lightning import LightningModule

from manafaln.common.constants import DefaultKeys
from manafaln.core.builders import (InfererBuilder, ModelBuilder,
                                    OptimizerBuilder, SchedulerBuilder)
from manafaln.core.loss import LossHelper
from manafaln.core.metricv2 import MetricCollection
from manafaln.core.transforms import build_transforms

DEFAULT_INPUT_KEY = DefaultKeys.INPUT_KEY
DEFAULT_OUTPUT_KEY = DefaultKeys.OUTPUT_KEY

class SupervisedLearningV2(LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        # Save all hyperparameters
        self.save_hyperparameters({ "workflow": config })

        # Get configurations for all components
        components = self.hparams.workflow["components"]
        self.build_model(components.get("model"))
        self.build_post_processing(components.get("post_processing"))
        self.build_loss_fn(components.get("loss"))
        self.build_inferer(components.get("inferer"))
        self.build_post_transforms(components.get("post_transforms"))
        self.build_metrics(components.get("metrics"))

        # Configure training settings
        settings = self.hparams.workflow["settings"]
        self.build_decollate_fn(settings.get("decollate"))

    def build_model(self, config):
        if config is None:
            raise KeyError(f"model is required in workflow {self.__class__.__name__}")
        builder = ModelBuilder()
        self.model = builder(config)
        self.model_input_keys = ensure_tuple(config.get("input_keys", DEFAULT_INPUT_KEY))
        self.model_output_keys = ensure_tuple(config.get("output_keys", DEFAULT_OUTPUT_KEY))

    def build_post_processing(self, config):
        # TODO: Check backpropagation with torch.Tensor.requires_grad = True
        if config is None:
            config = []
        self.post_processing = build_transforms(config)

    def build_loss_fn(self, config):
        if config is None:
            raise KeyError(f"loss is required in workflow {self.__class__.__name__}")
        self.loss_fn = LossHelper(config)

    def build_inferer(self, config):
        # If inferer is not given, use SimpleInferer
        if config is None:
            config = {"name": "SimpleInferer"}
        builder = InfererBuilder()
        self.inferer = builder(config)

    def build_post_transforms(self, config):
        if config is None:
            config = {}
        self.post_transforms = {
            phase: build_transforms(
                config.get(phase, [])
            )
            for phase in ["training", "validation", "predict"]
        }

    def build_metrics(self, config):
        if config is None:
            config = {}
        self.training_metrics = MetricCollection(config.get("training", []))
        self.validation_metrics = MetricCollection(config.get("validation", []))

    def build_decollate_fn(self, settings):
        decollate_fn = {}
        for phase in ["training", "validation", "predict"]:
            if (settings is None) or (phase not in settings):
                decollate_fn[phase] = None
            else:
                keys = settings[phase]
                decollate_fn[phase] = Decollated(keys=keys, allow_missing_keys=True)
        self.decollate_fn = decollate_fn

    def get_optimize_parameters(self):
        return self.model.parameters()

    def configure_optimizers(self):
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

    def forward(self, inputs, *args):
        result = self.inferer(inputs, self.model, *args)
        return result

    def training_step(self, batch: dict, batch_idx):
        # Get model input
        model_input = (batch[k] for k in self.model_input_keys)

        # Get model output
        preds = self.model(*model_input)

        # Update model output to batch
        preds = ensure_tuple(preds, wrap_array=True)
        batch.update(zip(self.model_output_keys, preds))

        # Apply post processing
        batch = self.post_processing(batch)

        # Compute loss
        loss = self.loss_fn(**batch)

        # Log current loss value
        self.log_dict(loss)

        if self.decollate_fn["training"] is not None:
            # Decolloate batch before post transform
            for item in self.decollate_fn["training"](batch):
                # Apply post transform on single item
                item = self.post_transforms["training"](item)
                # Compute metric for single item
                self.training_metrics.update(**item)
        else:
            # Apply post transforms on the whole batch
            batch = self.post_transforms["training"](batch)
            # Add result of whole batch to metric
            self.training_metrics.update(**batch)

        # Return {"loss": xxx, ...}
        return loss

    def training_epoch_end(self, outputs):
        m = self.training_metrics.compute()
        self.log_dict(m)
        self.training_metrics.reset()

    def validation_step(self, batch: dict, batch_idx):
        # Get model input
        model_input = (batch[k] for k in self.model_input_keys)

        # Run inference
        preds = self.forward(*model_input)

        # Update prediction to batch
        preds = ensure_tuple(preds, wrap_array=True)
        batch.update(zip(self.model_output_keys, preds))

        # Post transform & compute metrics
        if self.decollate_fn["validation"] is not None:
            for item in self.decollate_fn["validation"](batch):
                # Apply post transforms first
                item = self.post_transforms["validation"](item)
                # Calculate metrics
                self.validation_metrics.update(**item)
        else:
            batch = self.post_transforms["validation"](batch)
            self.validation_metrics.update(**batch)

    def validation_epoch_end(self, validation_step_outputs):
        m = self.validation_metrics.compute()
        self.log_dict(m)
        self.validation_metrics.reset()

    def predict_step(self, batch: dict, batch_idx):
        # No label for predict
        model_input = (batch[k] for k in self.model_input_keys)

        # Run inference
        preds = self.forward(*model_input)

        # Update prediction to batch
        preds = ensure_tuple(preds, wrap_array=True)
        batch.update(zip(self.model_output_keys, preds))

        if self.decollate_fn["predict"] is not None:
            outputs = []
            for item in self.decollate_fn["predict"](batch):
                item = self.post_transforms["predict"](item)
                outputs.append(item)
        else:
            outputs = self.post_transforms["predict"](batch)

        return outputs
