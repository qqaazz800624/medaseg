from monai.utils import ensure_tuple
import torch

from manafaln.core.loss import LossHelper
from manafaln.common.constants import DefaultKeys
from manafaln.core.builders import ModelBuilder
from manafaln.workflow.basev2 import SupervisedLearningV2
from manafaln.workflow.semi_supervised import SemiSupervisedLearning

DEFAULT_INPUT_KEY = DefaultKeys.INPUT_KEY
DEFAULT_OUTPUT_KEY = DefaultKeys.OUTPUT_KEY

class SelfSupervisedContrastiveLearning(SupervisedLearningV2):
    def build_model(self, config):
        if config is None:
            raise KeyError(f"model is required in workflow {self.__class__.__name__}")
        builder = ModelBuilder()
        self.model = builder(config)
        self.model_input_keys = ensure_tuple(config.get("input_keys", DEFAULT_INPUT_KEY))
        self.model_output_keys = ensure_tuple(config.get("output_keys", DEFAULT_OUTPUT_KEY))

        # Second keys for contrastive learning
        self.model_input_keys_contrast = \
            ensure_tuple(config.get("input_keys_contrast", [s+"_contrast" for s in self.model_input_keys]))
        self.model_output_keys_contrast = \
            ensure_tuple(config.get("output_keys_contrast", [s+"_contrast" for s in self.model_output_keys]))

    def training_step(self, batch: dict, batch_idx):
        # Get model input
        model_input = (batch[k] for k in self.model_input_keys)
        model_input_contrast = (batch[k] for k in self.model_input_keys_contrast)

        # Get model output
        preds = self.model(*model_input)
        preds_contrast = self.model(*model_input_contrast)

        # Update model output to batch
        preds = ensure_tuple(preds, wrap_array=True)
        batch.update(zip(self.model_output_keys, preds))
        preds_contrast = ensure_tuple(preds_contrast, wrap_array=True)
        batch.update(zip(self.model_output_keys_contrast, preds_contrast))

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

class SemiSupervisedContrastiveLearning(SelfSupervisedContrastiveLearning, SemiSupervisedLearning):
    def __init__(self, config):
        super().__init__(config)
        self.features_keys = config["settings"]["features_keys"]

    def build_loss_fn(self, config):
        super().build_loss_fn(config)
        contrastive_loss = LossHelper(config["contrastive"])
        self.loss_fn.update({"contrastive": contrastive_loss})

    def training_step(self, batch: dict, batch_idx):
        # Get model input
        labeled_model_input = (batch["labeled"][k] for k in self.model_input_keys)
        unlabeled_model_input = (batch["unlabeled"][k] for k in self.model_input_keys)
        labeled_model_input_contrast = (batch["labeled"][k] for k in self.model_input_keys_contrast)
        unlabeled_model_input_contrast = (batch["unlabeled"][k] for k in self.model_input_keys_contrast)

        # Get model output
        labeled_preds = self.model(*labeled_model_input)
        unlabeled_preds = self.model(*unlabeled_model_input)
        labeled_preds_contrast = self.model(*labeled_model_input_contrast)
        unlabeled_preds_contrast = self.model(*unlabeled_model_input_contrast)

        # Update model output to batch
        labeled_preds = ensure_tuple(labeled_preds, wrap_array=True)
        batch["labeled"].update(zip(self.model_output_keys, labeled_preds))
        unlabeled_preds = ensure_tuple(unlabeled_preds, wrap_array=True)
        batch["unlabeled"].update(zip(self.model_output_keys, unlabeled_preds))
        labeled_preds_contrast = ensure_tuple(labeled_preds_contrast, wrap_array=True)
        batch["labeled"].update(zip(self.model_output_keys_contrast, labeled_preds_contrast))
        unlabeled_preds_contrast = ensure_tuple(unlabeled_preds_contrast, wrap_array=True)
        batch["unlabeled"].update(zip(self.model_output_keys_contrast, unlabeled_preds_contrast))

        # Apply post processing
        batch["labeled"] = self.post_processing(batch["labeled"])
        batch["unlabeled"] = self.post_processing(batch["unlabeled"])

        # Compute loss
        labeled_loss = self.loss_fn["labeled"](**batch["labeled"])
        unlabeled_loss = self.loss_fn["unlabeled"](**batch["unlabeled"])

        # Combine features from labeled and unlabeled dataset
        # to compute contrastive loss together
        features = {}
        for features_key in self.features_keys:
            features[features_key] = torch.cat(
            [batch["labeled"][features_key], batch["unlabeled"][features_key]],
            dim=0
            )
        contrastive_loss = self.loss_fn["contrastive"](**features)

        total_loss = labeled_loss.pop("loss") + \
            unlabeled_loss.pop("loss") + \
            contrastive_loss.pop("loss")
        loss = {"loss": total_loss, **labeled_loss, **unlabeled_loss, **contrastive_loss}

        # Log current loss value
        self.log_dict(loss)

        # Post transform and compute metrics on labeled data only
        if self.decollate_fn["training"] is not None:
            # Decolloate batch before post transform
            for item in self.decollate_fn["training"](batch["labeled"]):
                # Apply post transform on single item
                item = self.post_transforms["training"](item)
                # Compute metric for single item
                self.training_metrics.update(**item)
        else:
            # Apply post transforms on the whole batch
            batch["labeled"] = self.post_transforms["training"](batch["labeled"])
            # Add result of whole batch to metric
            self.training_metrics.update(**batch["labeled"])

        # Return {"loss": xxx, ...}
        return loss
