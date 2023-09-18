import torch
from monai.utils import ensure_tuple

from manafaln.common.constants import DefaultKeys
from manafaln.core.builders import ModelBuilder
from manafaln.core.loss import LossHelper
from manafaln.utils import get_items, update_items
from manafaln.workflow.base_v2 import SupervisedLearningV2
from manafaln.workflow.semi_supervised import SemiSupervisedLearning


class SelfSupervisedContrastiveLearning(SupervisedLearningV2):
    DEFAULT_INPUT_KEY = DefaultKeys.INPUT_KEY
    DEFAULT_OUTPUT_KEY = DefaultKeys.OUTPUT_KEY

    """
    A class for self-supervised contrastive learning.

    Inherits from SupervisedLearningV2 class.
    """

    def build_model(self, config):
        """
        Build the model for self-supervised contrastive learning.

        Args:
            config (dict): The configuration for building the model.

        Raises:
            KeyError: If model is not provided in the workflow configuration.

        """
        if config is None:
            raise KeyError(f"model is required in workflow {self.__class__.__name__}")
        builder = ModelBuilder()
        self.model = builder(config)
        self.model_input_keys = ensure_tuple(
            config.get("input_keys", SelfSupervisedContrastiveLearning.DEFAULT_INPUT_KEY)
        )
        self.model_output_keys = ensure_tuple(
            config.get("output_keys", SelfSupervisedContrastiveLearning.DEFAULT_OUTPUT_KEY)
        )

        # Second keys for contrastive learning
        self.model_input_keys_contrast = \
            ensure_tuple(config.get("input_keys_contrast", [s+"_contrast" for s in self.model_input_keys]))
        self.model_output_keys_contrast = \
            ensure_tuple(config.get("output_keys_contrast", [s+"_contrast" for s in self.model_output_keys]))

    def _training_step(self, batch, model_input_keys, model_output_keys):
        """
        Perform a single training step.

        Args:
            batch (dict): The input batch.
            model_input_keys (tuple): The keys for model input.
            model_output_keys (tuple): The keys for model output.

        Returns:
            dict: The updated batch.

        """
        model_input = get_items(batch, model_input_keys)
        preds = self.model(*model_input)
        batch = update_items(batch, model_output_keys, preds)
        return batch

    def training_step(self, batch: dict, batch_idx):
        """
        Perform a training step.

        Args:
            batch (dict): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            dict: The loss value.

        """
        batch = self._training_step(batch, self.model_input_keys, self.model_output_keys)
        batch = self._training_step(batch, self.model_input_keys_contrast, self.model_output_keys_contrast)

        # Apply post processing
        batch = self.post_processing(batch)

        # Compute loss
        loss = self.loss_fn(**batch)

        # Log current loss value
        self.log_dict(loss, sync_dist=True)

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
    """
    A class for semi-supervised contrastive learning.

    Inherits from SelfSupervisedContrastiveLearning and SemiSupervisedLearning classes.
    """

    def __init__(self, config):
        """
        Initialize the SemiSupervisedContrastiveLearning class.

        Args:
            config (dict): The configuration for the class.

        """
        super().__init__(config)
        self.features_keys = config["settings"]["features_keys"]

    def build_loss_fn(self, config):
        """
        Build the loss function for semi-supervised contrastive learning.

        Args:
            config (dict): The configuration for building the loss function.

        """
        super().build_loss_fn(config)
        contrastive_loss = LossHelper(config["contrastive"])
        self.loss_fn.update({"contrastive": contrastive_loss})

    def training_step(self, batch: dict, batch_idx):
        """
        Perform a training step for semi-supervised contrastive learning.

        Args:
            batch (dict): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            dict: The loss value.

        """
        # Labeled data
        batch["labeled"] = self._training_step(
            batch["labeled"], self.model_input_keys, self.model_output_keys)
        # Unlabeled data
        batch["unlabeled"] = self._training_step(
            batch["unlabeled"], self.model_input_keys, self.model_output_keys)
        # Labeled data, contrast
        batch["labeled"] = self._training_step(
            batch["labeled"], self.model_input_keys_contrast, self.model_output_keys_contrast)
        # Unlabeled data, contrast
        batch["unlabeled"] = self._training_step(
            batch["unlabeled"], self.model_input_keys_contrast, self.model_output_keys_contrast)

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
            [
                batch["labeled"][features_key],
                batch["unlabeled"][features_key]
            ],
            dim=0
            )
        contrastive_loss = self.loss_fn["contrastive"](**features)

        total_loss = labeled_loss.pop("loss") + \
            unlabeled_loss.pop("loss") + \
            contrastive_loss.pop("loss")
        loss = {"loss": total_loss, **labeled_loss, **unlabeled_loss, **contrastive_loss}

        # Log current loss value
        self.log_dict(loss, sync_dist=True)

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
