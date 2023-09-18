from torch.nn import ModuleDict

from manafaln.core.loss import LossHelper
from manafaln.utils import get_items, update_items
from manafaln.workflow.base_v2 import SupervisedLearningV2

class SemiSupervisedLearning(SupervisedLearningV2):
    """
    A class for semi-supervised learning workflow.

    Inherits from SupervisedLearningV2 class.

    Attributes:
        loss_fn (ModuleDict): A dictionary of loss functions for labeled and unlabeled data.

    Methods:
        build_loss_fn(config): Builds the loss function dictionary.
        _training_step(batch): Performs a single training step.
        training_step(batch, batch_idx): Performs a training step and computes loss and metrics.
    """

    def build_loss_fn(self, config):
        """
        Builds the loss function dictionary.

        Args:
            config (dict): A dictionary containing the loss configurations for labeled and unlabeled data.

        Raises:
            KeyError: If loss is not provided in the workflow configuration.

        Returns:
            None
        """
        if config is None:
            raise KeyError(f"loss is required in workflow {self.__class__.__name__}")

        loss_fn = {}
        loss_fn["labeled"] = LossHelper(config.get("labeled", {}))
        loss_fn["unlabeled"] = LossHelper(config.get("unlabeled", {}))
        self.loss_fn = ModuleDict(loss_fn)

    def _training_step(self, batch):
        """
        Performs a single training step.

        Args:
            batch (dict): A dictionary containing the labeled or unlabeled batch data.

        Returns:
            dict: The updated batch data after performing the training step.
        """
        model_input = get_items(batch, self.model_input_keys)
        preds = self.model(*model_input)
        batch = update_items(batch, self.model_output_keys, preds)
        batch = self.post_processing(batch)
        return batch

    def training_step(self, batch: dict, batch_idx):
        """
        Performs a training step and computes loss and metrics.

        Args:
            batch (dict): A dictionary containing the labeled and unlabeled batch data.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the loss and metrics computed during the training step.
        """
        batch["labeled"] = self._training_step(batch["labeled"])
        batch["unlabeled"] = self._training_step(batch["unlabeled"])

        # Compute loss
        labeled_loss = self.loss_fn["labeled"](**batch["labeled"])
        unlabeled_loss = self.loss_fn["unlabeled"](**batch["unlabeled"])
        total_loss = labeled_loss.pop("loss") + unlabeled_loss.pop("loss")
        loss = {"loss": total_loss, **labeled_loss, **unlabeled_loss}

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
