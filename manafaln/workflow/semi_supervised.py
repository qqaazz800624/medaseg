from torch.nn import ModuleDict
from monai.utils import ensure_tuple

from manafaln.core.loss import LossHelper
from manafaln.workflow.basev2 import SupervisedLearningV2

class SemiSupervisedLearning(SupervisedLearningV2):
    def build_loss_fn(self, config):
        if config is None:
            raise KeyError(f"loss is required in workflow {self.__class__.__name__}")

        loss_fn = {}
        loss_fn["labeled"] = LossHelper(config.get("labeled", {}))
        loss_fn["unlabeled"] = LossHelper(config.get("unlabeled", {}))
        self.loss_fn = ModuleDict(loss_fn)

    def training_step(self, batch: dict, batch_idx):
        # Get model input
        labeled_model_input = (batch["labeled"][k] for k in self.model_input_keys)
        unlabeled_model_input = (batch["unlabeled"][k] for k in self.model_input_keys)

        # Get model output
        labeled_preds = self.model(*labeled_model_input)
        unlabeled_preds = self.model(*unlabeled_model_input)

        # Update model output to batch
        labeled_preds = ensure_tuple(labeled_preds, wrap_array=True)
        batch["labeled"].update(zip(self.model_output_keys, labeled_preds))
        unlabeled_preds = ensure_tuple(unlabeled_preds, wrap_array=True)
        batch["unlabeled"].update(zip(self.model_output_keys, unlabeled_preds))

        # Apply post processing
        batch["labeled"] = self.post_processing(batch["labeled"])
        batch["unlabeled"] = self.post_processing(batch["unlabeled"])

        # Compute loss
        labeled_loss = self.loss_fn["labeled"](**batch["labeled"])
        unlabeled_loss = self.loss_fn["unlabeled"](**batch["unlabeled"])
        total_loss = labeled_loss.pop("loss") + unlabeled_loss.pop("loss")
        loss = {"loss": total_loss, **labeled_loss, **unlabeled_loss}

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
