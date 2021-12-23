import warnings
import torch
from manafaln.workflow import SupervisedLearning
from manafaln.utils.builders import build_loss_fn

warnings.simplefilter("once")

class DistillationLearning(SupervisedLearning):
    def __init__(self, config: dict):
        super().__init__(config)

        self.kd_temp = config["settings"].get("kd_temperature", 1.0)

        # Setup teacher model
        self.teacher_model = torch.jit.load(config["settings"]["teacher_model_path"])
        self.teacher_model.eval()

        # Setup knowledge distillation loss
        self.kd_loss_fn = build_loss_fn(config["settings"]["kd_loss"])

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        # Model forward
        batch["preds"] = self.model(image)

        # Teacher model forward
        with torch.no_grad():
            device = next(self.model.parameters()).device
            if next(self.teacher_model.parameters()).device != device:
                print(f"Move teacher model to {device}")
                self.teacher_model = self.teacher_model.to(device)

            kd_preds = self.teacher_model(image)
            kd_preds = torch.nn.functional.softmax(kd_preds / self.kd_temp, dim=1)

        # Compute losses
        gt_loss = self.loss_fn(batch["preds"], batch["label"])
        kd_loss = self.kd_loss_fn(batch["preds"], kd_preds)
        loss = gt_loss + kd_loss

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
        self.log_dict({
            "train_loss": loss,
            "train_gt_loss": gt_loss,
            "train_kd_loss": kd_loss
        })

        return loss
