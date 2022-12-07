import warnings
from copy import deepcopy
from typing import Dict, Optional

import torch
from monai.networks import one_hot

from manafaln.core.builders import LossBuilder
from manafaln.utils import DummyTorchModule
from manafaln.workflow import SupervisedLearning

class TeacherModel(DummyTorchModule):
    def __init__(
        self,
        config: Dict,
        temperature: float = 1.0,
        sigmoid: bool = False,
        softmax: bool = False,
        argmax: bool = False,
        to_onehot: Optional[int] = None,
        thredhold_value: Optional[float] = None
    ):
        super().__init__(config=config)

        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        self.temperature = float(temperature)

        if sigmoid and softmax:
            raise ValueError("Cannot use sigmoid and softmax at the same time")

        self.sigmoid = sigmoid
        self.softmax = softmax
        self.argmax = argmax
        self.to_onehot = int(to_onehot) if to_onehot is not None else None
        self.thredhold_value = float(thredhold_value) if thredhold_value is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x) / self.temperature

        # For batch only or single channel output
        if logits.dim() == 1 or logits.shape[1] == 1:
            if self.sigmoid:
                logits = torch.sigmoid(logits)
            if self.thresold_value:
                logits = (logits > self.thresold_value).float()
            return logits

        # For multi-channel output
        if self.sigmoid:
            logits = torch.sigmoid(logits)

        if self.softmax:
            logits = torch.softmax(logits, dim=1)

        if self.argmax:
            logits = torch.argmax(logits, dim=1, keepdim=True)

        if self.to_onehot:
            logits = one_hot(logits, num_classes=self.to_onehot, dim=1)

        if self.thresold_value:
            logits = (logits > self.thresold_value).float()

        return logits

class DistillationLearning(SupervisedLearning):
    def __init__(self, config: dict):
        super().__init__(config)

        distill_config = config["settings"].get("distillation", {})

        loss_builder = LossBuilder()
        try:
            # Setup teacher model & inference
            self.teacher_model = TeacherModel(
                distill_config["model"],
                temperature=distill_config.get("temperature", 1.0),
                sigmoid=distill_config.get("sigmoid", False),
                softmax=distill_config.get("softmax", False),
                argmax=distill_config.get("argmax", False),
                to_onehot=distill_config.get("to_onehot", None),
                thredhold_value=distill_config.get("thredhold_value", None)
            )
            # Setup knowledge distillation loss
            self.kd_weight = distill_config.get("weight", 1.0)
            self.kd_loss_fn = loss_builder(distill_config["loss"])
        except ValueError:
            self.teacher_model = None
            self.kd_weight = 1.0
            self.kd_loss_fn = lambda x, y: 0.0

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        # Model forward
        batch["preds"] = self.model(image)

        # Compute losses
        gt_loss = self.loss_fn(batch["preds"], batch["label"])

        # Teacher model forward
        if self.teacher_model is not None:
            kd_preds = self.teacher_model(image)
        else:
            kd_preds = None

        kd_loss = self.kd_loss_fn(batch["preds"], kd_preds)
        loss = gt_loss + self.kd_weight * kd_loss

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

