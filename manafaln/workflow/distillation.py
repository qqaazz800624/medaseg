import warnings
from copy import deepcopy
from typing import Dict

import torch
from monai.networks import one_hot

from manafaln.workflow import SupervisedLearning
from manafaln.utils.builders import build_loss_fn

warnings.simplefilter("once")

class TeacherModel(object):
    def __init__(
        self,
        model: torch.nn.Module,
        sigmoid: bool = False,
        softmax: bool = False,
        temperature: float = 1.0
    ):
        self.model = model.eval()

        if sigmoid and softmax:
            raise ValueError("Only one of sigmoid and softmax can be selected")

        self.sigmoid = sigmoid
        self.softmax = softmax

        assert temperature > 0.0
        self.temperature = temperature

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model.forward(data).detach()
            if self.sigmoid:
                out = torch.nn.functional.sigmoid(out / self.temperature)
            if self.softmax:
                out = torch.nn.functional.softmax(
                    out / self.temperature, dim=1
                )
        return out

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)

class DistillationLearning(SupervisedLearning):
    def __init__(self, config: dict):
        super().__init__(config)

        self.kd_temp = config["settings"].get("kd_temperature", 1.0)

        # Setup teacher model
        try:
            self.teacher_model = torch.jit.load(
                config["settings"]["teacher_model_path"]
            )
            self.teacher_model.eval()
        except ValueError:
            self.teacher_model = None

        # Setup knowledge distillation loss
        self.kd_loss_fn = build_loss_fn(config["settings"]["kd_loss"])

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        # Model forward
        batch["preds"] = self.model(image)

        # Teacher model forward
        if self.teacher_model is not None:
            with torch.no_grad():
                kd_preds = self.teacher_model(image)
                kd_preds = torch.nn.functional.softmax(
                    kd_preds / self.kd_temp,
                    dim=1
                )
            #end
        #endif

        # Compute losses
        gt_loss = self.loss_fn(batch["preds"], batch["label"])

        if self.teacher_model is not None:
            kd_loss = self.kd_loss_fn(batch["preds"], kd_preds)
            loss = gt_loss + kd_loss
        else:
            loss = gt_loss

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

class PSKDLearning(SupervisedLearning):
    def __init__(self, config: Dict):
        super().__init__(config)

        self.pskd_alpha       = config["settings"].get("pskd_alpha", 0.7)
        self.pskd_epochs      = config["settings"].get("pskd_epochs", 1000)
        self.pskd_sigmoid     = config["settings"].get("pskd_sigmoid", False)
        self.pskd_softmax     = config["settings"].get("pskd_softmax", False)
        self.pskd_temperature = config["settings"].get("pskd_temperature", 1.0)

        self.teacher_model = None

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        # Model forward
        batch["preds"] = self.model(image)
        batch["label"] = one_hot(
            batch["label"], num_classes=batch["preds"].shape[1]
        )

        # Computer result for teacher model
        alpha = 0.0
        if self.teacher_model is not None:
            teacher_preds = self.teacher_model(image)

            alpha = self.pskd_alpha * min(self.current_epoch / self.pskd_epochs, 1.0)
            target = alpha * teacher_preds + (1.0 - alpha) * batch["label"]

            loss = self.loss_fn(batch["preds"], target)
        else:
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
        self.log_dict({
            "train_loss": loss,
            "alpha": alpha
        })

        return loss

    def training_epoch_end(self, train_epoch_output):
        if self.teacher_model is None:
            self.teacher_model = TeacherModel(
                deepcopy(self.model),
                sigmoid=self.pskd_sigmoid,
                softmax=self.pskd_softmax,
                temperature=self.pskd_temperature
            )
            device = next(self.teacher_model.model.parameters()).device
        else:
            self.teacher_model.load_state_dict(
                deepcopy(self.model.state_dict())
            )
            device = next(self.teacher_model.model.parameters()).device
        return None

