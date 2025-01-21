from copy import deepcopy
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from manafaln.workflow import SupervisedLearning

def ensure_length(x, length):
    if len(x) < length:
        x = x + [0.0] * (length - len(x))
    elif len(x) > length:
        x = x[:length]
    return x

class DeepSupervision(SupervisedLearning):
    def __init__(self, config: Dict):
        super().__init__(config)

        self.ds_weights = config["settings"].get("ds_weights", None)

    def model_infer(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if y.dim() > x.dim():
            return y[:, 0, ::]
        else:
            return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inferer(x, self.model_infer)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        outputs = self.model(image)

        loss = 0.0
        if outputs.dim() > image.dim():
            batch["preds"] = outputs[:, 0, ::]

            num_outputs = outputs.shape[1]
            if self.ds_weights is None:
                weights = [1.0] * num_outputs
            else:
                weights = ensure_length(self.ds_weights, num_outputs)

            for i, w in enumerate(weights):
                loss += w * self.loss_fn(outputs[:, i, ::], label)
        else:
            batch["preds"] = outputs
            loss += self.loss_fn(outputs, label)

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
        self.log_dict({ "train_loss": loss })

        return loss

    def validation_step(self, batch, batch_idx):
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

