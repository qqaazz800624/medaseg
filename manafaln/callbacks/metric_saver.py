from typing import Any, Union, List, Tuple

import torch
import pandas as pd
from monai.transforms import Decollated
from pytorch_lightning import Callback, LightningModule, Trainer

def ensure_list(arg: Union[Any, List[Any]]):
    if not isinstance(arg, List):
        return [arg]
    else:
        return arg

class IterationMetricSaver(Callback):
    def __init__(
        self,
        filename: str,
        metrics: Union[str, List[str]],
        meta_dict_key: str = "image_meta_dict",
        meta_dict_info: Union[str, List[str]] = "filename_or_obj",
        decollate=False,
        save_preds=False
    ) -> None:
        super().__init__()

        self.filename = filename
        self.metrics = ensure_list(metrics)
        self.meta_dict_key = meta_dict_key
        self.meta_dict_info = ensure_list(meta_dict_info)
        self.save_preds = save_preds

        if decollate:
            if self.save_preds:
                keys = [meta_dict_key, preds, metrics]
            else:
                keys = [meta_dict_key, metrics]

            self.decollate_fn = Decollated(
                keys=keys
            )
        else:
            self.decollate_fn = None

        self.buffer = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: List,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        if self.decollate_fn is not None:
            if len(outputs) > 1:
                raise ValueError("MetricSaver expected collated inputs if 'decollate' is enabled")

            if self.save_preds:
                outputs["preds"] = batch["preds"]

            for item in self.decollate_fn(outputs):
                row = []
                for key in self.meta_dict_info:
                    row.append(item[self.meta_dict_key][key])
                if self.save_preds:
                    row.append(item["preds"])
                for m in self.metrics:
                    row.append(item[m])
                self.buffer.append(row)
        else:
            for item, idx in enumerate(outputs):
                row = []
                for key in self.meta_dict_info:
                    row.append(item[self.meta_dict_key][key])
                if self.save_preds:
                    row.append(batch["preds"][idx])
                for m in self.metrics:
                    row.append(item[m])
                self.buffer.append(row)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        if len(self.buffer) > 0:
            # Make titles
            columns = self.meta_dict_info[:]
            if self.save_preds:
                columns.append("preds")
            columns += self.metrics

            # Create dataframe with data and title
            df = pd.DataFrame(self.buffer, columns=columns)

            # Save results to file
            df.to_csv(self.filename, index=False)

            # Cleanup
            self.buffer = []
