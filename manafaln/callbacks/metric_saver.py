import os
from typing import Dict, Union, List, Sequence

import pandas as pd
from monai.transforms import Decollated
from monai.utils import PostFix, ensure_tuple, convert_to_numpy
from monai.utils.misc import ImageMetaKey
from pytorch_lightning import Callback, LightningModule, Trainer

from manafaln.common.constants import DefaultKeys
from manafaln.utils.misc import ensure_list, ensure_python_value, get_attr, get_item

DEFAULT_META_KEY = PostFix.meta(DefaultKeys.INPUT_KEY) # "image_meta_dict"
DEFAULT_INFO_KEYS = ImageMetaKey.FILENAME_OR_OBJ  # "filename_or_obj"

class IterationMetricSaver(Callback):
    def __init__(
        self,
        filename: str,
        metrics: Union[str, List[str]],
        meta_dict_key: str = DEFAULT_META_KEY, # "image_meta_dict"
        meta_dict_info: Union[str, List[str]] = DEFAULT_INFO_KEYS, # "filename_or_obj"
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
                keys = [meta_dict_key, "preds"] + self.metrics
            else:
                keys = [meta_dict_key] + self.metrics

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
            outputs = outputs[0]

            for item in self.decollate_fn(outputs):
                row = []
                for key in self.meta_dict_info:
                    row.append(item[self.meta_dict_key][key])
                if self.save_preds:
                    preds = ensure_python_value(item["preds"])
                    row.append(preds)
                for m in self.metrics:
                    metric = ensure_python_value(item[m])
                    row.append(metric)
                self.buffer.append(row)
        else:
            for idx, item in enumerate(outputs):
                row = []
                for key in self.meta_dict_info:
                    row.append(item[self.meta_dict_key][key])
                if self.save_preds:
                    preds = ensure_python_value(outputs["preds"][idx])
                    row.append(preds)
                for m in self.metrics:
                    metric = ensure_python_value(item[m])
                    row.append(metric)
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

class CheckpointMetricSaver(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["logged_metrics"] = trainer.logged_metrics

class IterativeMetricSaver(Callback):
    """
    A callback to save metrics and info at the end of each validation epoch.
    Args:
        filename (os.PathLike): File name to save the metrics and info
        states (Union[str, Sequence[str]]): Metrics state(s) to save
        info_keys (Union[str, Sequence[str]], optional): Info key(s) to save.
            Defaults to "image_meta_dict.filename_or_obj".

    Attributes:
        buffer (Dict): A dictionary to buffer the metrics and info before saving
    """
    def __init__(
        self,
        filename: os.PathLike,
        states: Union[str, Sequence[str]],
        info_keys: Union[str, Sequence[str]] = f"{DEFAULT_META_KEY}.{DEFAULT_INFO_KEYS}",
    ):
        self.filename = filename
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        self.info_keys = ensure_tuple(info_keys)
        self.states = ensure_tuple(states)

        self.initialize_buffer()

    def initialize_buffer(self):
        self.buffer = {k: [] for k in self.info_keys + self.states}

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx
    ) -> None:

        # In each validation step, get the info from batch to be saved.
        for key in self.info_keys:
            info = get_item(batch, key)
            info = convert_to_numpy(info)
            self.buffer[key].extend(info)

        # In the last validation step,
        if batch_idx == trainer.num_val_batches[dataloader_idx] - 1:
            # get the states that is stored in metrics
            for state in self.states:
                metric = get_attr(pl_module.validation_metrics, state)
                metric = convert_to_numpy(metric)
                self.buffer[state].extend(metric)

            # save the stored info and metrics into csv
            df = pd.DataFrame(self.buffer)
            df.to_csv(self.filename, index=False)

            self.initialize_buffer()
