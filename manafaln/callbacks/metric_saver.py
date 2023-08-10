import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from monai.config.type_definitions import PathLike
from monai.transforms import Decollated
from monai.utils import PostFix, convert_to_numpy, ensure_tuple
from monai.utils.misc import ImageMetaKey
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import ConfusionMatrix

from manafaln.common.constants import DefaultKeys
from manafaln.core.metricv2 import DEFAULT_METRIC_INPUT_KEYS
from manafaln.utils import get_items
from manafaln.utils.misc import ensure_list, ensure_python_value, get_attr, get_item

DEFAULT_META_KEY = PostFix.meta(DefaultKeys.INPUT_KEY)  # "image_meta_dict"
DEFAULT_INFO_KEYS = ImageMetaKey.FILENAME_OR_OBJ  # "filename_or_obj"


class IterationMetricSaver(Callback):
    def __init__(
        self,
        filename: str,
        metrics: Union[str, List[str]],
        meta_dict_key: str = DEFAULT_META_KEY,  # "image_meta_dict"
        meta_dict_info: Union[str, List[str]] = DEFAULT_INFO_KEYS,  # "filename_or_obj"
        decollate=False,
        save_preds=False,
    ) -> None:
        """
        A callback to save metrics and info at the end of each validation batch.

        Args:
            filename (str): File name to save the metrics and info.
            metrics (Union[str, List[str]]): Metrics to save.
            meta_dict_key (str, optional): Key to access the meta dictionary. Defaults to DEFAULT_META_KEY.
            meta_dict_info (Union[str, List[str]], optional): Info keys to save. Defaults to DEFAULT_INFO_KEYS.
            decollate (bool, optional): Whether to decollate the inputs. Defaults to False.
            save_preds (bool, optional): Whether to save the predictions. Defaults to False.
        """
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

            self.decollate_fn = Decollated(keys=keys)
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
        dataloader_idx: int,
    ) -> None:
        """
        Called at the end of each validation batch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The lightning module.
            outputs (List): The outputs of the validation batch.
            batch (Dict): The validation batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.
        """
        if self.decollate_fn is not None:
            if len(outputs) > 1:
                raise ValueError(
                    "MetricSaver expected collated inputs if 'decollate' is enabled"
                )
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
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """
        Called at the end of each validation epoch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The lightning module.
        """
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
        """
        Called when saving a checkpoint.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The lightning module.
            checkpoint (Dict): The checkpoint dictionary.
        """
        checkpoint["logged_metrics"] = trainer.logged_metrics


class IterativeMetricSaver(Callback):
    def __init__(
        self,
        filename: PathLike,
        states: Union[str, Sequence[str]],
        info_keys: Union[
            str, Sequence[str]
        ] = f"{DEFAULT_META_KEY}.{DEFAULT_INFO_KEYS}",
    ):
        """
        A callback to save metrics and info at the end of each validation epoch.

        Args:
            filename (PathLike): File name to save the metrics and info.
            states (Union[str, Sequence[str]]): Metrics state(s) to save.
            info_keys (Union[str, Sequence[str]], optional): Info key(s) to save. Defaults to DEFAULT_META_KEY.DEFAULT_INFO_KEYS.
        """
        self.filename = filename

        dirname = os.path.dirname(self.filename)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)

        self.info_keys = ensure_tuple(info_keys)
        self.states = ensure_tuple(states)

        self.initialize_buffer()

    def initialize_buffer(self):
        self.buffer = {k: [] for k in self.info_keys + self.states}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """
        Called at the end of each validation batch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The lightning module.
            outputs (Any): The outputs of the validation batch.
            batch (Any): The validation batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.
        """
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

class ConfusionMatrixSaver(Callback):
    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        input_keys: Sequence[str] = DEFAULT_METRIC_INPUT_KEYS,
        output_file: PathLike = "confusion_matrix.png",
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ):
        """
        A callback to save the confusion matrix.

        Args:
            task (Literal["binary", "multiclass", "multilabel"]): The task type.
            input_keys (Sequence[str], optional): The input keys. Defaults to DEFAULT_METRIC_INPUT_KEYS.
            output_file (PathLike, optional): The output file name. Defaults to "confusion_matrix.png".
            threshold (float, optional): The threshold value. Defaults to 0.5.
            num_classes (Optional[int], optional): The number of classes. Defaults to None.
            num_labels (Optional[int], optional): The number of labels. Defaults to None.
            normalize (Optional[Literal["true", "pred", "all", "none"]], optional): The normalization type. Defaults to None.
            ignore_index (Optional[int], optional): The ignore index. Defaults to None.
            validate_args (bool, optional): Whether to validate the arguments. Defaults to True.
            **kwargs (Any): Additional keyword arguments.
        """
        self.input_keys = ensure_tuple(input_keys)
        self.confusion_matrix_metric = ConfusionMatrix(
            task=task,
            threshold=threshold,
            num_classes=num_classes,
            num_labels=num_labels,
            normalize=normalize,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        self.output_file = output_file

    def plot_cm(self, cm: np.ndarray) -> Figure:
        """
        Plot the confusion matrix.

        Args:
            cm (np.ndarray): The confusion matrix.

        Returns:
            Figure: The plotted figure.
        """
        fig, ax = plt.subplots()

        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_xlabel("Prediction")
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_ylabel("Ground Truth")

        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = [
            "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
        ]
        group_labels = [
            f"{name}\n{percent}"
            for name, percent in zip(group_counts, group_percentages)
        ]
        group_labels = np.asarray(group_labels).reshape(cm.shape[0], cm.shape[1])

        # Loop over data dimensions and create text annotations.
        color_thrs = cm.max() / 2.0
        for i in np.arange(cm.shape[0]):
            for j in np.arange(cm.shape[1]):
                color = "white" if cm[i, j] > color_thrs else "black"
                ax.text(j, i, group_labels[i, j], ha="center", va="center", color=color)

        # calculate confusion matrix related metrics
        tn, fp, fn, tp = (
            cm[:-1, :-1].sum(),
            cm[:-1, -1].sum(),
            cm[-1, :-1].sum(),
            cm[-1, -1],
        )
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)

        # plot sensitivity and specifiticy
        ax1 = ax.twinx()
        ax1.grid(False)
        ax1.set_yticks(np.array([0, len(cm) / 2]) + 0.5)
        ax1.set_yticklabels([f"Sens: {sens:.3f}", f"Spec: {spec:.3f}"])
        ax1.set(ylim=(0, len(cm)))
        ax1.tick_params(right=False, left=False)
        ax.tick_params(left=False)

        # plot ppv and npv
        ax2 = ax.twiny()
        ax2.grid(False)
        ax2.set_xticks(np.array([len(cm) / 2, len(cm)]) - 0.5)
        ax2.set_xticklabels([f"NPV: {npv:.3f}", f"PPV: {ppv:.3f}"])
        ax2.set(xlim=(0, len(cm)))
        ax2.xaxis.tick_bottom()
        ax2.xaxis.set_label_position("bottom")
        ax2.tick_params(bottom=False)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(top=False)

        ax.set_aspect(aspect="auto", adjustable="box")

        return fig

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Called at the end of each validation batch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The lightning module.
            outputs (Optional[STEP_OUTPUT]): The outputs of the validation batch.
            batch (Any): The validation batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.
        """
        preds, target = get_items(batch, self.input_keys)
        device = self.confusion_matrix_metric.device
        self.confusion_matrix_metric.update(preds.to(device), target.to(device))

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """
        Called at the end of each validation epoch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The lightning module.
        """
        cm = self.confusion_matrix_metric.compute().numpy()
        fig = self.plot_cm(cm)

        dirname = os.path.dirname(self.output_file)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(self.output_file, bbox_inches="tight")
