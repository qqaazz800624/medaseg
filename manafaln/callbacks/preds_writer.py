import os
from typing import Literal, Sequence, Union

import pandas as pd
from monai.config.type_definitions import PathLike
from monai.utils import PostFix, ensure_tuple
from monai.utils.misc import ImageMetaKey
from lightning.pytorch.callbacks import BasePredictionWriter

from manafaln.common.constants import DefaultKeys
from manafaln.utils.misc import get_item, get_items

DEFAULT_UID_KEY = (
    PostFix.meta(DefaultKeys.INPUT_KEY) + "." + ImageMetaKey.FILENAME_OR_OBJ
)
DEFAULT_OUTPUT_KEYS = DefaultKeys.OUTPUT_KEY


class ClassificationPredictionWriter(BasePredictionWriter):
    """A callback for writing classification predictions to a file during training or validation.

    Args:
        output_file (PathLike): The path to the output file.
        uid_key (str, optional): The key to extract the unique identifier from the batch. Defaults to DEFAULT_UID_KEY.
        pred_keys (Union[Sequence[str], str], optional): The keys to extract the predictions from the batch. Defaults to DEFAULT_OUTPUT_KEYS.
        sep (str, optional): The separator used to join the uid_key and pred_keys. Defaults to ".".
        write_interval (Literal["batch", "epoch", "batch_and_epoch"], optional): The interval at which to write the predictions. Defaults to "batch".
    """

    def __init__(
        self,
        output_file: PathLike,
        uid_key: str = DEFAULT_UID_KEY,
        pred_keys: Union[Sequence[str], str] = DEFAULT_OUTPUT_KEYS,
        sep: str = ".",
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ):
        super().__init__(write_interval)
        self.output_file = output_file

        dirname = os.path.dirname(self.output_file)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)

        self.uid_key = uid_key
        self.pred_keys = ensure_tuple(pred_keys)
        self.sep = sep
        self.preds = []

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Callback function called at the end of each prediction batch.

        Args:
            trainer: The PyTorch Lightning trainer object.
            pl_module: The PyTorch Lightning module object.
            outputs: The output predictions from the model.
            batch: The input batch.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the current dataloader.
        """
        uids = get_item(batch, self.uid_key, self.sep)
        if isinstance(outputs, list):
            for uid, output in zip(uids, outputs):
                pred = get_items(output, self.pred_keys, self.sep)
                self.preds.append((uid, *pred))
        else:
            preds = get_items(outputs, self.pred_keys, self.sep)
            self.preds.extend(zip(uids, *preds))


class CSVPredictionWriter(ClassificationPredictionWriter):
    """A callback for writing classification predictions to a CSV file during training or validation.

    Args:
        output_file (PathLike): The path to the output file.
        uid_key (str, optional): The key to extract the unique identifier from the batch. Defaults to DEFAULT_UID_KEY.
        pred_keys (Union[Sequence[str], str], optional): The keys to extract the predictions from the batch. Defaults to DEFAULT_OUTPUT_KEYS.
        sep (str, optional): The separator used to join the uid_key and pred_keys. Defaults to ".".
    """

    def __init__(
        self,
        output_file: PathLike,
        uid_key: str = DEFAULT_UID_KEY,
        pred_keys: Union[Sequence[str], str] = DEFAULT_OUTPUT_KEYS,
        sep: str = ".",
    ):
        super().__init__(output_file, uid_key, pred_keys, sep, write_interval="epoch")

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ) -> None:
        """Callback function called at the end of each epoch to write the predictions to a CSV file.

        Args:
            trainer: The PyTorch Lightning trainer object.
            pl_module: The PyTorch Lightning module object.
            predictions: The predictions made during the epoch.
            batch_indices: The indices of the batches used in the epoch.
        """
        df = pd.DataFrame(self.preds, columns=[self.uid_key, *self.pred_keys])
        df = df.set_index(self.uid_key)
        df.to_csv(self.output_file)
