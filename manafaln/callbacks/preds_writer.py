import os
from typing import Literal, Sequence, Union

import pandas as pd
from monai.utils import PostFix, ensure_tuple
from monai.utils.misc import ImageMetaKey
from pytorch_lightning.callbacks import BasePredictionWriter

from manafaln.common.constants import DefaultKeys
from manafaln.utils.misc import get_item, get_items

DEFAULT_UID_KEY = PostFix.meta(DefaultKeys.INPUT_KEY) + "." + ImageMetaKey.FILENAME_OR_OBJ
DEFAULT_OUTPUT_KEYS = DefaultKeys.OUTPUT_KEY

class ClassificationPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_file: os.PathLike,
        uid_key: str=DEFAULT_UID_KEY,
        pred_keys: Union[Sequence[str], str]=DEFAULT_OUTPUT_KEYS,
        sep: str=".",
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ):
        super().__init__(write_interval)
        self.output_file = output_file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.uid_key = uid_key
        self.pred_keys = ensure_tuple(pred_keys)
        self.sep = sep
        self.preds = []

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        uids = get_item(batch, self.uid_key, self.sep)
        if isinstance(outputs, list):
            for uid, output in zip(uids, outputs):
                pred = get_items(output, self.pred_keys, self.sep)
                self.preds.append((uid, *pred))
        else:
            preds = get_items(outputs, self.pred_keys, self.sep)
            self.preds.extend(zip(uids, *preds))

class CSVPredictionWriter(ClassificationPredictionWriter):
    def __init__(
        self,
        output_file: os.PathLike,
        uid_key: str=DEFAULT_UID_KEY,
        pred_keys: Union[Sequence[str], str]=DEFAULT_OUTPUT_KEYS,
        sep: str=".",
    ):
        super().__init__(output_file, uid_key, pred_keys, sep, write_interval="epoch")

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ) -> None:
        df = pd.DataFrame(self.preds, columns=[self.uid_key, *self.pred_keys])
        df = df.set_index(self.uid_key)
        df.to_csv(self.output_file)
