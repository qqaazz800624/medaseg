import os
from typing import Optional

import numpy as np
from monai.config.type_definitions import KeysCollection, PathLike
from monai.data.csv_saver import CSVSaver
from monai.transforms import MapTransform
from monai.utils import ImageMetaKey as Key
from monai.utils import PostFix, ensure_tuple_rep

DEFAULT_POST_FIX = PostFix.meta()


class RLECSVSaver(CSVSaver):
    def save(self, data, meta_data=None):
        save_key = (
            os.path.basename(meta_data[Key.FILENAME_OR_OBJ])
            if meta_data
            else str(self._data_index)
        )
        self._data_index += 1
        self._cache_dict[save_key] = data

    def finalize(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "a") as f:
            for k, v in self._cache_dict.items():
                f.write(k)
                f.write(self.delimiter + str(v))
                f.write("\n")
        # clear cache content after writing
        self.reset_cache()


class SaveRunLengthEncodingd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        id_column: str = "Id",
        data_column: str = "Predicted",
        output_dir: PathLike = "./",
        filename: str = "predictions.csv",
        delimiter: str = ",",
        flush: bool = True,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

        if len(self.keys) != 1:
            raise ValueError("only 1 key is allowed when saving the RLE.")

        self.saver = RLECSVSaver(
            output_dir=output_dir,
            filename=filename,
            flush=flush,
            delimiter=delimiter,
        )
        self.flush = flush
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

        # Write the header
        self.saver._cache_dict[id_column] = data_column
        self.saver.finalize()

    def encode_run_length(self, mask):
        """Encodes a binary mask as a run-length encoding (RLE) list.

        Args:
            mask (np.ndarray): A binary mask. Shape: (1, W, H) or (W, H)

        Returns:
            str: A str representing the run-length encoding of the mask.
                Each pair of integer indicate the start index and length of a run of 1s in the mask.
        """

        # Flatten the mask and append a 0 to the end to handle the case where the last pixel is 1.
        flat_mask = np.concatenate([[0], mask.flatten(), [0]])

        # Calculate the indices where the mask transitions from 0 to 1 or 1 to 0.
        transitions = np.nonzero(flat_mask[:-1] != flat_mask[1:])[0]

        # Pair the transition indices together to get the start and end of each run.
        run_starts = transitions[::2]
        run_lengths = transitions[1::2] - transitions[::2]

        # Concat the result into a string with format "s1 l1 s2 l2 s3 l3 ..."
        res = ""
        for run_start, run_length in zip(run_starts, run_lengths):
            res += str(run_start) + " " + str(run_length) + " "
        res = res[:-1]

        return res

    def __call__(self, data):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.meta_keys, self.meta_key_postfix
        ):
            if meta_key is None and meta_key_postfix is not None:
                meta_key = f"{key}_{meta_key_postfix}"
            meta_data = d[meta_key] if meta_key is not None else None
            rle = self.encode_run_length(d[key])
            self.saver.save(data=rle, meta_data=meta_data)
            if self.flush:
                self.saver.finalize()
        return d
