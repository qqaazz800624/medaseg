import os
from typing import Dict, Optional

import numpy as np
from monai.config.type_definitions import KeysCollection, PathLike
from monai.data.csv_saver import CSVSaver
from monai.transforms import MapTransform
from monai.utils import ImageMetaKey as Key
from monai.utils import PostFix, ensure_tuple_rep

DEFAULT_POST_FIX = PostFix.meta()


class RLECSVSaver(CSVSaver):
    """
    Overwrite monai.data.csv_saver.CSVSaver to
    1. Use the basename of filename as data index, e.g. path/to/data/liver_xx_xx.nii.gz -> liver_xx_xx.nii.gz
    2. Save the data as string, instead converting to np.ndarray
    3. Write string data to csv, instead of assumimg data to be np.ndarray
    """

    def save(self, data: str, meta_data: Optional[Dict] = None) -> None:
        """
        This overwrites `CSVSaver.save` to
        1. Use the basename of filename as data index, e.g. path/to/data/liver_xx_xx.nii.gz -> liver_xx_xx.nii.gz
        2. Save the data as string, instead converting to np.ndarray

        Save data into the cache dictionary. The metadata should have the following key:
            - ``'filename_or_obj'`` -- save the data corresponding to file name or object.
        If meta_data is None, use the default index from 0 to save data instead.

        Args:
            data: target data content that save into cache.
            meta_data: the metadata information corresponding to the data.
        """
        save_key = (
            # Use the basename of filename as data index
            # e.g. path/to/data/liver_xx_xx.nii.gz -> liver_xx_xx.nii.gz
            os.path.basename(meta_data[Key.FILENAME_OR_OBJ])
            if meta_data
            else str(self._data_index)
        )
        self._data_index += 1

        # Save the data as string, instead converting to np.ndarray
        self._cache_dict[save_key] = data

    def finalize(self) -> None:
        """
        This overwrites `CSVSaver.finalize` to
        1. Write string data to csv, instead of assumimg data to be np.ndarray

        Writes the cached dict to a csv
        """
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "a") as f:
            for k, v in self._cache_dict.items():
                f.write(k)
                # Write string data to csv, instead of assumimg data to be np.ndarray
                f.write(self.delimiter + str(v))
                f.write("\n")
        # clear cache content after writing
        self.reset_cache()


class SaveRunLengthEncodingd(MapTransform):
    """
    A map transform that saves the run-length encoding (RLE) of a binary mask in a CSV file.

    Args:
        keys (KeysCollection): Keys of the data to be encoded and saved, this transform only supports 1 key.
        meta_keys (Optional[KeysCollection]): explicitly indicate the key of the corresponding metadata dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            will extract the filename of input image to save RLE results.
        meta_key_postfix (str): `key_{postfix}` was used to store the metadata in `LoadImaged`.
            so need the key to extract the metadata of input image, like filename, etc. default is `meta_dict`.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            this arg only works when `meta_keys=None`. if no corresponding metadata, set to `None`.
        id_column (str): Name of the ID column in the CSV file. Defaults to "Id".
        data_column (str): Name of the data column in the CSV file. Defaults to "Predicted".
        output_dir (PathLike): The directory to save the CSV file. Defaults to "./".
        filename (str): Name of the CSV file. Defaults to "predictions.csv".
        delimiter (str): The delimiter character in the saved file, as the default output type is `csv`.
            to be consistent with: https://docs.python.org/3/library/csv.html#csv.Dialect.delimiter. Defaults to ",".
        flush (bool): Indicate whether to write the cache data to CSV file immediately in this transform and
            clear the cache. If True, flushes the CSV writer after each write operation. Defaults to True.
        allow_missing_keys (bool): If True, don't raise exception if key is missing. Defaults to False.

    Raises:
        ValueError: If more than 1 key is provided.
    """

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

    @staticmethod
    def encode_run_length(mask: np.ndarray) -> str:
        """
        Encodes a binary mask as a run-length encoding (RLE) string.

        Args:
            mask (np.ndarray): A binary mask with shape (1, W, H) or (W, H).

        Returns:
            str: A string representing the RLE of the mask.
            Each pair of integers indicates the start index and length of a run of 1s in the mask.
            The string format is "s1 l1 s2 l2 s3 l3 ...", where s1, s2, s3, etc. are the starting
            indices of each run, and l1, l2, l3, etc. are the lengths of each run.

        Example:
        >>> mask = np.array([
                [1, 0, 0, 1],
                [1, 1, 1, 0],
                [0, 1, 0, 0]
            ])
        >>> print(SaveRunLengthEncodingd.encode_run_length(mask))
        "0 1 3 4 9 1"
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


if __name__ == "__main__":
    mask = np.array([[1, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0]])
    print(SaveRunLengthEncodingd.encode_run_length(mask))
