import os
import json
from monai.data import Dataset, load_decathlon_datalist
from monai.transforms import Compose
from typing import List
from transforms.pinkcc_transforms import get_pinkcc_transforms

class PINKCCDataset(Dataset):
    def __init__(
        self,
        data_root: str = "/neodata/open_dataset/PINKCC",
        datalist_json: str = "/neodata/open_dataset/PINKCC/datalist.json",
        split: str = "training",  # "training", "validation", "test", or "predict"
        folds: List[str] = None,
        roi_size=(224, 224, 48),
        spacing=(1.5, 1.5, 5.0),
        intensity_min=-941.0,
        intensity_max=133.0,
        intensity_mean=38.804985821291375,
        intensity_std=92.42229571222241,
        num_samples=4,
    ):
        self.data_root = data_root
        self.datalist_json = datalist_json
        self.split = split
        self.folds = folds if folds is not None else self._default_folds(split)

        # Get MONAI-style transform
        self.transforms = Compose(get_pinkcc_transforms(
            stage=split,
            spacing=spacing,
            roi_size=roi_size,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            intensity_mean=intensity_mean,
            intensity_std=intensity_std,
            num_samples=num_samples,
        ))

        # Load raw datalist (list of dicts)
        self.data = self._load_fold_data()

        # Initialize MONAI Dataset base class
        super().__init__(data=self.data, transform=self.transforms)

    def _load_fold_data(self):
        if isinstance(self.folds, str):
            self.folds = [self.folds]

        all_data = []
        for fold_key in self.folds:
            fold_data = load_decathlon_datalist(
                data_list_file_path=self.datalist_json,
                base_dir=self.data_root,
                data_list_key=fold_key,
            )
            all_data.extend(fold_data)
        return all_data

    def _default_folds(self, split):
        return {
            "training": ["fold_0", "fold_1", "fold_2"],
            "validation": ["fold_3"],
            "test": ["fold_4"],
            "predict": ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
        }[split]
