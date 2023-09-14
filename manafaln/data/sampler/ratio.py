import random
from typing import Any, Dict, List, Sequence, Union

import torch
from monai.utils import ensure_tuple
from torch.utils.data.sampler import Sampler

from manafaln.common.constants import DefaultKeys

DEFAULT_LABEL_KEY = DefaultKeys.LABEL_KEY

class RatioSampler(Sampler):
    """
    Positive samples will be upsampled to the given ratio of negative samples.
    Arguments:
        data_list: list of dicts
        key: key of label
        ratio: ratio of desired positive to negative samples. Default to 1.0.
    """
    def __init__(self,
        data_list: List[Dict[str, Any]],
        key: str = DEFAULT_LABEL_KEY,
        ratio: float = 1.0
    ):
        self.key = key

        # Get labels from data list
        labels = self.get_labels(data_list)
        labels = torch.tensor(labels, dtype=bool)

        # Get indices of positive instances
        self.positives = labels.nonzero(as_tuple=True)[0].tolist()

        # Get indices of negative instances
        self.negatives = (labels == False).nonzero(as_tuple=True)[0].tolist()

        # Number of positive instances to sample
        self.positive_n = int(len(self.negatives) * ratio)

        # Number of times to duplicate positive instances
        self.positive_q = self.positive_n // len(self.positives)

        # Number of remaining positive instances to sample
        self.positive_r = self.positive_n % len(self.positives)

    def __iter__(self):
        """
        Returns an iterator that samples indices to create a balanced dataset.
        """
        # Oversample positive instances
        positive_samples = self.positives * self.positive_q + random.sample(self.positives, self.positive_r)

        # Total sampled dataset
        samples = self.negatives + positive_samples

        # Shuffle
        random.shuffle(samples)

        # Return iter of indices to sample data
        return iter(samples)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.negatives) + self.positive_n

    def get_labels(
        self,
        data_list: List[Dict[str, Any]]
    ) -> List[bool]:
        """
        Extracts labels from the data list and returns a list of boolean values.
        """
        labels = [
            bool(data[self.key])
            for data in data_list
            ]
        return labels
