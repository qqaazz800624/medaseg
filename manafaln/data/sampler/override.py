from torch.utils.data import SubsetRandomSampler as _SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler as _WeightedRandomSampler


class SubsetRandomSampler(_SubsetRandomSampler):
    """
    Override the default SubsetRandomSampler to ignore `data_source` argument used in SamplerBuilder.
    """

    def __init__(self, data_source, indices):
        super().__init__(indices)


class WeightedRandomSampler(_WeightedRandomSampler):
    """
    A custom implementation of WeightedRandomSampler that ignores the `data_source` argument used in SamplerBuilder.

    Args:
        weights (list): A list of weights for each sample.
        num_samples (int): The number of samples to draw.
        replacement (bool, optional): Whether to draw samples with replacement. Default is True.

    """
    def __init__(self, data_source, weights, num_samples, replacement=True):
        """
        Initializes a new instance of WeightedRandomSampler.

        Args:
            weights (list): A list of weights for each sample.
            num_samples (int): The number of samples to draw.
            replacement (bool, optional): Whether to draw samples with replacement. Default is True.

        """
        super().__init__(weights, num_samples, replacement)
