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
    Override the default WeightedRandomSampler to ignore `data_source` argument used in SamplerBuilder.
    """

    def __init__(self, data_source, weights, num_samples, replacement=True):
        super().__init__(weights, num_samples, replacement)
