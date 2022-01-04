import torch
from torch.utils.data import DataLoader

class RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

class MultiEpochDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_sampler = RepeatSampler(self.batch_sampler)
        self.iterator = super().__init__()

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
