import torch
import monai

from manafaln.workflow import SupervisedLearning

class SupervisedSegmentation(SupervisedLearning):
    def __init__(self, config: dict):
        super().__init__(config)

        settings = self.hparams.workflow["settings"]
        self.use_crf = settings.get("use_crf", False)
        if self.use_crf:
            crf_args = settings.get("crf_args", {})
            self.crf_module = monai.networks.blocks.CRF(**crf_args)

    def forward(self, data):
        if self.use_crf:
            out = self.inferer(data, self.model)
            out = self.crf_module(out, data)
        else:
            out = self.inferer(data, self.model)
        return out
