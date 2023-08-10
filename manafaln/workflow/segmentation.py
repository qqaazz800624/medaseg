import torch
import monai

from manafaln.workflow import SupervisedLearning

class SupervisedSegmentation(SupervisedLearning):
    """
    A class for performing supervised segmentation using a deep learning model.

    Args:
        config (dict): A dictionary containing configuration settings for the workflow.

    Attributes:
        use_crf (bool): A flag indicating whether to use Conditional Random Fields (CRF) for post-processing.
        crf_module (monai.networks.blocks.CRF): An instance of the CRF module.

    """

    def __init__(self, config: dict):
        """
        Initializes a new instance of the SupervisedSegmentation class.

        Args:
            config (dict): A dictionary containing configuration settings for the workflow.

        """
        super().__init__(config)

        settings = self.hparams.workflow["settings"]
        self.use_crf = settings.get("use_crf", False)
        if self.use_crf:
            crf_args = settings.get("crf_args", {})
            self.crf_module = monai.networks.blocks.CRF(**crf_args)

    def forward(self, data):
        """
        Performs the forward pass of the supervised segmentation workflow.

        Args:
            data: The input data to be processed.

        Returns:
            The output of the segmentation workflow.

        """
        if self.use_crf:
            out = self.inferer(data, self.model)
            out = self.crf_module(out, data)
        else:
            out = self.inferer(data, self.model)
        return out
