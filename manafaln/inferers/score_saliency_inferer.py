from typing import Any, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers.inferer import SaliencyInferer
from monai.visualize import CAM, GradCAM, GradCAMpp


class ScoreCAM(CAM):
    """
    Overwrites `monai.visualize.CAM` to return logits score along with CAM
    """

    def compute_map(self, x, class_idx=None, layer_idx=-1, **kwargs):
        logits, acti, _ = self.nn_module(x, **kwargs)
        acti = acti[layer_idx]
        if class_idx is None:
            class_idx = logits.max(1)[-1]
        b, c, *spatial = acti.shape
        acti = torch.split(acti.reshape(b, c, -1), 1, dim=2)  # make the spatial dims 1D
        fc_layers = self.nn_module.get_layer(self.fc_layers)
        output = torch.stack([fc_layers(a[..., 0]) for a in acti], dim=2)
        output = torch.stack(
            [output[i, b : b + 1] for i, b in enumerate(class_idx)], dim=0
        )
        return logits, output.reshape(
            b, 1, *spatial
        )  # resume the spatial dims on the selected class

    def __call__(self, x, class_idx=None, layer_idx=-1, **kwargs):
        logits, acti_map = self.compute_map(x, class_idx, layer_idx, **kwargs)
        return logits, self._upsample_and_post_process(acti_map, x)


class ScoreGradCAM(GradCAM):
    """
    Overwrites `monai.visualize.GradCAM` to return logits score along with GradCAM
    """

    def compute_map(
        self, x, class_idx=None, retain_graph=False, layer_idx=-1, **kwargs
    ):
        logits, acti, grad = self.nn_module(
            x, class_idx=class_idx, retain_graph=retain_graph, **kwargs
        )
        acti, grad = acti[layer_idx], grad[layer_idx]
        b, c, *spatial = grad.shape
        weights = grad.view(b, c, -1).mean(2).view(b, c, *[1] * len(spatial))
        acti_map = (weights * acti).sum(1, keepdim=True)
        return logits, F.relu(acti_map)

    def __call__(self, x, class_idx=None, layer_idx=-1, retain_graph=False, **kwargs):
        logits, acti_map = self.compute_map(
            x,
            class_idx=class_idx,
            retain_graph=retain_graph,
            layer_idx=layer_idx,
            **kwargs
        )
        return logits, self._upsample_and_post_process(acti_map, x)


class ScoreGradCAMpp(ScoreGradCAM):
    """
    Overwrites `monai.visualize.GradCAMpp` to return logits score along with GradCAMpp
    """

    def compute_map(
        self, x, class_idx=None, retain_graph=False, layer_idx=-1, **kwargs
    ):
        logits, acti, grad = self.nn_module(
            x, class_idx=class_idx, retain_graph=retain_graph, **kwargs
        )
        acti, grad = acti[layer_idx], grad[layer_idx]
        b, c, *spatial = grad.shape
        alpha_nr = grad.pow(2)
        alpha_dr = alpha_nr.mul(2) + acti.mul(grad.pow(3)).view(b, c, -1).sum(-1).view(
            b, c, *[1] * len(spatial)
        )
        alpha_dr = torch.where(alpha_dr != 0.0, alpha_dr, torch.ones_like(alpha_dr))
        alpha = alpha_nr.div(alpha_dr + 1e-7)
        relu_grad = F.relu(cast(torch.Tensor, self.nn_module.score).exp() * grad)
        weights = (
            (alpha * relu_grad).view(b, c, -1).sum(-1).view(b, c, *[1] * len(spatial))
        )
        acti_map = (weights * acti).sum(1, keepdim=True)
        return logits, F.relu(acti_map)


class ScoreSaliencyInferer(SaliencyInferer):
    """
    Overwrites `monai.inferers.inferer.SaliencyInferer` to
    1. disable inference mode that is enabled in `pytorch_lightning.Trainer.predict`
        to allow gradient-based CAM to be computed
    2. return logits score along with CAM
    """

    def __call__(self, inputs: torch.Tensor, network: nn.Module, *args: Any, **kwargs: Any):  # type: ignore
        cam: Union[CAM, GradCAM, GradCAMpp]

        if self.cam_name == "cam":
            cam = ScoreCAM(network, self.target_layers, *self.args, **self.kwargs)
            logits, _cam = cam(inputs, self.class_idx, *args, **kwargs)
        else:  # Gradient-based CAM
            with torch.inference_mode(False):
                if self.cam_name == "gradcam":
                    cam = ScoreGradCAM(
                        network, self.target_layers, *self.args, **self.kwargs
                    )
                else:
                    cam = ScoreGradCAMpp(
                        network, self.target_layers, *self.args, **self.kwargs
                    )
                # Need to clone inputs to get a non-InferenceMode tensor
                logits, _cam = cam(inputs.clone(), self.class_idx, *args, **kwargs)

        return logits, _cam
