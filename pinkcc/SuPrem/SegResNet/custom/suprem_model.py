from copy import deepcopy
from typing import Literal, Optional, Sequence
from pathlib import Path
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNet, SwinUNETR


logger = getLogger(__name__)

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"expected 5D input (got {x.dim()}D input)"
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["ReLU", "LeakyReLU", "ELU"]
    ) -> None:
        super().__init__()

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "ELU":
            self.activation = nn.ELU()
        else:
            raise ValueError("Invalid activation function.")

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn1(self.conv1(x)))

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        activation: Literal["ReLU", "LeakyReLU", "ELU"]
    ) -> None:
        super().__init__()

        self.ops = nn.Sequential(
            ConvBlock(
                in_channels,
                32 * (2 ** depth),
                activation
            ),
            ConvBlock(
                32 * (2 ** depth),
                32 * (2 ** (depth + 1)),
                activation
            )
        )
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.ops(x)
        if self.current_depth == 3:
            out = skip
        else:
            out = self.maxpool(skip)
        return out, skip

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int,
        depth: int,
        activation: Literal["ReLU", "LeakyReLU", "ELU"]
    ) -> None:
        super().__init__()

        self.up_conv = nn.ConvTranspose3d(
            in_channels,
            features,
            kernel_size=2,
            stride=2
        )
        self.ops = nn.Sequential(
            ConvBlock(
                in_channels + features // 2,
                32 * (2 ** (depth + 1)),
                activation
            ),
            ConvBlock(
                32 * (2 ** (depth + 1)),
                32 * (2 ** (depth + 1)),
                activation
            )
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.ops(torch.cat([self.up_conv(x), skip], 1))

class OutBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.final_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_conv(x)

class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["ReLU", "LeakyReLU", "ELU"]
    ) -> None:
        super().__init__()

        # Encoder blocks
        self.down_tr64  = DownBlock(in_channels, 0, activation)
        self.down_tr128 = DownBlock(64,  1, activation)
        self.down_tr256 = DownBlock(128, 2, activation)
        self.down_tr512 = DownBlock(256, 3, activation)

        # Decoder blocks
        self.up_tr256 = UpBlock(512, 512, 2, activation)
        self.up_tr128 = UpBlock(256, 256, 1, activation)
        self.up_tr64  = UpBlock(128, 128, 0, activation)

        # Output block
        self.out_tr = OutBlock(64, out_channels)

    def forward(self, x):
        # Encoder blocks
        skips = []
        for layer in [self.down_tr64, self.down_tr128, self.down_tr256]:
            x, skip = layer(x)
            skips.append(skip)

        # Bottleneck block
        x, _ = self.down_tr512(x)

        # Decoder blocks
        for layer in [self.up_tr256, self.up_tr128, self.up_tr64]:
            x = layer(x, skips.pop())

        return self.out_tr(x)

class SuPremModel(nn.Module):
    def __init__(
        self,
        backbone: Literal["SegResNet", "UNet", "SwinUNETR"],
        img_size: Sequence[int],
        in_channels: int,
        out_channels: int,
        pretrained: Optional[Path] = None
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels

        if backbone == "UNet":
            self.backbone = UNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                activation="ReLU"
            )
        elif backbone == "SwinUNETR":
            self.backbone = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=False
            )
        elif backbone == "SegResNet":
            self.backbone = SegResNet(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_prob=0.0
            )
        else:
            raise ValueError("Invalid backbone.")

        if pretrained is not None:
            self._load_pretrained_weights(pretrained)

        self.backbone = torch.compile(self.backbone)

    def _load_pretrained_weights(self, pretrained: Path):
        ckpt = torch.load(pretrained)

        load_vars = set()
        skip_vars = set()

        state_dict = self.backbone.state_dict()
        for k, v in ckpt["net"].items():
            k = k[7:] if k.startswith("module.") else k
            k = k[9:] if k.startswith("backbone.") else k
            if k in state_dict:
                if v.shape == state_dict[k].shape:
                    load_vars.add(k)
                    state_dict[k] = v
                else:
                    skip_vars.add(k)
            else:
                skip_vars.add(k)

        self.backbone.load_state_dict(state_dict)

        logger.info(f"Loaded {len(load_vars)} parameters from checkpoint.")
        logger.info(f"Skipped parameters: {skip_vars}")

    def export(
        self,
        mode: Literal["torchscript", "onnx"],
        path: str
    ) -> None:
        sample_input = torch.randn(
            (1, self.in_channels, *self.img_size),
            dtype=torch.float32,
            device="cpu"
        )
        model = deepcopy(self.backbone).cpu().eval()

        if mode == "torchscript":
            with torch.inference_mode():
                traced = torch.jit.trace(model, sample_input)
            traced.save(path)
        elif mode == "onnx":
            import torch.onnx as onnx
            onnx.export(
                model,
                (sample_input,),
                path,
                export_params=True,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "output": {0: "batch"}
                }
            )
        else:
            raise ValueError("Invalid export mode.")

    def forward(self, x):
        return self.backbone(x)
