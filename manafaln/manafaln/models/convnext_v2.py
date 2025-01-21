from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers import trunc_normal_, DropPath

class LayerNormBase(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["channels_first", "channels_last"] = "channels_last"
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.data_format = data_format

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def apply_weight(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                (self.normalized_shape,),
                self.weight,
                self.bias,
                self.eps
            )
        else:
            u = x.mean(dim=1, keepdim=True)
            s = (x - u).pow(2).mean(dim=1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.apply_weight(x)
        return x

class LayerNorm1D(LayerNormBase):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["channels_first", "channels_last"] = "channels_last"
    ):
        super().__init__(normalized_shape, eps, data_format)

        # For 1D channels_first input data
        self.w_shape = (1, self.normalized_shape, 1)

    def apply_weight(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.view(self.w_shape)
        b = self.bias.view(self.w_shape)

        return w * x + b

class LayerNorm2D(LayerNormBase):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["channels_first", "channels_last"] = "channels_last"
    ):
        super().__init__(normalized_shape, eps, data_format)

        # For 2D channels_first input data
        self.w_shape = (1, self.normalized_shape, 1, 1)

    def apply_weight(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.view(self.w_shape)
        b = self.bias.view(self.w_shape)

        return w * x + b

class LayerNorm3D(LayerNormBase):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["channels_first", "channels_last"] = "channels_last"
    ):
        super().__init__(normalized_shape, eps, data_format)

        # For 3D channels_first input data
        self.w_shape = (1, self.normalized_shape, 1, 1, 1)

    def apply_weight(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.view(self.w_shape)
        b = self.bias.view(self.w_shape)

        return w * x + b

class GRN(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        eps: float = 1e-6,
        data_format: Literal["channels_first", "channels_last"] = "channels_last"
    ):
        super().__init__()

        if spatial_dims < 0:
            raise ValueError("spatial_dims must be larger than 0.")

        self.eps = eps
        self.data_format = data_format

        if data_format == "channels_first":
            param_shape = (1, num_channels) + (1,) * spatial_dims
            self.channel_dim = 1
            self.reduce_axes = tuple(range(2, spatial_dims + 2))
        elif data_format == "channels_last":
            param_shape = (1,) * (spatial_dims + 1) + (num_channels,)
            self.channel_dim = -1
            self.reduce_axes = tuple(range(1, spatial_dims + 1))

        self.beta = nn.Parameter(torch.zeros(param_shape))
        self.gamma = nn.Parameter(torch.zeros(param_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=self.reduce_axes, keepdim=True)
        nx = gx / (gx.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        drop_path: float = 0.0
    ):
        super().__init__()

        if spatial_dims == 3:
            Conv = nn.Conv3d
            LayerNorm = LayerNorm3D
            self.to_channels_last = (0, 2, 3, 4, 1)
            self.to_channels_first = (0, 4, 1, 2, 3)
        elif spatial_dims == 2:
            Conv = nn.Conv2d
            LayerNorm = LayerNorm2D
            self.to_channels_last = (0, 2, 3, 1)
            self.to_channels_first = (0, 3, 1, 2)
        elif spatial_dims == 1:
            Conv = nn.Conv1d
            LayerNorm = LayerNorm1D
            self.to_channels_last = (0, 2, 1)
            self.to_channels_first = (0, 2, 1)
        else:
            raise ValueError(
                f"Unsupported ConvNeXt block dim ({spatial_dims})."
            )

        self.dwconv = Conv(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels
        )
        self.norm = LayerNorm(in_channels, data_format="channels_last")
        self.pwconv1 = nn.Linear(in_channels, in_channels * 4)
        self.act = nn.GELU()
        self.grn = GRN(spatial_dims, in_channels * 4)
        self.pwconv2 = nn.Linear(in_channels * 4, in_channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.dwconv(x)
        s = torch.permute(s, self.to_channels_last)
        s = self.norm(s)
        s = self.pwconv1(s)
        s = self.act(s)
        s = self.grn(s)
        s = self.pwconv2(s)
        s = torch.permute(s, self.to_channels_first)
        return x + self.drop_path(s)

class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_classes: int,
        depths: Sequence[int],
        dims: Sequence[int],
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        use_grad_checkpoint: bool = False
    ):
        super().__init__()

        if spatial_dims == 3:
            Conv = nn.Conv3d
            LayerNorm = LayerNorm3D
        elif spatial_dims == 2:
            Conv = nn.Conv2d
            LayerNorm = LayerNorm2D
        elif spatial_dims == 1:
            Conv = nn.Conv1d
            LayerNorm = LayerNorm1D
        else:
            raise ValueError(
                f"Unsupported ConvNeXt dim ({spatial_dims})."
            )

        self.use_grad_checkpoint = use_grad_checkpoint

        self.depths = depths
        self.down_layers = nn.ModuleList()

        stem = nn.Sequential(
            Conv(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.down_layers.append(stem)
        for i in range(len(self.depths) - 1):
            down_layer = nn.Sequential(
                LayerNorm(dims[i], data_format="channels_first"),
                Conv(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.down_layers.append(down_layer)

        self.stages = nn.ModuleList()
        cur = 0
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for dim, depth in zip(dims, depths):
            stage = [
                ConvNeXtBlock(spatial_dims, dim, drop_path=dp_rates[cur + j])
                for j in range(depth)
            ]
            stage = nn.Sequential(*stage)
            self.stages.append(stage)
            cur += depth

        self.norm = LayerNorm(dims[-1], data_format="channels_last")
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for down, stage in zip(self.down_layers, self.stages):
            if self.use_grad_checkpoint and self.training:
                x = checkpoint(down, x, use_reentrant=False)
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = down(x)
                x = stage(x)
        axes = list(range(2, x.dim()))
        return self.norm(x.mean(dim=axes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnextv2_atto(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    **kwargs
):
    model = ConvNeXtV2(
        spatial_dims,
        in_channels,
        num_classes,
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        **kwargs
    )
    return model

def convnextv2_fermo(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    **kwargs
):
    model = ConvNeXtV2(
        spatial_dims,
        in_channels,
        num_classes,
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        **kwargs
    )
    return model

def convnextv2_pico(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    **kwargs
):
    model = ConvNeXtV2(
        spatial_dims,
        in_channels,
        num_classes,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        **kwargs
    )
    return model

def convnextv2_nano(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    **kwargs
):
    model = ConvNeXtV2(
        spatial_dims,
        in_channels,
        num_classes,
        depths=[2, 2, 8, 2],
        dims=[80, 160, 320, 640],
        **kwargs
    )
    return model

def convnextv2_tiny(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    **kwargs
):
    model = ConvNeXtV2(
        spatial_dims,
        in_channels,
        num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )
    return model

def convnextv2_base(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    **kwargs
):
    model = ConvNeXtV2(
        spatial_dims,
        in_channels,
        num_classes,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        **kwargs
    )
    return model

def convnextv2_large(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    **kwargs
):
    model = ConvNeXtV2(
        spatial_dims,
        in_channels,
        num_classes,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        **kwargs
    )
    return model

def convnextv2_huge(
    spatial_dims: int,
    in_channels: int,
    num_classes: int,
    **kwargs
):
    model = ConvNeXtV2(
        spatial_dims,
        in_channels,
        num_classes,
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        **kwargs
    )
    return model

