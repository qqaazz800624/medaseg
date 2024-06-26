from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d, Upsample, InstanceNorm3d
from torch.nn import Sigmoid, Softmax

class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, mode="3d"):
        """
        Convolutional block module.

        Args:
        - inc: int - Number of input channels.
        - outc: int - Number of output channels.
        - kernel_size: int - Size of the kernel.
        - mode: str - Mode for convolution. Can be "2d", "3d", or "p3d".

        Raises:
        - ValueError: If the mode is not one of "2d", "3d", or "p3d".
        """
        super(ConvBlock, self).__init__()

        self.mode = mode.lower()
        if self.mode not in ["2d", "3d", "p3d"]:
            raise ValueError("Unknown mode for convolution")

        pad_size = (kernel_size - 1) // 2

        if self.mode == "2d":
            self.conv1 = Conv3d(
                inc,
                outc,
                kernel_size=(kernel_size, kernel_size, 1),
                stride=1,
                padding=(pad_size, pad_size, 0),
                bias=False
            )
            self.conv2 = None

        if self.mode == "3d":
            self.conv1 = Conv3d(
                inc,
                outc,
                kernel_size=kernel_size,
                stride=1,
                padding=pad_size,
                bias=False
            )
            self.conv2 = None

        if self.mode == "p3d":
            self.conv1 = Conv3d(
                inc,
                outc,
                kernel_size=(kernel_size, kernel_size, 1),
                stride=1,
                padding=(pad_size, pad_size, 0),
                bias=False
            )
            self.conv2 = Conv3d(
                outc,
                outc,
                kernel_size=(1, 1, kernel_size),
                stride=1,
                padding=(0, 0, pad_size),
                bias=False
            )

        self.norm = InstanceNorm3d(outc)

    def forward(self, x):
        """
        Forward pass of the convolutional block.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x = F.relu(x)
        x = self.conv1(x)
        if self.conv2 is not None:
            x = self.conv2(x)
        x = self.norm(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, inc, outc):
        """
        Upsample block module.

        Args:
        - inc: int - Number of input channels.
        - outc: int - Number of output channels.
        """
        super(UpsampleBlock, self).__init__()

        self.up = Upsample(
            scale_factor=2,
            mode="trilinear"
        )
        self.conv = Conv3d(
            inc,
            outc,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm = InstanceNorm3d(outc)

    def forward(self, x):
        """
        Forward pass of the upsample block.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x = self.up(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.norm(x)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, inc, outc):
        """
        Downsample block module.

        Args:
        - inc: int - Number of input channels.
        - outc: int - Number of output channels.
        """
        super(DownsampleBlock, self).__init__()

        self.conv = Conv3d(
            inc,
            outc,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm = InstanceNorm3d(outc)

    def forward(self, x):
        """
        Forward pass of the downsample block.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x = F.relu(x)
        x = self.conv(x)
        x = self.norm(x)
        return x

class Identity(nn.Module):
    def __init__(self, inc, outc, mode="3d"):
        """
        Identity module.

        Args:
        - inc: int - Number of input channels.
        - outc: int - Number of output channels.
        - mode: str - Mode for convolution. Can be "2d", "3d", or "p3d".
        """
        super(Identity, self).__init__()

        self.prep = ConvBlock(inc, outc, kernel_size=1)
        self.conv = ConvBlock(outc, outc, kernel_size=3, mode=mode)
        self.post = ConvBlock(outc, outc, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the identity module.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x = self.prep(x)
        x = self.conv(x)
        x = self.post(x)
        return x

class IdentityDownsample(nn.Module):
    def __init__(self, inc, outc, mode="3d"):
        """
        Identity downsample module.

        Args:
        - inc: int - Number of input channels.
        - outc: int - Number of output channels.
        - mode: str - Mode for convolution. Can be "2d", "3d", or "p3d".
        """
        super(IdentityDownsample, self).__init__()

        self.prep = DownsampleBlock(inc, outc)
        self.conv = ConvBlock(outc, outc, kernel_size=3, mode=mode)
        self.post = ConvBlock(outc, outc, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the identity downsample module.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x = self.prep(x)
        x = self.conv(x)
        x = self.post(x)
        return x

class Merge(nn.Module):
    def __init__(self, xc, prevc, outc, mode="3d"):
        """
        Merge module.

        Args:
        - xc: int - Number of input channels for the current tensor.
        - prevc: int - Number of input channels for the previous tensor.
        - outc: int - Number of output channels.
        - mode: str - Mode for convolution. Can be "2d", "3d", or "p3d".
        """
        super(Merge, self).__init__()

        self.prep0 = ConvBlock(xc,    outc, kernel_size=1)
        self.prep1 = ConvBlock(prevc, outc, kernel_size=1)

        self.conv0 = ConvBlock(outc, outc, kernel_size=3, mode=mode)
        self.conv1 = ConvBlock(outc, outc, kernel_size=3, mode=mode)

        self.post = ConvBlock(outc, outc, kernel_size=1)

    def forward(self, x, prev):
        """
        Forward pass of the merge module.

        Args:
        - x: torch.Tensor - Current tensor.
        - prev: torch.Tensor - Previous tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x0 = self.prep0(x)
        x1 = self.prep1(prev)

        s = self.conv0(x0) + self.conv1(x1)

        s = self.post(s)
        return s

class MergeUpsample(nn.Module):
    def __init__(self, xc, prevc, outc, mode="3d"):
        """
        Merge upsample module.

        Args:
        - xc: int - Number of input channels for the current tensor.
        - prevc: int - Number of input channels for the previous tensor.
        - outc: int - Number of output channels.
        - mode: str - Mode for convolution. Can be "2d", "3d", or "p3d".
        """
        super(MergeUpsample, self).__init__()

        self.prep0 = UpsampleBlock(xc, outc)
        self.prep1 = ConvBlock(prevc, outc, kernel_size=1)

        self.conv0 = ConvBlock(outc, outc, kernel_size=3, mode=mode)
        self.conv1 = ConvBlock(outc, outc, kernel_size=3, mode=mode)

        self.post = ConvBlock(outc, outc, kernel_size=1)

    def forward(self, x, prev):
        """
        Forward pass of the merge upsample module.

        Args:
        - x: torch.Tensor - Current tensor.
        - prev: torch.Tensor - Previous tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x0 = self.prep0(x)
        x1 = self.prep1(prev)

        s = self.conv0(x0) + self.conv1(x1)

        s = self.post(s)
        return s

class FirstStem(nn.Module):
    def __init__(self, inc, outc):
        """
        First stem module.

        Args:
        - inc: int - Number of input channels.
        - outc: int - Number of output channels.
        """
        super(FirstStem, self).__init__()

        filters = outc // 2

        self.conv1 = Conv3d(
            inc,
            filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm1 = InstanceNorm3d(filters)

        self.conv2 = Conv3d(
            filters,
            outc,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm2 = InstanceNorm3d(outc)

    def forward(self, x):
        """
        Forward pass of the first stem module.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        return x

class FinalStem(nn.Module):
    def __init__(self, inc, outc):
        """
        Final stem module.

        Args:
        - inc: int - Number of input channels.
        - outc: int - Number of output channels.
        """
        super(FinalStem, self).__init__()

        filters = outc * 2

        self.conv1 = Conv3d(
            inc,
            filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm1 = InstanceNorm3d(filters)

        self.upsample = Upsample(
            scale_factor=2,
            mode="trilinear"
        )

        self.conv2 = Conv3d(
            filters,
            outc,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm2 = InstanceNorm3d(outc)

    def forward(self, x, skip):
        """
        Forward pass of the final stem module.

        Args:
        - x: torch.Tensor - Input tensor.
        - skip: torch.Tensor - Skip connection tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x = torch.cat((x, skip), dim=1)

        x = F.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)

        x = self.upsample(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        return x

class C2FNAS(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        init_filters=32,
        final_activation="softmax"
    ):
        """
        C2FNAS model.

        Args:
        - in_channels: int - Number of input channels.
        - num_classes: int - Number of output classes.
        - init_filters: int - Number of initial filters.
        - final_activation: str - Final activation function. Can be "sigmoid" or "softmax".

        Raises:
        - ValueError: If the final_activation is "sigmoid" and num_classes is not 1.
        """
        super(C2FNAS, self).__init__()

        self.in_channels      = in_channels
        self.num_classes      = num_classes
        self.init_filters     = init_filters
        self.final_activation = final_activation

        if final_activation == "sigmoid" and num_classes != 1:
            raise ValueError("Output classes must be 1 when using sigmoid")

        filters = init_filters

        self.first_stem = FirstStem(in_channels, filters * 2)

        self.encode_layers = nn.ModuleList([
            Identity(filters * 2, filters * 2, mode="2d"),
            Merge(filters * 2, filters * 2, filters * 2, mode="p3d"),
            IdentityDownsample(filters * 2, filters * 4, mode="3d"),
            Identity(filters * 4, filters * 4,  mode="3d"),
            IdentityDownsample(filters * 4, filters * 8,  mode="3d"),
            IdentityDownsample(filters * 8, filters * 16,  mode="3d")
        ])

        self.decode_layers = nn.ModuleList([
             MergeUpsample(filters * 16, filters * 8, filters * 8, mode="2d"),
             Identity(filters * 8, filters * 8, mode="3d"),
             MergeUpsample(filters * 8, filters * 4, filters * 4, mode="p3d"),
             MergeUpsample(filters * 4, filters * 2, filters * 2, mode="3d"),
             Identity(filters * 2, filters * 2, mode="2d"),
             Merge(filters * 2, filters * 2, filters * 2, mode="3d")
        ])

        self.final_stem = FinalStem(filters * 4, filters)

        if final_activation.lower() == "sigmoid":
            self.final_conv = Conv3d(
                filters,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.output = Sigmoid()
        elif final_activation.lower() == "softmax":
            self.final_conv = Conv3d(
                filters,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.output = Softmax(dim=1)
        else:
            self.final_conv = Conv3d(
                filters,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.output = None

    def encoder(self, x):
        """
        Encoder part of the model.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Encoded tensor.
        - List[torch.Tensor] - List of skip connection tensors.
        """
        x = self.first_stem(x)

        skips = []

        stem = x
        x = self.encode_layers[0](x)
        x = self.encode_layers[1](x, stem)
        skips.append(x)

        x = self.encode_layers[2](x)
        x = self.encode_layers[3](x)
        skips.append(x)

        x = self.encode_layers[4](x)
        skips.append(x)

        x = self.encode_layers[5](x)

        return x, skips

    def decoder(self, x: torch.Tensor, skips: List[torch.Tensor]):
        """
        Decoder part of the model.

        Args:
        - x: torch.Tensor - Encoded tensor.
        - skips: List[torch.Tensor] - List of skip connection tensors.

        Returns:
        - torch.Tensor - Decoded tensor.
        """
        x = self.decode_layers[0](x, skips[2])

        x = self.decode_layers[1](x)
        x = self.decode_layers[2](x, skips[1])

        x = self.decode_layers[3](x, skips[0])

        prev = x
        x = self.decode_layers[4](x)
        s = self.decode_layers[5](x, prev)

        prev = x
        x = s
        x = self.final_stem(x, prev)

        return x

    def model(self, x):
        """
        Model part of the model.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        x = self.final_conv(x)
        if self.output is not None:
            x = self.output(x)
        return x

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: torch.Tensor - Input tensor.

        Returns:
        - torch.Tensor - Output tensor.
        """
        return self.model(x)
