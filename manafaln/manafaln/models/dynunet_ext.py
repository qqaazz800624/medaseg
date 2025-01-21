from typing import Tuple, Optional, Sequence, Union

import torch
from monai.networks.nets import DynUNet
from torch.nn.functional import interpolate


class DynUNetExtended(DynUNet):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
        output_features: bool = False
    ):
        super(DynUNetExtended, self).__init__(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            strides,
            upsample_kernel_size,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=res_block,
            trans_bias=trans_bias
        )
        self.output_features = output_features

    def encode(self, x):
        out = self.input_block(x)
        skips = [out]
        for block in self.downsamples:
            out = block(out)
            skips.append(out)
        out = self.bottleneck(out)
        return out, skips

    def decode(self, x, skips):
        temp = []
        skips = skips[::-1]
        for i, block in enumerate(self.upsamples):
            x = block(x, skips[i])
            temp.append(x)
        out = self.output_block(x)

        # Construct deep supervision output
        if self.training and self.deep_supervision:
            temp = temp[::-1]
            # Compute deep supervision heads
            for i in range(self.deep_supr_num):
                self.heads[i] = self.deep_supervision_heads[i](temp[i+1])
            # Combine deep supervision outputs
            out_all = torch.zeros(out.shape[0], len(self.heads) + 1, *out.shape[1:], device=out.device, dtype=out.dtype)
            out_all[:, 0] = out
            for idx, feature_map in enumerate(self.heads):
                out_all[:, idx + 1] = interpolate(feature_map, out.shape[2:])
        return out_all

        return out

    def forward(self, x):
        if not self.output_features:
            return super(DynUNetExtended, self).forward(x)
        else:
            v, skips = self.encode(x)
            out = self.decode(v, skips)
        return out, v

