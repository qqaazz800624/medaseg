import abc
import itertools
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.layers.factories import Conv
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.networks.nets.swin_unetr import PatchEmbed
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinTransformer"
]

def window_partition_3d(x: Tensor, window_size: Sequence[int]) -> Tensor:
    b, d, h, w, c = x.shape
    x = x.view(
        b,
        d // window_size[0], window_size[0],
        h // window_size[1], window_size[1],
        w // window_size[2], window_size[2],
        c
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size[0] * window_size[1] * window_size[2], c)
    return windows

def window_partition_2d(x: Tensor, window_size: Sequence[int]) -> Tensor:
    b, h, w, c = x.shape
    x = x.view(
        b,
        h // window_size[0], window_size[0],
        w // window_size[1], window_size[1],
        c
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size[0] * window_size[1], c)
    return windows

def window_partition_1d(x: Tensor, window_size: Sequence[int]) -> Tensor:
    b, h, c = x.shape
    x = x.view(b, h // window_size[0], window_size[0], c)
    windows = x.permute(0, 1, 2, 3).contiguous()
    windows = windows.view(-1, window_size[0], c)
    return windows

def window_reverse_3d(
    windows: Tensor,
    window_size: Sequence[int],
    dims: Sequence[int]
) -> Tensor:
    b, d, h, w = dims
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(b, d, h, w, -1)
    return x

def window_reverse_2d(
    windows: Tensor,
    window_size: Sequence[int],
    dims: Sequence[int]
) -> Tensor:
    b, h, w = dims
    x = windows.view(
        b,
        h // window_size[0],
        w // window_size[1],
        window_size[0],
        window_size[1],
        -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, h, w, -1)
    return x

def window_reverse_1d(
    windows: Tensor,
    window_size: Sequence[int],
    dims: Sequence[int]
) -> Tensor:
    b, h = dims
    x = windows.view(
        b,
        h // window_size[0],
        window_size[0],
        -1
    )
    x = x.permute(0, 1, 2, 3).contiguous()
    x = x.view(b, h, -1)
    return x

def get_window_size(
    x_size: Sequence[int],
    window_size: Sequence[int],
    shift_size: Optional[Sequence[int]] = None
):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        elif len(self.window_size) == 1:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(2 * window_size[0] - 1, num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module, abc.ABC):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    @abc.abstractmethod
    def forward_part1(self, x, mask_matrix):
        raise NotImplementedError

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class SwinTransformerBlock3D(SwinTransformerBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward_part1(self, x: Tensor, mask_matrix: Tensor):
        x_shape = x.size()
        x = self.norm1(x)

        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition_3d(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse_3d(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()
        return x

class SwinTransformerBlock2D(SwinTransformerBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)

        b, h, w, c = x.shape
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        pad_l = pad_t = 0
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape
        dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition_2d(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse_2d(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        return x

class SwinTransformerBlock1D(SwinTransformerBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)

        b, h, c = x.shape
        window_size, shift_size = get_window_size((h,), self.window_size, self.shift_size)
        pad_l = 0
        pad_r = (window_size[0] - h % window_size[0]) % window_size[0]
        x = F.pad(x, (0, 0, pad_l, pad_r))
        _, hp, _ = x.shape
        dims = [b, hp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0]), dims=(1,))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition_1d(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse_1d(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0]), dims=(1,))
        else:
            x = shifted_x

        if pad_r > 0:
            x = x[:, :h, :].contiguous()
        return x

class PatchMerging3D(nn.Module):
    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.size()
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x = torch.cat(
            [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchMerging2D(nn.Module):
    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.size()
        pad_input = (h % 2 == 1) or (w % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchMerging1D(nn.Module):
    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: Tensor) -> Tensor:
        b, h, c = x.size()
        pad_input = h % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, h % 2))
        x = torch.cat([x[:, i::2, :] for i in range(2)], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer3D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def compute_mask(self, dims, window_size, shift_size, device):
        cnt = 0

        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition_3d(img_mask, window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        b, c, d, h, w = x.size()
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = self.compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x

class BasicLayer2D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock2D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def compute_mask(self, dims, window_size, shift_size, device):
        cnt = 0

        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition_2d(img_mask, window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c h w -> b h w c")
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        wp = int(np.ceil(w / window_size[1])) * window_size[1]
        attn_mask = self.compute_mask([hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class BasicLayer1D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock1D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def compute_mask(self, dims, window_size, shift_size, device):
        cnt = 0

        h = dims[0]
        img_mask = torch.zeros((1, h, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            img_mask[:, h, :] = cnt
            cnt += 1

        mask_windows = window_partition_1d(img_mask, window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        b, c, h = x.size()
        window_size, shift_size = get_window_size((h,), self.window_size, self.shift_size)
        x = rearrange(x, "b c h -> b h c")
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        attn_mask = self.compute_mask([hp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, h, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b h c -> b c h")
        return x

class OutputProjection3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, normalize: bool = False):
        if normalize:
            n, c, d, h, w = x.size()
            x = rearrange(x, "n c d h w -> n d h w c")
            x = F.layer_norm(x, [c])
            x = rearrange(x, "n d h w c -> n c d h w")
        return x

class OutputProjection2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, normalize: bool = False):
        if normalize:
            n, c, h, w = x.size()
            x = rearrange(x, "n c h w -> n h w c")
            x = F.layer_norm(x, [c])
            x = rearrange(x, "n h w c -> n c h w")
        return x

class OutputProjection1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, normalize: bool = False):
        if normalize:
            n, c, h = x.size()
            x = rearrange(x, "n c h -> n h c")
            x = F.layer_norm(x, [c])
            x = rearrange(x, "n h c -> n c h")
        return x

class SwinTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_classes: int = 1000,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        use_v2=False,
    ) -> None:
        super().__init__()

        if spatial_dims == 3:
            PatchEmbed = PatchEmbed3D
            BasicLayer = BasicLayer3D
            PatchMerging = PatchMerging3D
            Pooling = nn.AdaptiveAvgPool3d
            Projection = OutputProjection3D
        elif spatial_dims == 2:
            PatchEmbed = PatchEmbed2D
            BasicLayer = BasicLayer2D
            PatchMerging = PatchMerging2D
            Pooling = nn.AdaptiveAvgPool2d
            Projection = OutputProjection2D
        elif spatial_dims == 1:
            PatchEmbed = PatchEmbed1D
            BasicLayer = BasicLayer1D
            PatchMerging = PatchMerging1D
            Pooling = nn.AdaptiveAvgPool1d
            Projection = OutputProjection1D
        else:
            raise ValueError(
                f"spatial_dims must be 0, 1 or 2, got {spatial_dims}"
            )

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None  # type: ignore
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.use_v2 = use_v2
        layers = []
        for i_layer in range(self.num_layers):
            if self.use_v2:
                layer_c = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                layers.append(layer_c)

            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer != self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            layers.append(layer)

        # Combine all layers into one module
        self.layers = nn.Sequential(*layers)

        self.proj_out = Projection()
        self.pooling = Pooling(1)
        self.flatten = nn.Flatten(1)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.fc = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x, normalize=True):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.proj_out(x, normalize=normalize)
        x = self.pooling(x)
        x = self.flatten(x)
        logits = self.fc(x)

        return logits

class PatchEmbed3D(PatchEmbed):
    def __init__(
        self,
        patch_size: Union[int, Sequence[int]] = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: type[LayerNorm] = nn.LayerNorm
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            spatial_dims=3
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, d, h, w = x.size()
        if w % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if d % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
            x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
        return x

class PatchEmbed2D(PatchEmbed):
    def __init__(
        self,
        patch_size: Union[int, Sequence[int]] = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: type[LayerNorm] = nn.LayerNorm
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            spatial_dims=2
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.size()
        if w % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
        if h % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            wh, ww = x_shape[2], x_shape[3]
            x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x

class PatchEmbed1D(PatchEmbed):
    def __init__(
        self,
        patch_size: Union[int, Sequence[int]] = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: type[LayerNorm] = nn.LayerNorm
    ) -> None:
        super().__init__()

        patch_size = ensure_tuple_rep(patch_size, 1)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        _, _, h = x.size()
        if h % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            wh = x_shape[2]
            x = x.transpose(1, 2).view(-1, self.embed_dim, wh)
        return x

