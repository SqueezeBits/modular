# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import numpy as np
from max.experimental import functional as F
from max.driver import CPU, Accelerator, Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import Weights
from max.nn.module_v3 import Conv2d, Conv3d, Module, ModuleList
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces import CompileWrapper
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.experimental.tensor import Tensor

from .model_config import AutoencoderKLWanConfig

logger = logging.getLogger(__name__)

CACHE_T = 2
WAN_DECODER_CACHE_SLOTS = 32


def _zero_cache_for(x: Tensor) -> Tensor:
    """Create a zero cache tensor shaped for a causal conv input."""
    return Tensor.zeros(
        [x.shape[0], x.shape[1], CACHE_T, x.shape[3], x.shape[4]],
        dtype=x.dtype,
        device=x.device,
    )


def _normalize_compiled_outputs(
    outputs: Tensor | list[Tensor] | tuple[Tensor, ...],
) -> list[Tensor]:
    if isinstance(outputs, Tensor):
        return [outputs]
    if isinstance(outputs, tuple):
        return list(outputs)
    return outputs


class WanRMSNorm(Module[[Tensor], Tensor]):
    """RMS norm used by Wan VAE blocks."""

    gamma: Tensor

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = False,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        self.channel_first = channel_first

        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        shape = [dim, *broadcastable_dims] if channel_first else [dim]
        self.gamma = Tensor.ones(
            shape,
            dtype=dtype,
            device=device.to_device() if device is not None else None,
        )

    def forward(self, x: Tensor) -> Tensor:
        axis = 1 if self.channel_first else x.rank - 1
        rms = F.mean(x * x, axis=axis)
        inv = F.rsqrt(rms + 1e-12)
        gamma = F.transfer_to(self.gamma, x.device)
        return x * inv * gamma


class WanCausalConv3d(Conv3d):
    """3D causal convolution for Wan VAE.

    Temporal causality is implemented via asymmetric padding: the front
    (temporal) dimension is padded on the left only, which the Conv3d
    padding parameter supports directly.

    Uses permute=True so that weights stay in FCQRS (PyTorch) layout and
    Conv3d.forward() can dispatch to cuDNN on NVIDIA GPUs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        has_bias: bool = True,
    ) -> None:
        if isinstance(padding, int):
            pad_t = pad_h = pad_w = padding
        else:
            pad_t, pad_h, pad_w = padding

        # Causal: pad only the front of the temporal axis (left=2*pad_t, right=0).
        super().__init__(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=stride,
            padding=(2 * pad_t, 0, pad_h, pad_h, pad_w, pad_w),
            dilation=1,
            num_groups=1,
            device=device,
            has_bias=has_bias,
            permute=True,
        )


class WanCausalConv3dCached(Conv3d):
    """3D causal convolution with explicit cache tensor I/O.

    Uses Conv3d's internal padding for spatial dimensions and handles
    temporal causal padding separately via concat/pad before calling
    super().forward().
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        has_bias: bool = True,
    ) -> None:
        if isinstance(padding, int):
            pad_t = pad_h = pad_w = padding
        else:
            pad_t, pad_h, pad_w = padding

        # Temporal causal padding: left=2*pad_t, right=0
        self._temporal_pad_left = 2 * pad_t

        # Let Conv3d handle spatial padding via F.conv3d internally.
        # Temporal padding = 0 here; we handle it ourselves.
        super().__init__(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=stride,
            padding=(0, 0, pad_h, pad_h, pad_w, pad_w),
            dilation=1,
            num_groups=1,
            device=device,
            has_bias=has_bias,
            permute=True,
        )

    def _apply_temporal_pad(self, x: Tensor, pad_left: int) -> Tensor:
        """Zero-pad the temporal dimension (axis=2) on the left only."""
        if pad_left <= 0:
            return x
        # F.pad expects 2*rank values: [d0_before, d0_after, d1_before, d1_after, ...]
        # For 5D [B, C, T, H, W]: pad only dim 2 (T) on the left.
        pad_vals = [0, 0, 0, 0, pad_left, 0, 0, 0, 0, 0]
        return F.pad(x, pad_vals)

    def forward(self, x: Tensor) -> Tensor:
        x = self._apply_temporal_pad(x, self._temporal_pad_left)
        return super().forward(x)

    def forward_cached(
        self, x: Tensor, cache_in: Tensor
    ) -> tuple[Tensor, Tensor]:
        x = F.concat([cache_in, x], axis=2)
        cache_out = x[:, :, -CACHE_T:, :, :]
        # Reduce temporal padding by the amount of context from cache.
        effective_pad = max(self._temporal_pad_left - int(cache_in.shape[2]), 0)
        x = self._apply_temporal_pad(x, effective_pad)
        return super().forward(x), cache_out

class WanResidualBlock(Module[[Tensor], Tensor]):
    """Residual block used in Wan VAE decoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        self.norm1 = WanRMSNorm(
            in_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv1 = WanCausalConv3d(
            in_dim,
            out_dim,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.norm2 = WanRMSNorm(
            out_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv2 = WanCausalConv3d(
            out_dim,
            out_dim,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.conv_shortcut = (
            WanCausalConv3d(
                in_dim,
                out_dim,
                1,
                padding=0,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
            if in_dim != out_dim
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class WanAttentionBlock(Module[[Tensor], Tensor]):
    """Per-frame windowed self-attention used in Wan decoder mid block.

    Uses window attention instead of full (H*W)^2 attention to avoid OOM
    at high resolutions. The spatial dimensions are partitioned into
    non-overlapping windows of size ws×ws, and attention is computed
    independently per window.

    Memory: O(b*t * num_windows * ws^2 * ws^2) instead of O(b*t * (H*W)^2).
    At 720p latent (90×160) with ws=8: ~158MB vs ~2.5GB+ per chunk.
    """

    _WINDOW_SIZE: int = 8

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        self.dim = dim
        self.norm = WanRMSNorm(
            dim,
            images=True,
            dtype=dtype,
            device=device,
        )
        self.to_qkv = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim * 3,
            dtype=dtype,
            stride=1,
            padding=0,
            has_bias=True,
            device=device,
            permute=False,
        )
        self.proj = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            dtype=dtype,
            stride=1,
            padding=0,
            has_bias=True,
            device=device,
            permute=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        b = int(x.shape[0])
        t = int(x.shape[2])
        h = int(x.shape[3])
        w = int(x.shape[4])
        c = self.dim
        ws = self._WINDOW_SIZE

        # [b, c, t, h, w] -> [b*t, c, h, w]
        x2d = F.permute(x, [0, 2, 1, 3, 4])
        x2d = F.reshape(x2d, [b * t, c, h, w])
        x2d = self.norm(x2d)

        x2d_nhwc = F.permute(x2d, [0, 2, 3, 1])  # [bt, h, w, c]
        qkv = self.to_qkv(x2d_nhwc)  # [bt, h, w, 3c]

        # Pad H and W to multiples of ws using Tensor.zeros (no broadcast).
        pad_h = (ws - (h % ws)) % ws
        pad_w = (ws - (w % ws)) % ws
        if pad_w:
            zw = Tensor.zeros(
                [b * t, h, pad_w, 3 * c],
                dtype=qkv.dtype,
                device=qkv.device,
            )
            qkv = F.concat([qkv, zw], axis=2)
        if pad_h:
            zh = Tensor.zeros(
                [b * t, pad_h, w + pad_w, 3 * c],
                dtype=qkv.dtype,
                device=qkv.device,
            )
            qkv = F.concat([qkv, zh], axis=1)

        h_p = h + pad_h
        w_p = w + pad_w
        hws = h_p // ws
        wws = w_p // ws
        nwin = hws * wws
        tok = ws * ws

        q = qkv[:, :, :, :c]
        k = qkv[:, :, :, c : 2 * c]
        v = qkv[:, :, :, 2 * c : 3 * c]

        def to_windows(y: Tensor) -> Tensor:
            y = F.reshape(y, [b * t, hws, ws, wws, ws, c])
            y = F.permute(y, [0, 1, 3, 2, 4, 5])  # [bt, hws, wws, ws, ws, c]
            return F.reshape(y, [b * t, nwin, tok, c])

        q_w = to_windows(q)
        k_w = to_windows(k)
        v_w = to_windows(v)

        attn_scores = F.matmul(
            q_w * (float(c) ** -0.5), F.permute(k_w, [0, 1, 3, 2])
        )
        attn = F.softmax(attn_scores, axis=-1)
        out = F.matmul(attn, v_w)  # [bt, nwin, tok, c]

        out = F.reshape(out, [b * t, hws, wws, ws, ws, c])
        out = F.permute(out, [0, 1, 3, 2, 4, 5])
        out = F.reshape(out, [b * t, h_p, w_p, c])

        if pad_h or pad_w:
            out = out[:, :h, :w, :]

        out = self.proj(out)  # [bt, h, w, c]
        out = F.permute(out, [0, 3, 1, 2])  # [bt, c, h, w]
        out = F.reshape(out, [b, t, c, h, w])
        out = F.permute(out, [0, 2, 1, 3, 4])
        return out + identity


class WanMidBlock(Module[[Tensor], Tensor]):
    """Middle decoder block with residual-attention-residual."""

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        self.resnets = ModuleList(
            [
                WanResidualBlock(dim, dim, dtype=dtype, device=device),
                WanResidualBlock(dim, dim, dtype=dtype, device=device),
            ]
        )
        self.attentions = ModuleList(
            [WanAttentionBlock(dim, dtype=dtype, device=device)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class WanUpsample2d(Module[[Tensor], Tensor]):
    """Nearest-neighbor 2D upsample by factor 2."""

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        # Reshape to [N, C, H, 1, W, 1], duplicate along new axes, reshape back
        x = F.reshape(x, [n, c, h, 1, w, 1])
        x = F.concat([x, x], axis=3)  # [N, C, H, 2, W, 1]
        x = F.concat([x, x], axis=5)  # [N, C, H, 2, W, 2]
        return F.reshape(x, [n, c, h * 2, w * 2])


class WanResample(Module[[Tensor], Tensor]):
    """Wan decoder upsampling module."""

    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int | None = None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        self._out_c = upsample_out_dim

        self.time_conv: WanCausalConv3d | None = None
        self.resample = ModuleList(
            [
                WanUpsample2d(),
                Conv2d(
                    kernel_size=3,
                    in_channels=dim,
                    out_channels=upsample_out_dim,
                    dtype=dtype,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    device=device,
                    permute=False,
                ),
            ]
        )

        if mode == "upsample3d":
            self.time_conv = WanCausalConv3d(
                in_channels=dim,
                out_channels=dim * 2,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
                dtype=dtype,
                device=device,
                has_bias=True,
            )
        elif mode != "upsample2d":
            raise ValueError(f"Unsupported WanResample mode: {mode}")

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        if self.mode == "upsample3d":
            if self.time_conv is None:
                raise ValueError("time_conv is required for upsample3d mode")
            x = self.time_conv(x)
            # x: [b, 2*dim, t, h, w] -> interleave temporal frames
            x = F.reshape(x, [b, 2, self.dim, t, h, w])
            x = F.permute(x, [0, 2, 3, 1, 4, 5])  # [b, dim, t, 2, h, w]
            t = t * 2
            x = F.reshape(x, [b, self.dim, t, h, w])

        # Per-frame 2D upsample + conv
        x = F.permute(x, [0, 2, 1, 3, 4])  # [b, t, c, h, w]
        x = F.reshape(x, [b * t, self.dim, h, w])
        x = self.resample[0](x)  # WanUpsample2d: [b*t, dim, h*2, w*2]
        # Conv2d(permute=False) expects NHWC input/output
        x = F.permute(x, [0, 2, 3, 1])
        x = self.resample[1](x)  # [b*t, h*2, w*2, out_c]
        x = F.permute(x, [0, 3, 1, 2])  # [b*t, out_c, h*2, w*2]

        x = F.reshape(x, [b, t, self._out_c, h * 2, w * 2])
        x = F.permute(x, [0, 2, 1, 3, 4])  # [b, out_c, t, h*2, w*2]
        return x


class WanUpBlock(Module[[Tensor], Tensor]):
    """Wan decoder up block composed of residual blocks and optional upsample."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        resnets: list[WanResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(
                    current_dim,
                    out_dim,
                    dtype=dtype,
                    device=device,
                )
            )
            current_dim = out_dim
        self.resnets = ModuleList(resnets)

        self.upsamplers: ModuleList[WanResample] | None = None
        if upsample_mode is not None:
            self.upsamplers = ModuleList(
                [
                    WanResample(
                        out_dim,
                        mode=upsample_mode,
                        upsample_out_dim=None,
                        dtype=dtype,
                        device=device,
                    )
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            x = self.upsamplers[0](x)

        return x


class WanDecoder3d(Module[[Tensor], Tensor]):
    """Wan 3D decoder module."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_upsample: tuple[bool, ...] = (False, True, True),
        out_channels: int = 3,
        is_residual: bool = False,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        del is_residual

        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]

        self.conv_in = WanCausalConv3d(
            z_dim,
            dims[0],
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        self.mid_block = WanMidBlock(dims[0], dtype=dtype, device=device)

        up_blocks: list[WanUpBlock] = []
        final_out_dim = dims[-1]
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            if i > 0:
                in_dim = in_dim // 2

            up_flag = i != len(dim_mult) - 1
            upsample_mode: str | None = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            up_blocks.append(
                WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    upsample_mode=upsample_mode,
                    dtype=dtype,
                    device=device,
                )
            )
            final_out_dim = out_dim

        self.up_blocks = ModuleList(up_blocks)

        self.norm_out = WanRMSNorm(
            final_out_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv_out = WanCausalConv3d(
            final_out_dim,
            out_channels,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class WanResidualBlockCached(Module[..., tuple[Tensor, Tensor, Tensor]]):
    """Wan residual block with explicit cache I/O for conv1/conv2."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        self.norm1 = WanRMSNorm(
            in_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv1 = WanCausalConv3dCached(
            in_dim,
            out_dim,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.norm2 = WanRMSNorm(
            out_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv2 = WanCausalConv3dCached(
            out_dim,
            out_dim,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.conv_shortcut = (
            WanCausalConv3d(
                in_dim,
                out_dim,
                1,
                padding=0,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
            if in_dim != out_dim
            else None
        )

    def forward(
        self,
        x: Tensor,
        cache1_in: Tensor | None = None,
        cache2_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )

        x = F.silu(self.norm1(x))
        if cache1_in is None:
            cache1_in = _zero_cache_for(x)
        x, cache1_out = self.conv1.forward_cached(x, cache1_in)

        x = F.silu(self.norm2(x))
        if cache2_in is None:
            cache2_in = _zero_cache_for(x)
        x, cache2_out = self.conv2.forward_cached(x, cache2_in)
        return x + residual, cache1_out, cache2_out


class WanMidBlockCached(Module[..., tuple[Tensor, ...]]):
    """Middle decoder block with cache threading."""

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        self.resnets = ModuleList(
            [
                WanResidualBlockCached(dim, dim, dtype=dtype, device=device),
                WanResidualBlockCached(dim, dim, dtype=dtype, device=device),
            ]
        )
        self.attentions = ModuleList(
            [WanAttentionBlock(dim, dtype=dtype, device=device)]
        )

    def forward(
        self, x: Tensor, *cache_inputs: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if len(cache_inputs) not in (0, 4):
            raise ValueError(
                f"WanMidBlockCached expected 0 or 4 cache tensors, got {len(cache_inputs)}"
            )

        cache1_in = cache_inputs[0] if len(cache_inputs) == 4 else None
        cache2_in = cache_inputs[1] if len(cache_inputs) == 4 else None
        x, cache1_out, cache2_out = self.resnets[0](x, cache1_in, cache2_in)
        x = self.attentions[0](x)

        cache3_in = cache_inputs[2] if len(cache_inputs) == 4 else None
        cache4_in = cache_inputs[3] if len(cache_inputs) == 4 else None
        x, cache3_out, cache4_out = self.resnets[1](x, cache3_in, cache4_in)
        return x, cache1_out, cache2_out, cache3_out, cache4_out


class WanResampleCached(Module[..., tuple[Tensor, Tensor]]):
    """Wan upsample3d module with explicit cache I/O."""

    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int | None = None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        if mode != "upsample3d":
            raise ValueError(
                "WanResampleCached only supports mode='upsample3d'"
            )

        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        self._out_c = upsample_out_dim

        self.time_conv = WanCausalConv3dCached(
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.resample = ModuleList(
            [
                WanUpsample2d(),
                Conv2d(
                    kernel_size=3,
                    in_channels=dim,
                    out_channels=upsample_out_dim,
                    dtype=dtype,
                    stride=1,
                    padding=1,
                    has_bias=True,
                    device=device,
                    permute=False,
                ),
            ]
        )

    def forward(
        self,
        x: Tensor,
        cache_in: Tensor | None = None,
        first_chunk: bool = False,
    ) -> tuple[Tensor, Tensor]:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        if cache_in is None:
            cache_in = _zero_cache_for(x)

        if first_chunk:
            cache_out = cache_in
        else:
            x, cache_out = self.time_conv.forward_cached(x, cache_in)
            x = F.reshape(x, [b, 2, self.dim, t, h, w])
            x = F.permute(x, [0, 2, 3, 1, 4, 5])
            t = t * 2
            x = F.reshape(x, [b, self.dim, t, h, w])

        x = F.permute(x, [0, 2, 1, 3, 4])
        x = F.reshape(x, [b * t, self.dim, h, w])
        x = self.resample[0](x)
        x = F.permute(x, [0, 2, 3, 1])
        x = self.resample[1](x)
        x = F.permute(x, [0, 3, 1, 2])
        x = F.reshape(x, [b, t, self._out_c, h * 2, w * 2])
        x = F.permute(x, [0, 2, 1, 3, 4])
        return x, cache_out


class WanUpBlockCached(Module[..., tuple[Tensor, ...]]):
    """Wan decoder up block with explicit cache threading."""

    cache_slots: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        resnets: list[WanResidualBlockCached] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlockCached(
                    current_dim,
                    out_dim,
                    dtype=dtype,
                    device=device,
                )
            )
            current_dim = out_dim
        self.resnets = ModuleList(resnets)

        self._has_temporal_upsample = upsample_mode == "upsample3d"
        self.cache_slots = len(resnets) * 2 + (
            1 if self._has_temporal_upsample else 0
        )

        self.upsamplers: ModuleList | None = None
        if upsample_mode is not None:
            if upsample_mode == "upsample3d":
                upsampler: Module[..., Any] = WanResampleCached(
                    out_dim,
                    mode=upsample_mode,
                    upsample_out_dim=None,
                    dtype=dtype,
                    device=device,
                )
            elif upsample_mode == "upsample2d":
                upsampler = WanResample(
                    out_dim,
                    mode=upsample_mode,
                    upsample_out_dim=None,
                    dtype=dtype,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Unsupported WanUpBlockCached upsample mode: {upsample_mode}"
                )

            self.upsamplers = ModuleList([upsampler])

    def forward(
        self,
        x: Tensor,
        *cache_inputs: Tensor,
        first_chunk: bool = False,
    ) -> tuple[Tensor, ...]:
        if len(cache_inputs) not in (0, self.cache_slots):
            raise ValueError(
                f"WanUpBlockCached expected 0 or {self.cache_slots} cache tensors, got {len(cache_inputs)}"
            )

        use_cache_inputs = len(cache_inputs) == self.cache_slots
        cache_outputs: list[Tensor] = []
        cache_idx = 0

        for resnet in self.resnets:
            cache1_in = cache_inputs[cache_idx] if use_cache_inputs else None
            cache2_in = (
                cache_inputs[cache_idx + 1] if use_cache_inputs else None
            )
            x, cache1_out, cache2_out = resnet(x, cache1_in, cache2_in)
            cache_outputs.extend([cache1_out, cache2_out])
            cache_idx += 2

        if self.upsamplers is not None:
            upsampler = self.upsamplers[0]
            if self._has_temporal_upsample:
                cache_in = cache_inputs[cache_idx] if use_cache_inputs else None
                if not isinstance(upsampler, WanResampleCached):
                    raise TypeError(
                        "Expected WanResampleCached for temporal upsample"
                    )
                x, cache_out = upsampler(
                    x,
                    cache_in,
                    first_chunk=first_chunk,
                )
                cache_outputs.append(cache_out)
            else:
                x = upsampler(x)

        return (x, *cache_outputs)


class WanDecoder3dCached(Module[..., tuple[Tensor, ...]]):
    """Wan 3D decoder with explicit cache tensor I/O."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_upsample: tuple[bool, ...] = (False, True, True),
        out_channels: int = 3,
        is_residual: bool = False,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        del is_residual

        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]

        self.conv_in = WanCausalConv3dCached(
            z_dim,
            dims[0],
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        self.mid_block = WanMidBlockCached(dims[0], dtype=dtype, device=device)

        up_blocks: list[WanUpBlockCached] = []
        final_out_dim = dims[-1]
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            if i > 0:
                in_dim = in_dim // 2

            up_flag = i != len(dim_mult) - 1
            upsample_mode: str | None = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            up_blocks.append(
                WanUpBlockCached(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    upsample_mode=upsample_mode,
                    dtype=dtype,
                    device=device,
                )
            )
            final_out_dim = out_dim

        self.up_blocks = ModuleList(up_blocks)
        self.norm_out = WanRMSNorm(
            final_out_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv_out = WanCausalConv3dCached(
            final_out_dim,
            out_channels,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def forward(
        self,
        x: Tensor,
        *cache_inputs: Tensor,
        first_chunk: bool = False,
    ) -> tuple[Tensor, ...]:
        if len(cache_inputs) not in (0, WAN_DECODER_CACHE_SLOTS):
            raise ValueError(
                "WanDecoder3dCached expected 0 or "
                f"{WAN_DECODER_CACHE_SLOTS} cache tensors, got {len(cache_inputs)}"
            )

        use_cache_inputs = len(cache_inputs) == WAN_DECODER_CACHE_SLOTS
        cache_outputs: list[Tensor] = []
        cache_idx = 0

        conv_in_cache = cache_inputs[cache_idx] if use_cache_inputs else None
        if conv_in_cache is None:
            conv_in_cache = _zero_cache_for(x)
        x, cache_out = self.conv_in.forward_cached(x, conv_in_cache)
        cache_outputs.append(cache_out)
        cache_idx += 1

        mid_cache_inputs: tuple[Tensor, ...] = (
            tuple(cache_inputs[cache_idx : cache_idx + 4])
            if use_cache_inputs
            else ()
        )
        mid_outputs = self.mid_block(x, *mid_cache_inputs)
        x = mid_outputs[0]
        cache_outputs.extend(mid_outputs[1:])
        cache_idx += 4

        for up_block in self.up_blocks:
            block_cache_inputs: tuple[Tensor, ...] = (
                tuple(
                    cache_inputs[cache_idx : cache_idx + up_block.cache_slots]
                )
                if use_cache_inputs
                else ()
            )
            block_outputs = up_block(
                x,
                *block_cache_inputs,
                first_chunk=first_chunk,
            )
            x = block_outputs[0]
            cache_outputs.extend(block_outputs[1:])
            cache_idx += up_block.cache_slots

        x = self.norm_out(x)
        x = F.silu(x)
        conv_out_cache = cache_inputs[cache_idx] if use_cache_inputs else None
        if conv_out_cache is None:
            conv_out_cache = _zero_cache_for(x)
        x, cache_out = self.conv_out.forward_cached(x, conv_out_cache)
        cache_outputs.append(cache_out)

        if len(cache_outputs) != WAN_DECODER_CACHE_SLOTS:
            raise ValueError(
                "WanDecoder3dCached produced "
                f"{len(cache_outputs)} cache tensors, expected {WAN_DECODER_CACHE_SLOTS}"
            )
        return (x, *cache_outputs)

    def cache_shapes(
        self,
        batch_size: int,
        latent_height: int,
        latent_width: int,
    ) -> list[list[int]]:
        h = latent_height
        w = latent_width
        shapes: list[list[int]] = [
            [batch_size, self.conv_in.in_channels, CACHE_T, h, w]
        ]

        for resnet in self.mid_block.resnets:
            shapes.append([batch_size, resnet.conv1.in_channels, CACHE_T, h, w])
            shapes.append([batch_size, resnet.conv2.in_channels, CACHE_T, h, w])

        for up_block in self.up_blocks:
            for resnet in up_block.resnets:
                shapes.append(
                    [batch_size, resnet.conv1.in_channels, CACHE_T, h, w]
                )
                shapes.append(
                    [batch_size, resnet.conv2.in_channels, CACHE_T, h, w]
                )

            if up_block.upsamplers is not None:
                if up_block._has_temporal_upsample:
                    upsampler = up_block.upsamplers[0]
                    if not isinstance(upsampler, WanResampleCached):
                        raise TypeError(
                            "Expected WanResampleCached for temporal upsample"
                        )
                    shapes.append(
                        [
                            batch_size,
                            upsampler.time_conv.in_channels,
                            CACHE_T,
                            h,
                            w,
                        ]
                    )
                h *= 2
                w *= 2

        shapes.append([batch_size, self.conv_out.in_channels, CACHE_T, h, w])
        if len(shapes) != WAN_DECODER_CACHE_SLOTS:
            raise ValueError(
                f"Expected {WAN_DECODER_CACHE_SLOTS} cache shapes, got {len(shapes)}"
            )
        return shapes


class _WanVAEPostQuantConv(Module[[Tensor], Tensor]):
    """Standalone post-quant conv graph (k=1, frame-independent)."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        self.post_quant_conv = WanCausalConv3d(
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            kernel_size=1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.post_quant_conv(z)


class _WanVAEDecoderFirstFrameCached(Module[[Tensor], tuple[Tensor, ...]]):
    """First-frame decoder graph returning pixels + initialized caches."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        self.decoder = WanDecoder3dCached(
            dim=config.base_dim,
            z_dim=config.z_dim,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            temperal_upsample=tuple(reversed(config.temperal_downsample)),
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            dtype=config.dtype,
            device=config.device,
        )

    def forward(self, z: Tensor) -> tuple[Tensor, ...]:
        outputs = self.decoder(z, first_chunk=True)
        x = outputs[0]
        x = F.max(x, -1.0)
        x = F.min(x, 1.0)
        return (x, *outputs[1:])


class _WanVAEDecoderRestFrameCached(Module[..., tuple[Tensor, ...]]):
    """Per-frame decoder graph with cache feedback for frames 1..T-1."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        self.decoder = WanDecoder3dCached(
            dim=config.base_dim,
            z_dim=config.z_dim,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            temperal_upsample=tuple(reversed(config.temperal_downsample)),
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            dtype=config.dtype,
            device=config.device,
        )

    def forward(self, z: Tensor, *cache_inputs: Tensor) -> tuple[Tensor, ...]:
        outputs = self.decoder(z, *cache_inputs, first_chunk=False)
        x = outputs[0]
        x = F.max(x, -1.0)
        x = F.min(x, 1.0)
        return (x, *outputs[1:])


class WanVAEDecoder(Module[[Tensor], Tensor]):
    """Wan VAE decoder graph used by AutoencoderKLWanModel."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        self._config = config
        self.post_quant_conv = WanCausalConv3d(
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            kernel_size=1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )
        self.decoder = WanDecoder3d(
            dim=config.base_dim,
            z_dim=config.z_dim,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            temperal_upsample=tuple(reversed(config.temperal_downsample)),
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            dtype=config.dtype,
            device=config.device,
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.post_quant_conv(z)
        x = self.decoder(x)
        x = F.max(x, -1.0)
        x = F.min(x, 1.0)
        return x


class _WanVAEDecoderFirstFrame(Module[[Tensor], Tensor]):
    """Wan VAE decoder for the FIRST latent frame.

    Identical to WanVAEDecoder but ALL temporal upsamples are replaced
    with spatial-only upsample2d (time_conv is omitted).  This means
    T=1 in -> T=1 out, matching the diffusers feat_cache behavior where
    the first frame skips temporal upsampling.
    """

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        self._config = config
        self.post_quant_conv = WanCausalConv3d(
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            kernel_size=1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )
        # Force all temporal upsamples to spatial-only.
        self.decoder = WanDecoder3d(
            dim=config.base_dim,
            z_dim=config.z_dim,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            temperal_upsample=(False,) * len(config.temperal_downsample),
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            dtype=config.dtype,
            device=config.device,
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.post_quant_conv(z)
        x = self.decoder(x)
        x = F.max(x, -1.0)
        x = F.min(x, 1.0)
        return x


class _PerFrameDecoder:
    """Per-frame VAE decode with cache feedback through graph I/O."""

    def __init__(
        self,
        post_quant_conv: CompileWrapper,
        first_decoder: CompileWrapper,
        rest_decoder: CompileWrapper,
    ) -> None:
        self.post_quant_conv = post_quant_conv
        self.first_decoder = first_decoder
        self.rest_decoder = rest_decoder

    def __call__(self, latents_5d: Tensor) -> Tensor:
        import time as _time

        t_total = int(latents_5d.shape[2])
        t0 = _time.perf_counter()

        accelerator: Accelerator | None = None
        if isinstance(latents_5d.device, Accelerator):
            accelerator = latents_5d.device

        post_quant_outputs = self.post_quant_conv(latents_5d)
        if isinstance(post_quant_outputs, (list, tuple)):
            if len(post_quant_outputs) != 1:
                raise ValueError(
                    "post_quant_conv returned "
                    f"{len(post_quant_outputs)} outputs, expected 1"
                )
            z = post_quant_outputs[0]
        else:
            z = post_quant_outputs

        if accelerator is not None:
            accelerator.synchronize()
        t_post = _time.perf_counter()

        logger.info(
            "post_quant_conv: input T=%d completed in %.1fs",
            t_total,
            t_post - t0,
        )

        first_outputs = _normalize_compiled_outputs(
            self.first_decoder(z[:, :, 0:1, :, :])
        )
        if len(first_outputs) != WAN_DECODER_CACHE_SLOTS + 1:
            raise ValueError(
                "first_decoder returned "
                f"{len(first_outputs)} outputs, expected {WAN_DECODER_CACHE_SLOTS + 1}"
            )
        first_pixels = first_outputs[0]
        caches = first_outputs[1:]

        if accelerator is not None:
            accelerator.synchronize()
        t_first_done = _time.perf_counter()

        first_np = np.from_dlpack(first_pixels.cast(DType.float32).to(CPU()))
        frames_np = [np.ascontiguousarray(first_np)]
        t_first_cpu = _time.perf_counter()

        logger.info(
            "first frame decode: exec=%.1fs D2H=%.1fs output T=%d",
            t_first_done - t_post,
            t_first_cpu - t_first_done,
            first_np.shape[2],
        )

        rest_exec_total = 0.0
        rest_cpu_total = 0.0

        for frame_idx in range(1, t_total):
            t_exec_start = _time.perf_counter()
            rest_outputs = _normalize_compiled_outputs(
                self.rest_decoder(
                    z[:, :, frame_idx : frame_idx + 1, :, :], *caches
                )
            )
            if len(rest_outputs) != WAN_DECODER_CACHE_SLOTS + 1:
                raise ValueError(
                    "rest_decoder returned "
                    f"{len(rest_outputs)} outputs, expected {WAN_DECODER_CACHE_SLOTS + 1}"
                )

            pixels = rest_outputs[0]
            caches = rest_outputs[1:]
            if accelerator is not None:
                accelerator.synchronize()
            t_exec_end = _time.perf_counter()

            frame_np = np.from_dlpack(pixels.cast(DType.float32).to(CPU()))
            frames_np.append(np.ascontiguousarray(frame_np))
            t_cpu_end = _time.perf_counter()

            rest_exec_total += t_exec_end - t_exec_start
            rest_cpu_total += t_cpu_end - t_exec_end

        result = np.concatenate(frames_np, axis=2)


        total = _time.perf_counter() - t0
        logger.info(
            "rest frames decode: frames=%d exec=%.1fs D2H=%.1fs",
            max(t_total - 1, 0),
            rest_exec_total,
            rest_cpu_total,
        )
        logger.info(
            "VAE output: shape=%s range=[%.4f,%.4f]",
            result.shape,
            result.min(),
            result.max(),
        )
        logger.info(
            "VAE decode: %d latent -> %d video frames in %.1fs",
            t_total,
            result.shape[2],
            total,
        )

        result_contiguous = np.ascontiguousarray(result).astype(
            np.float32, copy=False
        )
        return Tensor.from_dlpack(result_contiguous)


class AutoencoderKLWanModel(ComponentModel):
    """Wan VAE decoder model using MAX-native 3D modules."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = AutoencoderKLWanConfig.generate(config, encoding, devices)
        self._per_frame_decoder: _PerFrameDecoder | None = None
        self.load_model()

    def load_model(self) -> Callable[[Tensor], Tensor]:
        decoder_state_dict: dict[str, Any] = {}
        target_dtype = self.config.dtype

        assert self.weights is not None
        weights_obj: Any = self.weights

        for key, value in weights_obj.items():
            if not (
                key.startswith("decoder.") or key.startswith("post_quant_conv.")
            ):
                continue

            weight_data = value.data()

            # Wan checkpoints store Conv3d filters in FCQRS (PyTorch) layout.
            # Conv3d(permute=True) keeps FCQRS, enabling cuDNN on NVIDIA GPU.
            # Conv2d weights: PyTorch FCRS [out, in, H, W] -> RSCF
            # [H, W, in, out] for Conv2d(permute=False).
            if key.endswith(".weight") and len(weight_data.shape) == 4:
                weight_data = np.ascontiguousarray(
                    np.from_dlpack(weight_data).transpose(2, 3, 1, 0)
                )

            decoder_state_dict[key] = weight_data

        # Cast all weights to target dtype using a compiled graph (cached
        # per dtype pair, so the LLVM compilation only happens once).
        if target_dtype != DType.float32:
            cpu_device = CPU()
            for key in decoder_state_dict:
                decoder_state_dict[key] = cast_dlpack_to(
                    decoder_state_dict[key],
                    DType.float32,
                    target_dtype,
                    cpu_device,
                )

        logger.info("Loaded %d VAE decoder weights", len(decoder_state_dict))

        # Defer compilation to first decode_5d() call so we have concrete
        # dimensions.  This avoids symbolic-dim verification failures in
        # reshape ops that merge / redistribute spatial dimensions.
        self._decoder_state_dict = decoder_state_dict
        self._per_frame_decoder = None
        # Free the raw Weights object now that we have the state dict.
        self.weights = None  # type: ignore[assignment]

        return self.decode_4d

    def _compile_per_frame_decoder(
        self,
        shape: tuple[int, ...],
    ) -> None:
        """Compile post-quant, first-frame, and rest-frame decoder graphs."""
        import time as _time

        cfg = self.config
        sd = self._decoder_state_dict
        device_obj = self.devices[0]
        B, C, T_total, H, W = (int(s) for s in shape)

        logger.info(
            "Compiling per-frame VAE decoders: T_total=%d",
            T_total,
        )

        # --- post_quant_conv graph (k=1, all frames) ---
        t0 = _time.perf_counter()
        post_quant_input = TensorType(
            cfg.dtype,
            [B, C, T_total, H, W],
            device=cfg.device,
        )
        with F.lazy():
            post_quant_model = _WanVAEPostQuantConv(cfg)
            post_quant_model.to(device_obj)
        post_quant_keys = {name for name, _ in post_quant_model.parameters}
        post_quant_weights = {
            k: v for k, v in sd.items() if k in post_quant_keys
        }
        missing = sorted(post_quant_keys - set(post_quant_weights.keys()))
        if missing:
            logger.warning(
                "post_quant_conv: %d missing weights: %s",
                len(missing),
                missing[:10],
            )
        post_quant_compiled = CompileWrapper(
            post_quant_model,
            input_types=[post_quant_input],
            weights=post_quant_weights,
        )
        t1 = _time.perf_counter()
        logger.info(
            "Compiled post_quant_conv in %.1fs, %d weights",
            t1 - t0,
            len(post_quant_weights),
        )

        # --- first-frame decoder (T=1 -> T=1 + cache init) ---
        first_input = TensorType(cfg.dtype, [B, C, 1, H, W], device=cfg.device)
        with F.lazy():
            first_model = _WanVAEDecoderFirstFrameCached(cfg)
            first_model.to(device_obj)
        first_model_keys = {name for name, _ in first_model.parameters}
        first_weights = {k: v for k, v in sd.items() if k in first_model_keys}
        missing = sorted(first_model_keys - set(first_weights.keys()))
        if missing:
            logger.warning(
                "First-frame decoder: %d missing weights: %s",
                len(missing),
                missing[:10],
            )
        first_compiled = CompileWrapper(
            first_model,
            input_types=[first_input],
            weights=first_weights,
        )
        t2 = _time.perf_counter()
        logger.info(
            "Compiled first-frame decoder (cached, spatial-only path) in %.1fs, %d weights",
            t2 - t1,
            len(first_weights),
        )

        # --- rest-frame decoder (T=1 + 32 cache inputs -> T=4 + 32 cache outputs) ---
        rest_input = TensorType(
            cfg.dtype,
            [B, C, 1, H, W],
            device=cfg.device,
        )
        with F.lazy():
            rest_model = _WanVAEDecoderRestFrameCached(cfg)
            rest_model.to(device_obj)
        cache_shapes = rest_model.decoder.cache_shapes(B, H, W)
        cache_inputs = [
            TensorType(cfg.dtype, cache_shape, device=cfg.device)
            for cache_shape in cache_shapes
        ]

        rest_model_keys = {name for name, _ in rest_model.parameters}
        rest_weights = {k: v for k, v in sd.items() if k in rest_model_keys}
        missing = sorted(rest_model_keys - set(rest_weights.keys()))
        if missing:
            logger.warning(
                "Rest-frame decoder: %d missing weights: %s",
                len(missing),
                missing[:10],
            )
        rest_compiled = CompileWrapper(
            rest_model,
            input_types=[rest_input, *cache_inputs],
            weights=rest_weights,
        )
        t3 = _time.perf_counter()
        logger.info(
            "Compiled rest-frame decoder (per-frame cached) in %.1fs, %d weights, %d cache inputs",
            t3 - t2,
            len(rest_weights),
            len(cache_inputs),
        )

        self._per_frame_decoder = _PerFrameDecoder(
            post_quant_conv=post_quant_compiled,
            first_decoder=first_compiled,
            rest_decoder=rest_compiled,
        )
        # Free state dict after compilation.
        del self._decoder_state_dict

    def decode_5d(self, latents_5d: Tensor) -> Tensor:
        """Decode 5D latents [B, C, T, H, W] frame-by-frame."""
        import time as _time

        shape = tuple(int(s) for s in latents_5d.shape)
        logger.info(
            'VAE decode input: shape=%s dtype=%s', shape, latents_5d.dtype
        )
        if self._per_frame_decoder is None:
            t0 = _time.perf_counter()
            self._compile_per_frame_decoder(shape)
            logger.info('VAE compile total: %.1fs', _time.perf_counter() - t0)
        assert self._per_frame_decoder is not None
        return self._per_frame_decoder(latents_5d)

    def decode_4d(self, latents_4d: Tensor) -> Tensor:
        z5d = F.unsqueeze(latents_4d, axis=2)
        decoded_5d = self.decode_5d(z5d)
        return decoded_5d[:, :, 0, :, :]

    def decode(
        self, latents_4d: Tensor, return_dict: bool = False
    ) -> tuple[Tensor]:
        del return_dict
        if latents_4d.rank == 5:
            return (self.decode_5d(latents_4d),)
        return (self.decode_4d(latents_4d),)

    def __call__(self, latents_4d: Tensor) -> Tensor:
        if latents_4d.rank == 5:
            return self.decode_5d(latents_4d)
        return self.decode_4d(latents_4d)
