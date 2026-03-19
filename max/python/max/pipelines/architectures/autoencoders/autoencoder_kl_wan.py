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
import threading
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import numpy as np
from max.driver import CPU, Buffer, Device, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.type import FilterLayout
from max.graph.weights import Weights
from max.nn.layer import LayerList, Module
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import Tracer

from .model_config import AutoencoderKLWanConfig

logger = logging.getLogger(__name__)

CACHE_T = 2
WAN_DECODER_CACHE_SLOTS = 32
WAN_ENCODER_CHUNK_SIZE = 4  # Frames per encoder chunk (matching diffusers)


def _use_nvidia_fcrs_conv3d(device: DeviceRef | None) -> bool:
    return (
        device is not None and device.is_gpu() and accelerator_api() == "cuda"
    )


def _buffer_to_numpy_f32(buf: Buffer, cpu: CPU | None = None) -> np.ndarray:
    """Convert a Buffer (possibly bf16) to f32 numpy on CPU."""
    cpu_buf = buf.to(cpu or CPU())
    if cpu_buf.dtype == DType.bfloat16:
        u16 = np.from_dlpack(
            cpu_buf.view(dtype=DType.uint16, shape=cpu_buf.shape)
        )
        return (u16.astype(np.uint32) << 16).view(np.float32)
    return np.from_dlpack(cpu_buf).astype(np.float32, copy=False)


def _numpy_f32_to_buffer(
    arr: np.ndarray, target_dtype: DType, device: Device
) -> Buffer:
    """Convert f32 numpy to Buffer on device with target dtype."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if target_dtype == DType.bfloat16:
        u16 = float32_to_bfloat16_as_uint16(arr)
        return (
            Buffer.from_numpy(u16)
            .to(device)
            .view(dtype=DType.bfloat16, shape=arr.shape)
        )
    return Buffer.from_numpy(arr).to(device)


def _zero_cache_for(x: TensorValue) -> TensorValue:
    """Create a zero cache tensor shaped for a causal conv input."""
    shape = [
        int(x.shape[0]),
        int(x.shape[1]),
        CACHE_T,
        int(x.shape[3]),
        int(x.shape[4]),
    ]
    return ops.constant(0.0, dtype=x.dtype, device=x.device).broadcast_to(shape)


class WanRMSNorm(Module):
    """RMS norm used by Wan VAE blocks."""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = False,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.channel_first = channel_first

        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        shape = [dim, *broadcastable_dims] if channel_first else [dim]
        dev_ref = device if device is not None else DeviceRef.CPU()
        self.gamma = Weight(
            "gamma",
            dtype or DType.float32,
            shape,
            dev_ref,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        axis = 1 if self.channel_first else x.rank - 1
        rms = ops.mean(x * x, axis=axis)
        inv = ops.rsqrt(rms + 1e-12)
        gamma = ops.transfer_to(self.gamma, x.device)
        return x * inv * gamma


class WanCausalConv3d(Module):
    """3D causal convolution for Wan VAE.

    Temporal causality is implemented via asymmetric padding: the front
    (temporal) dimension is padded on the left only, which the conv3d
    padding parameter supports directly.

    Input is permuted from NCDHW to NDHWC before conv, and back after.
    On NVIDIA GPUs, weights stay in PyTorch FCQRS layout to use the cuDNN
    3D conv dispatch path. Other backends use MAX's native QRSCF layout.
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
        prefer_nvidia_fcrs: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            pad_t = pad_h = pad_w = padding
        else:
            pad_t, pad_h, pad_w = padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._stride = stride
        # Causal: pad only the front of the temporal axis (left=2*pad_t, right=0).
        self._padding = (2 * pad_t, 0, pad_h, pad_h, pad_w, pad_w)

        dev_ref = device if device is not None else DeviceRef.CPU()
        dt = dtype or DType.float32
        d, h, w = kernel_size
        self._use_nvidia_fcrs = (
            prefer_nvidia_fcrs and _use_nvidia_fcrs_conv3d(dev_ref)
        )
        filter_shape = (
            [out_channels, in_channels, d, h, w]
            if self._use_nvidia_fcrs
            else [d, h, w, in_channels, out_channels]
        )
        self.filter = Weight("weight", dt, filter_shape, dev_ref)
        self._has_bias = has_bias
        if has_bias:
            self.bias = Weight("bias", dt, [out_channels], dev_ref)

    def __call__(self, x: TensorValue) -> TensorValue:
        # NCDHW -> NDHWC
        x_ndhwc = ops.permute(x, [0, 2, 3, 4, 1])
        out = ops.conv3d(
            x_ndhwc,
            self.filter,
            stride=self._stride,
            padding=self._padding,
            filter_layout=(
                FilterLayout.FCRS
                if self._use_nvidia_fcrs
                else FilterLayout.QRSCF
            ),
        )
        # NDHWC -> NCDHW
        out = ops.permute(out, [0, 4, 1, 2, 3])
        if self._has_bias:
            bias_5d = ops.reshape(self.bias, [1, self.out_channels, 1, 1, 1])
            out = out + bias_5d
        return out


class WanCausalConv3dCached(Module):
    """3D causal convolution with explicit cache tensor I/O.

    Handles temporal causal padding separately via concat/pad before
    calling the conv, while spatial padding is handled by conv3d.
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
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            pad_t = pad_h = pad_w = padding
        else:
            pad_t, pad_h, pad_w = padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._stride = stride
        # Temporal causal padding: left=2*pad_t, right=0
        self._temporal_pad_left = 2 * pad_t
        # Let conv3d handle spatial padding. Temporal padding = 0 here.
        self._padding = (0, 0, pad_h, pad_h, pad_w, pad_w)

        dev_ref = device if device is not None else DeviceRef.CPU()
        dt = dtype or DType.float32
        d, h, w = kernel_size
        self._use_nvidia_fcrs = _use_nvidia_fcrs_conv3d(dev_ref)
        filter_shape = (
            [out_channels, in_channels, d, h, w]
            if self._use_nvidia_fcrs
            else [d, h, w, in_channels, out_channels]
        )
        self.filter = Weight("weight", dt, filter_shape, dev_ref)
        self._has_bias = has_bias
        if has_bias:
            self.bias = Weight("bias", dt, [out_channels], dev_ref)

    def _apply_temporal_pad(self, x: TensorValue, pad_left: int) -> TensorValue:
        """Zero-pad the temporal dimension (axis=2) on the left only."""
        if pad_left <= 0:
            return x
        # ops.pad expects 2*rank values: [d0_before, d0_after, d1_before, d1_after, ...]
        # For 5D [B, C, T, H, W]: pad only dim 2 (T) on the left.
        pad_vals = [0, 0, 0, 0, pad_left, 0, 0, 0, 0, 0]
        return ops.pad(x, pad_vals)

    def _forward_conv(self, x: TensorValue) -> TensorValue:
        # NCDHW -> NDHWC
        x_ndhwc = ops.permute(x, [0, 2, 3, 4, 1])
        out = ops.conv3d(
            x_ndhwc,
            self.filter,
            stride=self._stride,
            padding=self._padding,
            filter_layout=(
                FilterLayout.FCRS
                if self._use_nvidia_fcrs
                else FilterLayout.QRSCF
            ),
        )
        # NDHWC -> NCDHW
        out = ops.permute(out, [0, 4, 1, 2, 3])
        if self._has_bias:
            bias_5d = ops.reshape(self.bias, [1, self.out_channels, 1, 1, 1])
            out = out + bias_5d
        return out

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self._apply_temporal_pad(x, self._temporal_pad_left)
        return self._forward_conv(x)

    def forward_cached(
        self, x: TensorValue, cache_in: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        x = ops.concat([cache_in, x], axis=2)
        cache_out = x[:, :, -CACHE_T:, :, :]
        # Reduce temporal padding by the amount of context from cache.
        effective_pad = max(self._temporal_pad_left - int(cache_in.shape[2]), 0)
        x = self._apply_temporal_pad(x, effective_pad)
        return self._forward_conv(x), cache_out


class WanConv2dPermuted(Module):
    """2D convolution with NCHW input and FCRS weights (permute=True equivalent).

    Input is permuted from NCHW to NHWC before conv, and back after.
    Weights stay in FCRS (PyTorch) layout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(stride, int):
            self._stride = (stride, stride)
        else:
            self._stride = stride
        if isinstance(padding, int):
            self._padding = (padding, padding, padding, padding)
        else:
            self._padding = padding

        dev_ref = device if device is not None else DeviceRef.CPU()
        dt = dtype or DType.float32
        self.filter = Weight(
            "weight",
            dt,
            [out_channels, in_channels, kernel_size, kernel_size],
            dev_ref,
        )
        self._has_bias = has_bias
        if has_bias:
            self.bias = Weight("bias", dt, [out_channels], dev_ref)

    def __call__(self, x: TensorValue) -> TensorValue:
        # NCHW -> NHWC
        x_nhwc = ops.permute(x, [0, 2, 3, 1])
        out = ops.conv2d(
            x_nhwc,
            self.filter,
            stride=self._stride,
            padding=self._padding,
            filter_layout=FilterLayout.FCRS,
        )
        # NHWC -> NCHW
        out = ops.permute(out, [0, 3, 1, 2])
        if self._has_bias:
            bias_4d = ops.reshape(self.bias, [1, self.out_channels, 1, 1])
            out = out + bias_4d
        return out


class WanConv2d(Module):
    """2D convolution with NHWC input and RSCF weights (permute=False equivalent).

    Input is already in NHWC layout. Weights are in RSCF layout
    [H, W, in_channels, out_channels].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(stride, int):
            self._stride = (stride, stride)
        else:
            self._stride = stride
        if isinstance(padding, int):
            self._padding = (padding, padding, padding, padding)
        else:
            self._padding = padding

        dev_ref = device if device is not None else DeviceRef.CPU()
        dt = dtype or DType.float32
        self.filter = Weight(
            "weight",
            dt,
            [kernel_size, kernel_size, in_channels, out_channels],
            dev_ref,
        )
        self._has_bias = has_bias
        if has_bias:
            self.bias = Weight("bias", dt, [out_channels], dev_ref)

    def __call__(self, x: TensorValue) -> TensorValue:
        out = ops.conv2d(
            x,
            self.filter,
            stride=self._stride,
            padding=self._padding,
            filter_layout=FilterLayout.RSCF,
            bias=self.bias if self._has_bias else None,
        )
        return out


class WanResidualBlock(Module):
    """Residual block used in Wan VAE decoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        prefer_nvidia_fcrs: bool = True,
    ) -> None:
        super().__init__()
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
            prefer_nvidia_fcrs=prefer_nvidia_fcrs,
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
            prefer_nvidia_fcrs=prefer_nvidia_fcrs,
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
                prefer_nvidia_fcrs=prefer_nvidia_fcrs,
            )
            if in_dim != out_dim
            else None
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        residual = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )
        x = ops.silu(self.norm1(x))
        x = self.conv1(x)
        x = ops.silu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class WanAttentionBlock(Module):
    """Per-frame windowed self-attention used in Wan decoder mid block.

    Uses window attention instead of full (H*W)^2 attention to avoid OOM
    at high resolutions. The spatial dimensions are partitioned into
    non-overlapping windows of size ws*ws, and attention is computed
    independently per window.

    Memory: O(b*t * num_windows * ws^2 * ws^2) instead of O(b*t * (H*W)^2).
    At 720p latent (90x160) with ws=8: ~158MB vs ~2.5GB+ per chunk.
    """

    _WINDOW_SIZE: int = 8

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = WanRMSNorm(
            dim,
            images=True,
            dtype=dtype,
            device=device,
        )
        self.to_qkv = WanConv2d(
            in_channels=dim,
            out_channels=dim * 3,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.proj = WanConv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        identity = x
        b = int(x.shape[0])
        t = int(x.shape[2])
        h = int(x.shape[3])
        w = int(x.shape[4])
        c = self.dim
        ws = self._WINDOW_SIZE

        # [b, c, t, h, w] -> [b*t, c, h, w]
        x2d = ops.permute(x, [0, 2, 1, 3, 4])
        x2d = ops.reshape(x2d, [b * t, c, h, w])
        x2d = self.norm(x2d)

        x2d_nhwc = ops.permute(x2d, [0, 2, 3, 1])  # [bt, h, w, c]
        qkv = self.to_qkv(x2d_nhwc)  # [bt, h, w, 3c]

        # Pad H and W to multiples of ws using ops.constant (no broadcast).
        pad_h = (ws - (h % ws)) % ws
        pad_w = (ws - (w % ws)) % ws
        if pad_w:
            zw = ops.constant(
                0.0, dtype=qkv.dtype, device=qkv.device
            ).broadcast_to([b * t, h, pad_w, 3 * c])
            qkv = ops.concat([qkv, zw], axis=2)
        if pad_h:
            zh = ops.constant(
                0.0, dtype=qkv.dtype, device=qkv.device
            ).broadcast_to([b * t, pad_h, w + pad_w, 3 * c])
            qkv = ops.concat([qkv, zh], axis=1)

        h_p = h + pad_h
        w_p = w + pad_w
        hws = h_p // ws
        wws = w_p // ws
        nwin = hws * wws
        tok = ws * ws

        q = qkv[:, :, :, :c]
        k = qkv[:, :, :, c : 2 * c]
        v = qkv[:, :, :, 2 * c : 3 * c]

        def to_windows(y: TensorValue) -> TensorValue:
            y = ops.reshape(y, [b * t, hws, ws, wws, ws, c])
            y = ops.permute(y, [0, 1, 3, 2, 4, 5])  # [bt, hws, wws, ws, ws, c]
            return ops.reshape(y, [b * t, nwin, tok, c])

        q_w = to_windows(q)
        k_w = to_windows(k)
        v_w = to_windows(v)

        attn_scores = ops.matmul(
            q_w * (float(c) ** -0.5), ops.permute(k_w, [0, 1, 3, 2])
        )
        attn = ops.softmax(attn_scores, axis=-1)
        out = ops.matmul(attn, v_w)  # [bt, nwin, tok, c]

        out = ops.reshape(out, [b * t, hws, wws, ws, ws, c])
        out = ops.permute(out, [0, 1, 3, 2, 4, 5])
        out = ops.reshape(out, [b * t, h_p, w_p, c])

        if pad_h or pad_w:
            out = out[:, :h, :w, :]

        out = self.proj(out)  # [bt, h, w, c]
        out = ops.permute(out, [0, 3, 1, 2])  # [bt, c, h, w]
        out = ops.reshape(out, [b, t, c, h, w])
        out = ops.permute(out, [0, 2, 1, 3, 4])
        return out + identity


class WanMidBlock(Module):
    """Middle decoder block with residual-attention-residual."""

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        prefer_nvidia_fcrs: bool = True,
    ) -> None:
        super().__init__()
        self.resnets = LayerList(
            [
                WanResidualBlock(
                    dim,
                    dim,
                    dtype=dtype,
                    device=device,
                    prefer_nvidia_fcrs=prefer_nvidia_fcrs,
                ),
                WanResidualBlock(
                    dim,
                    dim,
                    dtype=dtype,
                    device=device,
                    prefer_nvidia_fcrs=prefer_nvidia_fcrs,
                ),
            ]
        )
        self.attentions = LayerList(
            [WanAttentionBlock(dim, dtype=dtype, device=device)]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class WanUpsample2d(Module):
    """Nearest-neighbor 2D upsample by factor 2."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: TensorValue) -> TensorValue:
        n = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        # Reshape to [N, C, H, 1, W, 1], duplicate along new axes, reshape back
        x = ops.reshape(x, [n, c, h, 1, w, 1])
        x = ops.concat([x, x], axis=3)  # [N, C, H, 2, W, 1]
        x = ops.concat([x, x], axis=5)  # [N, C, H, 2, W, 2]
        return ops.reshape(x, [n, c, h * 2, w * 2])


class WanResample(Module):
    """Wan decoder upsampling module."""

    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int | None = None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        self._out_c = upsample_out_dim

        self.time_conv: WanCausalConv3d | None = None
        self.resample = LayerList(
            [
                WanUpsample2d(),
                WanConv2dPermuted(
                    in_channels=dim,
                    out_channels=upsample_out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dtype=dtype,
                    device=device,
                    has_bias=True,
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

    def __call__(self, x: TensorValue) -> TensorValue:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        if self.mode == "upsample3d":
            if self.time_conv is None:
                raise ValueError("time_conv is required for upsample3d mode")
            x = self.time_conv(x)
            # x: [b, 2*dim, t, h, w] -> interleave temporal frames
            x = ops.reshape(x, [b, 2, self.dim, t, h, w])
            x = ops.permute(x, [0, 2, 3, 1, 4, 5])  # [b, dim, t, 2, h, w]
            t = t * 2
            x = ops.reshape(x, [b, self.dim, t, h, w])

        # Per-frame 2D upsample + conv
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, t, c, h, w]
        x = ops.reshape(x, [b * t, self.dim, h, w])
        x = self.resample[0](x)  # WanUpsample2d: [b*t, dim, h*2, w*2]
        # WanConv2dPermuted handles NCHW->NHWC->conv->NCHW internally.
        x = self.resample[1](x)  # [b*t, out_c, h*2, w*2]

        x = ops.reshape(x, [b, t, self._out_c, h * 2, w * 2])
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, out_c, t, h*2, w*2]
        return x


class WanUpBlock(Module):
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
        super().__init__()
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
        self.resnets = LayerList(resnets)

        self.upsamplers: LayerList | None = None
        if upsample_mode is not None:
            self.upsamplers = LayerList(
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

    def __call__(self, x: TensorValue) -> TensorValue:
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            x = self.upsamplers[0](x)

        return x


class WanDecoder3d(Module):
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
        super().__init__()
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

        self.up_blocks = LayerList(up_blocks)

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

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.conv_in(x)
        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.norm_out(x)
        x = ops.silu(x)
        x = self.conv_out(x)
        return x


class WanResidualBlockCached(Module):
    """Wan residual block with explicit cache I/O for conv1/conv2."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
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

    def __call__(
        self,
        x: TensorValue,
        cache1_in: TensorValue | None = None,
        cache2_in: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        residual = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )

        x = ops.silu(self.norm1(x))
        if cache1_in is None:
            cache1_in = _zero_cache_for(x)
        x, cache1_out = self.conv1.forward_cached(x, cache1_in)

        x = ops.silu(self.norm2(x))
        if cache2_in is None:
            cache2_in = _zero_cache_for(x)
        x, cache2_out = self.conv2.forward_cached(x, cache2_in)
        return x + residual, cache1_out, cache2_out


class WanMidBlockCached(Module):
    """Middle decoder block with cache threading."""

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.resnets = LayerList(
            [
                WanResidualBlockCached(dim, dim, dtype=dtype, device=device),
                WanResidualBlockCached(dim, dim, dtype=dtype, device=device),
            ]
        )
        self.attentions = LayerList(
            [WanAttentionBlock(dim, dtype=dtype, device=device)]
        )

    def __call__(
        self, x: TensorValue, *cache_inputs: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue, TensorValue]:
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


class WanResampleCached(Module):
    """Wan upsample3d module with explicit cache I/O."""

    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int | None = None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
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
        self.resample = LayerList(
            [
                WanUpsample2d(),
                WanConv2dPermuted(
                    in_channels=dim,
                    out_channels=upsample_out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                ),
            ]
        )

    def __call__(
        self,
        x: TensorValue,
        cache_in: TensorValue | None = None,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, TensorValue]:
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
            x = ops.reshape(x, [b, 2, self.dim, t, h, w])
            x = ops.permute(x, [0, 2, 3, 1, 4, 5])
            t = t * 2
            x = ops.reshape(x, [b, self.dim, t, h, w])

        x = ops.permute(x, [0, 2, 1, 3, 4])
        x = ops.reshape(x, [b * t, self.dim, h, w])
        x = self.resample[0](x)
        x = self.resample[1](x)
        x = ops.reshape(x, [b, t, self._out_c, h * 2, w * 2])
        x = ops.permute(x, [0, 2, 1, 3, 4])
        return x, cache_out


class WanUpBlockCached(Module):
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
        super().__init__()
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
        self.resnets = LayerList(resnets)

        self._has_temporal_upsample = upsample_mode == "upsample3d"
        self.cache_slots = len(resnets) * 2 + (
            1 if self._has_temporal_upsample else 0
        )

        self.upsamplers: LayerList | None = None
        if upsample_mode is not None:
            if upsample_mode == "upsample3d":
                upsampler: Module = WanResampleCached(
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

            self.upsamplers = LayerList([upsampler])

    def __call__(
        self,
        x: TensorValue,
        *cache_inputs: TensorValue,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, ...]:
        if len(cache_inputs) not in (0, self.cache_slots):
            raise ValueError(
                f"WanUpBlockCached expected 0 or {self.cache_slots} cache tensors, got {len(cache_inputs)}"
            )

        use_cache_inputs = len(cache_inputs) == self.cache_slots
        cache_outputs: list[TensorValue] = []
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


class WanDecoder3dCached(Module):
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
        super().__init__()
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

        self.up_blocks = LayerList(up_blocks)
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

    def __call__(
        self,
        x: TensorValue,
        *cache_inputs: TensorValue,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, ...]:
        if len(cache_inputs) not in (0, WAN_DECODER_CACHE_SLOTS):
            raise ValueError(
                "WanDecoder3dCached expected 0 or "
                f"{WAN_DECODER_CACHE_SLOTS} cache tensors, got {len(cache_inputs)}"
            )

        use_cache_inputs = len(cache_inputs) == WAN_DECODER_CACHE_SLOTS
        cache_outputs: list[TensorValue] = []
        cache_idx = 0

        conv_in_cache = cache_inputs[cache_idx] if use_cache_inputs else None
        if conv_in_cache is None:
            conv_in_cache = _zero_cache_for(x)
        x, cache_out = self.conv_in.forward_cached(x, conv_in_cache)
        cache_outputs.append(cache_out)
        cache_idx += 1

        mid_cache_inputs: tuple[TensorValue, ...] = (
            tuple(cache_inputs[cache_idx : cache_idx + 4])
            if use_cache_inputs
            else ()
        )
        mid_outputs = self.mid_block(x, *mid_cache_inputs)
        x = mid_outputs[0]
        cache_outputs.extend(mid_outputs[1:])
        cache_idx += 4

        for up_block in self.up_blocks:
            block_cache_inputs: tuple[TensorValue, ...] = (
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
        x = ops.silu(x)
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


class _WanVAEPostQuantConv(Module):
    """Standalone post-quant conv graph (k=1, frame-independent)."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self.post_quant_conv = WanCausalConv3d(
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            kernel_size=1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )

    def __call__(self, z: TensorValue) -> TensorValue:
        return self.post_quant_conv(z)


class _WanVAEDecoderFirstFrameCached(Module):
    """First-frame decoder graph returning pixels + initialized caches."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
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

    def __call__(self, z: TensorValue) -> tuple[TensorValue, ...]:
        outputs = self.decoder(z, first_chunk=True)
        x = outputs[0]
        x = ops.max(x, -1.0)
        x = ops.min(x, 1.0)
        return (x, *outputs[1:])


class _WanVAEDecoderRestFrameCached(Module):
    """Per-frame decoder graph with cache feedback for frames 1..T-1."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
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

    def __call__(
        self, z: TensorValue, *cache_inputs: TensorValue
    ) -> tuple[TensorValue, ...]:
        outputs = self.decoder(z, *cache_inputs, first_chunk=False)
        x = outputs[0]
        x = ops.max(x, -1.0)
        x = ops.min(x, 1.0)
        return (x, *outputs[1:])


class WanVAEDecoder(Module):
    """Wan VAE decoder graph used by AutoencoderKLWanModel."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
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

    def __call__(self, z: TensorValue) -> TensorValue:
        x = self.post_quant_conv(z)
        x = self.decoder(x)
        x = ops.max(x, -1.0)
        x = ops.min(x, 1.0)
        return x


class _WanVAEDecoderFirstFrame(Module):
    """Wan VAE decoder for the FIRST latent frame.

    Identical to WanVAEDecoder but ALL temporal upsamples are replaced
    with spatial-only upsample2d (time_conv is omitted).  This means
    T=1 in -> T=1 out, matching the diffusers feat_cache behavior where
    the first frame skips temporal upsampling.
    """

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
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

    def __call__(self, z: TensorValue) -> TensorValue:
        x = self.post_quant_conv(z)
        x = self.decoder(x)
        x = ops.max(x, -1.0)
        x = ops.min(x, 1.0)
        return x


class _CachedFramewiseDecoder:
    """Diffusers-like cached frame-by-frame VAE decode."""

    def __init__(
        self,
        post_quant_conv: Model,
        first_frame_decoder: Model,
        rest_frame_decoder: Model,
    ) -> None:
        self.post_quant_conv = post_quant_conv
        self.first_frame_decoder = first_frame_decoder
        self.rest_frame_decoder = rest_frame_decoder

    def __call__(self, latents_5d: Buffer) -> Buffer:
        t_total = int(latents_5d.shape[2])
        if t_total <= 0:
            raise ValueError("Expected non-empty temporal dimension for decode")

        cpu = CPU()
        decoded_frames: list[np.ndarray] = []
        caches: list[Buffer] | None = None

        # Slice temporal frames via numpy (Buffer doesn't support fancy indexing)
        latents_cpu_np = _buffer_to_numpy_f32(latents_5d, cpu)

        for t_idx in range(t_total):
            z_t_np = np.ascontiguousarray(
                latents_cpu_np[:, :, t_idx : t_idx + 1, :, :]
            )
            z_t_buf = _numpy_f32_to_buffer(
                z_t_np, latents_5d.dtype, latents_5d.device
            )

            pqc_outputs = self.post_quant_conv(z_t_buf)
            if len(pqc_outputs) != 1:
                raise ValueError(
                    f"Expected 1 output from post_quant_conv, got {len(pqc_outputs)}"
                )
            z_t_buf = pqc_outputs[0]

            if t_idx == 0:
                outputs = self.first_frame_decoder(z_t_buf)
            else:
                if caches is None:
                    raise ValueError(
                        "Cached framewise decoder expected caches after first frame."
                    )
                outputs = self.rest_frame_decoder(z_t_buf, *caches)

            if len(outputs) != 1 + WAN_DECODER_CACHE_SLOTS:
                raise ValueError(
                    "Cached framewise decoder produced "
                    f"{len(outputs)} tensors; expected {1 + WAN_DECODER_CACHE_SLOTS}."
                )

            decoded_buf = outputs[0]
            caches = list(outputs[1:])
            decoded_frames.append(_buffer_to_numpy_f32(decoded_buf, cpu))

        stitched = np.ascontiguousarray(np.concatenate(decoded_frames, axis=2))
        return Buffer.from_numpy(stitched)


_VAEFramewiseKey = tuple[int, int, int, int]


class WanDownResample(Module):
    """Wan encoder downsampling module.

    Matches diffusers WanResample downsample modes:
    - downsample2d: ZeroPad2d + Conv2d(stride=2) per frame
    - downsample3d: same spatial + CausalConv3d(stride=(2,1,1)) temporal
    """

    def __init__(
        self,
        dim: int,
        mode: str,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        prefer_nvidia_fcrs: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # Spatial: ZeroPad2d(0,1,0,1) + Conv2d(stride=2, padding=0)
        # Asymmetric padding: right=1, bottom=1 only.
        # Use index [1] to match state_dict key "resample.1".
        self.resample = LayerList(
            [
                WanUpsample2d(),  # Dummy at index 0 (no weights, not called)
                WanConv2dPermuted(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=2,
                    padding=0,  # We do manual asymmetric pad in __call__
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                ),
            ]
        )

        self.time_conv: WanCausalConv3d | None = None
        if mode == "downsample3d":
            self.time_conv = WanCausalConv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
                dtype=dtype,
                device=device,
                has_bias=True,
                # Encoder temporal downsample is the only conv pattern that
                # currently reproduces cuDNN aborts in VAE encode.
                prefer_nvidia_fcrs=False,
            )
        elif mode != "downsample2d":
            raise ValueError(f"Unsupported WanDownResample mode: {mode}")

    def __call__(self, x: TensorValue) -> TensorValue:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        if self.mode == "downsample3d":
            if self.time_conv is None:
                raise ValueError(
                    "time_conv is required for downsample3d mode"
                )
            # Temporal downsample via strided causal conv
            x = self.time_conv(x)
            t = x.shape[2]

        # Per-frame spatial downsample: ZeroPad2d(0,1,0,1) + Conv2d(stride=2)
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, t, c, h, w]
        x = ops.reshape(x, [b * t, self.dim, h, w])
        # ZeroPad2d(left=0, right=1, top=0, bottom=1) on NCHW
        # paddings format: [N_before, N_after, C_before, C_after, H_before, H_after, W_before, W_after]
        x = ops.pad(x, [0, 0, 0, 0, 0, 1, 0, 1])
        x = self.resample[1](x)  # Conv2d stride=2, padding=0
        new_h = (h + 1) // 2
        new_w = (w + 1) // 2
        x = ops.reshape(x, [b, t, self.dim, new_h, new_w])
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, dim, t, h/2, w/2]

        return x


class WanDownResampleCached(Module):
    """Encoder downsample with temporal cache for chunked encoding.

    Matches diffusers' WanResample cache behavior for the encoder:
    - downsample2d: spatial only, no temporal cache
    - downsample3d first chunk: spatial downsample, skip time_conv, cache last frame
    - downsample3d rest chunk: spatial downsample, prepend cached frame, apply time_conv

    Spatial downsample is done FIRST (matching diffusers order), then temporal.
    """

    cache_slots: int

    def __init__(
        self,
        dim: int,
        mode: str,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        self._has_temporal = mode == "downsample3d"
        self.cache_slots = 1 if self._has_temporal else 0

        self.resample = LayerList(
            [
                WanUpsample2d(),  # Dummy at index 0 (match weight naming)
                WanConv2dPermuted(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                ),
            ]
        )

        self.time_conv: WanCausalConv3d | None = None
        if self._has_temporal:
            self.time_conv = WanCausalConv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
                dtype=dtype,
                device=device,
                has_bias=True,
                prefer_nvidia_fcrs=False,
            )
        elif mode != "downsample2d":
            raise ValueError(f"Unsupported WanDownResampleCached mode: {mode}")

    def __call__(
        self,
        x: TensorValue,
        *,
        cache_in: TensorValue | None = None,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, ...]:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        # Spatial downsample first (matching diffusers order)
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, t, c, h, w]
        x = ops.reshape(x, [b * t, self.dim, h, w])
        x = ops.pad(x, [0, 0, 0, 0, 0, 1, 0, 1])  # ZeroPad2d(0,1,0,1)
        x = self.resample[1](x)  # Conv2d stride=2
        new_h = (h + 1) // 2
        new_w = (w + 1) // 2
        x = ops.reshape(x, [b, t, self.dim, new_h, new_w])
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, c, t, h', w']

        if self._has_temporal:
            assert self.time_conv is not None
            cache_out = x[:, :, -1:, :, :]  # Last frame after spatial
            if first_chunk:
                # Skip time_conv, return spatial output + cache
                return x, cache_out
            else:
                assert cache_in is not None
                # Prepend cached last frame, apply time_conv
                x_cat = ops.concat([cache_in, x], axis=2)
                x = self.time_conv(x_cat)
                return x, cache_out

        return (x,)


class WanDownBlock(Module):
    """Wan encoder down block (mirror of WanUpBlock)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        downsample_mode: str | None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
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
        self.resnets = LayerList(resnets)

        self.downsamplers: LayerList | None = None
        if downsample_mode is not None:
            self.downsamplers = LayerList(
                [
                    WanDownResample(
                        out_dim,
                        mode=downsample_mode,
                        dtype=dtype,
                        device=device,
                    )
                ]
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        for resnet in self.resnets:
            x = resnet(x)

        if self.downsamplers is not None:
            x = self.downsamplers[0](x)

        return x


class WanEncoder3d(Module):
    """Wan 3D encoder module (mirror of WanDecoder3d).

    Uses a flat ModuleList for down_blocks to match the diffusers
    safetensors key naming (encoder.down_blocks.{i}.{conv1,norm1,...}).
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        in_channels: int = 3,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: tuple[bool, ...] = (False, True, True),
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()

        dims = [dim * u for u in [1, *list(dim_mult)]]

        self.conv_in = WanCausalConv3d(
            in_channels,
            dims[0],
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
            prefer_nvidia_fcrs=False,
        )

        # Flat ModuleList matching diffusers weight naming:
        # down_blocks.{0,1} = ResidualBlock (first level, 2 blocks)
        # down_blocks.2 = Resample (downsample)
        # down_blocks.{3,4} = ResidualBlock (second level)
        # down_blocks.5 = Resample ...etc
        down_blocks: list[Module] = []
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            for j in range(num_res_blocks):
                down_blocks.append(
                    WanResidualBlock(
                        in_dim if j == 0 else out_dim,
                        out_dim,
                        dtype=dtype,
                        device=device,
                        prefer_nvidia_fcrs=False,
                    )
                )
            down_flag = i != len(dim_mult) - 1
            if down_flag:
                mode = (
                    "downsample3d" if temperal_downsample[i] else "downsample2d"
                )
                down_blocks.append(
                    WanDownResample(
                        out_dim,
                        mode=mode,
                        dtype=dtype,
                        device=device,
                        prefer_nvidia_fcrs=False,
                    )
                )

        self.down_blocks = LayerList(down_blocks)

        final_dim = dims[-1]
        self.mid_block = WanMidBlock(
            final_dim,
            dtype=dtype,
            device=device,
            prefer_nvidia_fcrs=False,
        )

        self.norm_out = WanRMSNorm(
            final_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        # Output 2*z_dim for mean + logvar
        self.conv_out = WanCausalConv3d(
            final_dim,
            z_dim * 2,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
            prefer_nvidia_fcrs=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)
        x = self.norm_out(x)
        x = ops.silu(x)
        x = self.conv_out(x)
        return x


class WanEncoder3dCached(Module):
    """Chunked encoder with explicit cache I/O for temporal context.

    Uses a flat ModuleList for down_blocks (matching WanEncoder3d weight naming).
    Each chunk processes either 1 frame (first) or CHUNK_SIZE frames (rest).
    Temporal context is maintained via cache tensors passed between chunks.
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        in_channels: int = 3,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: tuple[bool, ...] = (False, True, True),
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self._dim = dim
        self._in_channels = in_channels
        self._dim_mult = dim_mult
        self._num_res_blocks = num_res_blocks
        self._temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1, *list(dim_mult)]]

        self.conv_in = WanCausalConv3dCached(
            in_channels,
            dims[0],
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        # Flat list matching diffusers weight naming
        down_blocks: list[Module] = []
        self._block_cache_slots: list[int] = []
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            for j in range(num_res_blocks):
                down_blocks.append(
                    WanResidualBlockCached(
                        in_dim if j == 0 else out_dim,
                        out_dim,
                        dtype=dtype,
                        device=device,
                    )
                )
                self._block_cache_slots.append(2)
            down_flag = i != len(dim_mult) - 1
            if down_flag:
                mode = (
                    "downsample3d"
                    if temperal_downsample[i]
                    else "downsample2d"
                )
                ds = WanDownResampleCached(
                    out_dim,
                    mode=mode,
                    dtype=dtype,
                    device=device,
                )
                down_blocks.append(ds)
                self._block_cache_slots.append(ds.cache_slots)

        self.down_blocks = LayerList(down_blocks)

        final_dim = dims[-1]
        self.mid_block = WanMidBlockCached(
            final_dim, dtype=dtype, device=device
        )

        self.norm_out = WanRMSNorm(
            final_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv_out = WanCausalConv3dCached(
            final_dim,
            z_dim * 2,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    @property
    def total_cache_slots(self) -> int:
        return 1 + sum(self._block_cache_slots) + 4 + 1

    def cache_shapes(
        self, batch_size: int, height: int, width: int
    ) -> list[list[int]]:
        """Compute cache shapes for this encoder configuration."""
        dims = [self._dim * u for u in [1, *list(self._dim_mult)]]
        h, w = height, width
        shapes: list[list[int]] = []

        # conv_in cache
        shapes.append([batch_size, self._in_channels, CACHE_T, h, w])

        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            for j in range(self._num_res_blocks):
                block_in = in_dim if j == 0 else out_dim
                # cache1: input channels to conv1
                shapes.append([batch_size, block_in, CACHE_T, h, w])
                # cache2: input channels to conv2
                shapes.append([batch_size, out_dim, CACHE_T, h, w])

            down_flag = i != len(self._dim_mult) - 1
            if down_flag:
                new_h = (h + 1) // 2
                new_w = (w + 1) // 2
                if self._temperal_downsample[i]:
                    # Downsample3d cache: 1 frame at post-spatial resolution
                    shapes.append(
                        [batch_size, out_dim, 1, new_h, new_w]
                    )
                h, w = new_h, new_w

        # mid_block caches (2 resnets × 2 convs)
        final_dim = dims[-1]
        for _ in range(4):
            shapes.append([batch_size, final_dim, CACHE_T, h, w])

        # conv_out cache
        shapes.append([batch_size, final_dim, CACHE_T, h, w])

        assert len(shapes) == self.total_cache_slots, (
            f"cache_shapes produced {len(shapes)}, "
            f"expected {self.total_cache_slots}"
        )
        return shapes

    def __call__(
        self,
        x: TensorValue,
        *cache_inputs: TensorValue,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, ...]:
        use_cache = len(cache_inputs) == self.total_cache_slots
        if len(cache_inputs) not in (0, self.total_cache_slots):
            raise ValueError(
                f"WanEncoder3dCached expected 0 or {self.total_cache_slots} "
                f"cache tensors, got {len(cache_inputs)}"
            )

        cache_outputs: list[TensorValue] = []
        idx = 0

        # conv_in
        c_in = cache_inputs[idx] if use_cache else _zero_cache_for(x)
        x, c_out = self.conv_in.forward_cached(x, c_in)
        cache_outputs.append(c_out)
        idx += 1

        # down_blocks (flat list of ResidualBlockCached and DownResampleCached)
        block_idx = 0
        for block in self.down_blocks:
            if isinstance(block, WanResidualBlockCached):
                c1 = cache_inputs[idx] if use_cache else None
                c2 = cache_inputs[idx + 1] if use_cache else None
                x, co1, co2 = block(x, c1, c2)
                cache_outputs.extend([co1, co2])
                idx += 2
            elif isinstance(block, WanDownResampleCached):
                if block._has_temporal:
                    c = cache_inputs[idx] if use_cache else None
                    x, co = block(
                        x, cache_in=c, first_chunk=first_chunk
                    )
                    cache_outputs.append(co)
                    idx += 1
                else:
                    (x,) = block(x)
            block_idx += 1

        # mid_block
        mid_caches: tuple[TensorValue, ...] = (
            tuple(cache_inputs[idx : idx + 4]) if use_cache else ()
        )
        mid_out = self.mid_block(x, *mid_caches)
        x = mid_out[0]
        cache_outputs.extend(mid_out[1:])
        idx += 4

        # norm + silu + conv_out
        x = ops.silu(self.norm_out(x))
        c_in = cache_inputs[idx] if use_cache else _zero_cache_for(x)
        x, c_out = self.conv_out.forward_cached(x, c_in)
        cache_outputs.append(c_out)

        assert len(cache_outputs) == self.total_cache_slots, (
            f"Produced {len(cache_outputs)} caches, "
            f"expected {self.total_cache_slots}"
        )
        return (x, *cache_outputs)


class _WanVAEEncoder(Module):
    """Wrapper for VAE encoder graph compilation.

    Includes quant_conv (1x1 conv applied after encoder output).
    """

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self.encoder = WanEncoder3d(
            dim=config.base_dim,
            z_dim=config.z_dim,
            in_channels=3,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            temperal_downsample=config.temperal_downsample,
            dtype=config.dtype,
            device=config.device,
        )
        z2 = config.z_dim * 2
        self.quant_conv = WanCausalConv3d(
            z2,
            z2,
            1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
            prefer_nvidia_fcrs=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        h = self.encoder(x)
        return self.quant_conv(h)


class _WanVAEEncoderFirstChunk(Module):
    """First-chunk encoder graph: 1 frame in, mean latent + caches out."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self._z_dim = config.z_dim
        self.encoder = WanEncoder3dCached(
            dim=config.base_dim,
            z_dim=config.z_dim,
            in_channels=3,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            temperal_downsample=config.temperal_downsample,
            dtype=config.dtype,
            device=config.device,
        )
        z2 = config.z_dim * 2
        self.quant_conv = WanCausalConv3d(
            z2,
            z2,
            1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> tuple[TensorValue, ...]:
        outputs = self.encoder(x, first_chunk=True)
        moments = self.quant_conv(outputs[0])
        # Extract mean in-graph to avoid GPU→CPU transfer of full moments
        mean = moments[:, : self._z_dim, :, :, :]
        return (mean, *outputs[1:])


class _WanVAEEncoderRestChunk(Module):
    """Rest-chunk encoder graph: CHUNK_SIZE frames + caches in, mean latent + caches out."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self._z_dim = config.z_dim
        self.encoder = WanEncoder3dCached(
            dim=config.base_dim,
            z_dim=config.z_dim,
            in_channels=3,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            temperal_downsample=config.temperal_downsample,
            dtype=config.dtype,
            device=config.device,
        )
        z2 = config.z_dim * 2
        self.quant_conv = WanCausalConv3d(
            z2,
            z2,
            1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )

    def __call__(
        self, x: TensorValue, *cache_inputs: TensorValue
    ) -> tuple[TensorValue, ...]:
        outputs = self.encoder(x, *cache_inputs, first_chunk=False)
        moments = self.quant_conv(outputs[0])
        mean = moments[:, : self._z_dim, :, :, :]
        return (mean, *outputs[1:])


class _CachedChunkedEncoder:
    """Diffusers-like chunked encoder: first frame, then 4-frame chunks.

    Mean is extracted in-graph so only z_dim channels are transferred
    back to CPU per chunk (instead of 2*z_dim moments).
    """

    def __init__(
        self,
        first_chunk_model: Model,
        rest_chunk_model: Model,
    ) -> None:
        self.first_chunk_model = first_chunk_model
        self.rest_chunk_model = rest_chunk_model

    def __call__(
        self,
        video_np: np.ndarray,
        target_dtype: DType,
        device: Device,
    ) -> Buffer:
        """Encode video from numpy, avoiding redundant GPU→CPU round-trips.

        Args:
            video_np: Video as float32 numpy [B, 3, T, H, W].
            target_dtype: Model dtype (e.g. bfloat16) for GPU buffers.
            device: Target GPU device.
        """
        t_total = video_np.shape[2]
        cpu = CPU()

        latent_chunks: list[np.ndarray] = []
        caches: list[Buffer] | None = None

        num_chunks = 1 + (t_total - 1) // WAN_ENCODER_CHUNK_SIZE

        for i in range(num_chunks):
            if i == 0:
                chunk_np = np.ascontiguousarray(video_np[:, :, :1])
            else:
                start = 1 + WAN_ENCODER_CHUNK_SIZE * (i - 1)
                end = 1 + WAN_ENCODER_CHUNK_SIZE * i
                chunk_np = np.ascontiguousarray(video_np[:, :, start:end])

            chunk_buf = _numpy_f32_to_buffer(chunk_np, target_dtype, device)

            if i == 0:
                outputs = self.first_chunk_model.execute(chunk_buf)
            else:
                assert caches is not None
                outputs = self.rest_chunk_model.execute(chunk_buf, *caches)

            # Mean already extracted in-graph — just transfer to CPU
            latent_chunks.append(_buffer_to_numpy_f32(outputs[0], cpu))
            caches = list(outputs[1:])

        full_latent = np.ascontiguousarray(
            np.concatenate(latent_chunks, axis=2)
        )
        return _numpy_f32_to_buffer(full_latent, target_dtype, device)


class AutoencoderKLWanModel(ComponentModel):
    """Wan VAE model using MAX-native 3D modules (decoder + optional encoder)."""

    MAX_CACHED_DECODERS: int = 4

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession | None = None,
        eager_load: bool = True,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = AutoencoderKLWanConfig.generate(config, encoding, devices)
        self._framewise_decoder_cache: dict[
            _VAEFramewiseKey, _CachedFramewiseDecoder
        ] = {}
        self._decoder_state_dict: dict[str, Any] | None = None
        self._encoder_state_dict: dict[str, Any] | None = None
        self._chunked_encoder_cache: dict[
            tuple[int, int, int], _CachedChunkedEncoder
        ] = {}
        self._session = session or InferenceSession(devices=devices)
        self._load_lock = threading.Lock()
        self._compile_lock = threading.Lock()
        if eager_load:
            self.load_model()

    def load_model(self) -> Callable[[Buffer], Buffer]:
        with self._load_lock:
            if self._decoder_state_dict is not None:
                return self.decode_4d

            decoder_state_dict: dict[str, Any] = {}
            target_dtype = self.config.dtype

            assert self.weights is not None
            weights_obj: Any = self.weights

            encoder_state_dict: dict[str, Any] = {}
            has_encoder = False

            for key, value in weights_obj.items():
                is_decoder = key.startswith("decoder.") or key.startswith(
                    "post_quant_conv."
                )
                is_encoder = key.startswith("encoder.") or key.startswith(
                    "quant_conv."
                )
                if not (is_decoder or is_encoder):
                    continue

                weight_data = value.data()

                # Wan checkpoints store conv filters in PyTorch layout.
                # Keep 3D convs in FCQRS on NVIDIA GPUs to use the cuDNN path;
                # fall back to MAX QRSCF elsewhere.
                # Resample Conv2d (permute=True equivalent) stays in FCRS.
                # Attention Conv2d (permute=False equivalent) needs RSCF [H,W,in,out].
                if key.endswith(".weight") and len(weight_data.shape) == 5:
                    use_native_layout = not _use_nvidia_fcrs_conv3d(
                        self.config.device
                    )
                    # Encoder time_conv (stride-2 temporal) uses native layout
                    # to avoid cuDNN stability issues. All other encoder conv3d
                    # weights use cuDNN (FCQRS) via the chunked encoder.
                    if "time_conv" in key and (
                        key.startswith("encoder.")
                        or key.startswith("quant_conv.")
                    ):
                        use_native_layout = True
                    if use_native_layout:
                        weight_data = np.ascontiguousarray(
                            np.from_dlpack(weight_data).transpose(2, 3, 4, 1, 0)
                        )
                if key.endswith(".weight") and len(weight_data.shape) == 4:
                    is_resample_conv = "resample" in key
                    if not is_resample_conv:
                        weight_data = np.ascontiguousarray(
                            np.from_dlpack(weight_data).transpose(2, 3, 1, 0)
                        )

                if is_decoder:
                    decoder_state_dict[key] = weight_data
                if is_encoder:
                    encoder_state_dict[key] = weight_data
                    has_encoder = True

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
                for key in encoder_state_dict:
                    encoder_state_dict[key] = cast_dlpack_to(
                        encoder_state_dict[key],
                        DType.float32,
                        target_dtype,
                        cpu_device,
                    )

            # Defer compilation to first decode_5d() call so we have concrete
            # dimensions. Keep a small shape cache for mixed-resolution serving.
            self._decoder_state_dict = decoder_state_dict
            if has_encoder:
                self._encoder_state_dict = encoder_state_dict
            # Free the raw Weights object now that we have the state dict.
            self.weights = None  # type: ignore[assignment]

            return self.decode_4d

    def _compile_framewise_cached_decoder(
        self, shape: _VAEFramewiseKey
    ) -> _CachedFramewiseDecoder:
        cfg = self.config
        sd = self._decoder_state_dict
        assert sd is not None, "decoder state dict not initialized"
        batch_size, z_dim, latent_h, latent_w = shape

        latent_1f_type = TensorType(
            cfg.dtype,
            [batch_size, z_dim, 1, latent_h, latent_w],
            device=cfg.device,
        )

        # Post-quant conv
        pqc_module = _WanVAEPostQuantConv(cfg)
        pqc_module.load_state_dict(sd, weight_alignment=1, strict=False)
        with Graph(
            "wan_vae_pqc",
            input_types=[latent_1f_type],
        ) as pqc_graph:
            out = pqc_module(pqc_graph.inputs[0].tensor)
            pqc_graph.output(out)
        pqc_model = self._session.load(
            pqc_graph, weights_registry=pqc_module.state_dict()
        )

        # First-frame decoder
        first_module = _WanVAEDecoderFirstFrameCached(cfg)
        first_module.load_state_dict(sd, weight_alignment=1, strict=False)
        with Graph(
            "wan_vae_first_frame",
            input_types=[latent_1f_type],
        ) as first_graph:
            outputs = first_module(first_graph.inputs[0].tensor)
            first_graph.output(*outputs)
        first_model = self._session.load(
            first_graph, weights_registry=first_module.state_dict()
        )

        # Rest-frame decoder (with caches)
        cache_shapes = first_module.decoder.cache_shapes(
            batch_size=batch_size,
            latent_height=latent_h,
            latent_width=latent_w,
        )
        rest_input_types = [
            latent_1f_type,
            *[
                TensorType(cfg.dtype, cache_shape, device=cfg.device)
                for cache_shape in cache_shapes
            ],
        ]
        rest_module = _WanVAEDecoderRestFrameCached(cfg)
        rest_module.load_state_dict(sd, weight_alignment=1, strict=False)
        with Graph(
            "wan_vae_rest_frame",
            input_types=rest_input_types,
        ) as rest_graph:
            rest_inputs = [inp.tensor for inp in rest_graph.inputs]
            outputs = rest_module(rest_inputs[0], *rest_inputs[1:])
            rest_graph.output(*outputs)
        rest_model = self._session.load(
            rest_graph, weights_registry=rest_module.state_dict()
        )

        return _CachedFramewiseDecoder(
            post_quant_conv=pqc_model,
            first_frame_decoder=first_model,
            rest_frame_decoder=rest_model,
        )

    def _get_framewise_cached_decoder(
        self, shape: _VAEFramewiseKey
    ) -> _CachedFramewiseDecoder:
        with self._compile_lock:
            cached = self._framewise_decoder_cache.get(shape)
            if cached is not None:
                return cached

            decoder = self._compile_framewise_cached_decoder(shape)
            self._framewise_decoder_cache[shape] = decoder
            if len(self._framewise_decoder_cache) > self.MAX_CACHED_DECODERS:
                oldest_key = next(iter(self._framewise_decoder_cache))
                if oldest_key != shape:
                    self._framewise_decoder_cache.pop(oldest_key, None)
            return decoder

    def prepare_for_serving(self) -> None:
        # Materialize the decoder state dict eagerly, but keep framewise
        # decoder compilation deferred until the first decode request.
        self.load_model()

    def prewarm_for_latent_shape(
        self, shape: tuple[int, int, int, int, int]
    ) -> None:
        with Tracer("wan_vae_prewarm"):
            self.load_model()
            batch_size, z_dim, _frames, latent_h, latent_w = shape
            self._get_framewise_cached_decoder(
                (batch_size, z_dim, latent_h, latent_w)
            )

    def decode_5d(self, latents_5d: Buffer) -> Buffer:
        """Decode 5D latents [B, C, T, H, W] with the cached framewise path."""
        self.prepare_for_serving()
        shape = (
            int(latents_5d.shape[0]),
            int(latents_5d.shape[1]),
            int(latents_5d.shape[3]),
            int(latents_5d.shape[4]),
        )
        framewise_decoder = self._get_framewise_cached_decoder(shape)
        with Tracer("wan_vae_decode"):
            return framewise_decoder(latents_5d)

    def decode_4d(self, latents_4d: Buffer) -> Buffer:
        # Add temporal dimension: [B, C, H, W] -> [B, C, 1, H, W]
        shape_5d = (
            int(latents_4d.shape[0]),
            int(latents_4d.shape[1]),
            1,
            int(latents_4d.shape[2]),
            int(latents_4d.shape[3]),
        )
        z5d = latents_4d.view(dtype=latents_4d.dtype, shape=shape_5d)
        decoded_5d = self.decode_5d(z5d)
        # Remove temporal dimension from decoded output
        out_shape_4d = (
            int(decoded_5d.shape[0]),
            int(decoded_5d.shape[1]),
            int(decoded_5d.shape[3]),
            int(decoded_5d.shape[4]),
        )
        cpu = CPU()
        decoded_np = _buffer_to_numpy_f32(decoded_5d, cpu)
        return Buffer.from_numpy(
            np.ascontiguousarray(decoded_np[:, :, 0, :, :])
        )

    def decode(
        self, latents: Buffer, return_dict: bool = False
    ) -> tuple[Buffer]:
        del return_dict
        if latents.rank == 5:
            return (self.decode_5d(latents),)
        return (self.decode_4d(latents),)

    def _compile_chunked_encoder(
        self,
        batch_size: int,
        height: int,
        width: int,
    ) -> _CachedChunkedEncoder:
        """Compile chunked encoder graphs (first chunk + rest chunk)."""
        cfg = self.config
        sd = self._encoder_state_dict
        assert sd is not None, "encoder state dict not initialized"

        # First chunk: 1 frame, no cache inputs
        first_input_type = TensorType(
            cfg.dtype,
            [batch_size, 3, 1, height, width],
            device=cfg.device,
        )
        first_module = _WanVAEEncoderFirstChunk(cfg)
        first_module.load_state_dict(sd, weight_alignment=1, strict=False)

        with Graph(
            "wan_vae_enc_first", input_types=[first_input_type]
        ) as first_graph:
            outputs = first_module(first_graph.inputs[0].tensor)
            first_graph.output(*outputs)
        first_registry = first_module.state_dict()
        first_model = self._session.load(
            first_graph, weights_registry=first_registry
        )

        # Rest chunk: CHUNK_SIZE frames + cache inputs (shares weights)
        rest_input_type = TensorType(
            cfg.dtype,
            [batch_size, 3, WAN_ENCODER_CHUNK_SIZE, height, width],
            device=cfg.device,
        )
        cache_shapes = first_module.encoder.cache_shapes(
            batch_size, height, width
        )
        rest_input_types = [
            rest_input_type,
            *[
                TensorType(cfg.dtype, cs, device=cfg.device)
                for cs in cache_shapes
            ],
        ]
        rest_module = _WanVAEEncoderRestChunk(cfg)
        rest_module.load_state_dict(sd, weight_alignment=1, strict=False)
        with Graph(
            "wan_vae_enc_rest", input_types=rest_input_types
        ) as rest_graph:
            rest_inputs = [inp.tensor for inp in rest_graph.inputs]
            outputs = rest_module(rest_inputs[0], *rest_inputs[1:])
            rest_graph.output(*outputs)
        rest_model = self._session.load(
            rest_graph, weights_registry=rest_module.state_dict()
        )

        return _CachedChunkedEncoder(
            first_chunk_model=first_model,
            rest_chunk_model=rest_model,
        )

    def encode(self, video: Buffer) -> Buffer:
        """Encode a video tensor [B, 3, T, H, W] to latent space.

        Uses chunked encoding matching diffusers: first frame processed
        separately, then 4-frame chunks with temporal caching.
        Video is converted to numpy once for chunk slicing, avoiding
        redundant GPU→CPU→GPU round-trips.

        Returns the mean of the diagonal Gaussian (argmax mode),
        shape [B, z_dim, T_latent, H_latent, W_latent].
        """
        self.load_model()
        if self._encoder_state_dict is None:
            raise RuntimeError(
                "VAE encoder weights not available. "
                "Ensure the model checkpoint includes encoder weights."
            )

        batch_size = int(video.shape[0])
        height = int(video.shape[3])
        width = int(video.shape[4])
        shape_key = (batch_size, height, width)
        with self._compile_lock:
            encoder = self._chunked_encoder_cache.get(shape_key)
            if encoder is None:
                encoder = self._compile_chunked_encoder(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                )
                self._chunked_encoder_cache[shape_key] = encoder

        # Convert to numpy once for chunk slicing
        video_np = _buffer_to_numpy_f32(video, CPU())
        target_dtype = self.config.dtype
        device = self.devices[0]

        with Tracer("wan_vae_encode"):
            return encoder(video_np, target_dtype, device)

    def __call__(self, latents: Buffer) -> Buffer:
        if latents.rank == 5:
            return self.decode_5d(latents)
        return self.decode_4d(latents)
