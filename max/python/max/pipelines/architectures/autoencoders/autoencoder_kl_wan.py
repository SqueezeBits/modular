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
from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import Weights
from max.nn import Conv2d, Conv3d, Module, ModuleList
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces import CompileWrapper
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor

from .model_config import AutoencoderKLWanConfig

logger = logging.getLogger(__name__)


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


class _MonolithicFrameDecoder:
    """VAE decode using 2 monolithic compiled graphs.

    - first_decoder: spatial-only (no time_conv), T=1 -> T=1 (1 video frame)
    - rest_decoder: full temporal upsample, T=rest_T -> T=rest_T*4 video frames

    Total: 1 + (T-1)*4 = 4T-3 video frames.

    The rest decoder processes all remaining latent frames at once (or in
    chunks if VRAM is limited), amortizing the causal conv3d temporal
    padding overhead across many frames instead of paying it per-frame.
    """

    def __init__(
        self,
        first_decoder: CompileWrapper,
        rest_decoder: CompileWrapper,
        rest_chunk_size: int,
    ) -> None:
        self.first_decoder = first_decoder
        self.rest_decoder = rest_decoder
        self.rest_chunk_size = rest_chunk_size

    def __call__(self, latents_5d: Tensor) -> Tensor:
        """Decode [B, C, T, H, W] latents."""
        import time as _time

        T = int(latents_5d.shape[2])
        t0 = _time.perf_counter()

        # First frame: spatial-only (T=1 -> 1 video frame)
        z_first = latents_5d[:, :, 0:1, :, :]
        x_first = self.first_decoder(z_first)
        x_first_cpu = x_first.cast(DType.float32).to(CPU())
        parts: list[np.ndarray] = [np.from_dlpack(x_first_cpu)]
        logger.info(
            "First frame decoded in %.1fs, output T=%d",
            _time.perf_counter() - t0,
            parts[0].shape[2],
        )

        # Rest frames: full temporal upsample in chunks
        rest_T = T - 1
        chunk = self.rest_chunk_size
        n_chunks = (rest_T + chunk - 1) // chunk

        for ci in range(n_chunks):
            start = 1 + ci * chunk
            end = min(1 + (ci + 1) * chunk, T)
            z_rest = latents_5d[:, :, start:end, :, :]
            t_exec = _time.perf_counter()
            x_rest = self.rest_decoder(z_rest)
            t_done = _time.perf_counter()
            x_rest_cpu = x_rest.cast(DType.float32).to(CPU())
            t_cpu = _time.perf_counter()
            parts.append(np.from_dlpack(x_rest_cpu))
            logger.info(
                "Chunk %d/%d exec=%.2fs D2H=%.2fs elapsed=%.1fs",
                ci + 1,
                n_chunks,
                t_done - t_exec,
                t_cpu - t_done,
                t_cpu - t0,
            )

        result = np.concatenate(parts, axis=2)
        np.save('/tmp/wan_t2v_outputs/max_vae_raw_output.npy', result)
        logger.info('Saved raw VAE output: shape=%s range=[%.4f,%.4f]', result.shape, result.min(), result.max())
        total = _time.perf_counter() - t0
        logger.info(
            "VAE decode: %d latent -> %d video frames in %.1fs",
            T,
            result.shape[2],
            total,
        )
        return Tensor.from_dlpack(np.ascontiguousarray(result))


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
        self._mono_decoder: _MonolithicFrameDecoder | None = None
        self.load_model()

    def load_model(self) -> Callable[[Tensor], Tensor]:
        decoder_state_dict: dict[str, object] = {}
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
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
        self._mono_decoder = None
        # Free the raw Weights object now that we have the state dict.
        self.weights = None  # type: ignore[assignment]

        return self.decode_4d

    def _compile_monolithic_frame_decoder(
        self, shape: tuple[int, ...],
    ) -> None:
        """Compile 2 monolithic VAE decoder graphs for batched frame decode.

        - first_decoder: spatial-only (all temporal upsamples -> upsample2d).
          Weights exclude time_conv parameters.  T=1 in -> T=1 out.
        - rest_decoder: full decoder with upsample3d temporal upsamples.
          T=rest_chunk in -> T=rest_chunk*4 out (two temporal doublings).
          rest_chunk = T-1 to process all remaining frames in one shot.
        """
        import time as _time

        cfg = self.config
        sd = self._decoder_state_dict
        device_obj = self.devices[0]
        B, C, T_total, H, W = (int(s) for s in shape)
        # Larger chunks = better temporal context but more VRAM.
        # rest_chunk=5 balances quality and memory at 480p/720p.
        rest_chunk = 5
        logger.info(
            "Compiling VAE decoders: T_total=%d, rest_chunk=%d",
            T_total,
            rest_chunk,
        )

        # --- First-frame decoder (spatial-only, no time_conv) ---
        t0 = _time.perf_counter()
        first_input = TensorType(cfg.dtype, [B, C, 1, H, W], device=cfg.device)
        with F.lazy():
            first_model = _WanVAEDecoderFirstFrame(cfg)
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
            first_model, input_types=[first_input], weights=first_weights,
        )
        t1 = _time.perf_counter()
        logger.info(
            "Compiled first-frame decoder (spatial-only) in %.1fs, %d weights",
            t1 - t0,
            len(first_weights),
        )

        # --- Rest-frame decoder (full temporal upsample, batched) ---
        rest_input = TensorType(
            cfg.dtype, [B, C, rest_chunk, H, W], device=cfg.device,
        )
        with F.lazy():
            rest_model = WanVAEDecoder(cfg)
            rest_model.to(device_obj)
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
            rest_model, input_types=[rest_input], weights=rest_weights,
        )
        t2 = _time.perf_counter()
        logger.info(
            "Compiled rest-frame decoder (temporal, T=%d) in %.1fs, %d weights",
            rest_chunk,
            t2 - t1,
            len(rest_weights),
        )

        self._mono_decoder = _MonolithicFrameDecoder(
            first_decoder=first_compiled,
            rest_decoder=rest_compiled,
            rest_chunk_size=rest_chunk,
        )
        # Free state dict after compilation.
        del self._decoder_state_dict

    def decode_5d(self, latents_5d: Tensor) -> Tensor:
        """Decode 5D latents [B, C, T, H, W] frame-by-frame."""
        import time as _time

        shape = tuple(int(s) for s in latents_5d.shape)
        logger.info(
            "VAE decode input: shape=%s dtype=%s", shape, latents_5d.dtype
        )
        if self._mono_decoder is None:
            t0 = _time.perf_counter()
            self._compile_monolithic_frame_decoder(shape)
            logger.info(
                "VAE compile total: %.1fs", _time.perf_counter() - t0
            )
        assert self._mono_decoder is not None
        return self._mono_decoder(latents_5d)

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
