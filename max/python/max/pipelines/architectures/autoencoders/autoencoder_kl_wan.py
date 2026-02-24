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


class _VAEPreStage(Module[[Tensor], Tensor]):
    """VAE pre-processing: post_quant_conv + conv_in + mid_block."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        dim_mult = tuple(config.dim_mult)
        dims = [config.base_dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]
        self.post_quant_conv = WanCausalConv3d(
            config.z_dim, config.z_dim, 1, padding=0,
            dtype=config.dtype, device=config.device, has_bias=True,
        )
        self.conv_in = WanCausalConv3d(
            config.z_dim, dims[0], 3, padding=1,
            dtype=config.dtype, device=config.device, has_bias=True,
        )
        self.mid_block = WanMidBlock(
            dims[0], dtype=config.dtype, device=config.device,
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.post_quant_conv(z)
        x = self.conv_in(x)
        return self.mid_block(x)


class _VAEPostStage(Module[[Tensor], Tensor]):
    """VAE post-processing: last up_block + norm_out + silu + conv_out + clamp."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        out_channels: int,
        config: AutoencoderKLWanConfig,
    ) -> None:
        self.up_block = WanUpBlock(
            in_dim, out_dim, num_res_blocks, upsample_mode=None,
            dtype=config.dtype, device=config.device,
        )
        self.norm_out = WanRMSNorm(
            out_dim, images=False, dtype=config.dtype, device=config.device,
        )
        self.conv_out = WanCausalConv3d(
            out_dim, out_channels, 3, padding=1,
            dtype=config.dtype, device=config.device, has_bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_block(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        x = F.max(x, -1.0)
        return F.min(x, 1.0)


class _BlockLevelVAEDecoder:
    """Executes VAE decode as separate compiled stages.

    Each stage is compiled independently so only one stage's workspace
    is live at a time.  In sync mode, GPU memory is returned between
    stage executions.
    """

    def __init__(
        self,
        pre: CompileWrapper,
        up_blocks: list[CompileWrapper],
        post: CompileWrapper,
    ) -> None:
        self.pre = pre
        self.up_blocks = up_blocks
        self.post = post

    def __call__(self, latents: Tensor) -> Tensor:
        x = self.pre(latents)
        for block in self.up_blocks:
            x = block(x)
        return self.post(x)


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
        self.decoder_model: CompileWrapper | None = None
        self._block_decoder: _BlockLevelVAEDecoder | None = None
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
        self.decoder_model = None
        self._block_decoder = None
        # Free the raw Weights object now that we have the state dict.
        self.weights = None  # type: ignore[assignment]

        return self.decode_4d

    def _compile_decoder(self, shape: tuple[int, ...]) -> None:
        """Compile the VAE decoder graph for a concrete input shape."""
        input_type = TensorType(
            self.config.dtype,
            shape=list(shape),
            device=self.config.device,
        )
        with F.lazy():
            model = WanVAEDecoder(self.config)
            model.to(self.devices[0])

        # Verify weight key alignment between model and state dict.
        model_keys = {name for name, _ in model.parameters}
        sd_keys = set(self._decoder_state_dict.keys())
        only_model = sorted(model_keys - sd_keys)
        only_sd = sorted(sd_keys - model_keys)
        if only_model:
            logger.warning(
                "VAE weights MISSING from state dict (%d): %s",
                len(only_model),
                only_model[:20],
            )
        if only_sd:
            logger.warning(
                "VAE state dict keys NOT in model (%d): %s",
                len(only_sd),
                only_sd[:20],
            )
        if not only_model and not only_sd:
            logger.warning(
                "VAE weight keys match: %d parameters", len(model_keys)
            )

        self.decoder_model = CompileWrapper(
            model, input_types=[input_type], weights=self._decoder_state_dict
        )
        # Free state dict after compilation – weights are baked into the graph.
        del self._decoder_state_dict

    def _compile_decoder_block_level(self, shape: tuple[int, ...]) -> None:
        """Compile the VAE decoder as separate stages (block-level).

        Each stage is a separate compiled graph so only one stage's
        workspace is live at a time.  In sync mode, GPU memory is
        returned between stage executions.
        """
        cfg = self.config
        dtype = cfg.dtype
        dev = cfg.device
        device_obj = self.devices[0]
        dim = cfg.base_dim
        dim_mult = tuple(cfg.dim_mult)
        num_res = cfg.num_res_blocks
        temperal_upsample = tuple(reversed(cfg.temperal_downsample))
        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]

        sd = self._decoder_state_dict

        # --- Split state dict by stage ---
        pre_weights: dict[str, object] = {}
        up_block_weights: list[dict[str, object]] = [
            {} for _ in range(len(dim_mult))
        ]
        post_weights: dict[str, object] = {}

        for key, value in sd.items():
            if key.startswith("post_quant_conv."):
                pre_weights[key] = value
            elif key.startswith("decoder.conv_in."):
                pre_weights["conv_in." + key[len("decoder.conv_in."):]] = value
            elif key.startswith("decoder.mid_block."):
                pre_weights["mid_block." + key[len("decoder.mid_block."):]] = value
            elif key.startswith("decoder.up_blocks."):
                rest = key[len("decoder.up_blocks."):]
                dot = rest.index(".")
                idx = int(rest[:dot])
                sub_key = rest[dot + 1:]
                if idx < len(dim_mult) - 1:
                    # Up blocks 0..N-2 → separate stages
                    up_block_weights[idx]["up_block." + sub_key] = value
                else:
                    # Last up block → post stage
                    post_weights["up_block." + sub_key] = value
            elif key.startswith("decoder.norm_out."):
                post_weights["norm_out." + key[len("decoder.norm_out."):]] = value
            elif key.startswith("decoder.conv_out."):
                post_weights["conv_out." + key[len("decoder.conv_out."):]] = value

        B, C, T, H, W = (int(s) for s in shape)

        # --- Stage 0: pre (post_quant_conv + conv_in + mid_block) ---
        pre_input = TensorType(dtype, list(shape), device=dev)
        with F.lazy():
            pre_mod = _VAEPreStage(cfg)
            pre_mod.to(device_obj)
        pre_model = CompileWrapper(
            pre_mod, input_types=[pre_input], weights=pre_weights,
        )
        logger.info("Compiled pre stage (%d weights)", len(pre_weights))

        # Track shape through stages
        cur_C = dims[0]
        cur_T, cur_H, cur_W = T, H, W

        # --- Stages 1..N-2: up_blocks ---
        up_models: list[CompileWrapper] = []
        for i, (in_d, out_d) in enumerate(pairwise(dims)):
            if i >= len(dim_mult) - 1:
                break  # last up_block goes to post stage
            if i > 0:
                in_d = in_d // 2

            up_flag = i != len(dim_mult) - 1
            upsample_mode: str | None = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            block_input = TensorType(
                dtype, [B, cur_C, cur_T, cur_H, cur_W], device=dev,
            )
            with F.lazy():
                up_mod = WanUpBlock(
                    in_d, out_d, num_res, upsample_mode,
                    dtype=dtype, device=dev,
                )
                up_mod.to(device_obj)
            # Remap weights: "up_block.X" → "X" (WanUpBlock direct params)
            block_weights = {
                k[len("up_block."):]: v
                for k, v in up_block_weights[i].items()
            }
            up_models.append(CompileWrapper(
                up_mod, input_types=[block_input], weights=block_weights,
            ))
            logger.info(
                "Compiled up_block_%d (%d weights) input=[%d,%d,%d,%d,%d]",
                i, len(block_weights), B, cur_C, cur_T, cur_H, cur_W,
            )

            # Update shape for next stage
            if upsample_mode == "upsample3d":
                cur_T *= 2
                cur_H *= 2
                cur_W *= 2
                cur_C = out_d // 2
            elif upsample_mode == "upsample2d":
                cur_H *= 2
                cur_W *= 2
                cur_C = out_d // 2
            else:
                cur_C = out_d

        # --- Final stage: last up_block + norm_out + conv_out + clamp ---
        last_idx = len(dim_mult) - 1
        last_in_d = dims[last_idx]
        if last_idx > 0:
            last_in_d = last_in_d // 2
        last_out_d = dims[last_idx + 1]

        post_input = TensorType(
            dtype, [B, cur_C, cur_T, cur_H, cur_W], device=dev,
        )
        with F.lazy():
            post_mod = _VAEPostStage(
                last_in_d, last_out_d, num_res, cfg.out_channels, cfg,
            )
            post_mod.to(device_obj)
        # Remap weights: "up_block.X" → "up_block.X", "norm_out.X" → "norm_out.X"
        post_model = CompileWrapper(
            post_mod, input_types=[post_input], weights=post_weights,
        )
        logger.info(
            "Compiled post stage (%d weights) input=[%d,%d,%d,%d,%d]",
            len(post_weights), B, cur_C, cur_T, cur_H, cur_W,
        )

        self._block_decoder = _BlockLevelVAEDecoder(
            pre_model, up_models, post_model,
        )
        # Free state dict after compilation.
        del self._decoder_state_dict

    @staticmethod
    def upsample_temporal_linear(
        x: np.ndarray,
        scale_factor_temporal: int,
    ) -> np.ndarray:
        """Upsample [B, C, T, H, W] latents along time with linear interpolation."""
        if scale_factor_temporal <= 1 or x.shape[2] <= 1:
            return x

        num_frames = x.shape[2]
        out_frames = (num_frames - 1) * scale_factor_temporal + 1
        dst_idx = (
            np.arange(out_frames, dtype=np.float32) / scale_factor_temporal
        )
        left = np.floor(dst_idx).astype(np.int64)
        right = np.minimum(left + 1, num_frames - 1)
        alpha = (
            (dst_idx - left).reshape(1, 1, out_frames, 1, 1).astype(np.float32)
        )

        x_left = x[:, :, left, :, :]
        x_right = x[:, :, right, :, :]
        return ((1.0 - alpha) * x_left + alpha * x_right).astype(
            np.float32, copy=False
        )

    _TEMPORAL_SCALE: int = 4  # two upsample3d layers → ×4
    _OVERLAP: int = 1  # latent-frame overlap between chunks
    _MAX_LATENT_FRAMES: int = 6  # max latent frames per chunk

    def decode_5d(self, latents_5d: Tensor) -> Tensor:
        """Decode 5D latents using block-level compilation with temporal tiling.

        Each decoder stage (pre, up_blocks, post) is compiled and
        executed separately.  Window attention in the mid-block keeps
        memory bounded regardless of spatial resolution.  Temporal
        tiling keeps conv3d/upsample workspace bounded.
        """
        latents_5d = latents_5d.to(self.devices[0]).cast(self.config.dtype)
        total_t = int(latents_5d.shape[2])
        logger.info(
            "VAE decode input: shape=%s dtype=%s",
            tuple(int(s) for s in latents_5d.shape),
            latents_5d.dtype,
        )

        chunk_t = min(total_t, self._MAX_LATENT_FRAMES)

        # Compile block-level decoder on first call.
        if self._block_decoder is None:
            compile_shape = [int(s) for s in latents_5d.shape]
            compile_shape[2] = chunk_t
            logger.info(
                "Compiling block-level decoder for shape %s",
                tuple(compile_shape),
            )
            self._compile_decoder_block_level(tuple(compile_shape))
        assert self._block_decoder is not None

        # One-shot decode if input fits in a single chunk.
        if total_t <= self._MAX_LATENT_FRAMES:
            return self._block_decoder(latents_5d)

        # --- Temporal tiling ---
        overlap = min(self._OVERLAP, self._MAX_LATENT_FRAMES - 1)
        stride = max(1, self._MAX_LATENT_FRAMES - overlap)
        ts = self._TEMPORAL_SCALE
        overlap_out = overlap * ts

        chunk_starts: list[int] = []
        start = 0
        while start < total_t:
            chunk_starts.append(start)
            if start + self._MAX_LATENT_FRAMES >= total_t:
                break
            start += stride

        logger.info(
            "Temporal tiling: %d latent frames, chunk=%d, overlap=%d, "
            "stride=%d, chunks=%d",
            total_t, self._MAX_LATENT_FRAMES, overlap, stride,
            len(chunk_starts),
        )

        output_chunks: list[Tensor] = []
        for chunk_idx, start in enumerate(chunk_starts):
            end = min(start + self._MAX_LATENT_FRAMES, total_t)
            chunk = latents_5d[:, :, start:end, :, :]
            actual = end - start

            if actual < self._MAX_LATENT_FRAMES:
                pad_n = self._MAX_LATENT_FRAMES - actual
                last = chunk[:, :, -1:, :, :]
                chunk = F.concat([chunk] + [last] * pad_n, axis=2)

            logger.info("VAE chunk %d/%d (frames %d-%d)", chunk_idx + 1, len(chunk_starts), start, end - 1)
            decoded = self._block_decoder(chunk)

            if actual < self._MAX_LATENT_FRAMES:
                decoded = decoded[:, :, : actual * ts, :, :]

            if chunk_idx > 0:
                decoded = decoded[:, :, overlap_out:, :, :]

            output_chunks.append(decoded)

        return F.concat(output_chunks, axis=2)

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
