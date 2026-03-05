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

"""QwenImage VAE decoder for T=1 image generation.

The QwenImage VAE (Wan-2.1 architecture) uses 3D causal convolutions. For
single-image generation (T=1), these reduce to 2D convolutions: with causal
temporal padding on T=1 input, only the last temporal kernel slice contributes
non-zero values. This module implements the decoder using Conv2d with weights
extracted from the last temporal slice of the 3D kernels.

Weight transformations in load_model():
- 5D conv [O, I, D, H, W] -> 4D [O, I, H, W] (last temporal slice)
- Norm gamma [C, 1, 1, 1] or [C, 1, 1] -> [C]
- Fused to_qkv -> split into separate to_q, to_k, to_v
- time_conv weights -> skipped (irrelevant for T=1)
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.graph.weights import WeightData, Weights
from max.nn.module_v3 import Conv2d, Module
from max.nn.module_v3.norm.rms_norm import rms_norm
from max.nn.module_v3.sequential import ModuleList
from max.pipelines.architectures.autoencoders.layers.upsampling import (
    interpolate_2d_nearest,
)
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import (
    AutoencoderKLQwenImageConfig,
    AutoencoderKLQwenImageConfigBase,
)


class _QwenImageVAERMSNorm(Module[[Tensor], Tensor]):
    """RMS normalization with learnable gamma for NCHW tensors.

    The Wan VAE RMS norm computes mean(x^2) over the channel dimension.
    We permute to channels-last, apply standard rms_norm (over last dim),
    and permute back.

    HF weight key produces: {name}.gamma
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = Tensor.ones([dim])

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W] -> permute to [B, H, W, C]
        x_perm = F.permute(x, [0, 2, 3, 1])
        # rms_norm normalizes over last dim (C)
        x_normed = rms_norm(x_perm, self.gamma.to(x.device), self.eps)
        # Permute back to [B, C, H, W]
        return F.permute(x_normed, [0, 3, 1, 2])


class _QwenImageVAEResBlock(Module[[Tensor], Tensor]):
    """Residual block with RMS norm and Conv2d.

    HF keys: norm1.gamma, conv1.{weight,bias}, norm2.gamma, conv2.{weight,bias}
    Optional: conv_shortcut.{weight,bias} when in_ch != out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, eps: float = 1e-6):
        self.norm1 = _QwenImageVAERMSNorm(in_ch, eps)
        self.conv1 = Conv2d(
            kernel_size=3,
            in_channels=in_ch,
            out_channels=out_ch,
            padding=1,
            has_bias=True,
            permute=True,
        )
        self.norm2 = _QwenImageVAERMSNorm(out_ch, eps)
        self.conv2 = Conv2d(
            kernel_size=3,
            in_channels=out_ch,
            out_channels=out_ch,
            padding=1,
            has_bias=True,
            permute=True,
        )
        self.conv_shortcut: Conv2d | None = None
        if in_ch != out_ch:
            self.conv_shortcut = Conv2d(
                kernel_size=1,
                in_channels=in_ch,
                out_channels=out_ch,
                padding=0,
                has_bias=True,
                permute=True,
            )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + shortcut


class _QwenImageVAEAttention(Module[[Tensor], Tensor]):
    """Self-attention for VAE mid-block using 1x1 Conv2d.

    HF has fused to_qkv; we split into to_q/to_k/to_v during weight loading.
    HF keys: norm.gamma, to_qkv.{weight,bias} (split), proj.{weight,bias}
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self._dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        self.norm = _QwenImageVAERMSNorm(dim, eps)
        self.to_q = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            has_bias=True,
            permute=True,
        )
        self.to_k = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            has_bias=True,
            permute=True,
        )
        self.to_v = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            has_bias=True,
            permute=True,
        )
        self.proj = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            has_bias=True,
            permute=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)

        n, c, h, w = x.shape
        seq_len = h * w

        # Apply 1x1 convs for Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape [B, C, H, W] -> [B, H*W, C] for attention
        q = F.permute(F.reshape(q, [n, c, seq_len]), [0, 2, 1])
        k = F.permute(F.reshape(k, [n, c, seq_len]), [0, 2, 1])
        v = F.permute(F.reshape(v, [n, c, seq_len]), [0, 2, 1])

        # Scaled dot-product attention (single-head)
        attn = q @ F.permute(k, [0, 2, 1]) * self.scale
        attn = F.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back [B, H*W, C] -> [B, C, H, W]
        out = F.reshape(F.permute(out, [0, 2, 1]), [n, c, h, w])

        # Output projection
        out = self.proj(out)

        return residual + out


class _Interpolate2D(Module[[Tensor], Tensor]):
    """2x nearest-neighbor interpolation with no learnable parameters.

    Used as index 0 of the upsampler's resample ModuleList (Conv2d at index 1).
    This ensures weight keys match HF: resample.1.{weight,bias}.
    """

    def forward(self, x: Tensor) -> Tensor:
        return interpolate_2d_nearest(x, scale_factor=2)  # type: ignore[return-value]


class _QwenImageVAEUpsampler(Module[[Tensor], Tensor]):
    """Spatial upsampler: 2x interpolation then Conv2d.

    HF keys: resample.0 (no weights), resample.1.{weight,bias}
    """

    def __init__(self, in_ch: int, out_ch: int):
        self.resample: ModuleList = ModuleList(
            [
                _Interpolate2D(),
                Conv2d(
                    kernel_size=3,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    padding=1,
                    has_bias=True,
                    permute=True,
                ),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.resample[0](x)
        x = self.resample[1](x)
        return x


class _QwenImageVAEMidBlock(Module[[Tensor], Tensor]):
    """Mid block: ResBlock -> Attention -> ResBlock.

    HF keys: resnets.{0,1}.*, attentions.0.*
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.resnets: ModuleList[_QwenImageVAEResBlock] = ModuleList(
            [
                _QwenImageVAEResBlock(dim, dim, eps),
                _QwenImageVAEResBlock(dim, dim, eps),
            ]
        )
        self.attentions: ModuleList[_QwenImageVAEAttention] = ModuleList(
            [_QwenImageVAEAttention(dim, eps)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class _QwenImageVAEUpBlock(Module[[Tensor], Tensor]):
    """Up block: ResBlocks then optional upsampler.

    HF keys: resnets.{0,1,2}.*, upsamplers.0.resample.1.*
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_resnets: int,
        upsample_out_ch: int | None = None,
        eps: float = 1e-6,
    ):
        resnets = []
        for i in range(num_resnets):
            res_in = in_ch if i == 0 else out_ch
            resnets.append(_QwenImageVAEResBlock(res_in, out_ch, eps))
        self.resnets: ModuleList[_QwenImageVAEResBlock] = ModuleList(resnets)

        self.upsamplers: ModuleList[_QwenImageVAEUpsampler] | None = None
        if upsample_out_ch is not None:
            self.upsamplers = ModuleList(
                [_QwenImageVAEUpsampler(out_ch, upsample_out_ch)]
            )

    def forward(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(Module[[Tensor], Tensor]):
    """QwenImage VAE decoder for T=1 image generation.

    Converts latent [B, 16, H, W] to image [B, 3, 8H, 8W] via:
    post_quant_conv -> conv_in -> mid_block -> 4 up_blocks -> norm_out -> conv_out

    For base_dim=96, dim_mult=[1,2,4,4], the channel flow is:
    16 -> 384 (conv_in) -> 384 (mid_block) -> up_blocks:
      [384->384, upsample->192] -> [192->384, upsample->192] ->
      [192->192, upsample->96] -> [96->96] -> 3 (conv_out)
    """

    def __init__(self, config: AutoencoderKLQwenImageConfigBase):
        dims = [config.base_dim * m for m in config.dim_mult]
        num_levels = len(dims)
        num_resnets = config.num_res_blocks + 1
        deepest = dims[-1]

        self._z_dim = config.z_dim
        self._dtype = config.dtype
        self._device = config.device

        # Post-quantization 1x1 conv (from top-level VAE, routed to decoder)
        self.post_quant_conv = Conv2d(
            kernel_size=1,
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            has_bias=True,
            permute=True,
        )

        # Input projection: z_dim -> deepest
        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=config.z_dim,
            out_channels=deepest,
            padding=1,
            has_bias=True,
            permute=True,
        )

        # Mid block
        self.mid_block = _QwenImageVAEMidBlock(deepest)

        # Up blocks: compute channel configs from dim_mult
        dims_reversed = list(reversed(dims))
        up_blocks = []
        prev_ch = deepest

        for i in range(num_levels):
            block_ch = dims_reversed[i]

            # Determine upsample output channels (skip to next different dim)
            upsample_out = None
            if i < num_levels - 1:
                for k in range(i + 1, num_levels):
                    if dims_reversed[k] != block_ch:
                        upsample_out = dims_reversed[k]
                        break
                if upsample_out is None:
                    upsample_out = dims_reversed[-1]

            up_blocks.append(
                _QwenImageVAEUpBlock(
                    in_ch=prev_ch,
                    out_ch=block_ch,
                    num_resnets=num_resnets,
                    upsample_out_ch=upsample_out,
                )
            )

            prev_ch = upsample_out if upsample_out is not None else block_ch

        self.up_blocks: ModuleList[_QwenImageVAEUpBlock] = ModuleList(
            up_blocks
        )

        # Output
        shallowest = dims[0]
        self.norm_out = _QwenImageVAERMSNorm(shallowest)
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=shallowest,
            out_channels=3,
            padding=1,
            has_bias=True,
            permute=True,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch", self._z_dim, "height", "width"],
                device=self._device,
            ),
        )

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)
        h = self.conv_in(z)
        h = self.mid_block(h)
        for up_block in self.up_blocks:
            h = up_block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class AutoencoderKLQwenImage(Module[[Tensor], Tensor]):
    """QwenImage VAE wrapper for the decoder."""

    def __init__(self, config: AutoencoderKLQwenImageConfigBase) -> None:
        super().__init__()
        self.decoder = QwenImageDecoder3d(config)

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)


def _transform_decoder_weights(
    raw_weights: dict[str, Any],
    target_dtype: DType,
) -> dict[str, Any]:
    """Transform 3D VAE weights to 2D for T=1 image generation.

    Transformations applied:
    1. 5D conv [O, I, D, H, W] -> 4D [O, I, H, W] (last temporal slice for
       causal conv, or squeeze for 1x1x1 conv)
    2. Norm gamma [C, 1, 1, 1] or [C, 1, 1] -> [C]
    3. Fused to_qkv weight/bias -> split to to_q/to_k/to_v
    4. time_conv weights -> skipped (irrelevant for T=1)
    5. Float weights -> cast to target dtype
    """
    result: dict[str, Any] = {}

    def _to_numpy(wd: WeightData) -> np.ndarray:
        """Convert WeightData to numpy, casting bfloat16 to float32."""
        if wd.dtype == DType.bfloat16:
            wd = wd.astype(DType.float32)
        return np.from_dlpack(wd.data)  # type: ignore

    def _to_weight_data(
        arr: np.ndarray, name: str, dtype: DType
    ) -> WeightData:
        """Convert numpy array to WeightData at target dtype."""
        wd = WeightData.from_numpy(np.ascontiguousarray(arr), name)
        if dtype != DType.float32:
            wd = wd.astype(dtype)
        return wd

    for key, raw_data in raw_weights.items():
        # Skip temporal convolution (irrelevant for T=1)
        if "time_conv" in key:
            continue

        # Check if this weight needs numpy transformation
        shape = tuple(raw_data.shape)
        needs_transform = (
            ".to_qkv." in key  # QKV split
            or len(shape) == 5  # 5D conv -> 4D
            or (len(shape) == 4 and shape[1:] == (1, 1, 1))  # norm gamma
            or (len(shape) == 3 and shape[1:] == (1, 1))  # norm gamma
        )

        if not needs_transform:
            # No transformation needed - pass WeightData directly
            if (
                raw_data.dtype != target_dtype
                and raw_data.dtype.is_float()
                and target_dtype.is_float()
            ):
                result[key] = raw_data.astype(target_dtype)
            else:
                result[key] = raw_data
            continue

        # Need numpy for transformation
        data = _to_numpy(raw_data)

        # Split fused QKV into separate Q, K, V
        if ".to_qkv.weight" in key:
            if data.ndim == 5:
                data = (
                    data[:, :, -1, :, :]
                    if data.shape[2] > 1
                    else data[:, :, 0, :, :]
                )
            C = data.shape[0] // 3
            prefix = key.replace(".to_qkv.weight", "")
            result[f"{prefix}.to_q.weight"] = _to_weight_data(
                data[:C], f"{prefix}.to_q.weight", target_dtype
            )
            result[f"{prefix}.to_k.weight"] = _to_weight_data(
                data[C : 2 * C], f"{prefix}.to_k.weight", target_dtype
            )
            result[f"{prefix}.to_v.weight"] = _to_weight_data(
                data[2 * C :], f"{prefix}.to_v.weight", target_dtype
            )
            continue
        if ".to_qkv.bias" in key:
            C = data.shape[0] // 3
            prefix = key.replace(".to_qkv.bias", "")
            result[f"{prefix}.to_q.bias"] = _to_weight_data(
                data[:C], f"{prefix}.to_q.bias", target_dtype
            )
            result[f"{prefix}.to_k.bias"] = _to_weight_data(
                data[C : 2 * C], f"{prefix}.to_k.bias", target_dtype
            )
            result[f"{prefix}.to_v.bias"] = _to_weight_data(
                data[2 * C :], f"{prefix}.to_v.bias", target_dtype
            )
            continue

        # Transform 5D conv weights to 4D
        if data.ndim == 5:
            if data.shape[2] == 1:
                data = data[:, :, 0, :, :]  # 1x1x1 conv -> squeeze
            else:
                data = data[:, :, -1, :, :]  # Causal conv -> last slice
        # Squeeze norm gamma
        elif (
            data.ndim == 4
            and data.shape[1] == 1
            and data.shape[2] == 1
            and data.shape[3] == 1
        ):
            data = data.reshape(data.shape[0])  # [C, 1, 1, 1] -> [C]
        elif data.ndim == 3 and data.shape[1] == 1 and data.shape[2] == 1:
            data = data.reshape(data.shape[0])  # [C, 1, 1] -> [C]

        result[key] = _to_weight_data(data, key, target_dtype)

    return result


class AutoencoderKLQwenImageModel(ComponentModel):
    """ComponentModel wrapper for QwenImage VAE.

    Handles:
    - 3D to 2D weight transformation for T=1 image generation
    - Latent normalization via latents_mean/latents_std
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        self.latents_mean_tensor: Tensor | None = None
        self.latents_std_tensor: Tensor | None = None

        super().__init__(config, encoding, devices, weights)
        self.config = AutoencoderKLQwenImageConfig.generate(
            config, encoding, devices
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        target_dtype = self.config.dtype

        # Collect decoder weights: strip "decoder." prefix, keep "post_quant_conv."
        raw_decoder_weights: dict[str, Any] = {}
        for key, value in self.weights.items():
            weight_data = value.data()
            if key.startswith("decoder."):
                raw_decoder_weights[key.removeprefix("decoder.")] = weight_data
            elif key.startswith("post_quant_conv."):
                raw_decoder_weights[key] = weight_data

        # Transform 3D weights to 2D for T=1 image generation
        decoder_state_dict = _transform_decoder_weights(
            raw_decoder_weights, target_dtype
        )

        # Build and compile decoder
        with F.lazy():
            autoencoder = AutoencoderKLQwenImage(self.config)
            autoencoder.decoder.to(self.devices[0])

        self.model = autoencoder.decoder.compile(
            *autoencoder.decoder.input_types(), weights=decoder_state_dict
        )

        # Store latents_mean and latents_std as tensors on device
        if self.config.latents_mean:
            mean_np = np.array(self.config.latents_mean, dtype=np.float32)
            self.latents_mean_tensor = (
                Tensor.from_dlpack(mean_np)
                .to(self.devices[0])
                .cast(target_dtype)
            )

        if self.config.latents_std:
            std_np = np.array(self.config.latents_std, dtype=np.float32)
            self.latents_std_tensor = (
                Tensor.from_dlpack(std_np)
                .to(self.devices[0])
                .cast(target_dtype)
            )

        return self.model

    def decode(self, z: Tensor) -> Tensor:
        return self.model(z)

    def __call__(self, z: Tensor) -> Tensor:
        return self.decode(z)
