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

from typing import Any

import numpy as np
from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.graph.weights import Weights
from max.nn.module_v3 import Module
from max.nn.module_v3.norm import LayerNorm
from max.nn.module_v3.sequential import ModuleList
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import AutoencoderKLLTXVideoConfig
from .vae import DiagonalGaussianDistribution


def _channels_last(x: Tensor) -> Tensor:
    return F.permute(x, (0, 2, 3, 4, 1))


def _channels_first(x: Tensor) -> Tensor:
    return F.permute(x, (0, 4, 1, 2, 3))


class Conv3d(Module[[Tensor], Tensor]):
    """Module-v3 compatible 3D convolution with PyTorch weight layout."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int | tuple[int, int, int] = 1,
        padding: int
        | tuple[int, int, int]
        | tuple[int, int, int, int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        num_groups: int = 1,
        has_bias: bool = False,
        permute: bool = True,
    ) -> None:
        super().__init__()
        self.permute = permute
        self.num_groups = num_groups

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        k_d, k_h, k_w = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride

        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.dilation = dilation

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding, padding, padding)
        elif len(padding) == 3:
            p_d, p_h, p_w = padding
            padding = (p_d, p_d, p_h, p_h, p_w, p_w)
        self.padding = padding

        if permute:
            # PyTorch order: [out, in/groups, d, h, w]
            self.weight = Tensor.zeros(
                [out_channels, in_channels // num_groups, k_d, k_h, k_w],
                dtype=dtype,
            )
        else:
            # MAX order: [d, h, w, in/groups, out]
            self.weight = Tensor.zeros(
                [k_d, k_h, k_w, in_channels // num_groups, out_channels],
                dtype=dtype,
            )

        if has_bias:
            self.bias: Tensor | None = Tensor.zeros([out_channels], dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(x.device)

        if self.permute:
            # [N, C, D, H, W] -> [N, D, H, W, C]
            x = F.permute(x, (0, 2, 3, 4, 1))
            # [O, I, D, H, W] -> [D, H, W, I, O]
            weight = F.permute(weight, (2, 3, 4, 1, 0))

        y = F.conv3d(
            x,
            weight,
            self.stride,
            self.dilation,
            self.padding,
            self.num_groups,
            self.bias.to(x.device) if self.bias is not None else None,
        )

        if self.permute:
            # [N, D, H, W, C] -> [N, C, D, H, W]
            y = F.permute(y, (0, 4, 1, 2, 3))

        return y


class RMSNormNoAffine(Module[[Tensor], Tensor]):
    """RMSNorm without affine parameters (diffusers elementwise_affine=False)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_f32 = x.cast(DType.float32)
        eps = F.constant(self.eps, dtype=DType.float32, device=x.device)
        variance = F.mean(x_f32 * x_f32, axis=-1)
        if len(variance.shape) < len(x.shape):
            variance = F.unsqueeze(variance, -1)
        inv_rms = F.rsqrt(variance + eps)
        return (x_f32 * inv_rms).cast(x.dtype)


class LTXVideoCausalConv3d(Module[[Tensor], Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        is_causal: bool = True,
        dtype: DType = DType.bfloat16,
    ) -> None:
        super().__init__()
        self.is_causal = is_causal

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(dilation, int):
            dilation = (dilation, 1, 1)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, 0, height_pad, height_pad, width_pad, width_pad)

        self.conv = Conv3d(
            kernel_size=self.kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=stride,
            dilation=dilation,
            num_groups=groups,
            padding=padding,
            has_bias=True,
            permute=True,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        time_kernel = self.kernel_size[0]

        if self.is_causal:
            left_pad = time_kernel - 1
            if left_pad > 0:
                pad_left = F.tile(
                    hidden_states[:, :, :1, :, :],
                    (1, 1, left_pad, 1, 1),
                )
                hidden_states = F.concat([pad_left, hidden_states], axis=2)
        else:
            half = (time_kernel - 1) // 2
            if half > 0:
                pad_left = F.tile(
                    hidden_states[:, :, :1, :, :], (1, 1, half, 1, 1)
                )
                pad_right = F.tile(
                    hidden_states[:, :, -1:, :, :], (1, 1, half, 1, 1)
                )
                hidden_states = F.concat(
                    [pad_left, hidden_states, pad_right], axis=2
                )

        return self.conv(hidden_states)


class LTXVideoResnetBlock3d(Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        eps: float = 1e-6,
        is_causal: bool = True,
        dtype: DType = DType.bfloat16,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = RMSNormNoAffine(eps=1e-8)
        self.conv1 = LTXVideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            is_causal=is_causal,
            dtype=dtype,
        )

        self.norm2 = RMSNormNoAffine(eps=1e-8)
        self.conv2 = LTXVideoCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            is_causal=is_causal,
            dtype=dtype,
        )

        self.norm3: LayerNorm | None = None
        self.conv_shortcut: LTXVideoCausalConv3d | None = None
        if in_channels != out_channels:
            self.norm3 = LayerNorm(
                in_channels,
                eps=eps,
                keep_dtype=True,
                elementwise_affine=True,
                use_bias=True,
            )
            if self.norm3.weight is not None and self.norm3.weight.dtype != dtype:
                self.norm3.weight = self.norm3.weight.cast(dtype)
            if self.norm3.bias is not None and self.norm3.bias.dtype != dtype:
                self.norm3.bias = self.norm3.bias.cast(dtype)
            self.conv_shortcut = LTXVideoCausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_causal=is_causal,
                dtype=dtype,
            )

    def forward(
        self,
        inputs: Tensor,
        temb: Tensor | None = None,
        generator: Tensor | None = None,
    ) -> Tensor:
        _ = temb
        _ = generator

        hidden_states = _channels_first(self.norm1(_channels_last(inputs)))
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = _channels_first(self.norm2(_channels_last(hidden_states)))
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        residual = inputs
        if self.norm3 is not None:
            residual = _channels_first(self.norm3(_channels_last(residual)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


class LTXVideoUpsampler3d(Module[[Tensor], Tensor]):
    def __init__(
        self,
        in_channels: int,
        stride: int | tuple[int, int, int] = 1,
        is_causal: bool = True,
        residual: bool = False,
        upscale_factor: int = 1,
        dtype: DType = DType.bfloat16,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.upscale_factor = upscale_factor

        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride

        out_channels = (
            in_channels
            * self.stride[0]
            * self.stride[1]
            * self.stride[2]
            // upscale_factor
        )

        self.conv = LTXVideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size = hidden_states.shape[0]
        num_frames = hidden_states.shape[2]
        height = hidden_states.shape[3]
        width = hidden_states.shape[4]
        stride_t, stride_h, stride_w = self.stride

        if self.residual:
            residual = hidden_states.reshape(
                (
                    batch_size,
                    -1,
                    stride_t,
                    stride_h,
                    stride_w,
                    num_frames,
                    height,
                    width,
                )
            )
            residual = F.permute(residual, (0, 1, 5, 2, 6, 3, 7, 4))
            residual = F.reshape(
                residual,
                (
                    batch_size,
                    -1,
                    num_frames * stride_t,
                    height * stride_h,
                    width * stride_w,
                ),
            )
            repeats = (
                stride_t * stride_h * stride_w
            ) // self.upscale_factor
            residual = F.tile(residual, (1, repeats, 1, 1, 1))
            residual = residual[:, :, stride_t - 1 :, :, :]

        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.reshape(
            (
                batch_size,
                -1,
                stride_t,
                stride_h,
                stride_w,
                num_frames,
                height,
                width,
            )
        )
        hidden_states = F.permute(hidden_states, (0, 1, 5, 2, 6, 3, 7, 4))
        hidden_states = F.reshape(
            hidden_states,
            (
                batch_size,
                -1,
                num_frames * stride_t,
                height * stride_h,
                width * stride_w,
            ),
        )
        hidden_states = hidden_states[:, :, stride_t - 1 :, :, :]

        if self.residual:
            hidden_states = hidden_states + residual

        return hidden_states


class LTXVideoDownBlock3d(Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        resnet_eps: float,
        spatio_temporal_scale: bool,
        is_causal: bool,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.resnets = ModuleList(
            [
                LTXVideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    is_causal=is_causal,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.downsamplers: ModuleList[LTXVideoCausalConv3d] | None = None
        if spatio_temporal_scale:
            self.downsamplers = ModuleList(
                [
                    LTXVideoCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        is_causal=is_causal,
                        dtype=dtype,
                    )
                ]
            )

        self.conv_out: LTXVideoResnetBlock3d | None = None
        if in_channels != out_channels:
            self.conv_out = LTXVideoResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                eps=resnet_eps,
                is_causal=is_causal,
                dtype=dtype,
            )

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
        generator: Tensor | None = None,
    ) -> Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        if self.conv_out is not None:
            hidden_states = self.conv_out(hidden_states, temb, generator)

        return hidden_states


class LTXVideoMidBlock3d(Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        resnet_eps: float,
        is_causal: bool,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.resnets = ModuleList(
            [
                LTXVideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    is_causal=is_causal,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
        generator: Tensor | None = None,
    ) -> Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)
        return hidden_states


class LTXVideoUpBlock3d(Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        resnet_eps: float,
        spatio_temporal_scale: bool,
        is_causal: bool,
        upsample_residual: bool,
        upscale_factor: int,
        dtype: DType,
    ) -> None:
        super().__init__()

        self.conv_in: LTXVideoResnetBlock3d | None = None
        if in_channels != out_channels:
            self.conv_in = LTXVideoResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                eps=resnet_eps,
                is_causal=is_causal,
                dtype=dtype,
            )

        self.upsamplers: ModuleList[LTXVideoUpsampler3d] | None = None
        if spatio_temporal_scale:
            self.upsamplers = ModuleList(
                [
                    LTXVideoUpsampler3d(
                        out_channels * upscale_factor,
                        stride=(2, 2, 2),
                        is_causal=is_causal,
                        residual=upsample_residual,
                        upscale_factor=upscale_factor,
                        dtype=dtype,
                    )
                ]
            )

        self.resnets = ModuleList(
            [
                LTXVideoResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    is_causal=is_causal,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
        generator: Tensor | None = None,
    ) -> Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)

        return hidden_states


class LTXVideoEncoder3d(Module[[Tensor], Tensor]):
    def __init__(self, config: AutoencoderKLLTXVideoConfig) -> None:
        super().__init__()
        self.config = config

        patch_size = int(config.patch_size)
        patch_size_t = int(config.patch_size_t)
        block_out_channels = tuple(config.block_out_channels)
        layers_per_block = tuple(config.layers_per_block)
        spatio_temporal_scaling = tuple(config.spatio_temporal_scaling)

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = int(config.in_channels) * patch_size**2

        output_channel = block_out_channels[0]
        self.conv_in = LTXVideoCausalConv3d(
            in_channels=self.in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            is_causal=bool(config.encoder_causal),
            dtype=config.dtype,
        )

        self.down_blocks = ModuleList()
        num_block_out_channels = len(block_out_channels)
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = (
                block_out_channels[i + 1]
                if i + 1 < num_block_out_channels
                else block_out_channels[i]
            )
            down_block = LTXVideoDownBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i],
                resnet_eps=float(config.resnet_norm_eps),
                spatio_temporal_scale=bool(spatio_temporal_scaling[i]),
                is_causal=bool(config.encoder_causal),
                dtype=config.dtype,
            )
            self.down_blocks.append(down_block)

        self.mid_block = LTXVideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[-1],
            resnet_eps=float(config.resnet_norm_eps),
            is_causal=bool(config.encoder_causal),
            dtype=config.dtype,
        )

        self.norm_out = RMSNormNoAffine(eps=1e-8)
        self.conv_out = LTXVideoCausalConv3d(
            in_channels=output_channel,
            out_channels=int(config.latent_channels) + 1,
            kernel_size=3,
            stride=1,
            is_causal=bool(config.encoder_causal),
            dtype=config.dtype,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.config.dtype,
                shape=[
                    "batch",
                    int(self.config.in_channels),
                    1,
                    int(self.config.sample_height),
                    int(self.config.sample_width),
                ],
                device=self.config.device,
            ),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        p = self.patch_size
        p_t = self.patch_size_t

        batch_size = hidden_states.shape[0]
        num_channels = hidden_states.shape[1]
        num_frames = hidden_states.shape[2]
        height = hidden_states.shape[3]
        width = hidden_states.shape[4]

        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = hidden_states.reshape(
            (
                batch_size,
                num_channels,
                post_patch_num_frames,
                p_t,
                post_patch_height,
                p,
                post_patch_width,
                p,
            )
        )
        hidden_states = F.permute(hidden_states, (0, 1, 3, 7, 5, 2, 4, 6))
        hidden_states = hidden_states.reshape(
            (
                batch_size,
                -1,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
            )
        )
        hidden_states = self.conv_in(hidden_states)

        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)
        hidden_states = _channels_first(self.norm_out(_channels_last(hidden_states)))
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        # Match diffusers: expand latent moments to 2 * latent_channels by
        # repeating the final channel.
        last_channel = hidden_states[:, -1:, :, :, :]
        last_channel = F.tile(
            last_channel,
            (1, hidden_states.shape[1] - 2, 1, 1, 1),
        )
        hidden_states = F.concat([hidden_states, last_channel], axis=1)

        return hidden_states


class LTXVideoDecoder3d(Module[..., Tensor]):
    def __init__(self, config: AutoencoderKLLTXVideoConfig) -> None:
        super().__init__()
        self.config = config

        patch_size = int(config.patch_size)
        patch_size_t = int(config.patch_size_t)

        block_out_channels = tuple(reversed(tuple(config.block_out_channels)))
        spatio_temporal_scaling = tuple(
            reversed(tuple(config.decoder_spatio_temporal_scaling))
        )
        layers_per_block = tuple(reversed(tuple(config.decoder_layers_per_block)))
        inject_noise = tuple(reversed(tuple(config.decoder_inject_noise)))
        upsample_residual = tuple(reversed(tuple(config.upsample_residual)))
        upsample_factor = tuple(reversed(tuple(config.upsample_factor)))

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = int(config.out_channels) * patch_size**2

        output_channel = block_out_channels[0]

        self.conv_in = LTXVideoCausalConv3d(
            in_channels=int(config.latent_channels),
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            is_causal=bool(config.decoder_causal),
            dtype=config.dtype,
        )

        self.mid_block = LTXVideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=float(config.resnet_norm_eps),
            is_causal=bool(config.decoder_causal),
            dtype=config.dtype,
        )

        self.up_blocks = ModuleList()
        num_block_out_channels = len(block_out_channels)
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]

            up_block = LTXVideoUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=float(config.resnet_norm_eps),
                spatio_temporal_scale=bool(spatio_temporal_scaling[i]),
                is_causal=bool(config.decoder_causal),
                upsample_residual=bool(upsample_residual[i]),
                upscale_factor=int(upsample_factor[i]),
                dtype=config.dtype,
            )
            self.up_blocks.append(up_block)

        self.norm_out = RMSNormNoAffine(eps=1e-8)
        self.conv_out = LTXVideoCausalConv3d(
            in_channels=output_channel,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            is_causal=bool(config.decoder_causal),
            dtype=config.dtype,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        latent_frames = (
            (int(self.config.sample_num_frames) - 1)
            // int(self.config.temporal_compression_ratio)
        ) + 1
        latent_height = int(self.config.sample_height) // int(
            self.config.spatial_compression_ratio
        )
        latent_width = int(self.config.sample_width) // int(
            self.config.spatial_compression_ratio
        )
        input_types = [
            TensorType(
                self.config.dtype,
                shape=[
                    "batch",
                    int(self.config.latent_channels),
                    latent_frames,
                    latent_height,
                    latent_width,
                ],
                device=self.config.device,
            )
        ]
        if self.config.timestep_conditioning:
            input_types.append(
                TensorType(
                    DType.float32,
                    shape=["batch"],
                    device=self.config.device,
                )
            )
        return tuple(input_types)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid_block(hidden_states, temb)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, temb)

        hidden_states = _channels_first(self.norm_out(_channels_last(hidden_states)))
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size = hidden_states.shape[0]
        num_frames = hidden_states.shape[2]
        height = hidden_states.shape[3]
        width = hidden_states.shape[4]

        hidden_states = hidden_states.reshape(
            (batch_size, -1, p_t, p, p, num_frames, height, width)
        )
        hidden_states = F.permute(hidden_states, (0, 1, 5, 2, 6, 4, 7, 3))
        hidden_states = F.reshape(
            hidden_states,
            (batch_size, -1, num_frames * p_t, height * p, width * p),
        )

        return hidden_states


class AutoencoderKLLTXVideoModel(ComponentModel):
    """MAX-native LTX video VAE component (encoder + decoder)."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = AutoencoderKLLTXVideoConfig.generate(
            config,
            encoding,
            devices,
        )
        self.encoder_model = None
        self._encoder_cache: dict[tuple[int, int, int], Any] = {}
        self._encoder_state_dict: dict[str, Any] = {}
        self._decoder_cache: dict[tuple[int, int, int], Any] = {}
        self._decoder_state_dict: dict[str, Any] = {}

        self.latents_mean = (
            np.asarray(self.config.latents_mean, dtype=np.float32)
            if self.config.latents_mean is not None
            else np.zeros(int(self.config.latent_channels), dtype=np.float32)
        )
        self.latents_std = (
            np.asarray(self.config.latents_std, dtype=np.float32)
            if self.config.latents_std is not None
            else np.ones(int(self.config.latent_channels), dtype=np.float32)
        )

        self.load_model()

    @property
    def spatial_compression_ratio(self) -> int:
        return int(self.config.spatial_compression_ratio)

    @property
    def temporal_compression_ratio(self) -> int:
        return int(self.config.temporal_compression_ratio)

    @property
    def dtype(self) -> DType:
        return self.config.dtype

    @staticmethod
    def _to_numpy_weight(weight_data: Any) -> np.ndarray:
        return np.from_dlpack(weight_data).astype(np.float32, copy=True)

    def load_model(self):
        decoder_state_dict = {}
        encoder_state_dict = {}
        latents_mean = None
        latents_std = None
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
            if key == "latents_mean":
                latents_mean = self._to_numpy_weight(value.data())
                continue
            if key == "latents_std":
                latents_std = self._to_numpy_weight(value.data())
                continue

            weight_data = value.data()
            if weight_data.dtype != target_dtype:
                if weight_data.dtype.is_float() and target_dtype.is_float():
                    # LTX checkpoints are fp32; decoder runs in bf16 for memory/perf.
                    # Cast mismatched float tensors so module_v3 parameter loading
                    # remains dtype-compatible.
                    weight_data = weight_data.astype(target_dtype)

            if key.startswith("decoder."):
                decoder_state_dict[key.removeprefix("decoder.")] = weight_data
            elif key.startswith("encoder."):
                encoder_state_dict[key.removeprefix("encoder.")] = weight_data

        if latents_mean is not None:
            self.latents_mean = latents_mean.astype(np.float32, copy=False)
        if latents_std is not None:
            self.latents_std = latents_std.astype(np.float32, copy=False)

        with F.lazy():
            encoder = LTXVideoEncoder3d(self.config)
            encoder.to(self.devices[0])
            decoder = LTXVideoDecoder3d(self.config)
            decoder.to(self.devices[0])

        self._encoder_state_dict = encoder_state_dict
        self._decoder_state_dict = decoder_state_dict
        self.encoder_model = encoder.compile(
            *encoder.input_types(), weights=encoder_state_dict
        )
        self.model = decoder.compile(*decoder.input_types(), weights=decoder_state_dict)
        default_encoder_key = (
            1,
            int(self.config.sample_height),
            int(self.config.sample_width),
        )
        self._encoder_cache[default_encoder_key] = self.encoder_model
        default_key = (
            int(
                (int(self.config.sample_num_frames) - 1)
                // int(self.config.temporal_compression_ratio)
            )
            + 1,
            int(self.config.sample_height) // int(self.config.spatial_compression_ratio),
            int(self.config.sample_width) // int(self.config.spatial_compression_ratio),
        )
        self._decoder_cache[default_key] = self.model
        return self.model

    def _build_decoder_for_shape(
        self,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> Any:
        input_types = [
            TensorType(
                self.config.dtype,
                shape=[
                    "batch",
                    int(self.config.latent_channels),
                    int(latent_frames),
                    int(latent_height),
                    int(latent_width),
                ],
                device=self.config.device,
            )
        ]
        if self.config.timestep_conditioning:
            input_types.append(
                TensorType(
                    DType.float32,
                    shape=["batch"],
                    device=self.config.device,
                )
            )

        with F.lazy():
            decoder = LTXVideoDecoder3d(self.config)
            decoder.to(self.devices[0])
        return decoder.compile(*tuple(input_types), weights=self._decoder_state_dict)

    def _build_encoder_for_shape(
        self,
        sample_frames: int,
        sample_height: int,
        sample_width: int,
    ) -> Any:
        input_types = (
            TensorType(
                self.config.dtype,
                shape=[
                    "batch",
                    int(self.config.in_channels),
                    int(sample_frames),
                    int(sample_height),
                    int(sample_width),
                ],
                device=self.config.device,
            ),
        )
        with F.lazy():
            encoder = LTXVideoEncoder3d(self.config)
            encoder.to(self.devices[0])
        return encoder.compile(*input_types, weights=self._encoder_state_dict)

    def encode(
        self, sample: Tensor, return_dict: bool = True
    ) -> dict[str, DiagonalGaussianDistribution] | DiagonalGaussianDistribution:
        if self.encoder_model is None:
            raise ValueError("LTX VAE encoder is not initialized.")

        sample = sample.to(self.devices[0]).cast(self.dtype)
        sample_frames = int(sample.shape[2])
        sample_height = int(sample.shape[3])
        sample_width = int(sample.shape[4])
        encoder_key = (sample_frames, sample_height, sample_width)
        encoder = self._encoder_cache.get(encoder_key)
        if encoder is None:
            encoder = self._build_encoder_for_shape(
                sample_frames, sample_height, sample_width
            )
            self._encoder_cache[encoder_key] = encoder

        moments = encoder(sample)
        posterior = DiagonalGaussianDistribution(moments)
        if return_dict:
            return {"latent_dist": posterior}
        return posterior

    def decode(self, z: Tensor, timestep: Tensor | None = None) -> Tensor:
        latent_frames = int(z.shape[2])
        latent_height = int(z.shape[3])
        latent_width = int(z.shape[4])
        decoder_key = (latent_frames, latent_height, latent_width)
        decoder = self._decoder_cache.get(decoder_key)
        if decoder is None:
            decoder = self._build_decoder_for_shape(
                latent_frames, latent_height, latent_width
            )
            self._decoder_cache[decoder_key] = decoder

        if self.config.timestep_conditioning:
            if timestep is None:
                timestep = Tensor.zeros([z.shape[0]], dtype=DType.float32, device=z.device)
            return decoder(z, timestep.cast(DType.float32))
        return decoder(z)

    def __call__(self, z: Tensor, timestep: Tensor | None = None) -> Tensor:
        return self.decode(z, timestep)
