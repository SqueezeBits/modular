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

import math

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, Weight, ops
from max.nn import Conv2d, Linear
from max.nn.activation import activation_function_from_name
from max.nn.layer import LayerList, Module


def _normalize_activation_name(name: str) -> str:
    if name == "swish":
        return "silu"
    return name


class VAEGroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        dtype: DType,
        device: DeviceRef,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels({num_channels}) should be divisible by num_groups({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.weight: Weight | None = None
        self.bias: Weight | None = None

        if affine:
            self.weight = Weight(
                name="weight",
                shape=(num_channels,),
                dtype=dtype,
                device=device,
            )
            self.bias = Weight(
                name="bias",
                shape=(num_channels,),
                dtype=dtype,
                device=device,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        if len(x.shape) < 2:
            raise ValueError(
                f"Expected input tensor with >=2 dimensions, got shape {x.shape}"
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got shape {x.shape}"
            )

        if self.affine and self.weight is not None and self.bias is not None:
            gamma = self.weight.cast(x.dtype).to(x.device)
            beta = self.bias.cast(x.dtype).to(x.device)
        else:
            gamma = ops.broadcast_to(
                ops.constant(1.0, dtype=x.dtype, device=DeviceRef.CPU()),
                shape=(self.num_channels,),
            ).to(x.device)
            beta = ops.broadcast_to(
                ops.constant(0.0, dtype=x.dtype, device=DeviceRef.CPU()),
                shape=(self.num_channels,),
            ).to(x.device)

        return ops.custom(
            "group_norm",
            x.device,
            [
                x,
                gamma,
                beta,
                ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU()),
                ops.constant(
                    self.num_groups, dtype=DType.int32, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        )[0].tensor


def interpolate_2d_nearest(
    x: TensorValue,
    scale_factor: int = 2,
) -> TensorValue:
    if len(x.shape) != 4:
        raise ValueError(f"Input tensor must have rank 4, got {len(x.shape)}")
    if scale_factor != 2:
        raise NotImplementedError(
            f"Only scale_factor=2 is currently supported, got {scale_factor}"
        )

    n, c, h, w = x.shape
    x_reshaped = ops.reshape(x, [n, c, h, 1, w, 1])
    ones = ops.broadcast_to(
        ops.constant(1.0, dtype=x.dtype, device=DeviceRef.CPU()),
        shape=[1, 1, 1, scale_factor, 1, scale_factor],
    ).to(x.device)
    return ops.reshape(x_reshaped * ones, [n, c, h * 2, w * 2])


class Downsample2D(Module):
    def __init__(
        self,
        channels: int,
        *,
        use_conv: bool = False,
        out_channels: int | None = None,
        padding: int = 1,
        kernel_size: int = 3,
        norm_type: str | None = None,
        bias: bool = True,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        if norm_type is not None:
            raise NotImplementedError(
                "Downsample2D norm_type is not implemented in module v2."
            )
        self.conv: Conv2d | None = None
        if use_conv:
            self.conv = Conv2d(
                kernel_size=kernel_size,
                in_channels=channels,
                out_channels=self.out_channels,
                dtype=dtype,
                stride=2,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )
        elif channels != self.out_channels:
            raise ValueError(
                "When use_conv=False, channels must equal out_channels."
            )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        if self.use_conv and self.padding == 0:
            hidden_states = ops.pad(hidden_states, [0, 0, 0, 0, 0, 1, 0, 1])
        if self.use_conv:
            assert self.conv is not None
            return self.conv(hidden_states)
        hidden_states = ops.permute(hidden_states, [0, 2, 3, 1])
        hidden_states = ops.avg_pool2d(
            hidden_states,
            kernel_size=(2, 2),
            stride=2,
            padding=0,
        )
        return ops.permute(hidden_states, [0, 3, 1, 2])


class Upsample2D(Module):
    def __init__(
        self,
        channels: int,
        *,
        use_conv: bool = False,
        out_channels: int | None = None,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
        interpolate: bool = True,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        self.interpolate = interpolate
        self.conv: Conv2d | None = None
        if use_conv:
            self.conv = Conv2d(
                kernel_size=kernel_size,
                in_channels=channels,
                out_channels=out_channels or channels,
                dtype=dtype,
                stride=1,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        if self.interpolate:
            x = interpolate_2d_nearest(x)
        if self.use_conv and self.conv is not None:
            x = self.conv(x)
        return x


class ResnetBlock2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int | None,
        groups: int,
        groups_out: int,
        *,
        eps: float = 1e-6,
        non_linearity: str = "silu",
        use_conv_shortcut: bool = False,
        conv_shortcut_bias: bool = True,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        del temb_channels
        self.act = activation_function_from_name(
            _normalize_activation_name(non_linearity)
        )
        self.norm1 = VAEGroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            dtype=dtype,
            device=device,
        )
        self.conv1 = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )
        self.norm2 = VAEGroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
            dtype=dtype,
            device=device,
        )
        self.conv2 = Conv2d(
            kernel_size=3,
            in_channels=out_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )
        self.conv_shortcut: Conv2d | None = None
        if use_conv_shortcut or in_channels != out_channels:
            self.conv_shortcut = Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                has_bias=conv_shortcut_bias,
                device=device,
                permute=True,
            )

    def __call__(
        self, x: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        del temb
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + shortcut


class VAEAttention(Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.group_norm = VAEGroupNorm(
            num_groups=num_groups,
            num_channels=query_dim,
            eps=eps,
            affine=True,
            dtype=dtype,
            device=device,
        )
        self.to_q = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_k = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_v = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_out = LayerList(
            [
                Linear(
                    in_dim=self.inner_dim,
                    out_dim=query_dim,
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                )
            ]
        )
        self.scale = 1.0 / math.sqrt(dim_head)

    def __call__(self, x: TensorValue) -> TensorValue:
        residual = x
        x = self.group_norm(x)

        n, c, h, w = x.shape
        seq_len = h * w
        x = ops.permute(ops.reshape(x, [n, c, seq_len]), [0, 2, 1])

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = ops.permute(
            ops.reshape(q, [n, seq_len, self.heads, self.dim_head]),
            [0, 2, 1, 3],
        )
        k = ops.permute(
            ops.reshape(k, [n, seq_len, self.heads, self.dim_head]),
            [0, 2, 1, 3],
        )
        v = ops.permute(
            ops.reshape(v, [n, seq_len, self.heads, self.dim_head]),
            [0, 2, 1, 3],
        )

        attn = ops.softmax(
            (q @ ops.permute(k, [0, 1, 3, 2])) * self.scale,
            axis=-1,
        )
        out = attn @ v
        out = ops.reshape(
            ops.permute(out, [0, 2, 1, 3]),
            [n, seq_len, self.inner_dim],
        )
        out = self.to_out[0](out)
        out = ops.reshape(ops.permute(out, [0, 2, 1]), [n, c, h, w])
        return residual + out


class DownEncoderBlock2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        if resnet_time_scale_shift == "spatial":
            raise NotImplementedError(
                "resnet_time_scale_shift='spatial' is not supported in Max encoder."
            )

        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    groups=resnet_groups,
                    groups_out=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    use_conv_shortcut=False,
                    conv_shortcut_bias=True,
                    device=device,
                    dtype=dtype,
                )
            )
        self.resnets = LayerList(resnets)
        self.downsamplers = (
            LayerList(
                [
                    Downsample2D(
                        channels=out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        kernel_size=3,
                        norm_type=None,
                        bias=True,
                        device=device,
                        dtype=dtype,
                    )
                ]
            )
            if add_downsample
            else None
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, None)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
        return hidden_states


class UpDecoderBlock2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_upsample: bool = True,
        temb_channels: int | None = None,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    groups_out=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    use_conv_shortcut=False,
                    conv_shortcut_bias=True,
                    device=device,
                    dtype=dtype,
                )
            )
        self.resnets = LayerList(resnets)
        self.upsamplers = (
            LayerList(
                [
                    Upsample2D(
                        channels=out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        interpolate=True,
                        device=device,
                        dtype=dtype,
                    )
                ]
            )
            if add_upsample
            else None
        )

    def __call__(
        self, hidden_states: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class MidBlock2D(Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int | None,
        *,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                groups_out=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                use_conv_shortcut=False,
                conv_shortcut_bias=True,
                device=device,
                dtype=dtype,
            )
        ]
        attentions: list[VAEAttention] = []
        attention_indices: set[int] = set()
        for i in range(num_layers):
            if add_attention:
                attentions.append(
                    VAEAttention(
                        query_dim=in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        num_groups=resnet_groups,
                        eps=resnet_eps,
                        device=device,
                        dtype=dtype,
                    )
                )
                attention_indices.add(i)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    groups_out=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    use_conv_shortcut=False,
                    conv_shortcut_bias=True,
                    device=device,
                    dtype=dtype,
                )
            )

        self.resnets = LayerList(resnets)
        self.attentions = LayerList(attentions) if attentions else None
        self.attention_indices = attention_indices

    def __call__(
        self, hidden_states: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        hidden_states = self.resnets[0](hidden_states, temb)
        attention_idx = 0
        for i in range(len(self.resnets) - 1):
            if self.attentions is not None and i in self.attention_indices:
                hidden_states = self.attentions[attention_idx](hidden_states)
                attention_idx += 1
            hidden_states = self.resnets[i + 1](hidden_states, temb)
        return hidden_states


class Encoder(Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
        use_quant_conv: bool = False,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.dtype = dtype

        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if down_block_type != "DownEncoderBlock2D":
                raise ValueError(
                    f"Unsupported down_block_type: {down_block_type}"
                )
            down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    resnet_eps=1e-6,
                    resnet_time_scale_shift="default",
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    downsample_padding=0,
                    device=device,
                    dtype=dtype,
                )
            )
        self.down_blocks = LayerList(down_blocks)

        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            device=device,
            dtype=dtype,
        )
        self.conv_norm_out = VAEGroupNorm(
            num_groups=norm_num_groups,
            num_channels=block_out_channels[-1],
            eps=1e-6,
            affine=True,
            dtype=dtype,
            device=device,
        )
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[-1],
            out_channels=conv_out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )
        self.quant_conv = (
            Conv2d(
                kernel_size=1,
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                has_bias=True,
                device=device,
                permute=True,
            )
            if use_quant_conv
            else None
        )

    def __call__(self, sample: TensorValue) -> TensorValue:
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample, None)
        sample = ops.silu(self.conv_norm_out(sample))
        sample = self.conv_out(sample)
        if self.quant_conv is not None:
            sample = self.quant_conv(sample)
        return sample

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.dtype,
                shape=[
                    "batch_size",
                    self.in_channels,
                    "image_height",
                    "image_width",
                ],
                device=self.device,
            ),
        )


class Decoder(Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",
        mid_block_add_attention: bool = True,
        use_post_quant_conv: bool = True,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype

        self.post_quant_conv = (
            Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=in_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                has_bias=True,
                device=device,
                permute=True,
            )
            if use_post_quant_conv
            else None
        )

        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[-1],
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        temb_channels = in_channels if norm_type == "spatial" else None
        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=temb_channels,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            device=device,
            dtype=dtype,
        )

        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if up_block_type != "UpDecoderBlock2D":
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")
            up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block + 1,
                    resnet_eps=1e-6,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_upsample=not is_final_block,
                    temb_channels=temb_channels,
                    device=device,
                    dtype=dtype,
                )
            )
        self.up_blocks = LayerList(up_blocks)

        if norm_type == "spatial":
            raise NotImplementedError("SpatialNorm not implemented in MAX VAE")
        self.conv_norm_out = VAEGroupNorm(
            num_groups=norm_num_groups,
            num_channels=block_out_channels[0],
            eps=1e-6,
            affine=True,
            dtype=dtype,
            device=device,
        )
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )

    def __call__(
        self, z: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        sample = self.conv_in(z)
        sample = self.mid_block(sample, temb)
        for up_block in self.up_blocks:
            sample = up_block(sample, temb)
        sample = ops.silu(self.conv_norm_out(sample))
        return self.conv_out(sample)

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.dtype,
                shape=[
                    "batch_size",
                    self.in_channels,
                    "latent_height",
                    "latent_width",
                ],
                device=self.device,
            ),
        )


class DiagonalGaussianDistribution:
    def __init__(self, mean: Buffer, logvar: Buffer) -> None:
        self.mean = mean
        self.logvar = logvar

    def mode(self) -> Buffer:
        return self.mean

    def sample(self, generator: object | None = None) -> Buffer:
        del generator
        mean = self.mean.to_numpy().astype(np.float32, copy=False)
        logvar = np.clip(
            self.logvar.to_numpy().astype(np.float32, copy=False),
            -30.0,
            20.0,
        )
        sample = mean + np.exp(0.5 * logvar) * np.random.randn(*mean.shape)
        sample = np.ascontiguousarray(sample.astype(np.float32, copy=False))
        return Buffer.from_numpy(sample).to(self.mean.device)
