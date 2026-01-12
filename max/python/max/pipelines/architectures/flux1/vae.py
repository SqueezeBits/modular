# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import max.nn as nn
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.weights import SafetensorWeights
from max.nn.layer.layer_list import LayerList
from max.pipelines.lib.interfaces.configuration_utils import (
    ConfigMixin,
    register_to_config,
)

from .layers.normalizations import WeightedGroupNorm
from .layers.upsampling import Upsample2D
from .layers.vae_attention import Attention


class ResnetBlock2D(nn.Module):
    """Residual block for 2D VAE decoder.

    This module implements a residual block with two convolutional layers,
    group normalization, and optional shortcut connection. It supports
    time embedding conditioning and configurable activation functions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int | None,
        groups: int,
        groups_out: int,
        eps: float = 1e-6,
        non_linearity: str = "silu",
        use_conv_shortcut: bool = False,
        conv_shortcut_bias: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize ResnetBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            temb_channels: Number of time embedding channels (None if not used).
            groups: Number of groups for first GroupNorm.
            groups_out: Number of groups for second GroupNorm.
            eps: Epsilon value for GroupNorm layers.
            non_linearity: Activation function name (e.g., "silu").
            use_conv_shortcut: Whether to use convolutional shortcut.
            conv_shortcut_bias: Whether to use bias in shortcut convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = WeightedGroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            device=device,
            dtype=dtype,
        )

        self.conv1 = nn.Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.norm2 = WeightedGroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
            device=device,
            dtype=dtype,
        )

        self.conv2 = nn.Conv2d(
            kernel_size=3,
            in_channels=out_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.conv_shortcut = None
        if self.use_conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=conv_shortcut_bias,
                device=device,
                permute=True,
            )
        elif in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=conv_shortcut_bias,
                device=device,
                permute=True,
            )

    def __call__(
        self, x: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply ResnetBlock2D forward pass.

        Args:
            x: Input tensor of shape [N, C, H, W].
            temb: Optional time embedding tensor (currently unused).

        Returns:
            Output tensor of shape [N, C_out, H, W] with residual connection.
        """
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )

        h = ops.silu(self.norm1(x))
        h = self.conv1(h)

        h = ops.silu(self.norm2(h))
        h = self.conv2(h)

        return h + shortcut


class UpDecoderBlock2D(nn.Module):
    """Upsampling decoder block for 2D VAE.

    This module consists of multiple ResNet blocks followed by an optional
    upsampling layer. It progressively increases spatial resolution while
    processing features through residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: int | None = None,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize UpDecoderBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            resolution_idx: Optional resolution index for tracking.
            dropout: Dropout rate (currently unused).
            num_layers: Number of ResNet blocks in this decoder block.
            resnet_eps: Epsilon value for ResNet GroupNorm layers.
            resnet_time_scale_shift: Time embedding scale/shift mode.
            resnet_act_fn: Activation function for ResNet blocks.
            resnet_groups: Number of groups for ResNet GroupNorm.
            resnet_pre_norm: Whether to apply normalization before ResNet.
            output_scale_factor: Scaling factor for output (currently unused).
            add_upsample: Whether to add upsampling layer after ResNet blocks.
            temb_channels: Number of time embedding channels (None if not used).
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        resnets_list = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnet = ResnetBlock2D(
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
            resnets_list.append(resnet)
        self.resnets = LayerList(resnets_list)

        if add_upsample:
            upsampler = Upsample2D(
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                name="conv",
                kernel_size=3,
                padding=1,
                bias=True,
                interpolate=True,
                device=device,
                dtype=dtype,
            )
            self.upsamplers = LayerList([upsampler])
        else:
            self.upsamplers = None

    def __call__(
        self, hidden_states: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply UpDecoderBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C_in, H, W].
            temb: Optional time embedding tensor.

        Returns:
            Output tensor of shape [N, C_out, H*2, W*2] (if upsampling) or
            [N, C_out, H, W] (if no upsampling).
        """
        # Process through all resnet blocks
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

        # Apply upsampling if configured (compile-time decision)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class MidBlock2D(nn.Module):
    """Internal MAX module for MidBlock2D graph generation."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int | None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize MidBlock2D module."""
        super().__init__()
        resnets_list = []
        attentions_list = []

        resnet = ResnetBlock2D(
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
        resnets_list.append(resnet)

        for _i in range(num_layers):
            if add_attention:
                attn = Attention(
                    query_dim=in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    num_groups=resnet_groups,
                    eps=resnet_eps,
                    device=device,
                    dtype=dtype,
                )
                attentions_list.append(attn)
            else:
                attentions_list.append(None)

            resnet = ResnetBlock2D(
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
            resnets_list.append(resnet)

        self.resnets = LayerList(resnets_list)
        self.attentions = (
            LayerList(attentions_list) if attentions_list else None
        )

    def __call__(
        self, hidden_states: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply MidBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C, H, W].
            temb: Optional time embedding tensor.

        Returns:
            Output tensor of shape [N, C, H, W] with same spatial dimensions.
        """
        hidden_states = self.resnets[0](hidden_states, temb)

        for i in range(len(self.resnets) - 1):
            if self.attentions is not None and self.attentions[i] is not None:
                hidden_states = self.attentions[i](hidden_states)
            hidden_states = self.resnets[i + 1](hidden_states, temb)

        return hidden_states


@dataclass
class DecoderOutput:
    r"""Output of decoding method.

    Args:
        sample (`TensorValue` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: TensorValue
    commit_loss: TensorValue | None = None


class Decoder(nn.Module):
    """VAE decoder for generating images from latent representations.

    This decoder progressively upsamples latent features through multiple
    decoder blocks, applying ResNet layers and attention mechanisms to
    reconstruct high-resolution images from compressed latent codes.
    """

    def __init__(
        self,
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
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize Decoder module.

        Args:
            in_channels: Number of input channels (latent channels).
            out_channels: Number of output channels (image channels).
            up_block_types: Tuple of upsampling block types.
            block_out_channels: Tuple of channel counts for each decoder block.
            layers_per_block: Number of ResNet layers per decoder block.
            norm_num_groups: Number of groups for GroupNorm layers.
            act_fn: Activation function name (e.g., "silu").
            norm_type: Normalization type ("group" or "spatial").
            mid_block_add_attention: Whether to add attention in middle block.
            use_post_quant_conv: Whether to use post-quantization convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        self.layers_per_block = layers_per_block
        self.session = None

        self.post_quant_conv = None
        if use_post_quant_conv:
            self.post_quant_conv = nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=in_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=True,
                device=device,
                permute=True,
            )

        self.conv_in = nn.Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[-1],
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        temb_channels = in_channels if norm_type == "spatial" else None
        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=temb_channels,
            dropout=0.0,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_time_scale_shift=(
                "default" if norm_type == "group" else norm_type
            ),
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            resnet_pre_norm=True,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            output_scale_factor=1.0,
            device=device,
            dtype=dtype,
        )

        up_blocks_list = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "UpDecoderBlock2D":
                up_block = UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    resolution_idx=i,
                    dropout=0.0,
                    num_layers=self.layers_per_block + 1,
                    resnet_eps=1e-6,
                    resnet_time_scale_shift=norm_type,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    resnet_pre_norm=True,
                    output_scale_factor=1.0,
                    add_upsample=not is_final_block,
                    temb_channels=temb_channels,
                    device=device,
                    dtype=dtype,
                )
                up_blocks_list.append(up_block)
            else:
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

            prev_output_channel = output_channel

        self.up_blocks = LayerList(up_blocks_list)

        if norm_type == "spatial":
            raise NotImplementedError("SpatialNorm not implemented in MAX VAE")
        else:
            self.conv_norm_out = WeightedGroupNorm(
                num_groups=norm_num_groups,
                num_channels=block_out_channels[0],
                eps=1e-6,
                affine=True,
                device=device,
                dtype=dtype,
            )

        self.conv_out = nn.Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

    def __call__(
        self, z: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply Decoder forward pass.

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].
            temb: Optional time embedding tensor.

        Returns:
            Decoded image tensor of shape [N, C_out, H, W] where H and W are
            upsampled from H_latent and W_latent.
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        sample = self.conv_in(z)
        sample = self.mid_block(sample, temb)

        for up_block in self.up_blocks:
            sample = up_block(sample, temb)

        sample = self.conv_norm_out(sample)
        sample = ops.silu(sample)
        sample = self.conv_out(sample)

        return sample

    def input_types(
        self, in_channels: int, device: DeviceRef, dtype: DType
    ) -> tuple[TensorType, ...]:
        """Define input tensor types for the decoder model.

        Args:
            in_channels: Number of input channels (latent channels).
            device: Device reference for module placement.
            dtype: Data type for module parameters.

        Returns:
            Tuple of TensorType specifications for decoder input.
        """
        latent_type = TensorType(
            dtype,
            shape=[
                "batch_size",
                in_channels,
                "latent_height",
                "latent_width",
            ],
            device=device,
        )

        return (latent_type,)

    def load_model(
        self,
        pretrained_model_name_or_path: str,
        device: DeviceRef,
        dtype: DType,
        in_channels: int,
    ) -> None:
        """Load pretrained model weights and compile the decoder graph.

        This method loads SafeTensors weights from the specified path,
        filters decoder and post-quantization convolution weights, and
        compiles the decoder graph for inference.

        Args:
            pretrained_model_name_or_path: Path to pretrained model weights.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
            in_channels: Number of input channels (latent channels).
        """
        if device.is_cpu():
            session = InferenceSession([CPU()])
        else:
            session = InferenceSession([Accelerator()])

        safe_tensor_folder = os.path.join(pretrained_model_name_or_path, "vae")

        if not os.path.isdir(safe_tensor_folder):
            raise ValueError(
                f"VAE model directory not found: {safe_tensor_folder}. "
                f"Please check pretrained_model_name_or_path: {pretrained_model_name_or_path}"
            )

        safetensors_files = [
            Path(safe_tensor_folder) / file
            for file in os.listdir(safe_tensor_folder)
            if file.endswith(".safetensors")
        ]

        if not safetensors_files:
            available_files = os.listdir(safe_tensor_folder)
            raise ValueError(
                f"No .safetensors files found in {safe_tensor_folder}. "
                f"Available files: {available_files}"
            )

        weights = SafetensorWeights(safetensors_files)

        weight_registry = {}
        for name, weight in weights.items():
            if name.startswith("decoder.") or name.startswith(
                "post_quant_conv."
            ):
                # Remove "decoder." or "post_quant_conv." prefix for state_dict loading
                if name.startswith("decoder."):
                    weight_registry[name[len("decoder.") :]] = (
                        weight.data().data
                    )
                elif name.startswith("post_quant_conv."):
                    weight_registry[name[len("post_quant_conv.") :]] = (
                        weight.data().data
                    )

        self.load_state_dict(weight_registry)

        with Graph(
            "vae_decoder",
            input_types=self.input_types(in_channels, device, dtype),
        ) as graph:
            outputs = self(*graph.inputs)
            graph.output(outputs)
            compiled_graph = graph

        self.session = session.load(
            compiled_graph, weights_registry=self.state_dict()
        )


class AutoencoderKL(ConfigMixin):
    r"""A VAE model with KL loss for encoding images into latents and decoding latent representations into images."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: float | None = None,
        latents_mean: tuple[float] | None = None,
        latents_std: tuple[float] | None = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
        pretrained_model_name_or_path: str | None = None,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.bfloat16,
    ):
        """Initialize VAE AutoencoderKL model.

        Args:
            in_channels: Number of input image channels (default: 3 for RGB).
            out_channels: Number of output image channels (default: 3 for RGB).
            down_block_types: Types of downsampling blocks (encoder, currently unused).
            up_block_types: Types of upsampling blocks for decoder.
            block_out_channels: Tuple of channel counts for each decoder block.
            layers_per_block: Number of ResNet layers per decoder block.
            act_fn: Activation function name (e.g., "silu").
            latent_channels: Number of latent space channels.
            norm_num_groups: Number of groups for GroupNorm layers.
            sample_size: Input image size (currently unused).
            scaling_factor: Scaling factor for latent normalization.
            shift_factor: Optional shift factor for latent normalization.
            latents_mean: Optional mean values for latent normalization.
            latents_std: Optional std values for latent normalization.
            force_upcast: Whether to force upcast (currently unused).
            use_quant_conv: Whether to use quantization convolution (currently unused).
            use_post_quant_conv: Whether to use post-quantization convolution.
            mid_block_add_attention: Whether to add attention in decoder middle block.
            pretrained_model_name_or_path: Path to pretrained model weights.
            device: Device reference for model placement.
            dtype: Data type for model parameters.
        """
        self.latent_channels = latent_channels
        self.max_device = device
        self.max_dtype = dtype

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            norm_type="group",
            mid_block_add_attention=mid_block_add_attention,
            use_post_quant_conv=use_post_quant_conv,
            device=device,
            dtype=dtype,
        )

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path is None:
            raise ValueError(
                "pretrained_model_name_or_path is required to load model"
            )
        self.load_model()

    def load_model(self) -> None:
        """Load pretrained model weights and compile the model graph.

        This method delegates decoder graph compilation to the Decoder class.
        """
        self.decoder.load_model(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            device=self.max_device,
            dtype=self.max_dtype,
            in_channels=self.latent_channels,
        )

    def decode(
        self,
        z: Tensor,
        return_dict: bool = True,
        generator: Any | None = None,
    ) -> DecoderOutput | Tensor:
        """Decode a batch of images.

        Args:
            z (`Tensor`): Input batch of latent vectors (MAX Tensor).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.
            generator: Not used, kept for compatibility.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        # Execute decoder using compiled graph
        results = self.decoder.session.execute(z)

        dec = results[0]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
