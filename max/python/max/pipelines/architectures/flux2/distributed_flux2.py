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

"""Distributed (tensor-parallel) Flux2 Transformer."""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn.comm import Signals
from max.nn.layer import Module
from max.nn.linear import Linear

from .flux2 import Flux2Modulation, Flux2TimestepGuidanceEmbeddings
from .layers.distributed_flux2_attention import (
    DistributedFlux2Attention,
    DistributedFlux2FeedForward,
    DistributedFlux2ParallelSelfAttention,
)
from .layers.embeddings import TimestepEmbedding, Timesteps
from .layers.flux2_attention import Flux2PosEmbed
from .layers.normalizations import AdaLayerNormContinuous, LayerNorm
from .model_config import Flux2Config


class DistributedFlux2TransformerBlock(Module):
    """Tensor-parallel dual-stream transformer block."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dtype: DType,
        devices: list[DeviceRef],
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_devices = len(devices)

        # Norms are replicated (no learnable params, elementwise_affine=False)
        self.norm1_shards = [
            LayerNorm(
                dim, dtype=dtype, device=dev, eps=eps,
                elementwise_affine=False, use_bias=False,
            )
            for dev in devices
        ]
        self.norm1_context_shards = [
            LayerNorm(
                dim, dtype=dtype, device=dev, eps=eps,
                elementwise_affine=False, use_bias=False,
            )
            for dev in devices
        ]

        self.attn = DistributedFlux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            dtype=dtype,
            devices=devices,
        )

        self.norm2_shards = [
            LayerNorm(
                dim, dtype=dtype, device=dev, eps=eps,
                elementwise_affine=False, use_bias=False,
            )
            for dev in devices
        ]
        self.ff = DistributedFlux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias,
            dtype=dtype, devices=devices,
        )

        self.norm2_context_shards = [
            LayerNorm(
                dim, dtype=dtype, device=dev, eps=eps,
                elementwise_affine=False, use_bias=False,
            )
            for dev in devices
        ]
        self.ff_context = DistributedFlux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias,
            dtype=dtype, devices=devices,
        )

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        encoder_hidden_states_list: list[TensorValue],
        temb_mod_params_img: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        temb_mod_params_txt: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        signal_buffers: list,
        image_rotary_emb_list: list[
            tuple[TensorValue, TensorValue]
        ]
        | None = None,
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        # Modulation params are per-device lists after broadcast
        (shift_msa_list, scale_msa_list, gate_msa_list), (
            shift_mlp_list, scale_mlp_list, gate_mlp_list,
        ) = temb_mod_params_img
        (c_shift_msa_list, c_scale_msa_list, c_gate_msa_list), (
            c_shift_mlp_list, c_scale_mlp_list, c_gate_mlp_list,
        ) = temb_mod_params_txt

        # Apply norms + modulation on each device
        norm_hidden = [
            (1 + scale_msa_list[i]) * self.norm1_shards[i](hidden_states_list[i])
            + shift_msa_list[i]
            for i in range(self.num_devices)
        ]
        norm_encoder = [
            (1 + c_scale_msa_list[i])
            * self.norm1_context_shards[i](encoder_hidden_states_list[i])
            + c_shift_msa_list[i]
            for i in range(self.num_devices)
        ]

        # Distributed dual-stream attention
        attn_hidden, attn_encoder = self.attn(
            norm_hidden,
            signal_buffers,
            encoder_hidden_states_list=norm_encoder,
            image_rotary_emb_list=image_rotary_emb_list,
        )

        # Residual + gate for image stream
        hidden_states_list = [
            h + gate_msa_list[i] * a
            for i, (h, a) in enumerate(zip(hidden_states_list, attn_hidden))
        ]

        # MLP for image stream
        norm_hidden = [
            self.norm2_shards[i](hidden_states_list[i]) * (1 + scale_mlp_list[i])
            + shift_mlp_list[i]
            for i in range(self.num_devices)
        ]
        ff_output = self.ff(norm_hidden, signal_buffers)
        hidden_states_list = [
            h + gate_mlp_list[i] * f
            for i, (h, f) in enumerate(zip(hidden_states_list, ff_output))
        ]

        # Residual + gate for text stream
        encoder_hidden_states_list = [
            e + c_gate_msa_list[i] * a
            for i, (e, a) in enumerate(zip(encoder_hidden_states_list, attn_encoder))
        ]

        # MLP for text stream
        norm_encoder = [
            self.norm2_context_shards[i](encoder_hidden_states_list[i])
            * (1 + c_scale_mlp_list[i])
            + c_shift_mlp_list[i]
            for i in range(self.num_devices)
        ]
        context_ff_output = self.ff_context(norm_encoder, signal_buffers)
        encoder_hidden_states_list = [
            e + c_gate_mlp_list[i] * f
            for i, (e, f) in enumerate(zip(encoder_hidden_states_list, context_ff_output))
        ]

        # float16 clamping
        encoder_hidden_states_list = [
            ops.min(ops.max(e, -65504), 65504)
            if e.dtype == DType.float16
            else e
            for e in encoder_hidden_states_list
        ]

        return encoder_hidden_states_list, hidden_states_list


class DistributedFlux2SingleTransformerBlock(Module):
    """Tensor-parallel single-stream transformer block."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dtype: DType,
        devices: list[DeviceRef],
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_devices = len(devices)
        self.norm_shards = [
            LayerNorm(
                dim, dtype=dtype, device=dev, eps=eps,
                elementwise_affine=False, use_bias=False,
            )
            for dev in devices
        ]
        self.attn = DistributedFlux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            dtype=dtype,
            devices=devices,
        )

    def __call__(
        self,
        hidden_states_list: list[TensorValue],
        signal_buffers: list,
        encoder_hidden_states_list: list[TensorValue] | None = None,
        temb_mod_params: tuple[TensorValue, TensorValue, TensorValue]
        | None = None,
        image_rotary_emb_list: list[
            tuple[TensorValue, TensorValue]
        ]
        | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | Dim | None = None,
    ) -> list[TensorValue] | tuple[list[TensorValue], list[TensorValue]]:
        if encoder_hidden_states_list is not None:
            text_seq_len = encoder_hidden_states_list[0].shape[1]
            hidden_states_list = [
                ops.concat([e, h], axis=1)
                for e, h in zip(
                    encoder_hidden_states_list, hidden_states_list
                )
            ]

        if temb_mod_params is None:
            raise ValueError("temb_mod_params cannot be None")
        # Per-device lists after broadcast
        mod_shift_list, mod_scale_list, mod_gate_list = temb_mod_params

        norm_hidden = [
            (1 + mod_scale_list[i]) * self.norm_shards[i](hidden_states_list[i])
            + mod_shift_list[i]
            for i in range(self.num_devices)
        ]

        attn_output = self.attn(
            norm_hidden,
            signal_buffers,
            image_rotary_emb_list=image_rotary_emb_list,
        )

        hidden_states_list = [
            h + mod_gate_list[i] * a
            for i, (h, a) in enumerate(zip(hidden_states_list, attn_output))
        ]

        # float16 clamping
        hidden_states_list = [
            ops.min(ops.max(h, -65504), 65504)
            if h.dtype == DType.float16
            else h
            for h in hidden_states_list
        ]

        if split_hidden_states:
            if text_seq_len is None:
                raise ValueError(
                    "text_seq_len is required when splitting"
                )
            encoder_list = [h[:, :text_seq_len, :] for h in hidden_states_list]
            hidden_list = [h[:, text_seq_len:, :] for h in hidden_states_list]
            return encoder_list, hidden_list
        return hidden_states_list


class DistributedFlux2Transformer2DModel(Module):
    """Tensor-parallel Flux2 Transformer.

    Inputs are broadcast to all devices, transformer blocks run in TP,
    and the final output is gathered back to device 0.
    """

    def __init__(self, config: Flux2Config, devices: list[DeviceRef]) -> None:
        super().__init__()
        patch_size = config.patch_size
        in_channels = config.in_channels
        out_channels = config.out_channels
        num_layers = config.num_layers
        num_single_layers = config.num_single_layers
        attention_head_dim = config.attention_head_dim
        num_attention_heads = config.num_attention_heads
        joint_attention_dim = config.joint_attention_dim
        timestep_guidance_channels = config.timestep_guidance_channels
        mlp_ratio = config.mlp_ratio
        axes_dims_rope = config.axes_dims_rope
        rope_theta = config.rope_theta
        dtype = config.dtype
        eps = config.eps

        self.devices = devices
        self.num_devices = len(devices)
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim

        # These are small and replicated on device 0 only
        self.pos_embed = Flux2PosEmbed(
            theta=rope_theta, axes_dim=axes_dims_rope
        )
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=getattr(config, "guidance_embeds", True),
            dtype=dtype,
            device=devices[0],
        )

        # Modulations are replicated on device 0, results broadcast
        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim,
            dtype=dtype, device=devices[0], mod_param_sets=2, bias=False,
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim,
            dtype=dtype, device=devices[0], mod_param_sets=2, bias=False,
        )
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim,
            dtype=dtype, device=devices[0], mod_param_sets=1, bias=False,
        )

        # Input/context embedders: replicated on device 0, results broadcast
        self.x_embedder = Linear(
            in_dim=in_channels, out_dim=self.inner_dim,
            dtype=dtype, device=devices[0], has_bias=False,
        )
        self.context_embedder = Linear(
            in_dim=joint_attention_dim, out_dim=self.inner_dim,
            dtype=dtype, device=devices[0], has_bias=False,
        )

        # Distributed transformer blocks — must use LayerList for load_state_dict
        from max.nn.layer import LayerList

        self.transformer_blocks = LayerList(
            [
                DistributedFlux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    devices=devices,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = LayerList(
            [
                DistributedFlux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    devices=devices,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # Output norm and projection on device 0
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False,
            dtype=dtype, device=devices[0], eps=eps, bias=False,
        )
        self.proj_out = Linear(
            in_dim=self.inner_dim,
            out_dim=patch_size * patch_size * self.out_channels,
            dtype=dtype, device=devices[0], has_bias=False,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        device = self.devices[0]
        signals = Signals(devices=self.devices)

        input_types: list[TensorType | BufferType] = [
            TensorType(
                self.max_dtype,
                shape=["batch_size", "image_seq_len", self.in_channels],
                device=device,
            ),
            TensorType(
                self.max_dtype,
                shape=[
                    "batch_size",
                    "text_seq_len",
                    self.joint_attention_dim,
                ],
                device=device,
            ),
            TensorType(
                self.max_dtype, shape=["batch_size"], device=device
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "image_seq_len", 4],
                device=device,
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "text_seq_len", 4],
                device=device,
            ),
            TensorType(
                self.max_dtype, shape=["batch_size"], device=device
            ),
        ]
        input_types.extend(signals.input_types())
        return tuple(input_types)

    def _broadcast_mod_params(
        self,
        mod_params: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        signal_buffers: list,
    ) -> tuple[
        tuple[list[TensorValue], list[TensorValue], list[TensorValue]],
        tuple[list[TensorValue], list[TensorValue], list[TensorValue]],
    ]:
        """Broadcast double-stream modulation params to all devices."""
        (s0, sc0, g0), (s1, sc1, g1) = mod_params
        return (
            (
                ops.distributed_broadcast(s0, signal_buffers),
                ops.distributed_broadcast(sc0, signal_buffers),
                ops.distributed_broadcast(g0, signal_buffers),
            ),
            (
                ops.distributed_broadcast(s1, signal_buffers),
                ops.distributed_broadcast(sc1, signal_buffers),
                ops.distributed_broadcast(g1, signal_buffers),
            ),
        )

    def _broadcast_mod_single(
        self,
        mod_params: tuple[TensorValue, TensorValue, TensorValue],
        signal_buffers: list,
    ) -> tuple[list[TensorValue], list[TensorValue], list[TensorValue]]:
        """Broadcast single-stream modulation params to all devices."""
        s, sc, g = mod_params
        return (
            ops.distributed_broadcast(s, signal_buffers),
            ops.distributed_broadcast(sc, signal_buffers),
            ops.distributed_broadcast(g, signal_buffers),
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        guidance: TensorValue,
        signal_buffers: list,
    ) -> tuple[TensorValue]:
        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = ops.cast(timestep * 1000.0, hidden_states.dtype)
        guidance = ops.cast(guidance * 1000.0, hidden_states.dtype)
        temb = self.time_guidance_embed(timestep, guidance)

        # Compute modulations on device 0
        double_stream_mod_img_tuple = self.double_stream_modulation_img(temb)
        double_stream_mod_txt_tuple = self.double_stream_modulation_txt(temb)
        single_stream_mod_tuple = self.single_stream_modulation(temb)
        double_stream_mod_img = (
            double_stream_mod_img_tuple[0],
            double_stream_mod_img_tuple[1],
        )
        double_stream_mod_txt = (
            double_stream_mod_txt_tuple[0],
            double_stream_mod_txt_tuple[1],
        )
        single_stream_mod = single_stream_mod_tuple[0]

        # Embed inputs on device 0
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Compute RoPE on device 0
        ids = ops.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        # Broadcast to all devices
        hidden_states_list = ops.distributed_broadcast(
            hidden_states, signal_buffers
        )
        encoder_hidden_states_list = ops.distributed_broadcast(
            encoder_hidden_states, signal_buffers
        )
        # Broadcast RoPE to all devices
        cos, sin = image_rotary_emb
        cos_list = ops.distributed_broadcast(cos, signal_buffers)
        sin_list = ops.distributed_broadcast(sin, signal_buffers)
        image_rotary_emb_list = list(zip(cos_list, sin_list))

        # Broadcast modulation params to all devices
        double_stream_mod_img = self._broadcast_mod_params(
            double_stream_mod_img, signal_buffers
        )
        double_stream_mod_txt = self._broadcast_mod_params(
            double_stream_mod_txt, signal_buffers
        )
        single_stream_mod = self._broadcast_mod_single(
            single_stream_mod, signal_buffers
        )

        # Double-stream blocks
        for block in self.transformer_blocks:
            encoder_hidden_states_list, hidden_states_list = block(
                hidden_states_list=hidden_states_list,
                encoder_hidden_states_list=encoder_hidden_states_list,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                signal_buffers=signal_buffers,
                image_rotary_emb_list=image_rotary_emb_list,
            )

        # Concatenate text + image for single-stream
        hidden_states_list = [
            ops.concat([e, h], axis=1)
            for e, h in zip(
                encoder_hidden_states_list, hidden_states_list
            )
        ]

        # Single-stream blocks
        for single_block in self.single_transformer_blocks:
            hidden_states_list = single_block(
                hidden_states_list=hidden_states_list,
                signal_buffers=signal_buffers,
                encoder_hidden_states_list=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb_list=image_rotary_emb_list,
                split_hidden_states=False,
            )

        # Gather back to device 0: take only device 0's result
        hidden_states = hidden_states_list[0]
        hidden_states = hidden_states[:, num_txt_tokens:, :]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return (output,)
