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

"""Ring (context-parallel) Flux2 Transformer.

Ring CP shards the sequence across devices. In each attention layer,
K and V are allgathered so every device sees the full sequence, while
Q stays local. This is simpler than Ulysses (no head reshuffling) and
has no head-count divisibility requirement.

Note: This uses allgather emulation. True ring attention rotates KV
via P2P send/recv with online softmax, avoiding full materialization.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn.comm import Signals
from max.nn.layer import Module
from max.nn.linear import Linear

from .flux2 import Flux2Modulation, Flux2TimestepGuidanceEmbeddings
from .layers.flux2_attention import Flux2PosEmbed
from .layers.normalizations import AdaLayerNormContinuous, LayerNorm
from .layers.ring_flux2_attention import (
    RingFlux2Attention,
    RingFlux2FeedForward,
    RingFlux2ParallelSelfAttention,
)
from .model_config import Flux2Config


class RingFlux2TransformerBlock(Module):
    """Ring context-parallel dual-stream transformer block."""

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

        self.norm1_shards = [
            LayerNorm(dim, dtype=dtype, device=dev, eps=eps,
                      elementwise_affine=False, use_bias=False)
            for dev in devices
        ]
        self.norm1_context_shards = [
            LayerNorm(dim, dtype=dtype, device=dev, eps=eps,
                      elementwise_affine=False, use_bias=False)
            for dev in devices
        ]

        self.attn = RingFlux2Attention(
            query_dim=dim, added_kv_proj_dim=dim,
            dim_head=attention_head_dim, heads=num_attention_heads,
            out_dim=dim, bias=bias, added_proj_bias=bias, out_bias=bias,
            eps=eps, dtype=dtype, devices=devices,
        )

        self.norm2_shards = [
            LayerNorm(dim, dtype=dtype, device=dev, eps=eps,
                      elementwise_affine=False, use_bias=False)
            for dev in devices
        ]
        self.ff = RingFlux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias,
            dtype=dtype, devices=devices,
        )

        self.norm2_context_shards = [
            LayerNorm(dim, dtype=dtype, device=dev, eps=eps,
                      elementwise_affine=False, use_bias=False)
            for dev in devices
        ]
        self.ff_context = RingFlux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias,
            dtype=dtype, devices=devices,
        )

    def __call__(
        self,
        hidden_states_list, encoder_hidden_states_list,
        temb_mod_params_img, temb_mod_params_txt,
        signal_buffers, image_rotary_emb_list=None,
    ):
        (shift_msa_list, scale_msa_list, gate_msa_list), (
            shift_mlp_list, scale_mlp_list, gate_mlp_list,
        ) = temb_mod_params_img
        (c_shift_msa_list, c_scale_msa_list, c_gate_msa_list), (
            c_shift_mlp_list, c_scale_mlp_list, c_gate_mlp_list,
        ) = temb_mod_params_txt

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

        attn_hidden, attn_encoder = self.attn(
            norm_hidden, signal_buffers,
            encoder_hidden_states_list=norm_encoder,
            image_rotary_emb_list=image_rotary_emb_list,
        )

        hidden_states_list = [
            h + gate_msa_list[i] * a
            for i, (h, a) in enumerate(zip(hidden_states_list, attn_hidden))
        ]

        norm_hidden = [
            self.norm2_shards[i](hidden_states_list[i]) * (1 + scale_mlp_list[i])
            + shift_mlp_list[i]
            for i in range(self.num_devices)
        ]
        ff_output = self.ff(norm_hidden)
        hidden_states_list = [
            h + gate_mlp_list[i] * f
            for i, (h, f) in enumerate(zip(hidden_states_list, ff_output))
        ]

        encoder_hidden_states_list = [
            e + c_gate_msa_list[i] * a
            for i, (e, a) in enumerate(zip(encoder_hidden_states_list, attn_encoder))
        ]

        norm_encoder = [
            self.norm2_context_shards[i](encoder_hidden_states_list[i])
            * (1 + c_scale_mlp_list[i]) + c_shift_mlp_list[i]
            for i in range(self.num_devices)
        ]
        context_ff_output = self.ff_context(norm_encoder)
        encoder_hidden_states_list = [
            e + c_gate_mlp_list[i] * f
            for i, (e, f) in enumerate(zip(encoder_hidden_states_list, context_ff_output))
        ]

        encoder_hidden_states_list = [
            ops.min(ops.max(e, -65504), 65504) if e.dtype == DType.float16 else e
            for e in encoder_hidden_states_list
        ]

        return encoder_hidden_states_list, hidden_states_list


class RingFlux2SingleTransformerBlock(Module):
    """Ring context-parallel single-stream transformer block."""

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
            LayerNorm(dim, dtype=dtype, device=dev, eps=eps,
                      elementwise_affine=False, use_bias=False)
            for dev in devices
        ]
        self.attn = RingFlux2ParallelSelfAttention(
            query_dim=dim, dim_head=attention_head_dim,
            heads=num_attention_heads, out_dim=dim,
            bias=bias, out_bias=bias, eps=eps,
            mlp_ratio=mlp_ratio, mlp_mult_factor=2,
            dtype=dtype, devices=devices,
        )

    def __call__(
        self, hidden_states_list, signal_buffers,
        encoder_hidden_states_list=None, temb_mod_params=None,
        image_rotary_emb_list=None, split_hidden_states=False,
        text_seq_len=None,
    ):
        if encoder_hidden_states_list is not None:
            text_seq_len = encoder_hidden_states_list[0].shape[1]
            hidden_states_list = [
                ops.concat([e, h], axis=1)
                for e, h in zip(encoder_hidden_states_list, hidden_states_list)
            ]

        if temb_mod_params is None:
            raise ValueError("temb_mod_params cannot be None")
        mod_shift_list, mod_scale_list, mod_gate_list = temb_mod_params

        norm_hidden = [
            (1 + mod_scale_list[i]) * self.norm_shards[i](hidden_states_list[i])
            + mod_shift_list[i]
            for i in range(self.num_devices)
        ]

        attn_output = self.attn(
            norm_hidden, signal_buffers,
            image_rotary_emb_list=image_rotary_emb_list,
        )

        hidden_states_list = [
            h + mod_gate_list[i] * a
            for i, (h, a) in enumerate(zip(hidden_states_list, attn_output))
        ]

        hidden_states_list = [
            ops.min(ops.max(h, -65504), 65504) if h.dtype == DType.float16 else h
            for h in hidden_states_list
        ]

        if split_hidden_states:
            if text_seq_len is None:
                raise ValueError("text_seq_len is required when splitting")
            encoder_list = [h[:, :text_seq_len, :] for h in hidden_states_list]
            hidden_list = [h[:, text_seq_len:, :] for h in hidden_states_list]
            return encoder_list, hidden_list
        return hidden_states_list


class RingFlux2Transformer2DModel(Module):
    """Ring context-parallel Flux2 Transformer."""

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

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim, bias=False,
            guidance_embeds=getattr(config, "guidance_embeds", True),
            dtype=dtype, device=devices[0],
        )

        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim, dtype=dtype, device=devices[0], mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim, dtype=dtype, device=devices[0], mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim, dtype=dtype, device=devices[0], mod_param_sets=1, bias=False)

        self.x_embedder = Linear(
            in_dim=in_channels, out_dim=self.inner_dim,
            dtype=dtype, device=devices[0], has_bias=False)
        self.context_embedder = Linear(
            in_dim=joint_attention_dim, out_dim=self.inner_dim,
            dtype=dtype, device=devices[0], has_bias=False)

        from max.nn.layer import LayerList

        self.transformer_blocks = LayerList([
            RingFlux2TransformerBlock(
                dim=self.inner_dim, num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim, dtype=dtype,
                devices=devices, mlp_ratio=mlp_ratio, eps=eps, bias=False,
            )
            for _ in range(num_layers)
        ])

        self.single_transformer_blocks = LayerList([
            RingFlux2SingleTransformerBlock(
                dim=self.inner_dim, num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim, dtype=dtype,
                devices=devices, mlp_ratio=mlp_ratio, eps=eps, bias=False,
            )
            for _ in range(num_single_layers)
        ])

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
            TensorType(self.max_dtype, shape=["batch_size", "image_seq_len", self.in_channels], device=device),
            TensorType(self.max_dtype, shape=["batch_size", "text_seq_len", self.joint_attention_dim], device=device),
            TensorType(self.max_dtype, shape=["batch_size"], device=device),
            TensorType(DType.int64, shape=["batch_size", "image_seq_len", 4], device=device),
            TensorType(DType.int64, shape=["batch_size", "text_seq_len", 4], device=device),
            TensorType(self.max_dtype, shape=["batch_size"], device=device),
        ]
        input_types.extend(signals.input_types())
        return tuple(input_types)

    def _broadcast_mod_params(self, mod_params, signal_buffers):
        (s0, sc0, g0), (s1, sc1, g1) = mod_params
        return (
            (ops.distributed_broadcast(s0, signal_buffers),
             ops.distributed_broadcast(sc0, signal_buffers),
             ops.distributed_broadcast(g0, signal_buffers)),
            (ops.distributed_broadcast(s1, signal_buffers),
             ops.distributed_broadcast(sc1, signal_buffers),
             ops.distributed_broadcast(g1, signal_buffers)),
        )

    def _broadcast_mod_single(self, mod_params, signal_buffers):
        s, sc, g = mod_params
        return (
            ops.distributed_broadcast(s, signal_buffers),
            ops.distributed_broadcast(sc, signal_buffers),
            ops.distributed_broadcast(g, signal_buffers),
        )

    def _scatter_sequence(self, tensor, signal_buffers):
        n = self.num_devices
        replicated = ops.distributed_broadcast(tensor, signal_buffers)
        s_total = tensor.shape[1]
        s_local = s_total // n
        return [replicated[i][:, i * s_local : (i + 1) * s_local, :] for i in range(n)]

    def _scatter_ids(self, ids, signal_buffers):
        n = self.num_devices
        replicated = ops.distributed_broadcast(ids, signal_buffers)
        s_total = ids.shape[0]
        s_local = s_total // n
        return [replicated[i][i * s_local : (i + 1) * s_local, :] for i in range(n)]

    def __call__(
        self, hidden_states, encoder_hidden_states, timestep,
        img_ids, txt_ids, guidance, signal_buffers,
    ) -> tuple[TensorValue]:
        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = ops.cast(timestep * 1000.0, hidden_states.dtype)
        guidance = ops.cast(guidance * 1000.0, hidden_states.dtype)
        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod_tuple = self.single_stream_modulation(temb)
        double_stream_mod_img = (double_stream_mod_img[0], double_stream_mod_img[1])
        double_stream_mod_txt = (double_stream_mod_txt[0], double_stream_mod_txt[1])
        single_stream_mod = single_stream_mod_tuple[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = ops.concat([txt_ids, img_ids], axis=0)
        cos, sin = self.pos_embed(ids)

        # Scatter sequences and RoPE independently for text/image
        hidden_states_list = self._scatter_sequence(hidden_states, signal_buffers)
        encoder_hidden_states_list = self._scatter_sequence(encoder_hidden_states, signal_buffers)

        txt_cos = cos[:num_txt_tokens, :]
        img_cos = cos[num_txt_tokens:, :]
        txt_sin = sin[:num_txt_tokens, :]
        img_sin = sin[num_txt_tokens:, :]

        txt_cos_chunks = self._scatter_ids(txt_cos, signal_buffers)
        img_cos_chunks = self._scatter_ids(img_cos, signal_buffers)
        txt_sin_chunks = self._scatter_ids(txt_sin, signal_buffers)
        img_sin_chunks = self._scatter_ids(img_sin, signal_buffers)

        cos_chunks = [ops.concat([txt_cos_chunks[i], img_cos_chunks[i]], axis=0) for i in range(self.num_devices)]
        sin_chunks = [ops.concat([txt_sin_chunks[i], img_sin_chunks[i]], axis=0) for i in range(self.num_devices)]
        image_rotary_emb_list = list(zip(cos_chunks, sin_chunks))

        double_stream_mod_img = self._broadcast_mod_params(double_stream_mod_img, signal_buffers)
        double_stream_mod_txt = self._broadcast_mod_params(double_stream_mod_txt, signal_buffers)
        single_stream_mod = self._broadcast_mod_single(single_stream_mod, signal_buffers)

        for block in self.transformer_blocks:
            encoder_hidden_states_list, hidden_states_list = block(
                hidden_states_list, encoder_hidden_states_list,
                double_stream_mod_img, double_stream_mod_txt,
                signal_buffers, image_rotary_emb_list,
            )

        text_seq_lens = [encoder_hidden_states_list[i].shape[1] for i in range(self.num_devices)]

        hidden_states_list = [
            ops.concat([e, h], axis=1)
            for e, h in zip(encoder_hidden_states_list, hidden_states_list)
        ]

        for single_block in self.single_transformer_blocks:
            hidden_states_list = single_block(
                hidden_states_list, signal_buffers,
                encoder_hidden_states_list=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb_list=image_rotary_emb_list,
                split_hidden_states=False,
            )

        # Split text/image, gather image only
        image_chunks = [
            hidden_states_list[i][:, text_seq_lens[i]:, :]
            for i in range(self.num_devices)
        ]
        gathered = ops.allgather(image_chunks, signal_buffers, axis=1)
        hidden_states = gathered[0]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return (output,)
