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

"""Ulysses (context-parallel) Flux2 Transformer.

Unlike tensor parallelism which shards weights, context parallelism
shards the activation sequence across devices.  Each device holds a
chunk of the sequence with fully replicated weights.

Workflow:
  1. Compute embeddings on device 0.
  2. Scatter sequence chunks to all devices (each gets S/N tokens).
  3. In each attention layer:
     a. Project Q, K, V locally.
     b. All-to-all: gather full sequence, shard heads.
     c. Local flash attention on full sequence with local heads.
     d. All-to-all: gather full heads, shard sequence back.
  4. FFN runs locally (no communication needed).
  5. Gather results back to device 0.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn.comm import Signals
from max.nn.layer import Module
from max.nn.linear import Linear

from .flux2 import Flux2Modulation, Flux2TimestepGuidanceEmbeddings
from .layers.embeddings import TimestepEmbedding, Timesteps
from .layers.flux2_attention import Flux2PosEmbed
from .layers.normalizations import AdaLayerNormContinuous, LayerNorm
from .layers.ulysses_flux2_attention import (
    UlyssesFlux2Attention,
    UlyssesFlux2FeedForward,
    UlyssesFlux2ParallelSelfAttention,
)
from .model_config import Flux2Config


class UlyssesFlux2TransformerBlock(Module):
    """Context-parallel dual-stream transformer block."""

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

        self.attn = UlyssesFlux2Attention(
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
        # FFN is local per-device in CP mode (no communication)
        self.ff = UlyssesFlux2FeedForward(
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
        self.ff_context = UlyssesFlux2FeedForward(
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

        # Context-parallel dual-stream attention
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

        # MLP for image stream (local, no communication)
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

        # Residual + gate for text stream
        encoder_hidden_states_list = [
            e + c_gate_msa_list[i] * a
            for i, (e, a) in enumerate(zip(encoder_hidden_states_list, attn_encoder))
        ]

        # MLP for text stream (local)
        norm_encoder = [
            self.norm2_context_shards[i](encoder_hidden_states_list[i])
            * (1 + c_scale_mlp_list[i])
            + c_shift_mlp_list[i]
            for i in range(self.num_devices)
        ]
        context_ff_output = self.ff_context(norm_encoder)
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


class UlyssesFlux2SingleTransformerBlock(Module):
    """Context-parallel single-stream transformer block."""

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
        self.attn = UlyssesFlux2ParallelSelfAttention(
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
                raise ValueError("text_seq_len is required when splitting")
            encoder_list = [h[:, :text_seq_len, :] for h in hidden_states_list]
            hidden_list = [h[:, text_seq_len:, :] for h in hidden_states_list]
            return encoder_list, hidden_list
        return hidden_states_list


class UlyssesFlux2Transformer2DModel(Module):
    """Context-parallel Flux2 Transformer.

    Sequence is scattered across devices. Each transformer block uses
    Ulysses all-to-all for attention and local FFN. Output is gathered
    back to device 0.
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

        # Embeddings on device 0 (small, no need to distribute)
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

        # Modulations on device 0, results broadcast
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

        # Input/context embedders on device 0
        self.x_embedder = Linear(
            in_dim=in_channels, out_dim=self.inner_dim,
            dtype=dtype, device=devices[0], has_bias=False,
        )
        self.context_embedder = Linear(
            in_dim=joint_attention_dim, out_dim=self.inner_dim,
            dtype=dtype, device=devices[0], has_bias=False,
        )

        # Context-parallel transformer blocks
        from max.nn.layer import LayerList

        self.transformer_blocks = LayerList(
            [
                UlyssesFlux2TransformerBlock(
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
                UlyssesFlux2SingleTransformerBlock(
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
        s, sc, g = mod_params
        return (
            ops.distributed_broadcast(s, signal_buffers),
            ops.distributed_broadcast(sc, signal_buffers),
            ops.distributed_broadcast(g, signal_buffers),
        )

    def _scatter_sequence(
        self,
        tensor: TensorValue,
        signal_buffers: list,
    ) -> list[TensorValue]:
        """Scatter sequence dimension across devices.

        Input:  [B, S, D] on device 0
        Output: list of [B, S/N, D], one per device
        """
        n = self.num_devices
        # Broadcast full tensor to all devices first
        replicated = ops.distributed_broadcast(tensor, signal_buffers)
        # Each device slices its chunk of the sequence
        s_total = tensor.shape[1]
        s_local = s_total // n
        chunks = [
            replicated[i][:, i * s_local : (i + 1) * s_local, :]
            for i in range(n)
        ]
        return chunks

    def _scatter_ids(
        self,
        ids: TensorValue,
        signal_buffers: list,
    ) -> list[TensorValue]:
        """Scatter position IDs across devices.

        Input:  [S, D] on device 0
        Output: list of [S/N, D], one per device
        """
        n = self.num_devices
        replicated = ops.distributed_broadcast(ids, signal_buffers)
        s_total = ids.shape[0]
        s_local = s_total // n
        chunks = [
            replicated[i][i * s_local : (i + 1) * s_local, :]
            for i in range(n)
        ]
        return chunks

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

        # Compute full RoPE on device 0, then scatter chunks
        ids = ops.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)
        cos, sin = image_rotary_emb

        # Scatter sequence to all devices:
        # hidden_states [B, S_img, D] -> list of [B, S_img/N, D]
        # encoder_hidden_states [B, S_txt, D] -> list of [B, S_txt/N, D]
        hidden_states_list = self._scatter_sequence(
            hidden_states, signal_buffers
        )
        encoder_hidden_states_list = self._scatter_sequence(
            encoder_hidden_states, signal_buffers
        )

        # Scatter RoPE embeddings: cos/sin are [S_txt + S_img, D].
        # Text and image sequences are scattered INDEPENDENTLY, so we
        # must split RoPE into text/image parts, scatter each separately,
        # and recombine per device to match [text_chunk, image_chunk].
        txt_cos = cos[:num_txt_tokens, :]
        img_cos = cos[num_txt_tokens:, :]
        txt_sin = sin[:num_txt_tokens, :]
        img_sin = sin[num_txt_tokens:, :]

        txt_cos_chunks = self._scatter_ids(txt_cos, signal_buffers)
        img_cos_chunks = self._scatter_ids(img_cos, signal_buffers)
        txt_sin_chunks = self._scatter_ids(txt_sin, signal_buffers)
        img_sin_chunks = self._scatter_ids(img_sin, signal_buffers)

        cos_chunks = [
            ops.concat([txt_cos_chunks[i], img_cos_chunks[i]], axis=0)
            for i in range(self.num_devices)
        ]
        sin_chunks = [
            ops.concat([txt_sin_chunks[i], img_sin_chunks[i]], axis=0)
            for i in range(self.num_devices)
        ]
        image_rotary_emb_list = list(zip(cos_chunks, sin_chunks))

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

        # Save per-device text lengths before concatenation
        text_seq_lens = [
            encoder_hidden_states_list[i].shape[1]
            for i in range(self.num_devices)
        ]

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

        # Split text and image per device, then gather image chunks only.
        # Each device has [text_chunk_i, image_chunk_i] after single-stream.
        # We only need the image part for the final output.
        image_chunks = [
            hidden_states_list[i][:, text_seq_lens[i]:, :]
            for i in range(self.num_devices)
        ]

        # Gather image chunks: [B, S_img/N, D] per device -> [B, S_img, D]
        gathered = ops.allgather(image_chunks, signal_buffers, axis=1)
        hidden_states = gathered[0]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return (output,)
