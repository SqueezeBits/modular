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

"""USP (Unified Sequence Parallelism) Flux2 Transformer.

Combines Ulysses (all-to-all within subgroup) and Ring (KV allgather
within subgroup) for maximum scaling flexibility.

num_devices = ulysses_degree × ring_degree
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.nn.comm import Signals
from max.nn.layer import Module
from max.nn.linear import Linear

from .flux2 import Flux2Modulation, Flux2TimestepGuidanceEmbeddings
from .layers.flux2_attention import Flux2PosEmbed
from .layers.normalizations import AdaLayerNormContinuous, LayerNorm
from .layers.usp_flux2_attention import (
    USPFlux2Attention,
    USPFlux2FeedForward,
    USPFlux2ParallelSelfAttention,
)
from .model_config import Flux2Config


class USPFlux2TransformerBlock(Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, *,
                 dtype, devices, ulysses_degree, ring_degree,
                 mlp_ratio=3.0, eps=1e-6, bias=False):
        super().__init__()
        self.num_devices = len(devices)
        self.norm1_shards = [
            LayerNorm(dim, dtype=dtype, device=d, eps=eps,
                      elementwise_affine=False, use_bias=False) for d in devices]
        self.norm1_context_shards = [
            LayerNorm(dim, dtype=dtype, device=d, eps=eps,
                      elementwise_affine=False, use_bias=False) for d in devices]
        self.attn = USPFlux2Attention(
            query_dim=dim, added_kv_proj_dim=dim,
            dim_head=attention_head_dim, heads=num_attention_heads,
            out_dim=dim, bias=bias, added_proj_bias=bias, out_bias=bias,
            eps=eps, dtype=dtype, devices=devices,
            ulysses_degree=ulysses_degree, ring_degree=ring_degree)
        self.norm2_shards = [
            LayerNorm(dim, dtype=dtype, device=d, eps=eps,
                      elementwise_affine=False, use_bias=False) for d in devices]
        self.ff = USPFlux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias,
            dtype=dtype, devices=devices)
        self.norm2_context_shards = [
            LayerNorm(dim, dtype=dtype, device=d, eps=eps,
                      elementwise_affine=False, use_bias=False) for d in devices]
        self.ff_context = USPFlux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias,
            dtype=dtype, devices=devices)

    def __call__(self, hidden_states_list, encoder_hidden_states_list,
                 temb_mod_params_img, temb_mod_params_txt,
                 signal_buffers, image_rotary_emb_list=None):
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt
        n = self.num_devices

        norm_h = [(1 + scale_msa[i]) * self.norm1_shards[i](hidden_states_list[i]) + shift_msa[i] for i in range(n)]
        norm_e = [(1 + c_scale_msa[i]) * self.norm1_context_shards[i](encoder_hidden_states_list[i]) + c_shift_msa[i] for i in range(n)]

        attn_h, attn_e = self.attn(norm_h, signal_buffers, encoder_hidden_states_list=norm_e, image_rotary_emb_list=image_rotary_emb_list)

        hidden_states_list = [h + gate_msa[i] * a for i, (h, a) in enumerate(zip(hidden_states_list, attn_h))]
        norm_h = [self.norm2_shards[i](hidden_states_list[i]) * (1 + scale_mlp[i]) + shift_mlp[i] for i in range(n)]
        ff_out = self.ff(norm_h)
        hidden_states_list = [h + gate_mlp[i] * f for i, (h, f) in enumerate(zip(hidden_states_list, ff_out))]

        encoder_hidden_states_list = [e + c_gate_msa[i] * a for i, (e, a) in enumerate(zip(encoder_hidden_states_list, attn_e))]
        norm_e = [self.norm2_context_shards[i](encoder_hidden_states_list[i]) * (1 + c_scale_mlp[i]) + c_shift_mlp[i] for i in range(n)]
        ctx_ff = self.ff_context(norm_e)
        encoder_hidden_states_list = [e + c_gate_mlp[i] * f for i, (e, f) in enumerate(zip(encoder_hidden_states_list, ctx_ff))]

        encoder_hidden_states_list = [
            ops.min(ops.max(e, -65504), 65504) if e.dtype == DType.float16 else e
            for e in encoder_hidden_states_list]

        return encoder_hidden_states_list, hidden_states_list


class USPFlux2SingleTransformerBlock(Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, *,
                 dtype, devices, ulysses_degree, ring_degree,
                 mlp_ratio=3.0, eps=1e-6, bias=False):
        super().__init__()
        self.num_devices = len(devices)
        self.norm_shards = [
            LayerNorm(dim, dtype=dtype, device=d, eps=eps,
                      elementwise_affine=False, use_bias=False) for d in devices]
        self.attn = USPFlux2ParallelSelfAttention(
            query_dim=dim, dim_head=attention_head_dim, heads=num_attention_heads,
            out_dim=dim, bias=bias, out_bias=bias, eps=eps,
            mlp_ratio=mlp_ratio, mlp_mult_factor=2,
            dtype=dtype, devices=devices,
            ulysses_degree=ulysses_degree, ring_degree=ring_degree)

    def __call__(self, hidden_states_list, signal_buffers,
                 encoder_hidden_states_list=None, temb_mod_params=None,
                 image_rotary_emb_list=None, split_hidden_states=False,
                 text_seq_len=None):
        n = self.num_devices
        if encoder_hidden_states_list is not None:
            text_seq_len = encoder_hidden_states_list[0].shape[1]
            hidden_states_list = [
                ops.concat([e, h], axis=1)
                for e, h in zip(encoder_hidden_states_list, hidden_states_list)]

        if temb_mod_params is None:
            raise ValueError("temb_mod_params cannot be None")
        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_h = [(1 + mod_scale[i]) * self.norm_shards[i](hidden_states_list[i]) + mod_shift[i] for i in range(n)]
        attn_out = self.attn(norm_h, signal_buffers, image_rotary_emb_list=image_rotary_emb_list)
        hidden_states_list = [h + mod_gate[i] * a for i, (h, a) in enumerate(zip(hidden_states_list, attn_out))]

        hidden_states_list = [
            ops.min(ops.max(h, -65504), 65504) if h.dtype == DType.float16 else h
            for h in hidden_states_list]

        if split_hidden_states:
            if text_seq_len is None:
                raise ValueError("text_seq_len required")
            return [h[:, :text_seq_len, :] for h in hidden_states_list], \
                   [h[:, text_seq_len:, :] for h in hidden_states_list]
        return hidden_states_list


class USPFlux2Transformer2DModel(Module):
    """USP Flux2 Transformer (Ulysses + Ring combined)."""

    def __init__(self, config: Flux2Config, devices: list[DeviceRef],
                 ulysses_degree: int, ring_degree: int) -> None:
        super().__init__()
        self.devices = devices
        self.num_devices = len(devices)
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        self.max_dtype = config.dtype
        self.in_channels = config.in_channels
        self.joint_attention_dim = config.joint_attention_dim
        self.out_channels = config.out_channels or config.in_channels

        dtype = config.dtype
        eps = config.eps

        self.pos_embed = Flux2PosEmbed(theta=config.rope_theta, axes_dim=config.axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=config.timestep_guidance_channels,
            embedding_dim=self.inner_dim, bias=False,
            guidance_embeds=getattr(config, "guidance_embeds", True),
            dtype=dtype, device=devices[0])

        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, dtype=dtype, device=devices[0], mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, dtype=dtype, device=devices[0], mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, dtype=dtype, device=devices[0], mod_param_sets=1, bias=False)

        self.x_embedder = Linear(in_dim=config.in_channels, out_dim=self.inner_dim, dtype=dtype, device=devices[0], has_bias=False)
        self.context_embedder = Linear(in_dim=config.joint_attention_dim, out_dim=self.inner_dim, dtype=dtype, device=devices[0], has_bias=False)

        from max.nn.layer import LayerList
        self.transformer_blocks = LayerList([
            USPFlux2TransformerBlock(
                dim=self.inner_dim, num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim, dtype=dtype,
                devices=devices, ulysses_degree=ulysses_degree,
                ring_degree=ring_degree, mlp_ratio=config.mlp_ratio, eps=eps, bias=False)
            for _ in range(config.num_layers)])

        self.single_transformer_blocks = LayerList([
            USPFlux2SingleTransformerBlock(
                dim=self.inner_dim, num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim, dtype=dtype,
                devices=devices, ulysses_degree=ulysses_degree,
                ring_degree=ring_degree, mlp_ratio=config.mlp_ratio, eps=eps, bias=False)
            for _ in range(config.num_single_layers)])

        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim, conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False, dtype=dtype, device=devices[0], eps=eps, bias=False)
        self.proj_out = Linear(
            in_dim=self.inner_dim,
            out_dim=config.patch_size * config.patch_size * self.out_channels,
            dtype=dtype, device=devices[0], has_bias=False)

    def input_types(self):
        device = self.devices[0]
        signals = Signals(devices=self.devices)
        input_types = [
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
            (ops.distributed_broadcast(s0, signal_buffers), ops.distributed_broadcast(sc0, signal_buffers), ops.distributed_broadcast(g0, signal_buffers)),
            (ops.distributed_broadcast(s1, signal_buffers), ops.distributed_broadcast(sc1, signal_buffers), ops.distributed_broadcast(g1, signal_buffers)),
        )

    def _broadcast_mod_single(self, mod_params, signal_buffers):
        s, sc, g = mod_params
        return (ops.distributed_broadcast(s, signal_buffers), ops.distributed_broadcast(sc, signal_buffers), ops.distributed_broadcast(g, signal_buffers))

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

    def __call__(self, hidden_states, encoder_hidden_states, timestep,
                 img_ids, txt_ids, guidance, signal_buffers):
        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = ops.cast(timestep * 1000.0, hidden_states.dtype)
        guidance = ops.cast(guidance * 1000.0, hidden_states.dtype)
        temb = self.time_guidance_embed(timestep, guidance)

        ds_mod_img = self.double_stream_modulation_img(temb)
        ds_mod_txt = self.double_stream_modulation_txt(temb)
        ss_mod_tuple = self.single_stream_modulation(temb)
        ds_mod_img = (ds_mod_img[0], ds_mod_img[1])
        ds_mod_txt = (ds_mod_txt[0], ds_mod_txt[1])
        ss_mod = ss_mod_tuple[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = ops.concat([txt_ids, img_ids], axis=0)
        cos, sin = self.pos_embed(ids)

        hidden_states_list = self._scatter_sequence(hidden_states, signal_buffers)
        encoder_hidden_states_list = self._scatter_sequence(encoder_hidden_states, signal_buffers)

        # Scatter RoPE independently for text/image
        txt_cos, img_cos = cos[:num_txt_tokens, :], cos[num_txt_tokens:, :]
        txt_sin, img_sin = sin[:num_txt_tokens, :], sin[num_txt_tokens:, :]
        txt_cos_c = self._scatter_ids(txt_cos, signal_buffers)
        img_cos_c = self._scatter_ids(img_cos, signal_buffers)
        txt_sin_c = self._scatter_ids(txt_sin, signal_buffers)
        img_sin_c = self._scatter_ids(img_sin, signal_buffers)
        cos_chunks = [ops.concat([txt_cos_c[i], img_cos_c[i]], axis=0) for i in range(self.num_devices)]
        sin_chunks = [ops.concat([txt_sin_c[i], img_sin_c[i]], axis=0) for i in range(self.num_devices)]
        rope_list = list(zip(cos_chunks, sin_chunks))

        ds_mod_img = self._broadcast_mod_params(ds_mod_img, signal_buffers)
        ds_mod_txt = self._broadcast_mod_params(ds_mod_txt, signal_buffers)
        ss_mod = self._broadcast_mod_single(ss_mod, signal_buffers)

        for block in self.transformer_blocks:
            encoder_hidden_states_list, hidden_states_list = block(
                hidden_states_list, encoder_hidden_states_list,
                ds_mod_img, ds_mod_txt, signal_buffers, rope_list)

        text_seq_lens = [encoder_hidden_states_list[i].shape[1] for i in range(self.num_devices)]
        hidden_states_list = [
            ops.concat([e, h], axis=1)
            for e, h in zip(encoder_hidden_states_list, hidden_states_list)]

        for single_block in self.single_transformer_blocks:
            hidden_states_list = single_block(
                hidden_states_list, signal_buffers,
                temb_mod_params=ss_mod, image_rotary_emb_list=rope_list)

        image_chunks = [hidden_states_list[i][:, text_seq_lens[i]:, :] for i in range(self.num_devices)]
        gathered = ops.allgather(image_chunks, signal_buffers, axis=1)
        hidden_states = gathered[0]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return (output,)
