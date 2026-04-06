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

"""Z-Image DiT core model (Graph API / ModuleV2)."""

from __future__ import annotations

from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, Weight, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from max.pipelines.lib.interfaces.cache_mixin import (
    DenoisingCacheConfig,
    can_use_fbcache,
    teacache_rescaled_delta,
)

from .layers.attention import ZImageAttention
from .layers.embeddings import RopeEmbedder, TimestepEmbedder
from .model_config import ZImageConfig

ADALN_EMBED_DIM = 256


class LayerNorm(Module):
    """Layer normalisation with optional learned affine parameters."""

    weight: Weight | None
    bias: Weight | None

    def __init__(
        self,
        dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = Weight("weight", dtype, (dim,), device=device)
            self.bias = (
                Weight("bias", dtype, (dim,), device=device)
                if use_bias
                else None
            )
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: TensorValue) -> TensorValue:
        if self.weight is None:
            gamma = ops.broadcast_to(
                ops.constant(1.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            gamma = self.weight

        if self.bias is None:
            beta = ops.broadcast_to(
                ops.constant(0.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            beta = self.bias

        return ops.layer_norm(x, gamma=gamma, beta=beta, epsilon=self.eps)


class FeedForward(Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.w1 = Linear(
            in_dim=dim,
            out_dim=hidden_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.w2 = Linear(
            in_dim=hidden_dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.w3 = Linear(
            in_dim=dim,
            out_dim=hidden_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(Module):
    """Single transformer block with optional adaLN modulation."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        *,
        dtype: DType,
        device: DeviceRef,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        del n_kv_heads

        self.modulation = modulation
        self.dim = dim

        self.attention = ZImageAttention(
            dim=dim,
            n_heads=n_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            dtype=dtype,
            device=device,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=int(dim / 3 * 8),
            dtype=dtype,
            device=device,
        )
        self.attention_norm1 = RMSNorm(dim, dtype=dtype, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, dtype=dtype, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, dtype=dtype, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, dtype=dtype, eps=norm_eps)

        self.adaLN_modulation = (
            Linear(
                in_dim=min(dim, ADALN_EMBED_DIM),
                out_dim=4 * dim,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
            if modulation
            else None
        )

    def __call__(
        self,
        x: TensorValue,
        freqs_cis: TensorValue,
        adaln_input: TensorValue | None = None,
    ) -> TensorValue:
        if self.modulation:
            if adaln_input is None:
                raise ValueError("adaln_input is required when modulation=True")
            if self.adaLN_modulation is None:
                raise ValueError("adaLN_modulation is not initialized")

            mod = ops.unsqueeze(self.adaLN_modulation(adaln_input), 1)
            d = self.dim
            scale_msa = 1.0 + mod[:, :, :d]
            gate_msa = ops.tanh(mod[:, :, d : 2 * d])
            scale_mlp = 1.0 + mod[:, :, 2 * d : 3 * d]
            gate_mlp = ops.tanh(mod[:, :, 3 * d :])

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            x = x + gate_mlp * self.ffn_norm2(ffn_out)
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(Module):
    """Final projection layer with adaLN conditioning."""

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.norm_final = LayerNorm(
            hidden_size,
            dtype=dtype,
            device=device,
            eps=1e-6,
            elementwise_affine=False,
            use_bias=False,
        )
        self.linear = Linear(
            in_dim=hidden_size,
            out_dim=out_channels,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.adaLN_modulation = Linear(
            in_dim=min(hidden_size, ADALN_EMBED_DIM),
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue, c: TensorValue) -> TensorValue:
        scale = 1.0 + self.adaLN_modulation(ops.silu(c))
        x = self.norm_final(x) * ops.unsqueeze(scale, 1)
        return self.linear(x)


class ZImageTransformer2DModel(Module):
    """Z-Image diffusion transformer (DiT) model."""

    def __init__(
        self,
        config: ZImageConfig,
        *,
        cache_config: DenoisingCacheConfig | None = None,
    ) -> None:
        super().__init__()

        dim = config.dim
        n_heads = config.n_heads
        norm_eps = config.norm_eps
        qk_norm = config.qk_norm
        cap_feat_dim = config.cap_feat_dim
        n_layers = config.n_layers
        n_refiner_layers = config.n_refiner_layers
        axes_dims = config.axes_dims
        rope_theta = config.rope_theta
        dtype = config.dtype
        device = config.device

        patch_size = config.all_patch_size[0]
        f_patch_size = config.all_f_patch_size[0]
        in_channels = (
            config.in_channels * patch_size * patch_size * f_patch_size
        )
        out_channels = in_channels

        self.dim = dim
        self.packed_channels = in_channels
        self.out_channels_total = out_channels
        self.max_dtype = dtype
        self.max_device = device
        self.cap_feat_dim = cap_feat_dim
        self.t_scale = config.t_scale
        self.axes_dims = axes_dims

        self.x_embedder = Linear(
            in_dim=in_channels,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.final_layer = FinalLayer(
            hidden_size=dim,
            out_channels=out_channels,
            dtype=dtype,
            device=device,
        )

        self.noise_refiner = LayerList(
            [
                ZImageTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=config.n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=True,
                )
                for _ in range(n_refiner_layers)
            ]
        )
        self.context_refiner = LayerList(
            [
                ZImageTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=config.n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=False,
                )
                for _ in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(
            out_size=min(dim, ADALN_EMBED_DIM),
            mid_size=1024,
            dtype=dtype,
            device=device,
        )
        self.cap_norm = RMSNorm(cap_feat_dim, dtype=dtype, eps=norm_eps)
        self.cap_proj = Linear(
            in_dim=cap_feat_dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        self.layers = LayerList(
            [
                ZImageTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=config.n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=True,
                )
                for _ in range(n_layers)
            ]
        )

        head_dim = dim // n_heads
        if head_dim != sum(axes_dims):
            raise ValueError(
                f"head_dim ({head_dim}) must equal sum(axes_dims) ({sum(axes_dims)})"
            )

        self.rope_embedder = RopeEmbedder(
            theta=rope_theta,
            axes_dims=axes_dims,
        )

        # Cache mode routing.
        self._forward_impl: Callable[..., tuple[TensorValue, ...]] = (
            self._forward_standard
        )
        self._input_types_impl: Callable[..., tuple[TensorType, ...]] = (
            self._input_types_standard
        )
        self._teacache_rel_l1_thresh: float = 0.4
        self._teacache_coefficients: tuple[float, ...] = ()
        if cache_config is not None and cache_config.first_block_caching:
            self._forward_impl = self._forward_fbcache
            self._input_types_impl = self._input_types_fbcache
        elif cache_config is not None and cache_config.teacache:
            assert cache_config.teacache_rel_l1_thresh is not None
            assert cache_config.teacache_coefficients is not None
            self._teacache_rel_l1_thresh = cache_config.teacache_rel_l1_thresh
            self._teacache_coefficients = tuple(
                cache_config.teacache_coefficients
            )
            self._forward_impl = self._forward_teacache
            self._input_types_impl = self._input_types_teacache

    def _fbcache_output_types(self) -> list[TensorType]:
        """[residual_type, output_type] for FBCache conditional execution."""
        residual_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.dim],
            device=self.max_device,
        )
        output_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.out_channels_total],
            device=self.max_device,
        )
        return [residual_type, output_type]

    def _teacache_output_types(self) -> list[TensorType]:
        """[modulated_input, residual, accumulated, output] for TeaCache."""
        hidden_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.dim],
            device=self.max_device,
        )
        accum_type = TensorType(
            DType.float32, shape=[1], device=self.max_device
        )
        output_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.out_channels_total],
            device=self.max_device,
        )
        return [hidden_type, hidden_type, accum_type, output_type]

    def _base_input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.max_dtype,
                shape=["batch_size", "image_seq_len", self.packed_channels],
                device=self.max_device,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch_size", "text_seq_len", self.cap_feat_dim],
                device=self.max_device,
            ),
            TensorType(
                DType.float32,
                shape=["batch_size"],
                device=self.max_device,
            ),
            TensorType(
                DType.int64,
                shape=["image_seq_len", len(self.axes_dims)],
                device=self.max_device,
            ),
            TensorType(
                DType.int64,
                shape=["text_seq_len", len(self.axes_dims)],
                device=self.max_device,
            ),
        )

    def _input_types_standard(self) -> tuple[TensorType, ...]:
        return self._base_input_types()

    def _input_types_fbcache(self) -> tuple[TensorType, ...]:
        rdt_type = TensorType(
            DType.float32, shape=[], device=self.max_device
        )
        return (
            self._base_input_types()
            + tuple(self._fbcache_output_types())
            + (rdt_type,)
        )

    def _input_types_teacache(self) -> tuple[TensorType, ...]:
        hidden_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.dim],
            device=self.max_device,
        )
        accum_type = TensorType(
            DType.float32, shape=[1], device=self.max_device
        )
        force_type = TensorType(
            DType.bool, shape=[1], device=self.max_device
        )
        return self._base_input_types() + (
            hidden_type,  # prev_modulated_input
            hidden_type,  # prev_residual
            accum_type,  # accumulated_rel_l1
            force_type,  # force_compute
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return self._input_types_impl()

    def _forward_preamble(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
    ) -> tuple[TensorValue, int | Dim, TensorValue, TensorValue]:
        """Embed inputs, run refiners, return unified seq before layers[0]."""
        x = self.x_embedder(hidden_states)
        t_emb = ops.cast(self.t_embedder(timestep * self.t_scale), x.dtype)

        cap = self.cap_proj(self.cap_norm(encoder_hidden_states))

        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        img_seq_len = img_ids.shape[0]
        unified_ids = ops.concat([img_ids, txt_ids], axis=0)
        unified_freqs = ops.cast(self.rope_embedder(unified_ids), x.dtype)
        img_freqs = unified_freqs[:img_seq_len]
        txt_freqs = unified_freqs[img_seq_len:]

        for block in self.noise_refiner:
            x = block(x, freqs_cis=img_freqs, adaln_input=t_emb)

        for block in self.context_refiner:
            cap = block(cap, freqs_cis=txt_freqs)

        img_len = x.shape[1]
        unified0 = ops.concat([x, cap], axis=1)
        return unified0, img_len, t_emb, unified_freqs

    def _run_first_main_layer(
        self,
        unified0: TensorValue,
        t_emb: TensorValue,
        unified_freqs: TensorValue,
    ) -> TensorValue:
        return self.layers[0](
            unified0, freqs_cis=unified_freqs, adaln_input=t_emb
        )

    def _run_remaining_after_first(
        self,
        unified: TensorValue,
        *,
        img_len: int | Dim,
        t_emb: TensorValue,
        freqs_cis: TensorValue,
    ) -> TensorValue:
        u = unified
        for i in range(1, len(self.layers)):
            u = self.layers[i](u, freqs_cis=freqs_cis, adaln_input=t_emb)
        return u[:, :img_len, :]

    def _forward_postamble(
        self, hidden_states: TensorValue, t_emb: TensorValue
    ) -> TensorValue:
        return self.final_layer(hidden_states, t_emb)

    def _forward_standard(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
    ) -> tuple[TensorValue]:
        unified0, img_len, t_emb, unified_freqs = self._forward_preamble(
            hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids
        )
        u1 = self._run_first_main_layer(unified0, t_emb, unified_freqs)
        remaining = self._run_remaining_after_first(
            u1, img_len=img_len, t_emb=t_emb, freqs_cis=unified_freqs
        )
        return (self._forward_postamble(remaining, t_emb),)

    def _forward_fbcache(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        prev_residual: TensorValue,
        prev_output: TensorValue,
        residual_threshold: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        unified0, img_len, t_emb, unified_freqs = self._forward_preamble(
            hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids
        )
        unified1 = self._run_first_main_layer(unified0, t_emb, unified_freqs)
        first_block_residual = (
            unified1[:, :img_len, :] - unified0[:, :img_len, :]
        )

        use_cache = can_use_fbcache(
            first_block_residual, prev_residual, residual_threshold
        )

        def then_fn() -> tuple[TensorValue, TensorValue]:
            return first_block_residual, prev_output

        def else_fn() -> tuple[TensorValue, TensorValue]:
            remaining = self._run_remaining_after_first(
                unified1,
                img_len=img_len,
                t_emb=t_emb,
                freqs_cis=unified_freqs,
            )
            out = self._forward_postamble(remaining, t_emb)
            return first_block_residual, out

        result = ops.cond(
            use_cache, self._fbcache_output_types(), then_fn, else_fn
        )
        return (result[0], result[1])

    def _forward_teacache(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        prev_modulated_input: TensorValue,
        prev_residual: TensorValue,
        accumulated_rel_l1: TensorValue,
        force_compute: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
        unified0, img_len, t_emb, unified_freqs = self._forward_preamble(
            hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids
        )

        # Compute modulated input from first block's adaLN.
        projected = unified0[:, :img_len, :]
        block0 = self.layers[0]
        assert block0.adaLN_modulation is not None
        mod = ops.unsqueeze(block0.adaLN_modulation(t_emb), 1)
        d = self.dim
        scale_msa = 1.0 + mod[:, :, :d]
        modulated_input = block0.attention_norm1(projected) * scale_msa

        delta = teacache_rescaled_delta(
            modulated_input, prev_modulated_input, self._teacache_coefficients
        )
        next_accumulated = accumulated_rel_l1 + delta

        thresh = ops.constant(
            self._teacache_rel_l1_thresh,
            DType.float32,
            device=next_accumulated.device,
        )
        should_skip = ops.squeeze(
            ~force_compute & (next_accumulated < thresh), 0
        )

        def then_fn() -> (
            tuple[TensorValue, TensorValue, TensorValue, TensorValue]
        ):
            out = self._forward_postamble(projected + prev_residual, t_emb)
            return modulated_input, prev_residual, next_accumulated, out

        def else_fn() -> (
            tuple[TensorValue, TensorValue, TensorValue, TensorValue]
        ):
            u1 = self._run_first_main_layer(unified0, t_emb, unified_freqs)
            remaining = self._run_remaining_after_first(
                u1, img_len=img_len, t_emb=t_emb, freqs_cis=unified_freqs
            )
            residual = remaining - projected
            out = self._forward_postamble(remaining, t_emb)
            zero_accum = accumulated_rel_l1 - accumulated_rel_l1
            return modulated_input, residual, zero_accum, out

        result = ops.cond(
            should_skip, self._teacache_output_types(), then_fn, else_fn
        )
        return (result[0], result[1], result[2], result[3])

    def __call__(self, *args: TensorValue) -> tuple[TensorValue, ...]:
        return self._forward_impl(*args)
