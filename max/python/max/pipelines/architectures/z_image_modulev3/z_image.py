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

from collections.abc import Callable, Sequence
from typing import Any

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.norm import LayerNorm, RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.lib.interfaces.cache_mixin import (
    DenoisingCacheConfig,
    fbcache_conditional_execution,
)

from .layers.attention import ZImageAttention
from .layers.embeddings import RopeEmbedder, TimestepEmbedder
from .model_config import ZImageConfig

ADALN_EMBED_DIM = 256


class FeedForward(Module[[Tensor], Tensor]):
    """SwiGLU feed-forward network: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, dim: int, hidden_dim: int):
        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(Module[..., Tensor]):
    """Single Z-Image transformer block with optional AdaLN modulation.

    When ``modulation=True`` (main layers, noise refiner), the block uses
    adaptive layer-norm parameters (scale, gate) derived from the timestep
    embedding.  When ``modulation=False`` (context refiner), the block is a
    plain pre-norm residual block.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
    ):
        del n_kv_heads

        self.layer_id = layer_id
        self.modulation = modulation
        self.dim = dim

        self.attention = ZImageAttention(
            dim=dim,
            n_heads=n_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
        )
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = (
            Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            if modulation
            else None
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        adaln_input: Tensor | None = None,
    ) -> Tensor:
        if self.modulation:
            if adaln_input is None:
                raise ValueError("adaln_input is required when modulation=True")
            if self.adaLN_modulation is None:
                raise ValueError("adaLN_modulation is not initialized")

            mod = F.unsqueeze(self.adaLN_modulation(adaln_input), 1)
            d = self.dim
            scale_msa = 1.0 + mod[:, :, :d]
            gate_msa = F.tanh(mod[:, :, d : 2 * d])
            scale_mlp = 1.0 + mod[:, :, 2 * d : 3 * d]
            gate_mlp = F.tanh(mod[:, :, 3 * d :])

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


class FinalLayer(Module[..., Tensor]):
    """Output projection with AdaLN modulation for timestep conditioning."""

    def __init__(self, hidden_size: int, out_channels: int):
        self.norm_final = LayerNorm(
            hidden_size,
            eps=1e-6,
            elementwise_affine=False,
            use_bias=False,
        )
        self.linear = Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = Linear(
            min(hidden_size, ADALN_EMBED_DIM),
            hidden_size,
            bias=True,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        scale = 1.0 + self.adaLN_modulation(F.silu(c))
        x = self.norm_final(x) * F.unsqueeze(scale, 1)
        return self.linear(x)


class ZImageTransformer2DModel(Module[..., Sequence[Tensor]]):
    """Z-Image diffusion transformer backbone.

    Architecture: x_embedder → noise_refiner → context_refiner →
    concat(image, text) → main_layers → final_layer.

    Supports standard inference and first-block caching (FBCache) for
    accelerated denoising.
    """

    def __init__(
        self,
        config: ZImageConfig,
        cache_config: DenoisingCacheConfig | None = None,
    ):
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.t_scale = config.t_scale
        self.axes_dims = config.axes_dims
        self.axes_lens = config.axes_lens

        if len(config.all_patch_size) != len(config.all_f_patch_size):
            raise ValueError("all_patch_size and all_f_patch_size must align")

        self.patch_size = config.all_patch_size[0]
        self.f_patch_size = config.all_f_patch_size[0]
        self.packed_channels = (
            self.f_patch_size
            * self.patch_size
            * self.patch_size
            * self.in_channels
        )

        self.x_embedder = Linear(
            self.packed_channels,
            self.dim,
            bias=True,
        )

        self.final_layer = FinalLayer(
            self.dim,
            self.patch_size
            * self.patch_size
            * self.f_patch_size
            * self.out_channels,
        )

        self.noise_refiner: ModuleList[ZImageTransformerBlock] = ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    self.dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.norm_eps,
                    config.qk_norm,
                    modulation=True,
                )
                for layer_id in range(config.n_refiner_layers)
            ]
        )

        self.context_refiner: ModuleList[ZImageTransformerBlock] = ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    self.dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.norm_eps,
                    config.qk_norm,
                    modulation=False,
                )
                for layer_id in range(config.n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(
            min(self.dim, ADALN_EMBED_DIM),
            mid_size=1024,
        )
        self.cap_norm = RMSNorm(config.cap_feat_dim, eps=config.norm_eps)
        self.cap_proj = Linear(config.cap_feat_dim, self.dim, bias=True)

        self.layers: ModuleList[ZImageTransformerBlock] = ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    self.dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.norm_eps,
                    config.qk_norm,
                    modulation=True,
                )
                for layer_id in range(config.n_layers)
            ]
        )

        head_dim = self.dim // self.n_heads
        if head_dim != sum(self.axes_dims):
            raise ValueError(
                f"head_dim ({head_dim}) must equal sum(axes_dims) ({sum(self.axes_dims)})"
            )

        self.rope_embedder = RopeEmbedder(
            theta=config.rope_theta,
            axes_dims=config.axes_dims,
        )

        self.max_device = config.device
        self.max_dtype = config.dtype
        self.cap_feat_dim = config.cap_feat_dim

        self._forward_impl: Callable[..., tuple[Tensor, ...]] = (
            self._forward_standard
        )
        self._input_types_impl: Callable[..., tuple[TensorType, ...]] = (
            self._input_types_standard
        )
        if cache_config is not None and cache_config.first_block_caching:
            self._forward_impl = self._forward_step_cache
            self._input_types_impl = self._input_types_step_cache

    def _fbcache_conditional_execution_output_types(self) -> list[TensorType]:
        residual_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.dim],
            device=self.max_device,
        )
        out_ch = (
            self.patch_size
            * self.patch_size
            * self.f_patch_size
            * self.out_channels
        )
        output_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", out_ch],
            device=self.max_device,
        )
        return [residual_type, output_type]

    def _base_input_types(self) -> tuple[TensorType, ...]:
        hidden_states_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.packed_channels],
            device=self.max_device,
        )
        encoder_hidden_states_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "text_seq_len", self.cap_feat_dim],
            device=self.max_device,
        )
        timestep_type = TensorType(
            DType.float32,
            shape=["batch_size"],
            device=self.max_device,
        )
        img_ids_type = TensorType(
            DType.int64,
            shape=["image_seq_len", len(self.axes_dims)],
            device=self.max_device,
        )
        txt_ids_type = TensorType(
            DType.int64,
            shape=["text_seq_len", len(self.axes_dims)],
            device=self.max_device,
        )
        return (
            hidden_states_type,
            encoder_hidden_states_type,
            timestep_type,
            img_ids_type,
            txt_ids_type,
        )

    def _input_types_standard(self) -> tuple[TensorType, ...]:
        return self._base_input_types()

    def _input_types_step_cache(self) -> tuple[TensorType, ...]:
        rdt_type = TensorType(DType.float32, shape=[], device=self.max_device)
        return (
            self._base_input_types()
            + tuple(self._fbcache_conditional_execution_output_types())
            + (rdt_type,)
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return self._input_types_impl()

    def _forward_preamble(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
    ) -> tuple[Tensor, Any, Tensor, Tensor]:
        """Embed inputs, run refiners, return unified seq before main ``layers[0]``."""
        x = self.x_embedder(hidden_states)
        t_emb = self.t_embedder(timestep * self.t_scale).cast(x.dtype)

        cap = self.cap_proj(self.cap_norm(encoder_hidden_states))

        # Compute RoPE once on concatenated ids (interleaved [cos,sin] format),
        # cast to model dtype so the kernel operates in bfloat16 directly.
        img_seq_len = img_ids.shape[0]
        unified_ids = F.concat([img_ids, txt_ids], axis=0)
        unified_freqs = self.rope_embedder(unified_ids).cast(x.dtype)

        img_freqs = unified_freqs[:img_seq_len]
        txt_freqs = unified_freqs[img_seq_len:]

        for layer in self.noise_refiner:
            x = layer(
                x,
                freqs_cis=img_freqs,
                adaln_input=t_emb,
            )

        for layer in self.context_refiner:
            cap = layer(cap, freqs_cis=txt_freqs)

        img_len = x.shape[1]
        unified0 = F.concat([x, cap], axis=1)
        return (unified0, img_len, t_emb, unified_freqs)

    def _run_first_main_layer(
        self,
        unified0: Tensor,
        t_emb: Tensor,
        unified_freqs: Tensor,
    ) -> Tensor:
        return self.layers[0](
            unified0,
            freqs_cis=unified_freqs,
            adaln_input=t_emb,
        )

    def _run_remaining_after_first(
        self,
        unified: Tensor,
        *,
        img_len: Any,
        t_emb: Tensor,
        freqs_cis: Tensor,
    ) -> Tensor:
        u = unified
        for i in range(1, len(self.layers)):
            u = self.layers[i](
                u,
                freqs_cis=freqs_cis,
                adaln_input=t_emb,
            )
        return u[:, :img_len, :]

    def _forward_postamble(
        self, hidden_states: Tensor, t_emb: Tensor
    ) -> Tensor:
        """Final layer after the transformer backbone."""
        return self.final_layer(hidden_states, c=t_emb)

    def _forward_standard(self, *args: Tensor) -> tuple[Tensor]:
        hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids = args[
            :5
        ]
        unified0, img_len, t_emb, unified_freqs = self._forward_preamble(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
        )
        u1 = self._run_first_main_layer(unified0, t_emb, unified_freqs)
        remaining = self._run_remaining_after_first(
            u1,
            img_len=img_len,
            t_emb=t_emb,
            freqs_cis=unified_freqs,
        )
        return (self._forward_postamble(remaining, t_emb),)

    def _forward_step_cache(self, *args: Tensor) -> tuple[Tensor, Tensor]:
        (
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            prev_residual,
            prev_output,
            residual_threshold,
        ) = args
        unified0, img_len, t_emb, unified_freqs = self._forward_preamble(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
        )
        unified1 = self._run_first_main_layer(unified0, t_emb, unified_freqs)
        first_block_residual = (
            unified1[:, :img_len, :] - unified0[:, :img_len, :]
        )

        return fbcache_conditional_execution(
            first_block_residual,
            prev_residual,
            prev_output,
            residual_threshold,
            self._run_remaining_after_first,
            dict(
                unified=unified1,
                img_len=img_len,
                t_emb=t_emb,
                freqs_cis=unified_freqs,
            ),
            self._forward_postamble,
            t_emb,
            self._fbcache_conditional_execution_output_types(),
        )

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return self._forward_impl(*args)


# ---------------------------------------------------------------------------
# Split modules for batched-CFG: run shared preamble at batch=1, then
# duplicate and run the rest at batch=2.  Saves ~3% compute by avoiding
# redundant noise_refiner + embedder work on the duplicated batch.
# ---------------------------------------------------------------------------


class ZImageSharedPreamble(Module[..., Sequence[Tensor]]):
    """batch=1 shared computation: x_embed → t_embed → RoPE → noise_refiner."""

    def __init__(self, transformer: ZImageTransformer2DModel):
        self.x_embedder = transformer.x_embedder
        self.t_embedder = transformer.t_embedder
        self.noise_refiner = transformer.noise_refiner
        self.rope_embedder = transformer.rope_embedder
        self.t_scale = transformer.t_scale
        self.axes_dims = transformer.axes_dims
        self.max_dtype = transformer.max_dtype
        self.max_device = transformer.max_device
        self.packed_channels = transformer.packed_channels

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.max_dtype,
                shape=[1, "image_seq_len", self.packed_channels],
                device=self.max_device,
            ),
            TensorType(DType.float32, shape=[1], device=self.max_device),
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

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.x_embedder(hidden_states)
        t_emb = self.t_embedder(timestep * self.t_scale).cast(x.dtype)

        img_seq_len = img_ids.shape[0]
        unified_ids = F.concat([img_ids, txt_ids], axis=0)
        unified_freqs = self.rope_embedder(unified_ids).cast(x.dtype)
        img_freqs = unified_freqs[:img_seq_len]

        for layer in self.noise_refiner:
            x = layer(x, freqs_cis=img_freqs, adaln_input=t_emb)

        return (x, t_emb, unified_freqs, unified_freqs[img_seq_len:])


class ZImageCFGMain(Module[..., Sequence[Tensor]]):
    """batch=2 computation: cap_proj → context_refiner → main_layers → final."""

    def __init__(self, transformer: ZImageTransformer2DModel):
        self.cap_norm = transformer.cap_norm
        self.cap_proj = transformer.cap_proj
        self.context_refiner = transformer.context_refiner
        self.layers = transformer.layers
        self.final_layer = transformer.final_layer
        self.dim = transformer.dim
        self.cap_feat_dim = transformer.cap_feat_dim
        self.head_dim = sum(transformer.axes_dims)
        self.max_dtype = transformer.max_dtype
        self.max_device = transformer.max_device

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.max_dtype,
                shape=["batch_size", "image_seq_len", self.dim],
                device=self.max_device,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch_size", "text_seq_len", self.cap_feat_dim],
                device=self.max_device,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch_size", ADALN_EMBED_DIM],
                device=self.max_device,
            ),
            TensorType(
                self.max_dtype,
                shape=["total_seq_len", self.head_dim],
                device=self.max_device,
            ),
            TensorType(
                self.max_dtype,
                shape=["text_seq_len", self.head_dim],
                device=self.max_device,
            ),
        )

    def forward(
        self,
        x: Tensor,
        encoder_hidden_states: Tensor,
        t_emb: Tensor,
        unified_freqs: Tensor,
        txt_freqs: Tensor,
    ) -> tuple[Tensor]:
        cap = self.cap_proj(self.cap_norm(encoder_hidden_states))

        for layer in self.context_refiner:
            cap = layer(cap, freqs_cis=txt_freqs)

        img_len = x.shape[1]
        unified = F.concat([x, cap], axis=1)

        for layer in self.layers:
            unified = layer(
                unified, freqs_cis=unified_freqs, adaln_input=t_emb
            )

        hidden = unified[:, :img_len, :]
        return (self.final_layer(hidden, c=t_emb),)


class ZImageRoPEOnly(Module[..., Sequence[Tensor]]):
    """Compute RoPE frequencies for a given (img_ids, txt_ids) pair.

    Returns (unified_freqs, txt_freqs) in model dtype, ready for
    ZImageCFGMain.  Used by the explicit-negative split path so we
    can reuse the shared preamble (x, t_emb) while computing new
    RoPE for different negative txt_ids.
    """

    def __init__(self, transformer: ZImageTransformer2DModel):
        self.rope_embedder = transformer.rope_embedder
        self.axes_dims = transformer.axes_dims
        self.max_dtype = transformer.max_dtype
        self.max_device = transformer.max_device

    def input_types(self) -> tuple[TensorType, ...]:
        return (
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

    def forward(
        self, img_ids: Tensor, txt_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        img_seq_len = img_ids.shape[0]
        unified_ids = F.concat([img_ids, txt_ids], axis=0)
        unified_freqs = self.rope_embedder(unified_ids).cast(self.max_dtype)
        return (unified_freqs, unified_freqs[img_seq_len:])
