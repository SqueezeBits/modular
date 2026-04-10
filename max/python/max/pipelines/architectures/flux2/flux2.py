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

from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.pipelines.lib.interfaces.cache_mixin import (
    DenoisingCacheConfig,
    can_use_fbcache,
    teacache_conditional_execution,
    teacache_rescaled_delta,
)

from .layers.embeddings import TimestepEmbedding, Timesteps
from .layers.flux2_attention import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2ParallelSelfAttention,
    Flux2PosEmbed,
)
from .layers.normalizations import AdaLayerNormContinuous, LayerNorm
from .model_config import Flux2Config


class Flux2TimestepGuidanceEmbeddings(Module):
    def __init__(
        self,
        *,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize Flux2TimestepGuidanceEmbeddings.

        Args:
            in_channels: Number of sinusoidal channels.
            embedding_dim: Output embedding dimension.
            bias: Whether to use bias in MLP layers.
            guidance_embeds: If True, include guidance embedder.
            dtype: Weight dtype.
            device: Weight device.
        """
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
            dtype=dtype,
            device=device,
        )
        if guidance_embeds:
            self.guidance_embedder: TimestepEmbedding | None = (
                TimestepEmbedding(
                    in_channels=in_channels,
                    time_embed_dim=embedding_dim,
                    sample_proj_bias=bias,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.guidance_embedder = None

    def __call__(
        self, timestep: TensorValue, guidance: TensorValue
    ) -> TensorValue:
        """Compute combined timestep and guidance embeddings.

        Args:
            timestep: Timestep values of shape [B].
            guidance: Guidance scale values of shape [B].

        Returns:
            Combined embedding of shape [B, embedding_dim].
        """
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            ops.cast(timesteps_proj, timestep.dtype)
        )
        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(
                ops.cast(guidance_proj, guidance.dtype)
            )
            return timesteps_emb + guidance_emb
        return timesteps_emb


class Flux2Modulation(Module):
    def __init__(
        self,
        dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        mod_param_sets: int = 2,
        bias: bool = False,
    ) -> None:
        """Initialize Flux2Modulation.

        Args:
            dim: Input/output dimension.
            dtype: Weight dtype.
            device: Weight device.
            mod_param_sets: Number of modulation parameter sets.
            bias: Whether to use bias in the linear layer.
        """
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = Linear(
            in_dim=dim,
            out_dim=dim * 3 * mod_param_sets,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )

    def __call__(
        self, temb: TensorValue
    ) -> tuple[tuple[TensorValue, TensorValue, TensorValue], ...]:
        """Generate modulation parameters from timestep embedding.

        Args:
            temb: Timestep embedding of shape [B, dim] or [B, 1, dim].

        Returns:
            Tuple of modulation tuples, each containing (shift, scale, gate).
        """
        mod = self.linear(ops.silu(temb))
        if len(mod.shape) == 2:
            mod = ops.unsqueeze(mod, 1)
        mod_params = ops.split(
            mod,
            [temb.shape[-1]] * (3 * self.mod_param_sets),
            axis=-1,
        )
        return tuple(
            (mod_params[3 * i], mod_params[3 * i + 1], mod_params[3 * i + 2])
            for i in range(self.mod_param_sets)
        )


class Flux2TransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        """Initialize Flux2TransformerBlock.

        Args:
            dim: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            dtype: Weight dtype.
            device: Weight device.
            mlp_ratio: Multiplier for feedforward hidden dimension.
            eps: Epsilon for layer normalization.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.norm1 = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.norm1_context = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.attn = Flux2Attention(
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
            device=device,
        )
        self.norm2 = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.ff = Flux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.norm2_context = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.ff_context = Flux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        temb_mod_params_img: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        temb_mod_params_txt: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Forward pass for dual-stream transformer block.

        Args:
            hidden_states: Image tokens of shape [B, S_img, D].
            encoder_hidden_states: Text tokens of shape [B, S_txt, D].
            temb_mod_params_img: Image-stream modulation parameters.
            temb_mod_params_txt: Text-stream modulation parameters.
            image_rotary_emb: Optional (cos, sin) tuple for rotary embeddings.

        Returns:
            Tuple of (encoder_hidden_states, hidden_states).
        """
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = (
            temb_mod_params_img
        )
        (
            (c_shift_msa, c_scale_msa, c_gate_msa),
            (c_shift_mlp, c_scale_mlp, c_gate_mlp),
        ) = temb_mod_params_txt

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            1 + c_scale_msa
        ) * norm_encoder_hidden_states + c_shift_msa

        attn_result = self.attn(
            norm_hidden_states,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        if not isinstance(attn_result, tuple):
            raise ValueError("Expected tuple from dual-stream attention")
        attn_output, context_attn_output = attn_result

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp * context_ff_output
        )

        if encoder_hidden_states.dtype == DType.float16:
            encoder_hidden_states = ops.min(
                ops.max(encoder_hidden_states, -65504),
                65504,
            )

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        """Initialize Flux2SingleTransformerBlock.

        Args:
            dim: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            dtype: Weight dtype.
            device: Weight device.
            mlp_ratio: Multiplier for feedforward hidden dimension.
            eps: Epsilon for layer normalization.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.norm = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.attn = Flux2ParallelSelfAttention(
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
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue | None = None,
        temb_mod_params: tuple[
            TensorValue,
            TensorValue,
            TensorValue,
        ]
        | None = None,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | Dim | None = None,
    ) -> TensorValue | tuple[TensorValue, TensorValue]:
        """Forward pass for single-stream transformer block.

        Args:
            hidden_states: Image tokens or concatenated text+image tokens.
            encoder_hidden_states: Optional text tokens to concatenate.
            temb_mod_params: (shift, scale, gate) tuple for modulation.
            image_rotary_emb: Optional (cos, sin) tuple for rotary embeddings.
            split_hidden_states: If True, split output back into text and image.
            text_seq_len: Length of text sequence when splitting.

        Returns:
            Either concatenated hidden states or (encoder_hidden_states, hidden_states).
        """
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = ops.concat(
                [encoder_hidden_states, hidden_states],
                axis=1,
            )

        if temb_mod_params is None:
            raise ValueError("temb_mod_params cannot be None")
        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        attn_output = self.attn(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + mod_gate * attn_output

        if hidden_states.dtype == DType.float16:
            hidden_states = ops.min(ops.max(hidden_states, -65504), 65504)

        if split_hidden_states:
            if text_seq_len is None:
                raise ValueError("text_seq_len is required when splitting")
            encoder_hidden_states = hidden_states[:, :text_seq_len, :]
            hidden_states = hidden_states[:, text_seq_len:, :]
            return encoder_hidden_states, hidden_states
        return hidden_states


class Flux2Transformer2DModel(Module):
    def __init__(
        self,
        config: Flux2Config,
        cache_config: DenoisingCacheConfig | None = None,
    ) -> None:
        """Initialize Flux2Transformer2DModel.

        Args:
            config: Flux2 configuration containing model dimensions,
                attention settings, and device/dtype information.
            cache_config: Optional denoising cache config. When provided with
                first_block_caching or teacache enabled, the forward pass uses
                the corresponding cache path with ops.cond branching.
        """
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
        device = config.device
        dtype = config.dtype
        eps = config.eps

        self.device = device
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim

        self.pos_embed = Flux2PosEmbed(
            theta=rope_theta, axes_dim=axes_dims_rope
        )
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=getattr(config, "guidance_embeds", True),
            dtype=dtype,
            device=device,
        )
        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=2,
            bias=False,
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=2,
            bias=False,
        )
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=1,
            bias=False,
        )
        self.x_embedder = Linear(
            in_dim=in_channels,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.context_embedder = Linear(
            in_dim=joint_attention_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.transformer_blocks = LayerList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    device=device,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = LayerList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    device=device,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
            eps=eps,
            bias=False,
        )
        self.proj_out = Linear(
            in_dim=self.inner_dim,
            out_dim=patch_size * patch_size * self.out_channels,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

        # Step-cache routing: pick the forward/input_types path once at init.
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

    # -- Output type helpers for ops.cond -----------------------------------

    def _fbcache_output_types(self) -> list[TensorType]:
        """Return [residual_type, output_type] for FBCache ops.cond."""
        residual_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.inner_dim],
            device=self.device,
        )
        output_type = TensorType(
            self.max_dtype,
            shape=[
                "batch_size",
                "image_seq_len",
                self.patch_size * self.patch_size * self.out_channels,
            ],
            device=self.device,
        )
        return [residual_type, output_type]

    def _teacache_output_types(self) -> list[TensorType]:
        """Return TeaCache output types for ops.cond."""
        image_hidden_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.inner_dim],
            device=self.device,
        )
        accum_type = TensorType(DType.float32, shape=[1], device=self.device)
        output_type = TensorType(
            self.max_dtype,
            shape=[
                "batch_size",
                "image_seq_len",
                self.patch_size * self.patch_size * self.out_channels,
            ],
            device=self.device,
        )
        return [image_hidden_type, image_hidden_type, accum_type, output_type]

    # -- Input types ---------------------------------------------------------

    def _base_input_types(self) -> tuple[TensorType, ...]:
        """Return the base 6 input types shared by all forward paths."""
        return (
            TensorType(
                self.max_dtype,
                shape=["batch_size", "image_seq_len", self.in_channels],
                device=self.device,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch_size", "text_seq_len", self.joint_attention_dim],
                device=self.device,
            ),
            TensorType(
                self.max_dtype, shape=["batch_size"], device=self.device
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "image_seq_len", 4],
                device=self.device,
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "text_seq_len", 4],
                device=self.device,
            ),
            TensorType(
                self.max_dtype, shape=["batch_size"], device=self.device
            ),
        )

    def _input_types_standard(self) -> tuple[TensorType, ...]:
        return self._base_input_types()

    def _input_types_fbcache(self) -> tuple[TensorType, ...]:
        rdt_type = TensorType(DType.float32, shape=[], device=self.device)
        return (
            self._base_input_types()
            + tuple(self._fbcache_output_types())
            + (rdt_type,)
        )

    def _input_types_teacache(self) -> tuple[TensorType, ...]:
        image_hidden_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.inner_dim],
            device=self.device,
        )
        accum_type = TensorType(DType.float32, shape=[1], device=self.device)
        force_compute_type = TensorType(
            DType.bool, shape=[1], device=self.device
        )
        return self._base_input_types() + (
            image_hidden_type,  # prev_modulated_input
            image_hidden_type,  # prev_residual
            accum_type,  # accumulated_rel_l1
            force_compute_type,  # force_compute
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model with symbolic shapes."""
        return self._input_types_impl()

    # -- Factored sub-computations -------------------------------------------

    def _forward_preamble(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        guidance: TensorValue,
    ) -> tuple[
        TensorValue,
        TensorValue,
        TensorValue,
        tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        tuple[TensorValue, TensorValue, TensorValue],
        tuple[TensorValue, TensorValue],
        int | Dim,
    ]:
        """Embeddings, modulation, projection, RoPE.

        Returns:
            (projected_hidden_states, encoder_hidden_states,
             temb, double_stream_mod_img, double_stream_mod_txt,
             single_stream_mod, image_rotary_emb, num_txt_tokens).
        """
        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = ops.cast(timestep * 1000.0, hidden_states.dtype)
        guidance = ops.cast(guidance * 1000.0, hidden_states.dtype)
        temb = self.time_guidance_embed(timestep, guidance)

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

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        ids = ops.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        return (
            hidden_states,
            encoder_hidden_states,
            temb,
            double_stream_mod_img,
            double_stream_mod_txt,
            single_stream_mod,
            image_rotary_emb,
            num_txt_tokens,
        )

    def _run_first_block(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        double_stream_mod_img: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        double_stream_mod_txt: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        image_rotary_emb: tuple[TensorValue, TensorValue],
    ) -> tuple[TensorValue, TensorValue]:
        """Run the first dual-stream transformer block.

        Returns:
            (first_encoder_hidden_states, first_hidden_states).
        """
        return self.transformer_blocks[0](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=double_stream_mod_img,
            temb_mod_params_txt=double_stream_mod_txt,
            image_rotary_emb=image_rotary_emb,
        )

    def _run_remaining_blocks(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        double_stream_mod_img: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        double_stream_mod_txt: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        single_stream_mod: tuple[TensorValue, TensorValue, TensorValue],
        image_rotary_emb: tuple[TensorValue, TensorValue],
        num_txt_tokens: int | Dim,
    ) -> TensorValue:
        """Run dual-stream blocks 1..N and all single-stream blocks.

        Returns:
            Pre-tail image hidden states [B, image_seq_len, inner_dim].
        """
        for block in self.transformer_blocks[1:]:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = ops.concat(
            [encoder_hidden_states, hidden_states], axis=1
        )

        for single_block in self.single_transformer_blocks:
            hidden_states = single_block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=image_rotary_emb,
                split_hidden_states=False,
            )
            if isinstance(hidden_states, tuple):
                raise ValueError("Expected concatenated hidden states")

        return hidden_states[:, num_txt_tokens:, :]

    def _forward_postamble(
        self, hidden_states: TensorValue, temb: TensorValue
    ) -> TensorValue:
        """Final norm and projection after the transformer backbone."""
        return self.proj_out(self.norm_out(hidden_states, temb))

    def _teacache_modulated_input(
        self,
        hidden_states: TensorValue,
        double_stream_mod_img: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
    ) -> TensorValue:
        """Compute TeaCache's image-stream modulated input for the first block."""
        (shift_msa, scale_msa, _gate_msa), _ = double_stream_mod_img
        block0 = self.transformer_blocks[0]
        norm_hidden_states = block0.norm1(hidden_states)
        return (1 + scale_msa) * norm_hidden_states + shift_msa

    # -- Forward paths -------------------------------------------------------

    def _forward_standard(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        guidance: TensorValue,
    ) -> tuple[TensorValue]:
        """Standard forward pass (no cache)."""
        (
            projected,
            encoder_hidden_states,
            temb,
            double_stream_mod_img,
            double_stream_mod_txt,
            single_stream_mod,
            image_rotary_emb,
            num_txt_tokens,
        ) = self._forward_preamble(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )
        first_encoder, first_hidden = self._run_first_block(
            projected,
            encoder_hidden_states,
            double_stream_mod_img,
            double_stream_mod_txt,
            image_rotary_emb,
        )
        image_hidden = self._run_remaining_blocks(
            first_hidden,
            first_encoder,
            double_stream_mod_img,
            double_stream_mod_txt,
            single_stream_mod,
            image_rotary_emb,
            num_txt_tokens,
        )
        return (self._forward_postamble(image_hidden, temb),)

    def _forward_fbcache(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        guidance: TensorValue,
        prev_residual: TensorValue,
        prev_output: TensorValue,
        residual_threshold: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """FBCache forward pass with ops.cond branching for cache reuse.

        Uses the same default-parameter-binding trick as cache_mixin's
        fbcache_conditional_execution to capture outer-scope values
        inside ops.cond branch closures.
        """
        (
            projected_hidden_states,
            encoder_hidden_states,
            temb,
            double_stream_mod_img,
            double_stream_mod_txt,
            single_stream_mod,
            image_rotary_emb,
            num_txt_tokens,
        ) = self._forward_preamble(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )

        first_encoder, first_hidden = self._run_first_block(
            projected_hidden_states,
            encoder_hidden_states,
            double_stream_mod_img,
            double_stream_mod_txt,
            image_rotary_emb,
        )
        first_block_residual = first_hidden - projected_hidden_states

        use_fbcache = can_use_fbcache(
            first_block_residual, prev_residual, residual_threshold
        )

        output_types = self._fbcache_output_types()

        # Default parameter binding captures outer values correctly
        # in ops.cond branch closures (same pattern as cache_mixin.py).
        def then_fn(
            _prev_output: TensorValue = prev_output,
            _fbr: TensorValue = first_block_residual,
        ) -> tuple[TensorValue, TensorValue]:
            return (_fbr, _prev_output)

        def else_fn(
            _fbr: TensorValue = first_block_residual,
        ) -> tuple[TensorValue, TensorValue]:
            image_hidden = self._run_remaining_blocks(
                first_hidden,
                first_encoder,
                double_stream_mod_img,
                double_stream_mod_txt,
                single_stream_mod,
                image_rotary_emb,
                num_txt_tokens,
            )
            out = self._forward_postamble(image_hidden, temb)
            return (_fbr, out)

        result = ops.cond(use_fbcache, output_types, then_fn, else_fn)
        return (result[0], result[1])

    def _forward_teacache(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        guidance: TensorValue,
        prev_modulated_input: TensorValue,
        prev_residual: TensorValue,
        accumulated_rel_l1: TensorValue,
        force_compute: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
        """TeaCache forward pass using shared teacache utilities."""
        (
            projected_hidden_states,
            encoder_hidden_states,
            temb,
            double_stream_mod_img,
            double_stream_mod_txt,
            single_stream_mod,
            image_rotary_emb,
            num_txt_tokens,
        ) = self._forward_preamble(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )

        modulated_input = self._teacache_modulated_input(
            projected_hidden_states, double_stream_mod_img
        )

        delta = teacache_rescaled_delta(
            Tensor(modulated_input),
            Tensor(prev_modulated_input),
            self._teacache_coefficients,
        )
        next_accumulated = Tensor(accumulated_rel_l1) + delta

        return teacache_conditional_execution(
            modulated_input=Tensor(modulated_input),
            next_accumulated=next_accumulated,
            accumulated_rel_l1=Tensor(accumulated_rel_l1),
            force_compute=Tensor(force_compute),
            rel_l1_thresh=self._teacache_rel_l1_thresh,
            projected_hidden_states=Tensor(projected_hidden_states),
            prev_residual=Tensor(prev_residual),
            temb=Tensor(temb),
            run_first_block=self._run_first_block,
            first_block_kwargs=dict(
                hidden_states=projected_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                double_stream_mod_img=double_stream_mod_img,
                double_stream_mod_txt=double_stream_mod_txt,
                image_rotary_emb=image_rotary_emb,
            ),
            run_remaining_blocks=self._run_remaining_blocks,
            remaining_blocks_kwargs=dict(
                double_stream_mod_img=double_stream_mod_img,
                double_stream_mod_txt=double_stream_mod_txt,
                single_stream_mod=single_stream_mod,
                image_rotary_emb=image_rotary_emb,
                num_txt_tokens=num_txt_tokens,
            ),
            run_postamble=self._forward_postamble,
            output_types=self._teacache_output_types(),
        )

    def __call__(self, *args: TensorValue) -> tuple[TensorValue, ...]:
        """Forward pass, dispatched to standard or cache impl at init time."""
        return self._forward_impl(*args)
