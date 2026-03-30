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

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, Weight, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .layers.attention import ZImageAttention
from .layers.embeddings import RopeEmbedder, TimestepEmbedder
from .model_config import ZImageConfig

# Dimension of the adaLN conditioning input fed to each block.
ADALN_EMBED_DIM = 256


# ---------------------------------------------------------------------------
# Local LayerNorm (no learned affine params by default, matching FinalLayer
# and block usage that passes elementwise_affine=False).
# Mirrors the pattern from flux2/layers/normalizations.py.
# ---------------------------------------------------------------------------


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
        """Initialize LayerNorm.

        Args:
            dim: Size of the last dimension.
            dtype: Weight dtype.
            device: Weight device.
            eps: Epsilon for numerical stability.
            elementwise_affine: If True, learn scale and bias parameters.
            use_bias: If True and elementwise_affine, also learn a bias.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = use_bias
        if elementwise_affine:
            self.weight = Weight("weight", dtype, (dim,), device=device)
            self.bias = (
                Weight("bias", dtype, (dim,), device=device) if use_bias else None
            )
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply layer normalisation.

        Args:
            x: Input tensor of arbitrary shape; norm is applied over last dim.

        Returns:
            Normalised tensor with the same shape as x.
        """
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


# ---------------------------------------------------------------------------
# FeedForward (SwiGLU variant: w2(silu(w1(x)) * w3(x)))
# ---------------------------------------------------------------------------


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
        """Initialize FeedForward.

        Args:
            dim: Input and output dimension.
            hidden_dim: Hidden (gate/up) dimension.
            dtype: Weight dtype.
            device: Weight device.
        """
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
        """Compute SwiGLU feed-forward output.

        Args:
            x: Input tensor of shape [B, S, dim].

        Returns:
            Output tensor of shape [B, S, dim].
        """
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# ZImageTransformerBlock
# ---------------------------------------------------------------------------


class ZImageTransformerBlock(Module):
    """Single transformer block for Z-Image with optional adaLN modulation."""

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        norm_eps: float,
        qk_norm: bool,
        *,
        dtype: DType,
        device: DeviceRef,
        modulation: bool = True,
    ) -> None:
        """Initialize ZImageTransformerBlock.

        Args:
            layer_id: Index of this layer (informational).
            dim: Model hidden dimension.
            n_heads: Number of attention heads.
            norm_eps: Epsilon for RMSNorm layers.
            qk_norm: Whether to apply QK normalisation in attention.
            dtype: Weight dtype.
            device: Weight device.
            modulation: If True, use adaLN modulation from timestep conditioning.
        """
        super().__init__()
        self.layer_id = layer_id
        self.modulation = modulation

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

        if modulation:
            self.adaLN_modulation: Linear | None = Linear(
                in_dim=min(dim, ADALN_EMBED_DIM),
                out_dim=4 * dim,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
        else:
            self.adaLN_modulation = None

    def __call__(
        self,
        x: TensorValue,
        freqs_cis: tuple[TensorValue, TensorValue],
        adaln_input: TensorValue | None = None,
    ) -> TensorValue:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [B, S, dim].
            freqs_cis: Tuple of (cos, sin) rotary embedding tensors.
            adaln_input: Conditioning vector of shape [B, ADALN_EMBED_DIM]
                for adaLN modulation (required when modulation=True).

        Returns:
            Output tensor of shape [B, S, dim].
        """
        if self.modulation:
            if adaln_input is None:
                raise ValueError(
                    "adaln_input is required when modulation=True"
                )
            mod = self.adaLN_modulation(adaln_input)  # [B, 4*dim]
            mod = ops.unsqueeze(mod, 1)  # [B, 1, 4*dim]
            chunks = ops.chunk(mod, chunks=4, axis=2)
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                chunks[0],
                chunks[1],
                chunks[2],
                chunks[3],
            )
            gate_msa = ops.tanh(gate_msa)
            gate_mlp = ops.tanh(gate_mlp)
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp

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


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------


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
        """Initialize FinalLayer.

        Args:
            hidden_size: Input hidden dimension.
            out_channels: Output channel count (patch_size^2 * latent_channels).
            dtype: Weight dtype.
            device: Weight device.
        """
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
        """Apply final normalisation, modulation, and linear projection.

        Args:
            x: Input tensor of shape [B, S, hidden_size].
            c: Conditioning vector of shape [B, ADALN_EMBED_DIM].

        Returns:
            Output tensor of shape [B, S, out_channels].
        """
        scale = 1.0 + self.adaLN_modulation(ops.silu(c))  # [B, hidden_size]
        x = self.norm_final(x) * ops.unsqueeze(scale, 1)
        return self.linear(x)


# ---------------------------------------------------------------------------
# ZImageTransformer2DModel
# ---------------------------------------------------------------------------


class ZImageTransformer2DModel(Module):
    """Z-Image diffusion transformer (DiT) model.

    Implements the core denoising transformer for Z-Image, accepting noisy
    latent patches, text encoder hidden states, and a timestep, and returning
    predicted noise (or velocity) patches.
    """

    def __init__(self, config: ZImageConfig) -> None:
        """Initialize ZImageTransformer2DModel.

        Args:
            config: ZImageConfig with all model hyper-parameters, dtype, and
                device.
        """
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

        # patch_size is the spatial downsampling factor for latents
        patch_size: int = config.all_patch_size[0]
        f_patch_size: int = config.all_f_patch_size[0]
        # number of latent channels going into each patch token
        in_channels: int = (
            config.in_channels * patch_size * patch_size * f_patch_size
        )
        # output channels per patch token (same layout as input)
        out_channels: int = in_channels

        self.packed_channels = in_channels
        self.max_dtype = dtype
        self.max_device = device
        self.cap_feat_dim = cap_feat_dim
        self.t_scale = config.t_scale
        self.axes_dims = axes_dims

        # Patch embedding: project flattened patch to model dim.
        self.x_embedder = Linear(
            in_dim=in_channels,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        # Final output layer.
        self.final_layer = FinalLayer(
            hidden_size=dim,
            out_channels=out_channels,
            dtype=dtype,
            device=device,
        )

        # Noise refiner: modulated transformer blocks (with adaLN from timestep).
        self.noise_refiner = LayerList(
            [
                ZImageTransformerBlock(
                    layer_id=1000 + i,
                    dim=dim,
                    n_heads=n_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=True,
                )
                for i in range(n_refiner_layers)
            ]
        )
        # Context refiner: unmodulated transformer blocks (no adaLN).
        self.context_refiner = LayerList(
            [
                ZImageTransformerBlock(
                    layer_id=i,
                    dim=dim,
                    n_heads=n_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=False,
                )
                for i in range(n_refiner_layers)
            ]
        )

        # Timestep embedder: scalar timestep → ADALN_EMBED_DIM vector.
        self.t_embedder = TimestepEmbedder(
            out_size=min(dim, ADALN_EMBED_DIM),
            mid_size=1024,
            dtype=dtype,
            device=device,
        )

        # Caption normalisation and projection.
        self.cap_norm = RMSNorm(cap_feat_dim, dtype=dtype, eps=norm_eps)
        self.cap_proj = Linear(
            in_dim=cap_feat_dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        # Main transformer layers (adaLN modulation enabled).
        self.layers = LayerList(
            [
                ZImageTransformerBlock(
                    layer_id=i,
                    dim=dim,
                    n_heads=n_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=True,
                )
                for i in range(n_layers)
            ]
        )

        # Rotary position embedder.
        self.rope_embedder = RopeEmbedder(
            theta=rope_theta,
            axes_dims=axes_dims,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define symbolic input tensor types for graph compilation.

        Returns:
            Tuple of TensorType for (hidden_states, encoder_hidden_states,
            timestep, img_ids, txt_ids).
        """
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

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
    ) -> tuple[TensorValue]:
        """Run the Z-Image denoising forward pass.

        Args:
            hidden_states: Noisy latent patches of shape
                [B, image_seq_len, in_channels].
            encoder_hidden_states: Text encoder output of shape
                [B, text_seq_len, cap_feat_dim].
            timestep: Denoising timestep of shape [B].
            img_ids: Image position IDs of shape [image_seq_len, 3].
            txt_ids: Text position IDs of shape [text_seq_len, 3].

        Returns:
            Single-element tuple containing the denoised patch tensor of shape
            [B, image_seq_len, out_channels].
        """
        # Embed image patches.
        x = self.x_embedder(hidden_states)  # [B, image_seq_len, dim]

        # Embed timestep (scaled) to produce adaLN conditioning vector.
        t_emb = self.t_embedder(
            timestep * self.t_scale
        )  # [B, ADALN_EMBED_DIM]
        t_emb = ops.cast(t_emb, x.dtype)

        # Normalise and project caption features.
        cap = self.cap_proj(
            self.cap_norm(encoder_hidden_states)
        )  # [B, text_seq_len, dim]

        # Strip batch dim from IDs if present.
        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        # Compute RoPE embeddings for text and image separately.
        txt_freqs = self.rope_embedder(txt_ids)  # (cos, sin) [text_seq, D]
        img_freqs = self.rope_embedder(img_ids)  # (cos, sin) [img_seq, D]

        # Unified freqs for joint attention: [img + txt, D]
        unified_freqs = (
            ops.concat([img_freqs[0], txt_freqs[0]], axis=0),
            ops.concat([img_freqs[1], txt_freqs[1]], axis=0),
        )

        # Noise refiner: modulated blocks on image tokens.
        for block in self.noise_refiner:
            x = block(x, freqs_cis=img_freqs, adaln_input=t_emb)

        # Context refiner: unmodulated blocks on text tokens.
        for block in self.context_refiner:
            cap = block(cap, freqs_cis=txt_freqs)

        # Concatenate image and context tokens for joint attention.
        img_len = x.shape[1]
        x = ops.concat([x, cap], axis=1)  # [B, img+txt, dim]

        # Run main modulated transformer layers.
        for block in self.layers:
            x = block(x, freqs_cis=unified_freqs, adaln_input=t_emb)

        # Strip context tokens, apply final layer.
        x = x[:, :img_len, :]  # [B, image_seq_len, dim]
        x = self.final_layer(x, t_emb)  # [B, image_seq_len, out_channels]

        return (x,)
