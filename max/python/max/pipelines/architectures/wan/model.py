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

import gc
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.graph import TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces import CompileWrapper
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor

from .model_config import WanConfig
from .wan_transformer import (
    WanTransformerBlock,
    WanTransformerPostProcess,
    WanTransformerPreProcess,
)

logger = logging.getLogger(__name__)

# Weight key remapping from diffusers -> MAX module naming
_KEY_REMAP = [
    (".attn1.to_out.0.", ".attn1.to_out."),
    (".attn2.to_out.0.", ".attn2.to_out."),
    (".ffn.net.0.proj.", ".ffn.proj."),
    (".ffn.net.2.", ".ffn.linear_out."),
]

# Keys to skip (non-persistent buffers computed at runtime)
_SKIP_PREFIXES = ("rope.freqs_cos", "rope.freqs_sin")


def _remap_state_dict(
    weights: Weights,
    target_dtype: DType = DType.bfloat16,
) -> dict[str, object]:
    """Remap diffusers weight keys to MAX module naming, permute Conv3d,
    and cast weights to target dtype.

    The WAN safetensors store weights as float32. We cast to bfloat16
    to match the module parameter declarations (which must also be bfloat16).
    """
    state_dict: dict[str, object] = {}

    # First pass: collect all weights with key remapping.
    raw_dict: dict[str, object] = {}
    for key, value in weights.items():
        if any(key.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        new_key = key
        for old, new in _KEY_REMAP:
            new_key = new_key.replace(old, new)

        tensor = value.data()

        # Conv3d weight permutation for patch_embedding
        # Diffusers: [F, C, D, H, W] (PyTorch FCDHW)
        # MAX Conv3d(permute=False): [D, H, W, C, F] (QRSCF)
        if new_key == "patch_embedding.weight" and len(tensor.shape) == 5:
            tensor = np.ascontiguousarray(
                np.from_dlpack(tensor).transpose(2, 3, 4, 1, 0)
            )

        raw_dict[new_key] = tensor

    # Second pass: fuse attn2.to_k + attn2.to_v into attn2.to_kv
    fused_keys: set[str] = set()
    for key in list(raw_dict.keys()):
        if ".attn2.to_k." in key:
            k_key = key
            v_key = key.replace(".attn2.to_k.", ".attn2.to_v.")
            kv_key = key.replace(".attn2.to_k.", ".attn2.to_kv.")
            if v_key in raw_dict:
                k_np = np.from_dlpack(raw_dict[k_key])
                v_np = np.from_dlpack(raw_dict[v_key])
                kv_np = np.ascontiguousarray(
                    np.concatenate([k_np, v_np], axis=0)
                )
                state_dict[kv_key] = kv_np
                fused_keys.add(k_key)
                fused_keys.add(v_key)

    # Copy remaining unfused keys
    for key, tensor in raw_dict.items():
        if key not in fused_keys:
            state_dict[key] = tensor

    logger.info("_remap_state_dict: %d keys", len(state_dict))

    # Cast all weights to target dtype. The WAN safetensors are float32 but
    # the module parameters must be bfloat16 (for flash_attention_gpu and to
    # match the constant_external declarations in Module.compile).
    if target_dtype != DType.float32:
        cpu_device = CPU()
        for key in state_dict:
            state_dict[key] = cast_dlpack_to(
                state_dict[key], DType.float32, target_dtype, cpu_device
            )

    return state_dict


def _get_1d_rotary_pos_embed_np(
    dim: int,
    pos: np.ndarray,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D rotary position embeddings (numpy, for eager RoPE)."""
    freq_exponent = np.arange(0, dim, 2, dtype=np.float64) / dim
    freqs = 1.0 / (theta ** freq_exponent)
    # Outer product: [S, dim/2]
    angles = np.outer(pos.astype(np.float64), freqs)
    cos_emb = np.cos(angles).astype(np.float32)
    sin_emb = np.sin(angles).astype(np.float32)
    # repeat_interleave(2): [a,b,c] -> [a,a,b,b,c,c]
    cos_emb = np.repeat(cos_emb, 2, axis=1)
    sin_emb = np.repeat(sin_emb, 2, axis=1)
    return cos_emb, sin_emb


def _compute_wan_rope(
    num_frames: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
    head_dim: int,
    device: Device,
    theta: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """Compute 3D RoPE cos/sin tensors for Wan transformer.

    Splits head_dim into 3 parts for temporal, height, and width axes,
    computes per-axis 1D RoPE, then broadcasts to a 3D grid and concatenates.
    """
    p_t, p_h, p_w = patch_size
    ppf = num_frames // p_t  # post-patch frames
    pph = height // p_h  # post-patch height
    ppw = width // p_w  # post-patch width

    # Split head_dim across 3 axes (matching diffusers Wan RoPE)
    d_h = (head_dim // 3 // 2) * 2
    d_w = d_h
    d_t = head_dim - d_h - d_w

    # Compute per-axis 1D RoPE
    t_pos = np.arange(ppf, dtype=np.float32)
    h_pos = np.arange(pph, dtype=np.float32)
    w_pos = np.arange(ppw, dtype=np.float32)

    cos_t, sin_t = _get_1d_rotary_pos_embed_np(d_t, t_pos, theta)  # [ppf, d_t]
    cos_h, sin_h = _get_1d_rotary_pos_embed_np(d_h, h_pos, theta)  # [pph, d_h]
    cos_w, sin_w = _get_1d_rotary_pos_embed_np(d_w, w_pos, theta)  # [ppw, d_w]

    # Broadcast to 3D grid: [ppf, pph, ppw, head_dim]
    cos_t = cos_t[:, np.newaxis, np.newaxis, :]
    sin_t = sin_t[:, np.newaxis, np.newaxis, :]
    cos_h = cos_h[np.newaxis, :, np.newaxis, :]
    sin_h = sin_h[np.newaxis, :, np.newaxis, :]
    cos_w = cos_w[np.newaxis, np.newaxis, :, :]
    sin_w = sin_w[np.newaxis, np.newaxis, :, :]

    cos_t = np.broadcast_to(cos_t, (ppf, pph, ppw, d_t))
    sin_t = np.broadcast_to(sin_t, (ppf, pph, ppw, d_t))
    cos_h = np.broadcast_to(cos_h, (ppf, pph, ppw, d_h))
    sin_h = np.broadcast_to(sin_h, (ppf, pph, ppw, d_h))
    cos_w = np.broadcast_to(cos_w, (ppf, pph, ppw, d_w))
    sin_w = np.broadcast_to(sin_w, (ppf, pph, ppw, d_w))

    # Concatenate along last dim: [ppf, pph, ppw, head_dim]
    rope_cos = np.concatenate([cos_t, cos_h, cos_w], axis=-1)
    rope_sin = np.concatenate([sin_t, sin_h, sin_w], axis=-1)

    # Flatten spatial: [ppf*pph*ppw, head_dim]
    seq_len = ppf * pph * ppw
    rope_cos = np.ascontiguousarray(rope_cos.reshape(seq_len, head_dim))
    rope_sin = np.ascontiguousarray(rope_sin.reshape(seq_len, head_dim))

    cos_tensor = Tensor.from_dlpack(rope_cos).to(device)
    sin_tensor = Tensor.from_dlpack(rope_sin).to(device)
    return cos_tensor, sin_tensor


class _BlockLevelModel:
    """Executes transformer forward pass as pre -> N blocks -> post.

    Each component is a separately compiled graph, so only one block's
    workspace is live at any time.  This reduces peak VRAM from
    O(num_blocks * per_block_workspace) to O(per_block_workspace).
    """

    def __init__(
        self,
        pre: CompileWrapper,
        blocks: list[CompileWrapper],
        post: CompileWrapper,
    ) -> None:
        self.pre = pre
        self.blocks = blocks
        self.post = post

    def __call__(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        hs, temb, timestep_proj, text_emb = self.pre(
            hidden_states, timestep, encoder_hidden_states,
        )
        for block in self.blocks:
            hs = block(hs, text_emb, timestep_proj, rope_cos, rope_sin)
        return self.post(hs, temb)


class WanTransformerModel(ComponentModel):
    """MAX-native Wan DiT interface with block-level compilation.

    Instead of compiling the full 40-block transformer as a single graph
    (which requires O(N) workspace), each block is compiled independently.
    Only one block's execution workspace is live at a time, reducing peak
    VRAM from ~130 GB to ~40 GB.
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = WanConfig.generate(config, encoding, devices)
        self._state_dict = _remap_state_dict(
            self.weights,
            target_dtype=self.config.dtype,
        )
        # Free the raw Weights object (~56 GB fp32) now that we have the
        # remapped state dict.
        self.weights = None  # type: ignore[assignment]
        self.model: _BlockLevelModel | None = None

    def load_model(self) -> Callable[..., Any]:
        return lambda: None

    def compile_model(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        rope_cos: Tensor,
    ) -> None:
        """Compile the transformer as separate pre/block/post graphs.

        Called once before the denoising loop. Subsequent calls are no-ops.
        """
        if self.model is not None:
            return

        hs_shape = [int(s) for s in hidden_states.shape]
        batch_size = hs_shape[0]
        ts_shape = [batch_size]
        enc_shape = [int(s) for s in encoder_hidden_states.shape]
        rope_shape = [int(s) for s in rope_cos.shape]

        p_t, p_h, p_w = self.config.patch_size
        ppf = hs_shape[2] // p_t
        pph = hs_shape[3] // p_h
        ppw = hs_shape[4] // p_w
        seq_len = ppf * pph * ppw
        dim = self.config.num_attention_heads * self.config.attention_head_dim

        dtype = self.config.dtype
        dev = self.config.device

        logger.info(
            "Compiling transformer (block-level): "
            "seq_len=%d, %d blocks, hidden=%s enc=%s",
            seq_len, self.config.num_layers, hs_shape, enc_shape,
        )

        # --- Split state dict by component ---
        pre_weights: dict[str, object] = {}
        post_weights: dict[str, object] = {}
        block_weights_list: list[dict[str, object]] = [
            {} for _ in range(self.config.num_layers)
        ]

        for key, value in self._state_dict.items():
            if key.startswith("patch_embedding.") or key.startswith(
                "condition_embedder."
            ):
                pre_weights[key] = value
            elif key.startswith("blocks."):
                # "blocks.3.attn1.to_q.weight" -> block_idx=3, sub_key="attn1.to_q.weight"
                rest = key[len("blocks."):]
                dot = rest.index(".")
                block_idx = int(rest[:dot])
                sub_key = rest[dot + 1:]
                block_weights_list[block_idx][sub_key] = value
            else:
                # scale_shift_table, norm_out.*, proj_out.*
                post_weights[key] = value

        # --- Compile pre-processing ---
        pre_input_types = (
            TensorType(dtype, hs_shape, device=dev),
            TensorType(DType.float32, ts_shape, device=dev),  # float32 to avoid bf16 timestep quantization
            TensorType(dtype, enc_shape, device=dev),
        )
        with F.lazy():
            pre_module = WanTransformerPreProcess(self.config)
            pre_module.to(self.devices[0])
        pre_model = CompileWrapper(
            pre_module, input_types=pre_input_types, weights=pre_weights,
        )
        logger.info("Compiled pre-processing (%d weights)", len(pre_weights))

        # --- Compile each transformer block ---
        block_hs_type = TensorType(dtype, [batch_size, seq_len, dim], device=dev)
        text_emb_type = TensorType(
            dtype, [batch_size, enc_shape[1], dim], device=dev,
        )
        ts_proj_type = TensorType(dtype, [batch_size, 6, dim], device=dev)
        rope_type = TensorType(DType.float32, rope_shape, device=dev)

        block_input_types = (
            block_hs_type,
            text_emb_type,
            ts_proj_type,
            rope_type,
            rope_type,
        )

        block_models: list[CompileWrapper] = []
        for i in range(self.config.num_layers):
            with F.lazy():
                block = WanTransformerBlock(
                    dim=dim,
                    ffn_dim=self.config.ffn_dim,
                    num_heads=self.config.num_attention_heads,
                    head_dim=self.config.attention_head_dim,
                    text_dim=dim,
                    cross_attn_norm=self.config.cross_attn_norm,
                    eps=self.config.eps,
                )
                block.to(self.devices[0])
            block_models.append(
                CompileWrapper(
                    block,
                    input_types=block_input_types,
                    weights=block_weights_list[i],
                )
            )
        logger.info("Compiled %d transformer blocks", len(block_models))

        # --- Compile post-processing ---
        post_input_types = (
            block_hs_type,
            TensorType(dtype, [batch_size, dim], device=dev),  # temb
        )
        with F.lazy():
            post_module = WanTransformerPostProcess(
                self.config, ppf=ppf, pph=pph, ppw=ppw,
            )
            post_module.to(self.devices[0])
        post_model = CompileWrapper(
            post_module, input_types=post_input_types, weights=post_weights,
        )
        logger.info("Compiled post-processing (%d weights)", len(post_weights))

        self.model = _BlockLevelModel(pre_model, block_models, post_model)

    def compute_rope(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[Tensor, Tensor]:
        """Compute 3D RoPE cos/sin tensors for the given latent dimensions."""
        return _compute_wan_rope(
            num_frames=num_frames,
            height=height,
            width=width,
            patch_size=self.config.patch_size,
            head_dim=self.config.attention_head_dim,
            device=self.devices[0],
        )
