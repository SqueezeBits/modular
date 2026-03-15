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

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import WanConfig
from .wan_transformer import (
    WanTransformerBlock,
    WanTransformerPostProcess,
    WanTransformerPreProcess,
)

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
    freqs = 1.0 / (theta**freq_exponent)
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
) -> tuple[Buffer, Buffer]:
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

    cos_buf = Buffer.from_numpy(rope_cos).to(device)
    sin_buf = Buffer.from_numpy(rope_sin).to(device)
    return cos_buf, sin_buf


class _BlockLevelModel:
    """Executes transformer forward pass as pre -> N blocks -> post.

    Each component is a separately compiled graph, so only one block's
    workspace is live at any time.  This reduces peak VRAM from
    O(num_blocks * per_block_workspace) to O(per_block_workspace).
    """

    def __init__(
        self,
        pre: Model,
        blocks: list[Model],
        post: Model,
    ) -> None:
        self.pre = pre
        self.blocks = blocks
        self.post = post

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
    ) -> Buffer:
        pre_out = self.pre.execute(
            hidden_states,
            timestep,
            encoder_hidden_states,
        )
        hs, temb, timestep_proj, text_emb = (
            pre_out[0],
            pre_out[1],
            pre_out[2],
            pre_out[3],
        )
        for block in self.blocks:
            block_out = block.execute(
                hs, text_emb, timestep_proj, rope_cos, rope_sin
            )
            hs = block_out[0]
        post_out = self.post.execute(hs, temb, spatial_shape)
        return post_out[0]


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
        session: InferenceSession | None = None,
        eager_load: bool = True,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = WanConfig.generate(config, encoding, devices)
        self._state_dict: dict[str, object] | None = None
        self.model: _BlockLevelModel | None = None
        self._rope_cache: dict[tuple[int, int, int], tuple[Buffer, Buffer]] = {}
        self._max_rope_cache_entries = 8
        self.session = session or InferenceSession(devices=devices)
        self._load_lock = threading.Lock()
        if eager_load:
            self.load_model()

    def _ensure_state_dict(self) -> dict[str, object]:
        if self._state_dict is None:
            self._state_dict = _remap_state_dict(
                self.weights,
                target_dtype=self.config.dtype,
            )
            self.weights = None  # type: ignore[assignment]
        return self._state_dict

    def prepare_state_dict(self) -> dict[str, object]:
        """Materialize the remapped state dict without compiling graphs."""
        with self._load_lock:
            return self._ensure_state_dict()

    def _split_state_dict(
        self, state_dict: dict[str, object]
    ) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        pre_weights: dict[str, object] = {}
        post_weights: dict[str, object] = {}
        block_weights_list: list[dict[str, object]] = [
            {} for _ in range(self.config.num_layers)
        ]

        for key, value in state_dict.items():
            if key.startswith("patch_embedding.") or key.startswith(
                "condition_embedder."
            ):
                pre_weights[key] = value
            elif key.startswith("blocks."):
                rest = key[len("blocks.") :]
                dot = rest.index(".")
                block_idx = int(rest[:dot])
                sub_key = rest[dot + 1 :]
                block_weights_list[block_idx][sub_key] = value
            else:
                post_weights[key] = value

        return pre_weights, block_weights_list, post_weights

    def _build_weight_registries(
        self, state_dict: dict[str, object]
    ) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        dim = self.config.num_attention_heads * self.config.attention_head_dim
        dtype = self.config.dtype
        dev_ref = DeviceRef.from_device(self.config.device)
        pre_weights, block_weights_list, post_weights = self._split_state_dict(
            state_dict
        )

        pre_module = WanTransformerPreProcess(
            self.config, dtype=dtype, device=dev_ref
        )
        pre_module.load_state_dict(pre_weights, weight_alignment=1, strict=True)

        # Reuse a single block module instance to build all weight registries
        block_registries: list[dict[str, object]] = []
        block_module = WanTransformerBlock(
            dim=dim,
            ffn_dim=self.config.ffn_dim,
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.attention_head_dim,
            text_dim=dim,
            cross_attn_norm=self.config.cross_attn_norm,
            eps=self.config.eps,
            dtype=dtype,
            device=dev_ref,
        )
        for block_weights in block_weights_list:
            block_module.load_state_dict(
                block_weights, weight_alignment=1, strict=True
            )
            block_registries.append(block_module.state_dict())

        post_module = WanTransformerPostProcess(
            self.config, dtype=dtype, device=dev_ref
        )
        post_module.load_state_dict(
            post_weights, weight_alignment=1, strict=True
        )

        return (
            pre_module.state_dict(),
            block_registries,
            post_module.state_dict(),
        )

    def reload_model_weights(
        self, state_dict: dict[str, object] | None = None
    ) -> None:
        """Reload weights into already-compiled models via private Model._load.

        Uses Model._load (private API) to hot-swap weight buffers without
        recompiling the graph.  This is needed for MoE weight switching
        between high-noise and low-noise transformer experts.
        """
        with self._load_lock:
            if self.model is None:
                self.load_model()
            if self.model is None:
                raise RuntimeError("Wan transformer model failed to load.")

            target_state_dict = state_dict or self._ensure_state_dict()
            pre_registry, block_registries, post_registry = (
                self._build_weight_registries(target_state_dict)
            )

            self.model.pre._load(pre_registry)
            for compiled_block, block_registry in zip(
                self.model.blocks, block_registries, strict=True
            ):
                compiled_block._load(block_registry)
            self.model.post._load(post_registry)

    def load_model(self) -> Callable[..., Any]:
        """Compile the transformer as separate pre/block/post graphs.

        Uses symbolic dimensions so the compiled graphs work for any
        resolution / frame count without recompilation.
        """
        with self._load_lock:
            if self.model is not None:
                return self.__call__

            state_dict = self._ensure_state_dict()

            dim = (
                self.config.num_attention_heads * self.config.attention_head_dim
            )
            dtype = self.config.dtype
            dev = self.config.device
            dev_ref = DeviceRef.from_device(dev)

            pre_weights, block_weights_list, post_weights = (
                self._split_state_dict(state_dict)
            )

            # --- Compile pre-processing (symbolic spatial dims) ---
            pre_input_types = [
                TensorType(
                    dtype,
                    [
                        "batch",
                        self.config.in_channels,
                        "frames",
                        "height",
                        "width",
                    ],
                    device=dev,
                ),
                TensorType(DType.float32, ["batch"], device=dev),
                TensorType(
                    dtype,
                    ["batch", "seq_text", self.config.text_dim],
                    device=dev,
                ),
            ]
            pre_module = WanTransformerPreProcess(
                self.config, dtype=dtype, device=dev_ref
            )
            pre_module.load_state_dict(
                pre_weights, weight_alignment=1, strict=True
            )
            with Graph("wan_pre", input_types=pre_input_types) as pre_graph:
                outs = pre_module(*(v.tensor for v in pre_graph.inputs))
                pre_graph.output(*outs)
            pre_model = self.session.load(
                pre_graph, weights_registry=pre_module.state_dict()
            )

            # --- Compile transformer block graph ONCE, reuse for all layers ---
            # All 40 blocks have identical structure (same dims, heads, FFN).
            # Compile the graph once, then use Model._load() to swap weights
            # for each subsequent block — avoids 39 redundant compilations.
            block_input_types = [
                TensorType(dtype, ["batch", "seq_len", dim], device=dev),
                TensorType(dtype, ["batch", "seq_text", dim], device=dev),
                TensorType(dtype, ["batch", 6, dim], device=dev),
                TensorType(
                    DType.float32,
                    ["seq_len", self.config.attention_head_dim],
                    device=dev,
                ),
                TensorType(
                    DType.float32,
                    ["seq_len", self.config.attention_head_dim],
                    device=dev,
                ),
            ]

            # Build block module template (used for graph + all weight registries)
            block_template = WanTransformerBlock(
                dim=dim,
                ffn_dim=self.config.ffn_dim,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.attention_head_dim,
                text_dim=dim,
                cross_attn_norm=self.config.cross_attn_norm,
                eps=self.config.eps,
                dtype=dtype,
                device=dev_ref,
            )

            # Build and compile graph with block 0's weights
            block_template.load_state_dict(
                block_weights_list[0], weight_alignment=1, strict=True
            )
            with Graph(
                "wan_block", input_types=block_input_types
            ) as block_graph:
                block_out = block_template(
                    *(v.tensor for v in block_graph.inputs)
                )
                block_graph.output(block_out)
            block_0_model = self.session.load(
                block_graph, weights_registry=block_template.state_dict()
            )
            block_models: list[Model] = [block_0_model]

            # Reuse compiled graph for blocks 1..N-1 via _load()
            for i in range(1, self.config.num_layers):
                block_template.load_state_dict(
                    block_weights_list[i], weight_alignment=1, strict=True
                )
                block_i_model = self.session.load(
                    block_graph,
                    weights_registry=block_template.state_dict(),
                )
                block_models.append(block_i_model)

            # --- Compile post-processing (spatial shape tensor carries ppf/pph/ppw) ---
            post_input_types = [
                TensorType(dtype, ["batch", "seq_len", dim], device=dev),
                TensorType(dtype, ["batch", dim], device=dev),  # temb
                TensorType(DType.int8, ["ppf", "pph", "ppw"], device=dev),
            ]
            post_module = WanTransformerPostProcess(
                self.config, dtype=dtype, device=dev_ref
            )
            post_module.load_state_dict(
                post_weights, weight_alignment=1, strict=True
            )
            with Graph("wan_post", input_types=post_input_types) as post_graph:
                post_out = post_module(*(v.tensor for v in post_graph.inputs))
                post_graph.output(post_out)
            post_model = self.session.load(
                post_graph, weights_registry=post_module.state_dict()
            )
            self.model = _BlockLevelModel(pre_model, block_models, post_model)
            return self.__call__

    def warmup(self) -> None:
        """Run a tiny forward pass to force GPU kernel initialization.

        Uses minimal dimensions (1 frame, 2x2 spatial) to trigger lazy kernel
        compilation without wasting time on large tensors.
        """
        if self.model is None:
            self.load_model()
        assert self.model is not None

        import logging

        logger = logging.getLogger(__name__)
        from time import perf_counter

        t0 = perf_counter()

        dev = self.devices[0]
        dtype = self.config.dtype
        p_t, p_h, p_w = self.config.patch_size
        dim = self.config.num_attention_heads * self.config.attention_head_dim
        # Minimal latent: 1 frame, patch_size spatial
        warmup_frames, warmup_h, warmup_w = p_t, p_h * 2, p_w * 2
        seq_len = (warmup_frames // p_t) * (warmup_h // p_h) * (warmup_w // p_w)

        from max.pipelines.lib.bfloat16_utils import (
            float32_to_bfloat16_as_uint16,
        )

        def _zeros_buf(shape: tuple[int, ...], dt: DType) -> Buffer:
            arr = np.zeros(shape, dtype=np.float32)
            if dt == DType.bfloat16:
                u16 = float32_to_bfloat16_as_uint16(arr)
                return (
                    Buffer.from_numpy(u16)
                    .to(dev)
                    .view(dtype=DType.bfloat16, shape=shape)
                )
            return Buffer.from_numpy(arr).to(dev)

        hs = _zeros_buf(
            (1, self.config.in_channels, warmup_frames, warmup_h, warmup_w),
            dtype,
        )
        ts = Buffer.from_numpy(np.zeros((1,), dtype=np.float32)).to(dev)
        enc = _zeros_buf((1, 4, self.config.text_dim), dtype)
        rope_cos, rope_sin = self.compute_rope(
            warmup_frames, warmup_h, warmup_w
        )
        spatial = Buffer.from_numpy(
            np.zeros(
                (warmup_frames // p_t, warmup_h // p_h, warmup_w // p_w),
                dtype=np.int8,
            )
        ).to(dev)

        self.model(hs, ts, enc, rope_cos, rope_sin, spatial)
        logger.info(
            "Wan transformer warmup complete in %.2fs (seq_len=%d)",
            perf_counter() - t0,
            seq_len,
        )

    def compute_rope(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[Buffer, Buffer]:
        """Compute 3D RoPE cos/sin tensors for the given latent dimensions."""
        key = (num_frames, height, width)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached

        rope = _compute_wan_rope(
            num_frames=num_frames,
            height=height,
            width=width,
            patch_size=self.config.patch_size,
            head_dim=self.config.attention_head_dim,
            device=self.devices[0],
        )
        self._rope_cache[key] = rope
        if len(self._rope_cache) > self._max_rope_cache_entries:
            self._rope_cache.pop(next(iter(self._rope_cache)))
        return rope

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
    ) -> Buffer:
        if self.model is None:
            self.load_model()
        if self.model is None:
            raise RuntimeError("Wan transformer model failed to load.")
        return self.model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            spatial_shape,
        )
