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

"""Wan-Animate transformer model with block-level compilation.

Extends the base WanTransformerModel with:
- Pose patch embedding injection in pre-processing
- CLIP image embedding in pre-processing (passed to block cross-attention)
- Face adapter injection every `inject_face_latents_blocks` blocks
- Face encoder (run once per segment, before denoising loop)
- Motion encoder (PyTorch bridge, run once per segment)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model import (
    WanTransformerModel,
    _BlockLevelModel,
    _compute_wan_rope,
    _remap_state_dict,
)
from .model_config import WanConfig
from .wan_animate_transformer import (
    WanAnimateFaceBlock,
    WanAnimateFaceEncoder,
    WanAnimateMotionEncoder,
    WanAnimatePreProcess,
    _MOTION_ENCODER_CHANNEL_SIZES,
)
from .wan_transformer import (
    WanTransformerBlock,
    WanTransformerPostProcess,
)

logger = logging.getLogger(__name__)

class _AnimateBlockLevelModel:
    """Executes animate transformer: pre -> N blocks (with face adapter) -> post.

    Face adapter cross-attention is injected every `inject_interval` blocks.
    """

    def __init__(
        self,
        pre: Model,
        blocks: list[Model],
        face_adapters: list[Model],
        post: Model,
        inject_interval: int = 5,
    ) -> None:
        self.pre = pre
        self.blocks = blocks
        self.face_adapters = face_adapters
        self.post = post
        self.inject_interval = inject_interval

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        clip_features: Buffer,
        pose_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        face_emb: Buffer,
        num_temporal_frames: Buffer,
    ) -> Buffer:
        # Pre: patch_embed + pose injection + condition embedding + image embedding
        pre_out = self.pre.execute(
            hidden_states,
            timestep,
            encoder_hidden_states,
            clip_features,
            pose_hidden_states,
        )
        hs, temb, timestep_proj, text_emb, image_embeds = (
            pre_out[0],
            pre_out[1],
            pre_out[2],
            pre_out[3],
            pre_out[4],
        )

        # Block loop with face adapter injection
        adapter_idx = 0
        for i, block in enumerate(self.blocks):
            block_out = block.execute(
                hs, text_emb, timestep_proj, rope_cos, rope_sin, image_embeds
            )
            hs = block_out[0]

            if (
                i % self.inject_interval == 0
                and adapter_idx < len(self.face_adapters)
            ):
                adapter_out = self.face_adapters[adapter_idx].execute(
                    hs, face_emb, num_temporal_frames
                )
                hs = adapter_out[0]
                adapter_idx += 1

        # Post: output modulation + unpatchify
        post_out = self.post.execute(hs, temb, spatial_shape)
        return post_out[0]


class WanAnimateTransformerModel(ComponentModel):
    """MAX-native Wan-Animate transformer with block-level compilation.

    Extends WanTransformerModel with pose injection, CLIP conditioning,
    face encoder, and face adapter blocks.
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
        self.config.dtype = DType.bfloat16
        self._state_dict: dict[str, Any] | None = None
        self.model: _AnimateBlockLevelModel | None = None
        self._rope_cache: dict[tuple[int, int, int], tuple[Buffer, Buffer]] = {}
        self._max_rope_cache_entries = 8
        self.session = session or InferenceSession(devices=devices)
        self._load_lock = threading.Lock()

        # Face encoder model (compiled MAX graph, run once per segment)
        self._face_encoder_model: Model | None = None
        # Motion encoder model (compiled MAX graph, run once per segment)
        self._motion_encoder_model: Model | None = None

        if eager_load:
            self.load_model()

    def _ensure_state_dict(self) -> dict[str, Any]:
        if self._state_dict is None:
            if self.weights is None:
                raise RuntimeError(
                    "WanAnimateTransformerModel weights unavailable."
                )
            self._state_dict = _remap_state_dict(
                self.weights, target_dtype=DType.bfloat16
            )

            # Post-remap transformations on Buffer objects (already bfloat16).
            from max.driver import CPU
            from max.graph.buffer_utils import cast_dlpack_to

            cpu = CPU()

            def _buf_to_np(buf: Any) -> np.ndarray:
                """Convert a Buffer (possibly bfloat16) to float32 numpy."""
                f32 = cast_dlpack_to(buf, DType.bfloat16, DType.float32, cpu)
                return np.from_dlpack(f32)

            def _np_to_buf(arr: np.ndarray) -> Any:
                """Convert float32 numpy to bfloat16 Buffer."""
                return cast_dlpack_to(
                    np.ascontiguousarray(arr),
                    DType.float32, DType.bfloat16, cpu,
                )

            # Permute pose_patch_embedding Conv3d weight
            # [F, C, D, H, W] -> [D, H, W, C, F]
            pose_key = "pose_patch_embedding.weight"
            if pose_key in self._state_dict:
                arr = _buf_to_np(self._state_dict[pose_key])
                if arr.ndim == 5:
                    arr = arr.transpose(2, 3, 4, 1, 0)
                    self._state_dict[pose_key] = _np_to_buf(arr)

            # Reshape face_encoder Conv1d weights:
            # [out, in, k] (PyTorch) -> [k*in, out] (matmul-based conv1d)
            for k in list(self._state_dict.keys()):
                if k.startswith("face_encoder.") and k.endswith(".weight"):
                    buf = self._state_dict[k]
                    shape = tuple(int(d) for d in buf.shape)
                    if len(shape) == 3:  # Conv1d weight [out, in, k]
                        arr = _buf_to_np(buf)
                        # [out, in, k] -> [k, in, out] -> [k*in, out]
                        arr = arr.transpose(2, 1, 0)  # [k, in, out]
                        k_size, in_ch, out_ch = arr.shape
                        arr = arr.reshape(k_size * in_ch, out_ch)
                        self._state_dict[k] = _np_to_buf(arr)

            # --- Motion encoder weight preprocessing (Phase 7) ---
            self._preprocess_motion_encoder_weights(
                self._state_dict, _buf_to_np, _np_to_buf
            )

            self.weights = None  # type: ignore[assignment]
        return self._state_dict

    @staticmethod
    def _preprocess_motion_encoder_weights(
        state_dict: dict[str, Any],
        buf_to_np: Any,
        np_to_buf: Any,
    ) -> None:
        """Pre-process motion encoder weights in-place.

        Transforms:
        - Conv weights: OIHW → RSCF, scale ``1/sqrt(fan_in)`` baked in.
        - Linear weights: ``[out, in] → [in, out]``, scale baked in.
        - ``motion_synthesis_weight`` → ``q_matrix`` via QR decomposition.
        - Key remapping: ``res_blocks.{i}.`` → ``res_blocks_{i}.``,
          ``motion_network.{i}.`` → ``motion_network_{i}.``.
        """
        import math
        import re

        me_keys = [k for k in state_dict if k.startswith("motion_encoder.")]
        if not me_keys:
            return

        # --- Transform weights ---
        for key in me_keys:
            sub = key[len("motion_encoder."):]
            buf = state_dict[key]
            shape = tuple(int(d) for d in buf.shape)
            arr = buf_to_np(buf)

            if sub == "motion_synthesis_weight":
                # QR decompose: [out_dim, motion_dim] → Q [out_dim, motion_dim]
                q, _ = np.linalg.qr((arr + 1e-8).astype(np.float64))
                q = q.astype(np.float32)
                # Store as float32 Buffer (not bfloat16)
                from max.driver import CPU, Buffer
                q_buf = Buffer(q, device=CPU())
                new_key = "motion_encoder.q_matrix"
                state_dict[new_key] = q_buf
                del state_dict[key]
                continue

            if sub.endswith(".weight") and len(shape) == 4:
                # Conv2d weight: [O, I, H, W] → bake scale → RSCF [H, W, I, O]
                o, i, kh, kw = shape
                scale = 1.0 / math.sqrt(i * kh * kw)
                arr = arr * scale
                arr = arr.transpose(2, 3, 1, 0)  # OIHW → HWIO = RSCF
                state_dict[key] = np_to_buf(arr)

            elif sub.endswith(".weight") and len(shape) == 2:
                # Linear weight: [out, in] → bake scale → transpose [in, out]
                out_dim, in_dim = shape
                scale = 1.0 / math.sqrt(in_dim)
                arr = arr * scale
                arr = arr.T  # [out, in] → [in, out]
                state_dict[key] = np_to_buf(arr)

            # Biases and other 1-D tensors: no transform needed.

        # --- Remap indexed keys: res_blocks.{i}. → res_blocks_{i}. etc. ---
        idx_pattern = re.compile(
            r"^(motion_encoder\.)(res_blocks|motion_network)\.(\d+)\."
        )
        for key in list(state_dict.keys()):
            m = idx_pattern.match(key)
            if m:
                prefix, group_name, idx = m.group(1), m.group(2), m.group(3)
                rest = key[m.end():]
                new_key = f"{prefix}{group_name}_{idx}.{rest}"
                state_dict[new_key] = state_dict.pop(key)

        # --- Generate FIR blur filter weights for conv layers that need them ---
        # The blur kernel is [1,3,3,1] outer product, normalized. Not in checkpoint.
        k1d = np.array([1, 3, 3, 1], dtype=np.float32)
        k2d = np.outer(k1d, k1d)
        k2d = k2d / k2d.sum()

        # Identify blur convs by looking for conv2 and conv_skip in res_blocks.
        # Channel sizes for the encoder:
        channels = _MOTION_ENCODER_CHANNEL_SIZES
        size = 512
        log_size = int(math.log(size, 2))
        in_ch = channels[size]  # 32
        for block_idx in range(log_size - 2):  # 7 blocks
            out_ch = channels[2 ** (log_size - 1 - block_idx)]
            # conv2.blur_filter: depthwise on in_channels
            blur_arr = np.tile(k2d[:, :, np.newaxis, np.newaxis], [1, 1, 1, in_ch])
            prefix = f"motion_encoder.res_blocks_{block_idx}"
            state_dict[f"{prefix}.conv2.blur_filter"] = np_to_buf(blur_arr)
            # conv_skip.blur_filter: depthwise on in_channels (same)
            state_dict[f"{prefix}.conv_skip.blur_filter"] = np_to_buf(blur_arr)
            in_ch = out_ch

    def _split_animate_state_dict(
        self, state_dict: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Split state dict into pre, blocks, post, face_adapters, face_encoder, motion_encoder."""
        pre_weights: dict[str, Any] = {}
        post_weights: dict[str, Any] = {}
        face_encoder_weights: dict[str, Any] = {}
        motion_encoder_weights: dict[str, Any] = {}
        block_weights_list: list[dict[str, Any]] = [
            {} for _ in range(self.config.num_layers)
        ]
        num_adapters = self.config.num_layers // self.config.inject_face_latents_blocks
        face_adapter_weights_list: list[dict[str, Any]] = [
            {} for _ in range(num_adapters)
        ]

        for key, value in state_dict.items():
            if key.startswith("patch_embedding.") or key.startswith(
                "condition_embedder."
            ) or key.startswith("pose_patch_embedding."):
                pre_weights[key] = value
            elif key.startswith("blocks."):
                rest = key[len("blocks."):]
                dot = rest.index(".")
                block_idx = int(rest[:dot])
                sub_key = rest[dot + 1:]
                block_weights_list[block_idx][sub_key] = value
            elif key.startswith("face_adapter."):
                rest = key[len("face_adapter."):]
                dot = rest.index(".")
                adapter_idx = int(rest[:dot])
                sub_key = rest[dot + 1:]
                face_adapter_weights_list[adapter_idx][sub_key] = value
            elif key.startswith("face_encoder."):
                sub_key = key[len("face_encoder."):]
                face_encoder_weights[sub_key] = value
            elif key.startswith("motion_encoder."):
                sub_key = key[len("motion_encoder."):]
                motion_encoder_weights[sub_key] = value
            else:
                # proj_out, scale_shift_table -> post weights
                post_weights[key] = value

        return (
            pre_weights,
            block_weights_list,
            post_weights,
            face_adapter_weights_list,
            face_encoder_weights,
            motion_encoder_weights,
        )

    def load_model(self) -> Callable[..., Any]:
        """Compile the animate transformer as separate graphs."""
        with self._load_lock:
            if self.model is not None:
                return self.__call__

            state_dict = self._ensure_state_dict()
            dim = self.config.num_attention_heads * self.config.attention_head_dim
            dtype = self.config.dtype
            dev = self.config.device
            dev_ref = DeviceRef.from_device(dev)
            inject_interval = self.config.inject_face_latents_blocks

            (
                pre_weights,
                block_weights_list,
                post_weights,
                face_adapter_weights_list,
                face_encoder_weights,
                motion_encoder_weights,
            ) = self._split_animate_state_dict(state_dict)

            # --- Compile animate pre-processing ---
            pre_input_types = [
                # hidden_states [B, 36, T+1, H, W]
                TensorType(
                    dtype,
                    ["batch", self.config.in_channels, "frames", "height", "width"],
                    device=dev,
                ),
                # timestep [B]
                TensorType(DType.float32, ["batch"], device=dev),
                # encoder_hidden_states [B, seq_text, 4096]
                TensorType(
                    dtype, ["batch", "seq_text", self.config.text_dim], device=dev,
                ),
                # clip_features [B, 257, image_dim]
                TensorType(
                    dtype,
                    ["batch", "clip_seq", self.config.image_dim or 1280],
                    device=dev,
                ),
                # pose_hidden_states [B, 16, T_pose, H, W]
                TensorType(
                    dtype,
                    ["batch", self.config.latent_channels, "pose_frames", "height", "width"],
                    device=dev,
                ),
            ]
            pre_module = WanAnimatePreProcess(
                self.config, dtype=dtype, device=dev_ref
            )
            pre_module.load_state_dict(
                pre_weights, weight_alignment=1, strict=True
            )
            with Graph("wan_animate_pre", input_types=pre_input_types) as pre_graph:
                outs = pre_module(*(v.tensor for v in pre_graph.inputs))
                pre_graph.output(*outs)
            pre_model = self.session.load(
                pre_graph, weights_registry=pre_module.state_dict()
            )

            # --- Compile block graph (6 inputs: hs, text, timestep, cos, sin, image) ---
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
                # image_embeds [B, clip_seq, dim]
                TensorType(dtype, ["batch", "clip_seq", dim], device=dev),
            ]

            block_template = WanTransformerBlock(
                dim=dim,
                ffn_dim=self.config.ffn_dim,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.attention_head_dim,
                text_dim=dim,
                cross_attn_norm=self.config.cross_attn_norm,
                eps=self.config.eps,
                added_kv_proj_dim=self.config.added_kv_proj_dim,
                dtype=dtype,
                device=dev_ref,
            )

            block_template.load_state_dict(
                block_weights_list[0],
                weight_alignment=1, strict=True,
            )
            with Graph(
                "wan_animate_block", input_types=block_input_types
            ) as block_graph:
                block_out = block_template(
                    *(v.tensor for v in block_graph.inputs)
                )
                block_graph.output(block_out)
            block_0_model = self.session.load(
                block_graph, weights_registry=block_template.state_dict()
            )
            block_models: list[Model] = [block_0_model]

            for i in range(1, self.config.num_layers):
                block_template.load_state_dict(
                    block_weights_list[i],
                    weight_alignment=1, strict=True,
                )
                block_i_model = self.session.load(
                    block_graph, weights_registry=block_template.state_dict()
                )
                block_models.append(block_i_model)

            # --- Compile face adapter graphs ---
            face_adapter_input_types = [
                # hidden_states [B, S, D]
                TensorType(dtype, ["batch", "seq_len", dim], device=dev),
                # face_emb [B, T_face, 5, D]
                TensorType(dtype, ["batch", "t_face", 5, dim], device=dev),
                # num_temporal_frames (scalar)
                TensorType(DType.int32, [1], device=dev),
            ]

            face_adapter_template = WanAnimateFaceBlock(
                dim=dim,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.attention_head_dim,
                eps=self.config.eps,
                dtype=dtype,
                device=dev_ref,
            )

            # Compile first adapter, reuse graph for others
            face_adapter_template.load_state_dict(
                face_adapter_weights_list[0], weight_alignment=1, strict=True
            )
            with Graph(
                "wan_face_adapter", input_types=face_adapter_input_types
            ) as adapter_graph:
                hs_in = adapter_graph.inputs[0].tensor
                face_in = adapter_graph.inputs[1].tensor
                n_frames = adapter_graph.inputs[2].tensor
                adapter_out = face_adapter_template(hs_in, face_in, n_frames)
                # Rebind adapter_out to match hs_in shape (seq_len equivalence)
                adapter_out = ops.rebind(adapter_out, shape=[
                    hs_in.shape[0], hs_in.shape[1], hs_in.shape[2],
                ])
                # Output hs + adapter residual
                adapter_graph.output(hs_in + adapter_out)
            adapter_0_model = self.session.load(
                adapter_graph,
                weights_registry=face_adapter_template.state_dict(),
            )
            face_adapter_models: list[Model] = [adapter_0_model]

            for i in range(1, len(face_adapter_weights_list)):
                face_adapter_template.load_state_dict(
                    face_adapter_weights_list[i],
                    weight_alignment=1,
                    strict=True,
                )
                adapter_i = self.session.load(
                    adapter_graph,
                    weights_registry=face_adapter_template.state_dict(),
                )
                face_adapter_models.append(adapter_i)

            # --- Compile post-processing (same as base) ---
            post_input_types = [
                TensorType(dtype, ["batch", "seq_len", dim], device=dev),
                TensorType(dtype, ["batch", dim], device=dev),
                TensorType(DType.int8, ["ppf", "pph", "ppw"], device=dev),
            ]
            post_module = WanTransformerPostProcess(
                self.config, dtype=dtype, device=dev_ref
            )
            post_module.load_state_dict(
                post_weights, weight_alignment=1, strict=True
            )
            with Graph(
                "wan_animate_post", input_types=post_input_types
            ) as post_graph:
                post_out = post_module(*(v.tensor for v in post_graph.inputs))
                post_graph.output(post_out)
            post_model = self.session.load(
                post_graph, weights_registry=post_module.state_dict()
            )

            # --- Compile face encoder graph ---
            face_enc_dim = self.config.motion_encoder_dim  # 512
            face_enc_out_dim = dim  # 5120
            face_enc_hidden = self.config.face_encoder_hidden_dim  # 1024
            face_enc_heads = self.config.face_encoder_num_heads  # 4

            face_enc_input_types = [
                # motion_vectors [B, T, 512]
                TensorType(dtype, ["batch", "t_motion", face_enc_dim], device=dev),
            ]
            face_enc_module = WanAnimateFaceEncoder(
                in_dim=face_enc_dim,
                hidden_dim=face_enc_hidden,
                out_dim=face_enc_out_dim,
                num_heads=face_enc_heads,
                dtype=dtype,
                device=dev_ref,
            )
            face_enc_module.load_state_dict(
                face_encoder_weights, weight_alignment=1, strict=True
            )
            with Graph(
                "wan_face_encoder", input_types=face_enc_input_types
            ) as face_enc_graph:
                face_out = face_enc_module(face_enc_graph.inputs[0].tensor)
                face_enc_graph.output(face_out)
            self._face_encoder_model = self.session.load(
                face_enc_graph, weights_registry=face_enc_module.state_dict()
            )

            # --- Compile motion encoder graph ---
            me_size = self.config.motion_encoder_size  # 512
            me_style_dim = self.config.motion_style_dim  # 512
            me_motion_dim = self.config.motion_dim  # 20
            me_blocks = 5  # fixed in diffusers

            me_input_types = [
                # face_image [B, 3, size, size] in [-1, 1]
                TensorType(dtype, ["batch", 3, me_size, me_size], device=dev),
            ]
            me_module = WanAnimateMotionEncoder(
                size=me_size,
                style_dim=me_style_dim,
                motion_dim=me_motion_dim,
                out_dim=me_style_dim,
                motion_blocks=me_blocks,
                dtype=dtype,
                device=dev_ref,
            )
            me_module.load_state_dict(
                motion_encoder_weights, weight_alignment=1, strict=True,
            )
            with Graph(
                "wan_motion_encoder", input_types=me_input_types
            ) as me_graph:
                me_out = me_module(me_graph.inputs[0].tensor)
                me_graph.output(me_out)
            self._motion_encoder_model = self.session.load(
                me_graph, weights_registry=me_module.state_dict()
            )

            self.model = _AnimateBlockLevelModel(
                pre_model,
                block_models,
                face_adapter_models,
                post_model,
                inject_interval=inject_interval,
            )
            return self.__call__

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

    def encode_face(self, motion_vectors: Buffer) -> Buffer:
        """Run face encoder graph: motion_vectors -> face embeddings."""
        if self._face_encoder_model is None:
            raise RuntimeError("Face encoder model not compiled.")
        return self._face_encoder_model.execute(motion_vectors)[0]

    def encode_motion(self, face_pixels: Buffer) -> Buffer:
        """Run motion encoder graph: face_pixels [B, 3, size, size] -> [B, 512]."""
        if self._motion_encoder_model is None:
            raise RuntimeError("Motion encoder model not compiled.")
        return self._motion_encoder_model.execute(face_pixels)[0]

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        clip_features: Buffer,
        pose_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        face_emb: Buffer,
        num_temporal_frames: Buffer,
    ) -> Buffer:
        if self.model is None:
            self.load_model()
        if self.model is None:
            raise RuntimeError("Wan animate transformer model failed to load.")
        return self.model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            clip_features,
            pose_hidden_states,
            rope_cos,
            rope_sin,
            spatial_shape,
            face_emb,
            num_temporal_frames,
        )
