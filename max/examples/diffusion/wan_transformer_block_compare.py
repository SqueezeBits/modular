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

import argparse
import gc
import html
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
from max.driver import CPU, Buffer, Device, DeviceSpec, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight
from max.graph.weights import load_weights
from max.nn.layer import Module
from max.pipelines import MAXModelConfig, PipelineConfig
from max.pipelines.architectures.wan.model import WanTransformerModel
from max.pipelines.architectures.wan.model_config import WanConfig
from max.pipelines.architectures.wan.wan_transformer import (
    WanCrossAttention,
    WanFeedForward,
    WanLayerNorm,
    WanSelfAttention,
)
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.pipeline_variants.utils import get_weight_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep block-level Wan transformer comparison."
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="low quality")
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "outputs" / timestamp


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return _default_output_dir()


def _np_float32(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32).cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _max_tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    cpu_tensor = tensor.cast(DType.float32).to(CPU())
    return np.from_dlpack(cpu_tensor)


def _max_buffer_to_numpy(buffer: Buffer) -> np.ndarray:
    return _max_tensor_to_numpy(Tensor.from_dlpack(buffer))


def _l2_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32)
    act = np.asarray(actual, dtype=np.float32)
    diff = act - ref
    ref_norm = float(np.linalg.norm(ref.reshape(-1)))
    diff_norm = float(np.linalg.norm(diff.reshape(-1)))
    denom = max(ref_norm, 1e-12)
    return {
        "shape": list(ref.shape),
        "l2_norm": diff_norm,
        "relative_l2_norm": diff_norm / denom,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
    }


def _basic_clean(text: str) -> str:
    return html.unescape(html.unescape(text)).strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _diffusers_prompt_batch(prompt: str) -> list[str]:
    return [_whitespace_clean(_basic_clean(prompt))]


def _prepare_timestep_tensor_diffusers(
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    expand_timesteps: bool,
) -> torch.Tensor:
    if expand_timesteps:
        mask = torch.ones(latents.shape, dtype=torch.float32, device=latents.device)
        temp_ts = (mask[0][0][:, ::2, ::2] * timestep).flatten()
        return temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
    return timestep.expand(latents.shape[0])


def _run_diffusers_reference(args: argparse.Namespace) -> dict[str, Any]:
    module_path = Path(__file__).with_name("wan_compare_against_diffusers.py")
    spec = importlib.util.spec_from_file_location(
        "wan_compare_against_diffusers", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._run_diffusers_reference(args)


def _max_stage_component(
    args: argparse.Namespace,
    component_name: str,
    *,
    eager_load: bool = True,
) -> WanTransformerModel:
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=args.model,
            device_specs=[DeviceSpec.accelerator()],
        )
    )
    devices = load_devices(config.model.device_specs)
    session = InferenceSession(devices=devices)
    config.configure_session(session)
    component_cfg = (
        (config.model.diffusers_config or {})
        .get("components", {})
        .get(component_name, {})
        .get("config_dict", {})
    )
    weight_paths = get_weight_paths(config.model)
    relative_paths = [
        str(path)
        for path in config.model.weight_path
        if str(path).split("/")[0] == component_name
    ]
    abs_paths = [
        abs_path
        for abs_path in weight_paths
        for rel_path in relative_paths
        if rel_path in str(abs_path)
    ]
    return WanTransformerModel(
        config=component_cfg,
        encoding=cast(
            SupportedEncoding, config.model.quantization_encoding
        ),
        devices=devices,
        weights=load_weights(abs_paths),
        session=session,
        eager_load=eager_load,
    )


def _max_timestep_buffer(
    timestep_value: float,
    *,
    batch_size: int,
    device: Device,
) -> Buffer:
    return Buffer.from_numpy(
        np.full([batch_size], timestep_value, dtype=np.float32)
    ).to(device)


def _diffusers_transformer_inputs(
    model: Any,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep_value: float,
    *,
    expand_timesteps: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
]:
    timestep = torch.tensor([timestep_value], device=latents.device, dtype=torch.float32)
    timestep_tensor = _prepare_timestep_tensor_diffusers(
        latents, timestep, expand_timesteps=expand_timesteps
    )
    latents = latents.to(torch.bfloat16)
    rotary_emb = model.rope(latents)
    hidden_states = model.patch_embedding(latents)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    if timestep_tensor.ndim == 2:
        ts_seq_len = timestep_tensor.shape[1]
        timestep_flat = timestep_tensor.flatten()
    else:
        ts_seq_len = None
        timestep_flat = timestep_tensor

    temb, timestep_proj, text_emb, _ = model.condition_embedder(
        timestep_flat, prompt_embeds, None, timestep_seq_len=ts_seq_len
    )
    if ts_seq_len is not None:
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    return hidden_states, temb, timestep_proj, text_emb, rotary_emb


def _max_transformer_inputs(
    model: WanTransformerModel,
    latents: np.ndarray,
    prompt_embeds: np.ndarray,
    timestep_value: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.load_model()
    assert model.model is not None
    device = model.devices[0]
    latents_buf = (
        Tensor.from_dlpack(np.ascontiguousarray(latents, dtype=np.float32))
        .cast(model.config.dtype)
        .to(device)
        .driver_tensor
    )
    prompt_buf = (
        Tensor.from_dlpack(np.ascontiguousarray(prompt_embeds, dtype=np.float32))
        .cast(model.config.dtype)
        .to(device)
        .driver_tensor
    )
    timestep_buf = _max_timestep_buffer(
        timestep_value, batch_size=int(latents.shape[0]), device=device
    )
    pre_out = model.model.pre.execute(latents_buf, timestep_buf, prompt_buf)
    hs = _max_tensor_to_numpy(Tensor.from_dlpack(pre_out[0]))
    temb = _max_tensor_to_numpy(Tensor.from_dlpack(pre_out[1]))
    timestep_proj = _max_tensor_to_numpy(Tensor.from_dlpack(pre_out[2]))
    text_emb = _max_tensor_to_numpy(Tensor.from_dlpack(pre_out[3]))
    rope_cos, rope_sin = model.compute_rope(
        num_frames=int(latents.shape[2]),
        height=int(latents.shape[3]),
        width=int(latents.shape[4]),
    )
    return (
        hs,
        temb,
        timestep_proj,
        text_emb,
        _max_buffer_to_numpy(rope_cos),
        _max_buffer_to_numpy(rope_sin),
    )


class DebugWanTransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        text_dim: int,
        cross_attn_norm: bool,
        eps: float,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.scale_shift_table = Weight("scale_shift_table", dtype, [1, 6, dim], device)
        self.norm1 = WanLayerNorm(
            dim, eps=eps, elementwise_affine=False, dtype=dtype, device=device
        )
        self.attn1 = WanSelfAttention(
            dim, num_heads, head_dim, eps, dtype=dtype, device=device
        )
        self.norm2 = WanLayerNorm(
            dim,
            eps=eps,
            elementwise_affine=cross_attn_norm,
            use_bias=cross_attn_norm,
            dtype=dtype,
            device=device,
        )
        self.attn2 = WanCrossAttention(
            dim, text_dim, num_heads, head_dim, eps, dtype=dtype, device=device
        )
        self.norm3 = WanLayerNorm(
            dim, eps=eps, elementwise_affine=False, dtype=dtype, device=device
        )
        self.ffn = WanFeedForward(dim, ffn_dim, dtype=dtype, device=device)

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep_proj: TensorValue,
        rope_cos: TensorValue,
        rope_sin: TensorValue,
    ) -> tuple[TensorValue, ...]:
        rotary_emb = (rope_cos, rope_sin)
        mod = self.scale_shift_table + timestep_proj
        shift_sa = mod[:, 0:1, :]
        scale_sa = mod[:, 1:2, :]
        gate_sa = mod[:, 2:3, :]
        shift_ff = mod[:, 3:4, :]
        scale_ff = mod[:, 4:5, :]
        gate_ff = mod[:, 5:6, :]

        norm1_out = self.norm1(hidden_states)
        sa_input = norm1_out
        sa_input = sa_input * (1 + scale_sa) + shift_sa
        sa_out = self.attn1(sa_input, rotary_emb)
        after_sa = hidden_states + gate_sa * sa_out

        norm2_out = self.norm2(after_sa)
        ca_input = norm2_out
        ca_out = self.attn2(ca_input, encoder_hidden_states)
        after_ca = after_sa + ca_out

        norm3_out = self.norm3(after_ca)
        ff_input = norm3_out
        ff_input = ff_input * (1 + scale_ff) + shift_ff
        ff_out = self.ffn(ff_input)
        out = after_ca + gate_ff * ff_out

        return (
            norm1_out,
            shift_sa,
            scale_sa,
            gate_sa,
            sa_input,
            sa_out,
            after_sa,
            norm2_out,
            ca_input,
            ca_out,
            after_ca,
            norm3_out,
            shift_ff,
            scale_ff,
            gate_ff,
            ff_input,
            ff_out,
            out,
        )


def _extract_block_weights(
    state_dict: dict[str, object], block_idx: int
) -> dict[str, object]:
    prefix = f"blocks.{block_idx}."
    return {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _compile_debug_block(
    model: WanTransformerModel, block_idx: int
) -> Any:
    model.load_model()
    assert model._state_dict is not None
    block_weights = _extract_block_weights(model._state_dict, block_idx)
    cfg = model.config
    dim = cfg.num_attention_heads * cfg.attention_head_dim
    dev = cfg.device
    device_ref = DeviceRef.from_device(dev)
    debug_block = DebugWanTransformerBlock(
        dim=dim,
        ffn_dim=cfg.ffn_dim,
        num_heads=cfg.num_attention_heads,
        head_dim=cfg.attention_head_dim,
        text_dim=dim,
        cross_attn_norm=cfg.cross_attn_norm,
        eps=cfg.eps,
        dtype=cfg.dtype,
        device=device_ref,
    )
    debug_block.load_state_dict(
        cast(dict[str, Any], block_weights), weight_alignment=1, strict=True
    )
    input_types = [
        TensorType(cfg.dtype, ["batch", "seq_len", dim], device=dev),
        TensorType(cfg.dtype, ["batch", "seq_text", dim], device=dev),
        TensorType(cfg.dtype, ["batch", 6, dim], device=dev),
        TensorType(DType.float32, ["seq_len", cfg.attention_head_dim], device=dev),
        TensorType(DType.float32, ["seq_len", cfg.attention_head_dim], device=dev),
    ]
    with Graph("wan_debug_block", input_types=input_types) as graph:
        outputs = debug_block(*(value.tensor for value in graph.inputs))
        graph.output(*outputs)
    return model.session.load(graph, weights_registry=debug_block.state_dict())


def _extract_prefixed_weights(
    state_dict: dict[str, object], prefix: str
) -> dict[str, object]:
    return {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


class DebugWanSelfAttention(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.inner_dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn = WanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        rope_cos: TensorValue,
        rope_sin: TensorValue,
    ) -> tuple[TensorValue, ...]:
        rotary_emb = (rope_cos, rope_sin)
        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(hidden_states)
        value = self.attn.to_v(hidden_states)
        query_norm = self.attn.norm_q(query)
        key_norm = self.attn.norm_k(key)

        batch_size = query.shape[0]
        seq_len = query.shape[1]
        query_heads = ops.reshape(
            query_norm, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        key_heads = ops.reshape(
            key_norm, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        value_heads = ops.reshape(
            value, [batch_size, seq_len, self.num_heads, self.head_dim]
        )

        query_rope = apply_rotary_emb(
            query_heads,
            rotary_emb,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        key_rope = apply_rotary_emb(
            key_heads,
            rotary_emb,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        scale = 1.0 / (self.head_dim**0.5)
        attn_heads = flash_attention_gpu(
            query_rope,
            key_rope,
            value_heads,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )
        attn_flat = ops.reshape(
            attn_heads, [attn_heads.shape[0], attn_heads.shape[1], self.inner_dim]
        )
        out = self.attn.to_out(attn_flat)
        query_rope_flat = ops.reshape(
            query_rope, [batch_size, seq_len, self.inner_dim]
        )
        key_rope_flat = ops.reshape(
            key_rope, [batch_size, seq_len, self.inner_dim]
        )
        return (
            query,
            key,
            value,
            query_norm,
            key_norm,
            query_rope_flat,
            key_rope_flat,
            attn_flat,
            out,
        )


class DebugWanCrossAttention(Module):
    def __init__(
        self,
        *,
        dim: int,
        text_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.inner_dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn = WanCrossAttention(
            dim=dim,
            text_dim=text_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
    ) -> tuple[TensorValue, ...]:
        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(encoder_hidden_states)
        value = self.attn.to_v(encoder_hidden_states)
        query_norm = self.attn.norm_q(query)
        key_norm = self.attn.norm_k(key)

        batch_size = query.shape[0]
        q_seq_len = query.shape[1]
        kv_seq_len = key.shape[1]
        query_heads = ops.reshape(
            query_norm, [batch_size, q_seq_len, self.num_heads, self.head_dim]
        )
        key_heads = ops.reshape(
            key_norm, [batch_size, kv_seq_len, self.num_heads, self.head_dim]
        )
        value_heads = ops.reshape(
            value, [batch_size, kv_seq_len, self.num_heads, self.head_dim]
        )
        scale = 1.0 / (self.head_dim**0.5)
        attn_heads = flash_attention_gpu(
            query_heads,
            key_heads,
            value_heads,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )
        attn_flat = ops.reshape(
            attn_heads, [attn_heads.shape[0], attn_heads.shape[1], self.inner_dim]
        )
        out = self.attn.to_out(attn_flat)
        return (
            query,
            key,
            value,
            query_norm,
            key_norm,
            attn_flat,
            out,
        )


def _compile_debug_attention_modules(
    model: WanTransformerModel, block_idx: int
) -> tuple[Any, Any]:
    model.load_model()
    assert model._state_dict is not None
    block_weights = _extract_block_weights(model._state_dict, block_idx)
    cfg = model.config
    dim = cfg.num_attention_heads * cfg.attention_head_dim
    dev = cfg.device
    device_ref = DeviceRef.from_device(dev)

    self_attn = DebugWanSelfAttention(
        dim=dim,
        num_heads=cfg.num_attention_heads,
        head_dim=cfg.attention_head_dim,
        eps=cfg.eps,
        dtype=cfg.dtype,
        device=device_ref,
    )
    self_attn.load_state_dict(
        cast(
            dict[str, Any], _extract_prefixed_weights(block_weights, "attn1.")
        ),
        weight_alignment=1,
        strict=True,
    )
    self_input_types = [
        TensorType(cfg.dtype, ["batch", "seq_len", dim], device=dev),
        TensorType(DType.float32, ["seq_len", cfg.attention_head_dim], device=dev),
        TensorType(DType.float32, ["seq_len", cfg.attention_head_dim], device=dev),
    ]
    with Graph("wan_debug_self_attn", input_types=self_input_types) as graph:
        outputs = self_attn(*(value.tensor for value in graph.inputs))
        graph.output(*outputs)
    self_model = model.session.load(
        graph, weights_registry=self_attn.state_dict()
    )

    cross_attn = DebugWanCrossAttention(
        dim=dim,
        text_dim=dim,
        num_heads=cfg.num_attention_heads,
        head_dim=cfg.attention_head_dim,
        eps=cfg.eps,
        dtype=cfg.dtype,
        device=device_ref,
    )
    cross_attn.load_state_dict(
        cast(
            dict[str, Any], _extract_prefixed_weights(block_weights, "attn2.")
        ),
        weight_alignment=1,
        strict=True,
    )
    cross_input_types = [
        TensorType(cfg.dtype, ["batch", "seq_len", dim], device=dev),
        TensorType(cfg.dtype, ["batch", "seq_text", dim], device=dev),
    ]
    with Graph("wan_debug_cross_attn", input_types=cross_input_types) as graph:
        outputs = cross_attn(*(value.tensor for value in graph.inputs))
        graph.output(*outputs)
    cross_model = model.session.load(
        graph, weights_registry=cross_attn.state_dict()
    )
    return self_model, cross_model


def _diffusers_block_debug(
    block: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, np.ndarray]:
    captures: dict[str, torch.Tensor] = {}

    def _pre_hook(name: str):
        def hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
            captures[name] = inputs[0].detach().to(torch.float32).cpu()

        return hook

    def _post_hook(name: str):
        def hook(
            _module: Any, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            captures[name] = output.detach().to(torch.float32).cpu()

        return hook

    hooks = [
        block.norm1.register_forward_pre_hook(_pre_hook("norm1_input")),
        block.norm1.register_forward_hook(_post_hook("norm1_out")),
        block.attn1.register_forward_pre_hook(_pre_hook("sa_input")),
        block.attn1.register_forward_hook(_post_hook("sa_out")),
        block.norm2.register_forward_pre_hook(_pre_hook("norm2_input")),
        block.norm2.register_forward_hook(_post_hook("norm2_out")),
        block.attn2.register_forward_pre_hook(_pre_hook("ca_input")),
        block.attn2.register_forward_hook(_post_hook("ca_out")),
        block.norm3.register_forward_pre_hook(_pre_hook("norm3_input")),
        block.norm3.register_forward_hook(_post_hook("norm3_out")),
        block.ffn.register_forward_pre_hook(_pre_hook("ff_input")),
        block.ffn.register_forward_hook(_post_hook("ff_out")),
    ]

    if temb.ndim == 4:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            block.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            block.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    try:
        out = block(hidden_states, encoder_hidden_states, temb, rotary_emb)
    finally:
        for hook in hooks:
            hook.remove()

    return {
        "norm1_in": _np_float32(captures["norm1_input"]),
        "norm1_out": _np_float32(captures["norm1_out"]),
        "shift_sa": _np_float32(shift_msa),
        "scale_sa": _np_float32(scale_msa),
        "gate_sa": _np_float32(gate_msa),
        "sa_input": _np_float32(captures["sa_input"]),
        "sa_out": _np_float32(captures["sa_out"]),
        "after_sa": _np_float32(captures["norm2_input"]),
        "norm2_out": _np_float32(captures["norm2_out"]),
        "ca_input": _np_float32(captures["ca_input"]),
        "ca_out": _np_float32(captures["ca_out"]),
        "after_ca": _np_float32(captures["norm3_input"]),
        "norm3_out": _np_float32(captures["norm3_out"]),
        "shift_ff": _np_float32(c_shift_msa),
        "scale_ff": _np_float32(c_scale_msa),
        "gate_ff": _np_float32(c_gate_msa),
        "ff_input": _np_float32(captures["ff_input"]),
        "ff_out": _np_float32(captures["ff_out"]),
        "out": _np_float32(out),
    }


def _run_stage_analysis(
    args: argparse.Namespace,
    *,
    component_name: str,
    step_idx: int,
    timestep_value: float,
    latents_in: np.ndarray,
    prompt_embeds: np.ndarray,
) -> dict[str, Any]:
    stage_name = "high_noise" if component_name == "transformer" else "low_noise"
    device = torch.device("cuda")

    diff_model = WanTransformer3DModel.from_pretrained(
        args.model, subfolder=component_name, torch_dtype=torch.bfloat16
    ).to(device)
    diff_model.eval()

    latents_t = torch.from_numpy(np.ascontiguousarray(latents_in)).to(
        device=device, dtype=torch.float32
    )
    prompt_embeds_t = torch.from_numpy(np.ascontiguousarray(prompt_embeds)).to(
        device=device, dtype=torch.bfloat16
    )
    diff_hs, diff_temb, diff_timestep_proj, diff_text_emb, diff_rope = _diffusers_transformer_inputs(
        diff_model,
        latents_t,
        prompt_embeds_t,
        timestep_value,
        expand_timesteps=False,
    )

    diff_block_outputs: list[np.ndarray] = []
    diff_block_inputs: list[np.ndarray] = []
    hs = diff_hs
    for block in diff_model.blocks:
        diff_block_inputs.append(_np_float32(hs))
        hs = block(hs, diff_text_emb, diff_timestep_proj, diff_rope)
        diff_block_outputs.append(_np_float32(hs))

    del diff_model
    gc.collect()
    torch.cuda.empty_cache()

    max_model = _max_stage_component(args, component_name, eager_load=True)
    max_hs, _max_temb, max_timestep_proj, max_text_emb, max_rope_cos, max_rope_sin = (
        _max_transformer_inputs(max_model, latents_in, prompt_embeds, timestep_value)
    )
    assert max_model.model is not None
    max_block_outputs: list[np.ndarray] = []
    max_block_inputs: list[np.ndarray] = []
    hs_np = max_hs
    text_emb_buf = (
        Tensor.from_dlpack(np.ascontiguousarray(max_text_emb, dtype=np.float32))
        .cast(max_model.config.dtype)
        .to(max_model.devices[0])
        .driver_tensor
    )
    rope_cos_buf = (
        Tensor.from_dlpack(np.ascontiguousarray(max_rope_cos, dtype=np.float32))
        .to(max_model.devices[0])
        .driver_tensor
    )
    rope_sin_buf = (
        Tensor.from_dlpack(np.ascontiguousarray(max_rope_sin, dtype=np.float32))
        .to(max_model.devices[0])
        .driver_tensor
    )
    for block_model in max_model.model.blocks:
        max_block_inputs.append(np.asarray(hs_np, dtype=np.float32))
        hs_buf = (
            Tensor.from_dlpack(np.ascontiguousarray(hs_np, dtype=np.float32))
            .cast(max_model.config.dtype)
            .to(max_model.devices[0])
            .driver_tensor
        )
        timestep_proj_buf = (
            Tensor.from_dlpack(np.ascontiguousarray(max_timestep_proj, dtype=np.float32))
            .cast(max_model.config.dtype)
            .to(max_model.devices[0])
            .driver_tensor
        )
        out_buf = block_model.execute(
            hs_buf,
            text_emb_buf,
            timestep_proj_buf,
            rope_cos_buf,
            rope_sin_buf,
        )[0]
        hs_np = _max_tensor_to_numpy(Tensor.from_dlpack(out_buf))
        max_block_outputs.append(np.asarray(hs_np, dtype=np.float32))

    block_metrics = [
        _l2_metrics(diff_block_outputs[i], max_block_outputs[i])
        for i in range(len(diff_block_outputs))
    ]
    first_large_block = next(
        (
            i
            for i, metric in enumerate(block_metrics)
            if metric["relative_l2_norm"] > 0.02
        ),
        0,
    )

    debug_model = _compile_debug_block(max_model, first_large_block)
    max_block_input_buf = (
        Tensor.from_dlpack(np.ascontiguousarray(max_block_inputs[first_large_block], dtype=np.float32))
        .cast(max_model.config.dtype)
        .to(max_model.devices[0])
        .driver_tensor
    )
    timestep_proj_buf = (
        Tensor.from_dlpack(np.ascontiguousarray(max_timestep_proj, dtype=np.float32))
        .cast(max_model.config.dtype)
        .to(max_model.devices[0])
        .driver_tensor
    )
    max_debug_outputs = debug_model.execute(
        max_block_input_buf,
        text_emb_buf,
        timestep_proj_buf,
        rope_cos_buf,
        rope_sin_buf,
    )
    max_debug = {
        name: _max_tensor_to_numpy(Tensor.from_dlpack(value))
        for name, value in zip(
            (
                "norm1_out",
                "shift_sa",
                "scale_sa",
                "gate_sa",
                "sa_input",
                "sa_out",
                "after_sa",
                "norm2_out",
                "ca_input",
                "ca_out",
                "after_ca",
                "norm3_out",
                "shift_ff",
                "scale_ff",
                "gate_ff",
                "ff_input",
                "ff_out",
                "out",
            ),
            max_debug_outputs,
            strict=False,
        )
    }
    max_debug["norm1_in"] = np.asarray(
        max_block_inputs[first_large_block], dtype=np.float32
    )

    diff_debug: dict[str, np.ndarray] = {}
    hs = diff_hs
    diff_debug_model = WanTransformer3DModel.from_pretrained(
        args.model, subfolder=component_name, torch_dtype=torch.bfloat16
    ).to(device)
    diff_debug_model.eval()
    for idx, block in enumerate(diff_debug_model.blocks):
        if idx == first_large_block:
            diff_debug = _diffusers_block_debug(
                block, hs, diff_text_emb, diff_timestep_proj, diff_rope
            )
            break
        hs = block(hs, diff_text_emb, diff_timestep_proj, diff_rope)
    del diff_debug_model
    gc.collect()
    torch.cuda.empty_cache()

    detail_metrics = {
        name: _l2_metrics(diff_debug[name], max_debug[name])
        for name in diff_debug
    }

    return {
        "stage": stage_name,
        "step_index": step_idx,
        "pre_metrics": {
            "hidden_states": _l2_metrics(_np_float32(diff_hs), max_hs),
            "timestep_proj": _l2_metrics(
                _np_float32(diff_timestep_proj), max_timestep_proj
            ),
            "text_emb": _l2_metrics(_np_float32(diff_text_emb), max_text_emb),
        },
        "block_metrics": block_metrics,
        "first_large_block": first_large_block,
        "detail_metrics": detail_metrics,
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Wan Transformer Block 심화 비교 리포트",
        "",
        "## 요약",
        f"- high-noise 첫 큰 오차 block: {report['high_noise']['first_large_block']}",
        f"- low-noise 첫 큰 오차 block: {report['low_noise']['first_large_block']}",
        "",
        "## 판단",
        "- pre-process(hidden/timestep_proj)보다 block 내부에서 오차가 더 빨리 커지면 block 내부 연산이 원인이다.",
        "- self-attn / cross-attn / ffn 중 어느 출력이 먼저 커지는지로 위치를 좁힌다.",
        "",
    ]
    for stage_key in ("high_noise", "low_noise"):
        stage = report[stage_key]
        lines.extend(
            [
                f"## {stage_key}",
                f"- 첫 큰 오차 block index: {stage['first_large_block']}",
                f"- pre hidden rel L2: {stage['pre_metrics']['hidden_states']['relative_l2_norm']:.6e}",
                f"- pre timestep_proj rel L2: {stage['pre_metrics']['timestep_proj']['relative_l2_norm']:.6e}",
                "",
                "### block 내부 detail",
            ]
        )
        if "norm1_in" in stage["detail_metrics"]:
            lines.append(
                f"- norm1_in: rel_l2={stage['detail_metrics']['norm1_in']['relative_l2_norm']:.6e}, max_abs={stage['detail_metrics']['norm1_in']['max_abs']:.6f}"
            )
        for name, metrics in stage["detail_metrics"].items():
            if name == "norm1_in":
                continue
            lines.append(
                f"- {name}: rel_l2={metrics['relative_l2_norm']:.6e}, max_abs={metrics['max_abs']:.6f}"
            )
        lines.append("")
    (output_dir / "block_report_ko.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_ref = _run_diffusers_reference(args)
    high_step = 0
    low_step = 2

    report = {
        "settings": {
            "model": args.model,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2,
        },
        "high_noise": _run_stage_analysis(
            args,
            component_name="transformer",
            step_idx=high_step,
            timestep_value=float(base_ref["step_debug"]["step_0"]["timestep"][0]),
            latents_in=np.asarray(base_ref["step_debug"]["step_0"]["latents_in"], dtype=np.float32),
            prompt_embeds=np.asarray(base_ref["prompt_embeds"], dtype=np.float32),
        ),
        "low_noise": _run_stage_analysis(
            args,
            component_name="transformer_2",
            step_idx=low_step,
            timestep_value=float(base_ref["step_debug"]["step_2"]["timestep"][0]),
            latents_in=np.asarray(base_ref["step_debug"]["step_2"]["latents_in"], dtype=np.float32),
            prompt_embeds=np.asarray(base_ref["prompt_embeds"], dtype=np.float32),
        ),
    }

    (output_dir / "block_comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    _write_report(output_dir, report)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
