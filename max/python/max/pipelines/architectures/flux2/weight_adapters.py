from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from max.driver import Buffer
from max.graph.weights import WeightData

Flux2ActivationScheme = Literal["static", "dynamic"]

_DYNAMIC_ADAPTED_FORMAT = "dynamic_block_fp8_v1"
_STATIC_ADAPTED_FORMAT = "legacy_scalar_static_v1"


def _clone_weight(weight: WeightData, new_name: str) -> WeightData:
    return WeightData(
        data=weight.data,
        name=new_name,
        dtype=weight.dtype,
        shape=weight.shape,
        quantization_encoding=weight.quantization_encoding,
    )


def _legacy_fp8_input_scale_weight(
    weight: WeightData, new_name: str
) -> WeightData:
    if not weight.dtype.is_float():
        return _clone_weight(weight, new_name)

    scale_np = np.asarray(
        Buffer.from_dlpack(weight.data).to_numpy(), dtype=np.float32
    )
    inv_scale_np = np.asarray(
        np.reciprocal(scale_np, dtype=np.float32), dtype=np.float32
    )
    return WeightData.from_numpy(inv_scale_np, new_name)


def _slice_rows(
    weight: WeightData, start: int, end: int, new_name: str
) -> WeightData:
    tensor = Buffer.from_dlpack(weight.data)[start:end, :].contiguous()
    return WeightData(
        data=tensor,
        name=new_name,
        dtype=weight.dtype,
        shape=weight.shape.__class__(tensor.shape),
        quantization_encoding=weight.quantization_encoding,
    )


def is_bflabs_flux2_transformer_checkpoint(
    state_dict: Mapping[str, WeightData],
) -> bool:
    return "img_in.weight" in state_dict and "txt_in.weight" in state_dict


def _is_bflabs_flux2_transformer_tensor_checkpoint(
    state_dict,
) -> bool:
    return "img_in.weight" in state_dict and "txt_in.weight" in state_dict


def uses_legacy_scalar_fp8_scales(
    state_dict: Mapping[str, WeightData],
) -> bool:
    return any(
        key.endswith(".weight_scale") and tuple(int(d) for d in value.shape) == ()
        for key, value in state_dict.items()
    )


def _adapted_flux2_transformer_path(
    path: Path, activation_scheme: Flux2ActivationScheme
) -> Path:
    if activation_scheme == "dynamic":
        return path.with_name(f"{path.stem}.max.safetensors")
    return path.with_name(f"{path.stem}.max.static.safetensors")


def adapt_bflabs_flux2_transformer_weights(
    path: Path, *, activation_scheme: Flux2ActivationScheme = "dynamic"
) -> Path:
    """Persist a MAX-native Flux2 transformer checkpoint beside the source file.

    The adapted checkpoint is created once and then reused on subsequent loads.
    Non-BFLabs checkpoints are returned unchanged.
    """
    adapted_path = _adapted_flux2_transformer_path(path, activation_scheme)
    expected_format = (
        _DYNAMIC_ADAPTED_FORMAT
        if activation_scheme == "dynamic"
        else _STATIC_ADAPTED_FORMAT
    )
    if adapted_path.exists():
        from safetensors import safe_open

        with safe_open(str(adapted_path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        if metadata.get("max_flux2_adapted_format") == expected_format:
            return adapted_path

    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(str(path), framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
        metadata = f.metadata()

    if not _is_bflabs_flux2_transformer_tensor_checkpoint(state_dict):
        return path

    if activation_scheme == "dynamic":
        tensors = _convert_safetensor_torch_state_dict_dynamic(state_dict)
    else:
        tensors = _convert_safetensor_torch_state_dict_static(state_dict)

    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = adapted_path.with_name(f"{adapted_path.name}.tmp")
    out_metadata = dict(metadata or {})
    out_metadata["max_flux2_adapted"] = "true"
    out_metadata["max_flux2_adapted_format"] = expected_format
    out_metadata["max_flux2_activation_scheme"] = activation_scheme
    save_file(tensors, str(tmp_path), metadata=out_metadata)
    tmp_path.replace(adapted_path)
    return adapted_path


def _clone_torch_weight(tensor, /):
    return tensor.detach().clone()


def _legacy_fp8_input_scale_tensor(tensor, /):
    import torch

    if not torch.is_floating_point(tensor):
        return _clone_torch_weight(tensor)
    return torch.reciprocal(tensor.to(torch.float32)).contiguous()


def _slice_rows_torch(tensor, start: int, end: int, /):
    return tensor[start:end, :].contiguous()


def _quantize_blockwise_fp8_tensor(
    weight, *, block_n: int = 128, block_k: int = 128
):
    import torch

    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 tensor, got {tuple(weight.shape)}")

    out_dim = int(weight.shape[0])
    in_dim = int(weight.shape[1])
    out_blocks = (out_dim + block_n - 1) // block_n
    in_blocks = (in_dim + block_k - 1) // block_k

    w = weight.to(torch.float32)
    padded_out = out_blocks * block_n
    padded_in = in_blocks * block_k
    if padded_out != out_dim or padded_in != in_dim:
        w = torch.nn.functional.pad(
            w,
            (0, padded_in - in_dim, 0, padded_out - out_dim),
            mode="constant",
            value=0.0,
        )

    w_blocks = w.reshape(out_blocks, block_n, in_blocks, block_k)
    block_absmax = w_blocks.abs().amax(dim=(1, 3))
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scales = torch.clamp(block_absmax / fp8_max, min=1e-8).to(torch.float32)
    w_scaled = w_blocks / scales[:, None, :, None]
    w_q = (
        w_scaled.reshape(padded_out, padded_in)[:out_dim, :in_dim]
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return w_q, scales.contiguous()


def _convert_safetensor_torch_state_dict_dynamic(state_dict):
    import torch

    if not _is_bflabs_flux2_transformer_tensor_checkpoint(state_dict):
        return {name: _clone_torch_weight(tensor) for name, tensor in state_dict.items()}

    converted = {}

    def _convert_weight(old_key: str, tensor):
        scale_key = old_key[: -len(".weight")] + ".weight_scale"
        scale = state_dict.get(scale_key)
        if tensor.dtype == torch.float8_e4m3fn and scale is not None and scale.numel() == 1:
            dequantized = tensor.to(torch.float32) * scale.to(torch.float32)
            return _quantize_blockwise_fp8_tensor(dequantized)
        return _clone_torch_weight(tensor), None

    direct_mappings = {
        "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
        "img_in.weight": "x_embedder.weight",
        "txt_in.weight": "context_embedder.weight",
        "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.linear.weight": "proj_out.weight",
    }
    for old_name, new_name in direct_mappings.items():
        if old_name in state_dict:
            weight, weight_scale = _convert_weight(old_name, state_dict[old_name])
            converted[new_name] = weight
            if weight_scale is not None:
                converted[new_name[: -len(".weight")] + ".weight_scale"] = weight_scale

    for key, tensor in state_dict.items():
        if key.endswith(".input_scale") or key.endswith(".weight_scale"):
            continue

        img_match = re.fullmatch(r"double_blocks\.(\d+)\.img_attn\.qkv\.weight", key)
        if img_match:
            idx = img_match.group(1)
            dim = int(tensor.shape[0]) // 3
            for new_name, start, end in (
                (f"transformer_blocks.{idx}.attn.to_q.weight", 0, dim),
                (f"transformer_blocks.{idx}.attn.to_k.weight", dim, 2 * dim),
                (f"transformer_blocks.{idx}.attn.to_v.weight", 2 * dim, 3 * dim),
            ):
                weight, weight_scale = _convert_weight(
                    key, _slice_rows_torch(tensor, start, end)
                )
                converted[new_name] = weight
                if weight_scale is not None:
                    converted[new_name[: -len(".weight")] + ".weight_scale"] = (
                        weight_scale
                    )
            continue

        txt_match = re.fullmatch(r"double_blocks\.(\d+)\.txt_attn\.qkv\.weight", key)
        if txt_match:
            idx = txt_match.group(1)
            dim = int(tensor.shape[0]) // 3
            for new_name, start, end in (
                (f"transformer_blocks.{idx}.attn.add_q_proj.weight", 0, dim),
                (f"transformer_blocks.{idx}.attn.add_k_proj.weight", dim, 2 * dim),
                (f"transformer_blocks.{idx}.attn.add_v_proj.weight", 2 * dim, 3 * dim),
            ):
                weight, weight_scale = _convert_weight(
                    key, _slice_rows_torch(tensor, start, end)
                )
                converted[new_name] = weight
                if weight_scale is not None:
                    converted[new_name[: -len(".weight")] + ".weight_scale"] = (
                        weight_scale
                    )
            continue

        replacements = (
            (
                r"double_blocks\.(\d+)\.img_attn\.proj\.weight",
                r"transformer_blocks.\1.attn.to_out.0.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.proj\.weight",
                r"transformer_blocks.\1.attn.to_add_out.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.scale",
                r"transformer_blocks.\1.attn.norm_q.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.scale",
                r"transformer_blocks.\1.attn.norm_k.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.scale",
                r"transformer_blocks.\1.attn.norm_added_q.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.scale",
                r"transformer_blocks.\1.attn.norm_added_k.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.0\.weight",
                r"transformer_blocks.\1.ff.linear_in.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.2\.weight",
                r"transformer_blocks.\1.ff.linear_out.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.0\.weight",
                r"transformer_blocks.\1.ff_context.linear_in.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.2\.weight",
                r"transformer_blocks.\1.ff_context.linear_out.weight",
            ),
            (
                r"single_blocks\.(\d+)\.linear1\.weight",
                r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.weight",
            ),
            (
                r"single_blocks\.(\d+)\.linear2\.weight",
                r"single_transformer_blocks.\1.attn.to_out.weight",
            ),
            (
                r"single_blocks\.(\d+)\.norm\.query_norm\.scale",
                r"single_transformer_blocks.\1.attn.norm_q.weight",
            ),
            (
                r"single_blocks\.(\d+)\.norm\.key_norm\.scale",
                r"single_transformer_blocks.\1.attn.norm_k.weight",
            ),
        )
        for pattern, replacement in replacements:
            mapped = re.sub(pattern, replacement, key)
            if mapped != key:
                if mapped.endswith(".weight"):
                    weight, weight_scale = _convert_weight(key, tensor)
                    converted[mapped] = weight
                    if weight_scale is not None:
                        converted[mapped[: -len(".weight")] + ".weight_scale"] = (
                            weight_scale
                        )
                else:
                    converted[mapped] = _clone_torch_weight(tensor)
                break

    _validate_required_flux2_prefixes(converted)
    return converted


def _validate_required_flux2_prefixes(
    converted: Mapping[str, object],
) -> None:
    required_prefixes = (
        "time_guidance_embed.timestep_embedder.",
        "x_embedder.",
        "context_embedder.",
        "double_stream_modulation_img.",
        "double_stream_modulation_txt.",
        "single_stream_modulation.",
        "transformer_blocks.0.attn.",
        "single_transformer_blocks.0.attn.",
        "norm_out.",
        "proj_out.",
    )
    for prefix in required_prefixes:
        if not any(name.startswith(prefix) for name in converted):
            raise ValueError(
                f"Missing required flux2 transformer weights with prefix '{prefix}'"
            )

def _convert_safetensor_torch_state_dict_static(state_dict):
    if not _is_bflabs_flux2_transformer_tensor_checkpoint(state_dict):
        return {
            name: _clone_torch_weight(tensor) for name, tensor in state_dict.items()
        }

    converted = {}

    direct_mappings = {
        "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
        "img_in.weight": "x_embedder.weight",
        "txt_in.weight": "context_embedder.weight",
        "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.linear.weight": "proj_out.weight",
    }
    for old_name, new_name in direct_mappings.items():
        if old_name in state_dict:
            converted[new_name] = _clone_torch_weight(state_dict[old_name])

    for key, tensor in state_dict.items():
        img_match = re.fullmatch(r"double_blocks\.(\d+)\.img_attn\.qkv\.weight", key)
        if img_match:
            idx = img_match.group(1)
            dim = int(tensor.shape[0]) // 3
            converted[f"transformer_blocks.{idx}.attn.to_q.weight"] = _slice_rows_torch(
                tensor, 0, dim
            )
            converted[f"transformer_blocks.{idx}.attn.to_k.weight"] = _slice_rows_torch(
                tensor, dim, 2 * dim
            )
            converted[f"transformer_blocks.{idx}.attn.to_v.weight"] = _slice_rows_torch(
                tensor, 2 * dim, 3 * dim
            )
            continue

        txt_match = re.fullmatch(r"double_blocks\.(\d+)\.txt_attn\.qkv\.weight", key)
        if txt_match:
            idx = txt_match.group(1)
            dim = int(tensor.shape[0]) // 3
            converted[
                f"transformer_blocks.{idx}.attn.add_q_proj.weight"
            ] = _slice_rows_torch(tensor, 0, dim)
            converted[
                f"transformer_blocks.{idx}.attn.add_k_proj.weight"
            ] = _slice_rows_torch(tensor, dim, 2 * dim)
            converted[
                f"transformer_blocks.{idx}.attn.add_v_proj.weight"
            ] = _slice_rows_torch(tensor, 2 * dim, 3 * dim)
            continue

        img_match = re.fullmatch(
            r"double_blocks\.(\d+)\.img_attn\.qkv\.(input_scale|weight_scale)",
            key,
        )
        if img_match:
            idx, suffix = img_match.groups()
            for proj in ("to_q", "to_k", "to_v"):
                new_name = f"transformer_blocks.{idx}.attn.{proj}.{suffix}"
                converted[new_name] = (
                    _legacy_fp8_input_scale_tensor(tensor)
                    if suffix == "input_scale"
                    else _clone_torch_weight(tensor)
                )
            continue

        txt_match = re.fullmatch(
            r"double_blocks\.(\d+)\.txt_attn\.qkv\.(input_scale|weight_scale)",
            key,
        )
        if txt_match:
            idx, suffix = txt_match.groups()
            for proj in ("add_q_proj", "add_k_proj", "add_v_proj"):
                new_name = f"transformer_blocks.{idx}.attn.{proj}.{suffix}"
                converted[new_name] = (
                    _legacy_fp8_input_scale_tensor(tensor)
                    if suffix == "input_scale"
                    else _clone_torch_weight(tensor)
                )
            continue

        replacements = (
            (
                r"double_blocks\.(\d+)\.img_attn\.proj\.weight",
                r"transformer_blocks.\1.attn.to_out.0.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_attn\.proj\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.attn.to_out.0.\2",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.proj\.weight",
                r"transformer_blocks.\1.attn.to_add_out.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.proj\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.attn.to_add_out.\2",
            ),
            (
                r"double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.scale",
                r"transformer_blocks.\1.attn.norm_q.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.scale",
                r"transformer_blocks.\1.attn.norm_k.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.scale",
                r"transformer_blocks.\1.attn.norm_added_q.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.scale",
                r"transformer_blocks.\1.attn.norm_added_k.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.0\.weight",
                r"transformer_blocks.\1.ff.linear_in.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.0\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.ff.linear_in.\2",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.2\.weight",
                r"transformer_blocks.\1.ff.linear_out.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.2\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.ff.linear_out.\2",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.0\.weight",
                r"transformer_blocks.\1.ff_context.linear_in.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.0\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.ff_context.linear_in.\2",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.2\.weight",
                r"transformer_blocks.\1.ff_context.linear_out.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.2\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.ff_context.linear_out.\2",
            ),
            (
                r"single_blocks\.(\d+)\.linear1\.weight",
                r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.weight",
            ),
            (
                r"single_blocks\.(\d+)\.linear1\.(input_scale|weight_scale)",
                r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.\2",
            ),
            (
                r"single_blocks\.(\d+)\.linear2\.weight",
                r"single_transformer_blocks.\1.attn.to_out.weight",
            ),
            (
                r"single_blocks\.(\d+)\.linear2\.(input_scale|weight_scale)",
                r"single_transformer_blocks.\1.attn.to_out.\2",
            ),
            (
                r"single_blocks\.(\d+)\.norm\.query_norm\.scale",
                r"single_transformer_blocks.\1.attn.norm_q.weight",
            ),
            (
                r"single_blocks\.(\d+)\.norm\.key_norm\.scale",
                r"single_transformer_blocks.\1.attn.norm_k.weight",
            ),
        )
        for pattern, replacement in replacements:
            mapped = re.sub(pattern, replacement, key)
            if mapped != key:
                converted[mapped] = (
                    _legacy_fp8_input_scale_tensor(tensor)
                    if mapped.endswith(".input_scale")
                    else _clone_torch_weight(tensor)
                )
                break

    _validate_required_flux2_prefixes(converted)
    return converted


def convert_safetensor_state_dict(
    state_dict: Mapping[str, WeightData],
) -> dict[str, WeightData]:
    if not is_bflabs_flux2_transformer_checkpoint(state_dict):
        return dict(state_dict)

    converted: dict[str, WeightData] = {}

    direct_mappings = {
        "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
        "img_in.weight": "x_embedder.weight",
        "txt_in.weight": "context_embedder.weight",
        "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.linear.weight": "proj_out.weight",
    }
    for old_name, new_name in direct_mappings.items():
        if old_name in state_dict:
            converted[new_name] = _clone_weight(state_dict[old_name], new_name)

    for key, weight in state_dict.items():
        img_match = re.fullmatch(r"double_blocks\.(\d+)\.img_attn\.qkv\.weight", key)
        if img_match:
            idx = img_match.group(1)
            dim = int(weight.shape[0]) // 3
            converted[f"transformer_blocks.{idx}.attn.to_q.weight"] = _slice_rows(
                weight,
                0,
                dim,
                f"transformer_blocks.{idx}.attn.to_q.weight",
            )
            converted[f"transformer_blocks.{idx}.attn.to_k.weight"] = _slice_rows(
                weight,
                dim,
                2 * dim,
                f"transformer_blocks.{idx}.attn.to_k.weight",
            )
            converted[f"transformer_blocks.{idx}.attn.to_v.weight"] = _slice_rows(
                weight,
                2 * dim,
                3 * dim,
                f"transformer_blocks.{idx}.attn.to_v.weight",
            )
            continue

        txt_match = re.fullmatch(r"double_blocks\.(\d+)\.txt_attn\.qkv\.weight", key)
        if txt_match:
            idx = txt_match.group(1)
            dim = int(weight.shape[0]) // 3
            converted[f"transformer_blocks.{idx}.attn.add_q_proj.weight"] = _slice_rows(
                weight,
                0,
                dim,
                f"transformer_blocks.{idx}.attn.add_q_proj.weight",
            )
            converted[f"transformer_blocks.{idx}.attn.add_k_proj.weight"] = _slice_rows(
                weight,
                dim,
                2 * dim,
                f"transformer_blocks.{idx}.attn.add_k_proj.weight",
            )
            converted[f"transformer_blocks.{idx}.attn.add_v_proj.weight"] = _slice_rows(
                weight,
                2 * dim,
                3 * dim,
                f"transformer_blocks.{idx}.attn.add_v_proj.weight",
            )
            continue

        img_match = re.fullmatch(
            r"double_blocks\.(\d+)\.img_attn\.qkv\.(input_scale|weight_scale)",
            key,
        )
        if img_match:
            idx, suffix = img_match.groups()
            for proj in ("to_q", "to_k", "to_v"):
                new_name = f"transformer_blocks.{idx}.attn.{proj}.{suffix}"
                converted[new_name] = (
                    _legacy_fp8_input_scale_weight(weight, new_name)
                    if suffix == "input_scale"
                    else _clone_weight(weight, new_name)
                )
            continue

        txt_match = re.fullmatch(
            r"double_blocks\.(\d+)\.txt_attn\.qkv\.(input_scale|weight_scale)",
            key,
        )
        if txt_match:
            idx, suffix = txt_match.groups()
            for proj in ("add_q_proj", "add_k_proj", "add_v_proj"):
                new_name = f"transformer_blocks.{idx}.attn.{proj}.{suffix}"
                converted[new_name] = (
                    _legacy_fp8_input_scale_weight(weight, new_name)
                    if suffix == "input_scale"
                    else _clone_weight(weight, new_name)
                )
            continue

        replacements = (
            (
                r"double_blocks\.(\d+)\.img_attn\.proj\.weight",
                r"transformer_blocks.\1.attn.to_out.0.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_attn\.proj\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.attn.to_out.0.\2",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.proj\.weight",
                r"transformer_blocks.\1.attn.to_add_out.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.proj\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.attn.to_add_out.\2",
            ),
            (
                r"double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.scale",
                r"transformer_blocks.\1.attn.norm_q.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.scale",
                r"transformer_blocks.\1.attn.norm_k.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.scale",
                r"transformer_blocks.\1.attn.norm_added_q.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.scale",
                r"transformer_blocks.\1.attn.norm_added_k.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.0\.weight",
                r"transformer_blocks.\1.ff.linear_in.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.0\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.ff.linear_in.\2",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.2\.weight",
                r"transformer_blocks.\1.ff.linear_out.weight",
            ),
            (
                r"double_blocks\.(\d+)\.img_mlp\.2\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.ff.linear_out.\2",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.0\.weight",
                r"transformer_blocks.\1.ff_context.linear_in.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.0\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.ff_context.linear_in.\2",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.2\.weight",
                r"transformer_blocks.\1.ff_context.linear_out.weight",
            ),
            (
                r"double_blocks\.(\d+)\.txt_mlp\.2\.(input_scale|weight_scale)",
                r"transformer_blocks.\1.ff_context.linear_out.\2",
            ),
            (
                r"single_blocks\.(\d+)\.linear1\.weight",
                r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.weight",
            ),
            (
                r"single_blocks\.(\d+)\.linear1\.(input_scale|weight_scale)",
                r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.\2",
            ),
            (
                r"single_blocks\.(\d+)\.linear2\.weight",
                r"single_transformer_blocks.\1.attn.to_out.weight",
            ),
            (
                r"single_blocks\.(\d+)\.linear2\.(input_scale|weight_scale)",
                r"single_transformer_blocks.\1.attn.to_out.\2",
            ),
            (
                r"single_blocks\.(\d+)\.norm\.query_norm\.scale",
                r"single_transformer_blocks.\1.attn.norm_q.weight",
            ),
            (
                r"single_blocks\.(\d+)\.norm\.key_norm\.scale",
                r"single_transformer_blocks.\1.attn.norm_k.weight",
            ),
        )
        for pattern, replacement in replacements:
            mapped = re.sub(pattern, replacement, key)
            if mapped != key:
                converted[mapped] = (
                    _legacy_fp8_input_scale_weight(weight, mapped)
                    if mapped.endswith(".input_scale")
                    else _clone_weight(weight, mapped)
                )
                break

    required_prefixes = (
        "time_guidance_embed.timestep_embedder.",
        "x_embedder.",
        "context_embedder.",
        "double_stream_modulation_img.",
        "double_stream_modulation_txt.",
        "single_stream_modulation.",
        "transformer_blocks.0.attn.",
        "single_transformer_blocks.0.attn.",
        "norm_out.",
        "proj_out.",
    )
    for prefix in required_prefixes:
        if not any(name.startswith(prefix) for name in converted):
            raise ValueError(
                f"Missing required flux2 transformer weights with prefix '{prefix}'"
            )

    return converted
