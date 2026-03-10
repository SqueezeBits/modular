from __future__ import annotations

import re
from collections.abc import Mapping

from max.driver import Buffer
from max.graph.weights import WeightData


def _clone_weight(weight: WeightData, new_name: str) -> WeightData:
    return WeightData(
        data=weight.data,
        name=new_name,
        dtype=weight.dtype,
        shape=weight.shape,
        quantization_encoding=weight.quantization_encoding,
    )


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


def uses_legacy_scalar_fp8_scales(
    state_dict: Mapping[str, WeightData],
) -> bool:
    return any(
        key.endswith(".weight_scale") and tuple(int(d) for d in value.shape) == ()
        for key, value in state_dict.items()
    )


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
                converted[new_name] = _clone_weight(weight, new_name)
            continue

        txt_match = re.fullmatch(
            r"double_blocks\.(\d+)\.txt_attn\.qkv\.(input_scale|weight_scale)",
            key,
        )
        if txt_match:
            idx, suffix = txt_match.groups()
            for proj in ("add_q_proj", "add_k_proj", "add_v_proj"):
                new_name = f"transformer_blocks.{idx}.attn.{proj}.{suffix}"
                converted[new_name] = _clone_weight(weight, new_name)
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
                converted[mapped] = _clone_weight(weight, mapped)
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
