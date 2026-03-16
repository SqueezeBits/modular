from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from max.driver import Buffer
from max.graph.weights import WeightsFormat
from max.graph.weights import WeightData

Flux2ActivationScheme = Literal["static", "dynamic"]

_DYNAMIC_ADAPTED_FORMAT = "dynamic_block_fp8_v1"
_STATIC_ADAPTED_FORMAT = "legacy_scalar_static_v2"
_STATIC_REPO_FORMAT = "legacy_scalar_static_repo_v1"


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
    # Static FP8 kernels consume the checkpoint's direct scalar input_scale.
    # Keep the source value unchanged when adapting legacy scalar checkpoints.
    return _clone_weight(weight, new_name)


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


def _materialized_static_flux2_repo_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.max.static.repo")


def _materialized_static_flux2_repo_manifest_path(repo_root: Path) -> Path:
    return repo_root / ".max_flux2_static_repo.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f"{dst.name}.tmp")
    if tmp_dst.exists() or tmp_dst.is_symlink():
        tmp_dst.unlink()
    try:
        os.symlink(src, tmp_dst)
    except OSError:
        shutil.copy2(src, tmp_dst)
    tmp_dst.replace(dst)


def _resolve_repo_file_paths(
    *,
    repo_id: str,
    revision: str,
    filenames: list[str],
    force_download: bool,
) -> dict[str, Path]:
    if not filenames:
        return {}

    from max.pipelines.lib.hf_utils import (
        HuggingFaceRepo,
        download_weight_files,
    )

    repo = HuggingFaceRepo(repo_id=repo_id, revision=revision)
    if repo.repo_type == "local":
        return {
            filename: (Path(repo.repo_id) / filename)
            for filename in filenames
        }

    resolved = download_weight_files(
        huggingface_model_id=repo.repo_id,
        filenames=filenames,
        revision=revision,
        force_download=force_download,
    )
    return dict(zip(filenames, resolved, strict=True))


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


def materialize_bflabs_flux2_klein_static_repo(
    path: Path,
    *,
    base_repo_id: str,
    base_revision: str,
    diffusers_config: Mapping[str, Any],
    force_download: bool = False,
) -> Path:
    """Create a minimal diffusers-style local repo for HF flat Klein FP8 weights.

    The raw HF checkpoint is adapted once into MAX's static scalar-FP8 format
    and then exposed through a local repo layout so the runtime can use the
    normal diffusers component loading path.
    """
    repo_root = _materialized_static_flux2_repo_path(path)
    manifest_path = _materialized_static_flux2_repo_manifest_path(repo_root)
    adapted_transformer = adapt_bflabs_flux2_transformer_weights(
        path, activation_scheme="static"
    )

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("format") == _STATIC_REPO_FORMAT
            and manifest.get("source_checkpoint") == str(path)
            and manifest.get("base_repo_id") == base_repo_id
            and manifest.get("base_revision") == base_revision
            and (repo_root / "model_index.json").exists()
            and (repo_root / "transformer/config.json").exists()
            and (
                repo_root / "transformer/diffusion_pytorch_model.fp8.safetensors"
            ).exists()
        ):
            return repo_root

    components_config = diffusers_config.get("components")
    if not isinstance(components_config, Mapping):
        raise ValueError(
            "diffusers_config['components'] is required to materialize the "
            "Flux2 Klein static repo."
        )

    required_components = ("vae", "text_encoder", "transformer")
    for component_name in required_components:
        if component_name not in components_config:
            raise ValueError(
                f"diffusers_config is missing required component '{component_name}'."
            )

    model_index = {
        "_class_name": diffusers_config.get("_class_name"),
        "_diffusers_version": diffusers_config.get("_diffusers_version"),
    }
    for component_name in required_components:
        component_info = components_config[component_name]
        model_index[component_name] = [
            component_info["library"],
            component_info["class_name"],
        ]

    repo_root.mkdir(parents=True, exist_ok=True)
    _write_json(repo_root / "model_index.json", model_index)

    for component_name in ("vae", "text_encoder"):
        component_cfg = deepcopy(
            components_config[component_name].get("config_dict") or {}
        )
        _write_json(
            repo_root / component_name / "config.json",
            component_cfg,
        )

    transformer_cfg = deepcopy(
        components_config["transformer"].get("config_dict") or {}
    )
    transformer_quant_cfg = dict(
        transformer_cfg.get("quantization_config") or {}
    )
    transformer_quant_cfg["quant_method"] = "fp8"
    transformer_quant_cfg["activation_scheme"] = "static"
    transformer_cfg["quantization_config"] = transformer_quant_cfg
    transformer_cfg["activation_scheme"] = "static"
    _write_json(repo_root / "transformer" / "config.json", transformer_cfg)

    from max.pipelines.lib.hf_utils import HuggingFaceRepo

    base_repo = HuggingFaceRepo(repo_id=base_repo_id, revision=base_revision)
    base_weight_files = base_repo.weight_files.get(WeightsFormat.safetensors, [])
    passthrough_files = [
        filename
        for filename in base_weight_files
        if filename.startswith("vae/") or filename.startswith("text_encoder/")
    ]
    passthrough_paths = _resolve_repo_file_paths(
        repo_id=base_repo_id,
        revision=base_revision,
        filenames=passthrough_files,
        force_download=force_download,
    )
    for filename, resolved_path in passthrough_paths.items():
        _link_or_copy_file(resolved_path, repo_root / filename)

    _link_or_copy_file(
        adapted_transformer,
        repo_root / "transformer" / "diffusion_pytorch_model.fp8.safetensors",
    )

    _write_json(
        manifest_path,
        {
            "format": _STATIC_REPO_FORMAT,
            "source_checkpoint": str(path),
            "base_repo_id": base_repo_id,
            "base_revision": base_revision,
            "activation_scheme": "static",
        },
    )
    return repo_root


def _clone_torch_weight(tensor, /):
    return tensor.detach().clone()


def _legacy_fp8_input_scale_tensor(tensor, /):
    # Static FP8 kernels consume the checkpoint's direct scalar input_scale.
    # Keep the source value unchanged when adapting legacy scalar checkpoints.
    return _clone_torch_weight(tensor)


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
            # Tile the scalar weight_scale to blockwise format without re-quantizing
            # the FP8 weights. The previous approach (dequant -> blockwise requant)
            # added a second round of quantization error on top of the original.
            # Instead, keep the exact original FP8 values and broadcast the scalar
            # scale across all 128x128 blocks: the kernel dequantizes each element
            # as w_fp8 * block_scale, so tiling the scalar reproduces the original
            # scalar dequantization exactly with zero additional quantization error.
            block_n, block_k = 128, 128
            out_blocks = (int(tensor.shape[0]) + block_n - 1) // block_n
            in_blocks = (int(tensor.shape[1]) + block_k - 1) // block_k
            tiled_scale = (
                scale.to(torch.float32)
                .expand(out_blocks, in_blocks)
                .clone()
                .contiguous()
            )
            return tensor.contiguous(), tiled_scale
        return _clone_torch_weight(tensor), None

    direct_mappings = {
        "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
        "img_in.weight": "x_embedder.weight",
        "txt_in.weight": "context_embedder.weight",
        "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
        "final_layer.linear.weight": "proj_out.weight",
    }
    for old_name, new_name in direct_mappings.items():
        if old_name in state_dict:
            weight, weight_scale = _convert_weight(old_name, state_dict[old_name])
            converted[new_name] = weight
            if weight_scale is not None:
                converted[new_name[: -len(".weight")] + ".weight_scale"] = weight_scale

    # BFLabs stores adaLN_modulation output as [shift; scale] but diffusers/MAX
    # AdaLayerNormContinuous expects [scale; shift] (scale first). Swap the halves.
    if "final_layer.adaLN_modulation.1.weight" in state_dict:
        raw, weight_scale = _convert_weight(
            "final_layer.adaLN_modulation.1.weight",
            state_dict["final_layer.adaLN_modulation.1.weight"],
        )
        half = int(raw.shape[0]) // 2
        converted["norm_out.linear.weight"] = torch.cat(
            [raw[half:], raw[:half]], dim=0
        ).contiguous()
        if weight_scale is not None:
            converted["norm_out.linear.weight_scale"] = weight_scale

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
    import torch

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
        "final_layer.linear.weight": "proj_out.weight",
    }
    for old_name, new_name in direct_mappings.items():
        if old_name in state_dict:
            converted[new_name] = _clone_torch_weight(state_dict[old_name])

    # BFLabs stores adaLN_modulation output as [shift; scale] but diffusers/MAX
    # AdaLayerNormContinuous expects [scale; shift] (scale first). Swap the halves.
    if "final_layer.adaLN_modulation.1.weight" in state_dict:
        raw = _clone_torch_weight(state_dict["final_layer.adaLN_modulation.1.weight"])
        half = int(raw.shape[0]) // 2
        converted["norm_out.linear.weight"] = torch.cat(
            [raw[half:], raw[:half]], dim=0
        ).contiguous()

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
        "final_layer.linear.weight": "proj_out.weight",
    }
    for old_name, new_name in direct_mappings.items():
        if old_name in state_dict:
            converted[new_name] = _clone_weight(state_dict[old_name], new_name)

    # BFLabs stores adaLN_modulation output as [shift; scale] but diffusers/MAX
    # AdaLayerNormContinuous expects [scale; shift] (scale first). Swap the halves.
    if "final_layer.adaLN_modulation.1.weight" in state_dict:
        import torch

        src = state_dict["final_layer.adaLN_modulation.1.weight"]
        half = int(src.shape[0]) // 2
        # WeightData implements __dlpack__ so torch.from_dlpack works directly.
        raw_t = torch.from_dlpack(src).clone()
        swapped_t = torch.cat([raw_t[half:], raw_t[:half]], dim=0).contiguous()
        converted["norm_out.linear.weight"] = WeightData(
            data=Buffer.from_dlpack(swapped_t),
            name="norm_out.linear.weight",
            dtype=src.dtype,
            shape=src.shape,
            quantization_encoding=src.quantization_encoding,
        )

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
