from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import struct
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightsFormat
from max.graph.weights import WeightData

Flux2ActivationScheme = Literal["static", "dynamic"]

_DYNAMIC_ADAPTED_FORMAT = "dynamic_block_fp8_v1"
_STATIC_ADAPTED_FORMAT = "legacy_scalar_static_v2"
_STATIC_REPO_FORMAT = "legacy_scalar_static_repo_v1"

# Mapping from MAX DType to the dtype name string expected by safetensors'
# framework-agnostic serialize_file (numpy/torch style names, e.g. "float32").
_DTYPE_TO_SAFETENSORS: dict[DType, str] = {
    DType.bool: "bool",
    DType.uint8: "uint8",
    DType.int8: "int8",
    DType.int16: "int16",
    DType.int32: "int32",
    DType.int64: "int64",
    DType.float16: "float16",
    DType.bfloat16: "bfloat16",
    DType.float32: "float32",
    DType.float8_e4m3fn: "float8_e4m3fn",
    DType.float8_e5m2: "float8_e5m2",
}


# ---------------------------------------------------------------------------
# Low-level Buffer helpers (ctypes only, no numpy / torch)
# ---------------------------------------------------------------------------


def _buffer_to_bytes(buf: Buffer) -> bytes:
    """Return a copy of *buf*'s raw memory as a Python :class:`bytes` object."""
    nbytes = buf.num_elements * buf.element_size
    return bytes(ctypes.string_at(buf._data_ptr(), nbytes))


def _memmove_buffer(dst: Buffer, dst_offset_bytes: int, src: Buffer) -> None:
    """Copy all bytes of *src* into *dst* at *dst_offset_bytes*."""
    nbytes = src.num_elements * src.element_size
    ctypes.memmove(dst._data_ptr() + dst_offset_bytes, src._data_ptr(), nbytes)


def _fill_float32_buffer(shape: list[int], value: float) -> Buffer:
    """Allocate a float32 Buffer and fill every element with *value*."""
    n = 1
    for d in shape:
        n *= d
    raw = struct.pack(f"<{n}f", *([value] * n))
    buf = Buffer(DType.float32, shape)
    ctypes.memmove(buf._data_ptr(), raw, len(raw))
    return buf


def _read_scalar_float32(buf: Buffer) -> float:
    """Read a single float32 value from a scalar (0-d or 1-element) Buffer."""
    return struct.unpack_from("<f", ctypes.string_at(buf._data_ptr(), 4))[0]


# ---------------------------------------------------------------------------
# WeightData helpers
# ---------------------------------------------------------------------------


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
    """Slice rows [start, end) from a 2-D WeightData using raw memory copy."""
    src_buf = weight.data
    cols = int(weight.shape[1])
    elem_size = src_buf.element_size
    new_rows = end - start
    offset_bytes = start * cols * elem_size
    slice_bytes = new_rows * cols * elem_size
    new_buf = Buffer(weight.dtype, [new_rows, cols])
    ctypes.memmove(new_buf._data_ptr(), src_buf._data_ptr() + offset_bytes, slice_bytes)
    new_shape = weight.shape.__class__([new_rows, cols])
    return WeightData(
        data=new_buf,
        name=new_name,
        dtype=weight.dtype,
        shape=new_shape,
        quantization_encoding=weight.quantization_encoding,
    )


def _swap_row_halves(weight: WeightData, new_name: str) -> WeightData:
    """Return a new WeightData whose first and second row-halves are swapped.

    BFLabs stores adaLN_modulation output as [shift; scale] but MAX expects
    [scale; shift].  This function performs the swap without torch or numpy by
    working directly on the underlying Buffer bytes via ctypes.
    """
    src_buf = weight.data
    N = int(weight.shape[0])
    half = N // 2
    cols = int(weight.shape[1]) if len(weight.shape) > 1 else 1
    half_bytes = half * cols * src_buf.element_size

    new_buf = Buffer(weight.dtype, list(weight.shape))
    # second half of src → first position of dst
    ctypes.memmove(new_buf._data_ptr(), src_buf._data_ptr() + half_bytes, half_bytes)
    # first half of src → second position of dst
    ctypes.memmove(new_buf._data_ptr() + half_bytes, src_buf._data_ptr(), half_bytes)

    return WeightData(
        data=new_buf,
        name=new_name,
        dtype=weight.dtype,
        shape=weight.shape,
        quantization_encoding=weight.quantization_encoding,
    )


def _tile_scalar_fp8_scale(
    scale: WeightData,
    weight: WeightData,
    new_name: str,
    block_n: int = 128,
    block_k: int = 128,
) -> WeightData:
    """Expand a per-tensor (scalar) FP8 weight_scale to blockwise 2-D format.

    The dynamic FP8 kernel expects scale shape [out_blocks, in_blocks] where
    every entry equals the original scalar.  Broadcasting the scalar this way
    reproduces the original scalar dequantization (``w_fp8 * scale``) exactly
    across all blocks with zero additional quantization error.
    """
    scalar_val = _read_scalar_float32(scale.data)
    out_blocks = (int(weight.shape[0]) + block_n - 1) // block_n
    in_blocks = (int(weight.shape[1]) + block_k - 1) // block_k
    new_shape = [out_blocks, in_blocks]
    new_buf = _fill_float32_buffer(new_shape, scalar_val)
    return WeightData(
        data=new_buf,
        name=new_name,
        dtype=DType.float32,
        shape=scale.shape.__class__(new_shape),
        quantization_encoding=scale.quantization_encoding,
    )


# ---------------------------------------------------------------------------
# Safetensors I/O (MAX-native reader, framework-agnostic writer)
# ---------------------------------------------------------------------------


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    """Read the ``__metadata__`` dict from a safetensors header.

    Parses the binary header using :mod:`struct` and :mod:`json` — no
    ML framework required.
    """
    with open(path, "rb") as f:
        header_size = struct.unpack_from("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header.get("__metadata__") or {}


def _load_safetensors_as_weight_data(path: Path) -> dict[str, WeightData]:
    """Load a safetensors file into a ``{name: WeightData}`` dict.

    Uses MAX's own ``max._core.safetensors.safe_open`` which returns
    :class:`~max.driver.Buffer` objects directly — no torch or numpy.
    """
    from max._core.safetensors import safe_open

    result: dict[str, WeightData] = {}
    with safe_open(path) as f:
        for key in f.keys():
            buf = f.get_buffer(key)
            result[key] = WeightData(
                data=buf,
                name=key,
                dtype=buf.dtype,
                shape=buf.shape,
            )
    return result


def _save_weight_data_to_safetensors(
    tensors: dict[str, WeightData],
    path: Path,
    metadata: dict[str, str],
) -> None:
    """Write *tensors* to a safetensors file using the framework-agnostic API.

    ``safetensors.serialize_file`` accepts raw ``{"dtype", "shape", "data"}``
    dicts, so no torch or numpy is needed — raw bytes are extracted from each
    Buffer via :mod:`ctypes`.
    """
    from safetensors import serialize_file

    raw: dict[str, dict[str, Any]] = {}
    for name, wd in tensors.items():
        dtype_str = _DTYPE_TO_SAFETENSORS.get(wd.dtype)
        if dtype_str is None:
            raise ValueError(
                f"Unsupported dtype {wd.dtype} for safetensors serialisation "
                f"(tensor '{name}')."
            )
        raw[name] = {
            "dtype": dtype_str,
            "shape": list(int(d) for d in wd.shape),
            "data": _buffer_to_bytes(wd.data),
        }
    serialize_file(raw, str(path), metadata=metadata)


# ---------------------------------------------------------------------------
# BFLabs checkpoint detection helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main adaptation entry-point
# ---------------------------------------------------------------------------


def adapt_bflabs_flux2_transformer_weights(
    path: Path, *, activation_scheme: Flux2ActivationScheme = "dynamic"
) -> Path:
    """Persist a MAX-native Flux2 transformer checkpoint beside the source file.

    The adapted checkpoint is created once and then reused on subsequent loads.
    Non-BFLabs checkpoints are returned unchanged.

    Uses only MAX-native types (Buffer / WeightData / DType) for all tensor
    operations.  File I/O uses ``max._core.safetensors`` for reading and
    ``safetensors.serialize_file`` (framework-agnostic) for writing.
    """
    adapted_path = _adapted_flux2_transformer_path(path, activation_scheme)
    expected_format = (
        _DYNAMIC_ADAPTED_FORMAT
        if activation_scheme == "dynamic"
        else _STATIC_ADAPTED_FORMAT
    )
    if adapted_path.exists():
        metadata = _read_safetensors_metadata(adapted_path)
        if metadata.get("max_flux2_adapted_format") == expected_format:
            return adapted_path

    state_dict = _load_safetensors_as_weight_data(path)

    if not is_bflabs_flux2_transformer_checkpoint(state_dict):
        return path

    converted = convert_safetensor_state_dict(
        state_dict, activation_scheme=activation_scheme
    )

    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = adapted_path.with_name(f"{adapted_path.name}.tmp")
    out_metadata: dict[str, str] = {
        "max_flux2_adapted": "true",
        "max_flux2_adapted_format": expected_format,
        "max_flux2_activation_scheme": activation_scheme,
    }
    _save_weight_data_to_safetensors(converted, tmp_path, out_metadata)
    tmp_path.replace(adapted_path)
    return adapted_path


# ---------------------------------------------------------------------------
# Repo materialisation (legacy — kept for backward compatibility)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Runtime weight adapter  (WeightData → WeightData)
# ---------------------------------------------------------------------------


def convert_safetensor_state_dict(
    state_dict: Mapping[str, WeightData],
    activation_scheme: Flux2ActivationScheme = "static",
) -> dict[str, WeightData]:
    """Remap BFLabs-format Flux2 transformer keys to MAX-native names.

    All tensor operations use MAX's :class:`~max.driver.Buffer` and
    :class:`~max.graph.weights.WeightData` — no torch or numpy.

    When *activation_scheme* is ``"dynamic"``, any per-tensor (scalar)
    ``weight_scale`` associated with an FP8 weight is tiled to the blockwise
    2-D format expected by the dynamic FP8 kernel.
    """
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

    # BFLabs stores adaLN_modulation output as [shift; scale] but MAX
    # AdaLayerNormContinuous expects [scale; shift] (scale first). Swap rows.
    if "final_layer.adaLN_modulation.1.weight" in state_dict:
        converted["norm_out.linear.weight"] = _swap_row_halves(
            state_dict["final_layer.adaLN_modulation.1.weight"],
            "norm_out.linear.weight",
        )

    for key, weight in state_dict.items():
        img_match = re.fullmatch(r"double_blocks\.(\d+)\.img_attn\.qkv\.weight", key)
        if img_match:
            idx = img_match.group(1)
            dim = int(weight.shape[0]) // 3
            converted[f"transformer_blocks.{idx}.attn.to_q.weight"] = _slice_rows(
                weight, 0, dim, f"transformer_blocks.{idx}.attn.to_q.weight"
            )
            converted[f"transformer_blocks.{idx}.attn.to_k.weight"] = _slice_rows(
                weight, dim, 2 * dim, f"transformer_blocks.{idx}.attn.to_k.weight"
            )
            converted[f"transformer_blocks.{idx}.attn.to_v.weight"] = _slice_rows(
                weight, 2 * dim, 3 * dim, f"transformer_blocks.{idx}.attn.to_v.weight"
            )
            continue

        txt_match = re.fullmatch(r"double_blocks\.(\d+)\.txt_attn\.qkv\.weight", key)
        if txt_match:
            idx = txt_match.group(1)
            dim = int(weight.shape[0]) // 3
            converted[f"transformer_blocks.{idx}.attn.add_q_proj.weight"] = _slice_rows(
                weight, 0, dim, f"transformer_blocks.{idx}.attn.add_q_proj.weight"
            )
            converted[f"transformer_blocks.{idx}.attn.add_k_proj.weight"] = _slice_rows(
                weight, dim, 2 * dim, f"transformer_blocks.{idx}.attn.add_k_proj.weight"
            )
            converted[f"transformer_blocks.{idx}.attn.add_v_proj.weight"] = _slice_rows(
                weight, 2 * dim, 3 * dim, f"transformer_blocks.{idx}.attn.add_v_proj.weight"
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

    # For the dynamic activation scheme, tile any scalar (per-tensor)
    # weight_scale associated with an FP8 weight to blockwise 2-D format.
    if activation_scheme == "dynamic":
        for name in list(converted.keys()):
            if not name.endswith(".weight_scale"):
                continue
            scale_wd = converted[name]
            if tuple(int(d) for d in scale_wd.shape) != ():
                continue  # already blockwise
            weight_name = name[: -len("_scale")]  # strip "_scale" → ".weight"
            weight_wd = converted.get(weight_name)
            if weight_wd is not None and weight_wd.dtype == DType.float8_e4m3fn:
                converted[name] = _tile_scalar_fp8_scale(
                    scale_wd, weight_wd, name
                )

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
        if not any(n.startswith(prefix) for n in converted):
            raise ValueError(
                f"Missing required flux2 transformer weights with prefix '{prefix}'"
            )

    return converted
