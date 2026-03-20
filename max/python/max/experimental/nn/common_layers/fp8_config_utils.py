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

from collections.abc import Mapping
from typing import Any

from max.dtype import DType
from max.nn.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)


def build_dynamic_block_fp8_config(
    config_dict: Mapping[str, Any],
    *,
    component_name: str,
) -> Float8Config:
    """Build Float8Config from a HF-style quantization_config dict.

    Expected contract:
      - quant_method: "fp8"
      - activation_scheme: "dynamic"
      - weight_block_size: [N, K]
    """
    quant_cfg = config_dict.get("quantization_config")
    if not isinstance(quant_cfg, Mapping):
        raise ValueError(
            f"{component_name}: missing required 'quantization_config' for FP8 encoding."
        )

    quant_method = quant_cfg.get("quant_method")
    if quant_method != "fp8":
        raise ValueError(
            f"{component_name}: unsupported quant_method={quant_method!r}. Expected 'fp8'."
        )

    # Some legacy/runtime-only FP8 checkpoints store float8 weights without
    # explicit weight scales. These should run via metadata-less fallback
    # (no scaled FP8 kernel path).
    if bool(quant_cfg.get("metadata_less_fp8", False)):
        raise ValueError(
            f"{component_name}: metadata_less_fp8 requested; use fallback FP8 path."
        )

    activation_scheme = quant_cfg.get("activation_scheme", "dynamic")
    if activation_scheme != "dynamic":
        raise ValueError(
            f"{component_name}: unsupported activation_scheme={activation_scheme!r}. Expected 'dynamic'."
        )

    block = quant_cfg.get("weight_block_size", [128, 128])
    if (
        not isinstance(block, list | tuple)
        or len(block) != 2
        or not all(isinstance(v, int) and v > 0 for v in block)
    ):
        raise ValueError(
            f"{component_name}: invalid weight_block_size={block!r}. Expected two positive integers."
        )
    weight_block = (int(block[0]), int(block[1]))

    return Float8Config(
        input_scale=Float8InputScaleSpec(
            granularity=Float8ScaleGranularity.BLOCK,
            origin=Float8ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, weight_block[1]),
        ),
        weight_scale=Float8WeightScaleSpec(
            granularity=Float8ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=weight_block,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
        quant_method="fp8",
    )


def build_legacy_scalar_fp8_config(*, component_name: str) -> Float8Config:
    """Build Float8Config for legacy scalar-scale FP8 checkpoints.

    These checkpoints provide per-layer scalar `input_scale` and
    `weight_scale` tensors rather than blockwise scales.
    """
    return Float8Config(
        input_scale=Float8InputScaleSpec(
            granularity=Float8ScaleGranularity.TENSOR,
            origin=Float8ScaleOrigin.STATIC,
            dtype=DType.float32,
        ),
        weight_scale=Float8WeightScaleSpec(
            granularity=Float8ScaleGranularity.TENSOR,
            dtype=DType.float32,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
        quant_method="fp8",
    )


def validate_fp8_weight_scale_contract(
    state_dict: Mapping[str, Any],
    *,
    float8_config: Float8Config,
    component_name: str,
) -> None:
    """Validate `.weight`/`.weight_scale` presence and block scale shape."""
    assert float8_config.weight_scale.block_size is not None
    block_n, block_k = float8_config.weight_scale.block_size

    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    missing_scales: list[str] = []
    bad_scale_shape: list[str] = []
    bad_scale_dtype: list[str] = []
    fp8_weight_count = 0

    for key, weight in state_dict.items():
        if not key.endswith(".weight"):
            continue
        dtype = getattr(weight, "dtype", None)
        if dtype is None or not dtype.is_float8():
            continue
        fp8_weight_count += 1

        scale_key = key[: -len(".weight")] + ".weight_scale"
        scale = state_dict.get(scale_key)
        if scale is None:
            missing_scales.append(scale_key)
            continue

        if getattr(scale, "dtype", None) != DType.float32:
            bad_scale_dtype.append(scale_key)

        try:
            out_dim = int(weight.shape[0])
            in_dim = int(weight.shape[1])
            expected_shape = (ceildiv(out_dim, block_n), ceildiv(in_dim, block_k))
            scale_shape = tuple(int(d) for d in scale.shape)
        except Exception:
            bad_scale_shape.append(
                f"{scale_key} (failed to inspect weight/scale shapes)"
            )
            continue

        if scale_shape != expected_shape:
            bad_scale_shape.append(
                f"{scale_key} expected {expected_shape}, got {scale_shape}"
            )

    if fp8_weight_count == 0:
        raise ValueError(
            f"{component_name}: FP8 encoding selected but no float8 weights found in state_dict."
        )

    if missing_scales or bad_scale_shape or bad_scale_dtype:
        lines = [
            f"{component_name}: FP8 weight/scale contract validation failed.",
            f"fp8_weight_count={fp8_weight_count}",
        ]
        if missing_scales:
            lines.append(
                "missing_scales=" + ", ".join(missing_scales[:12])
            )
        if bad_scale_shape:
            lines.append(
                "bad_scale_shape=" + ", ".join(bad_scale_shape[:12])
            )
        if bad_scale_dtype:
            lines.append(
                "bad_scale_dtype=" + ", ".join(bad_scale_dtype[:12])
            )
        raise ValueError(" ".join(lines))
