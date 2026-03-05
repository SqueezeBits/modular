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

"""Registry for step-cache model adapters."""

from __future__ import annotations

from typing import Any

from .adapters import Flux1StepCacheMissAdapter, Flux2StepCacheMissAdapter

_ADAPTERS_BY_KEY: dict[str, type[Any]] = {}


def register_step_cache_adapter(model_key: str, adapter_cls: type[Any]) -> None:
    """Register miss adapter class for a model key."""
    _ADAPTERS_BY_KEY[model_key] = adapter_cls


def resolve_step_cache_adapter(model_or_config: Any) -> type[Any]:
    """Resolve a registered adapter from model/config identity."""
    candidates: list[str] = []
    if isinstance(model_or_config, str):
        candidates.append(model_or_config)
    else:
        arch = getattr(model_or_config, "architecture", None)
        if isinstance(arch, str):
            candidates.append(arch)
        candidates.append(type(model_or_config).__name__)
        cfg = getattr(model_or_config, "config", None)
        if cfg is not None:
            arch2 = getattr(cfg, "architecture", None)
            if isinstance(arch2, str):
                candidates.append(arch2)
            candidates.append(type(cfg).__name__)

    normalized: list[str] = []
    for key in candidates:
        normalized.append(key)
        normalized.append(key.lower())

    for key in normalized:
        if key in _ADAPTERS_BY_KEY:
            return _ADAPTERS_BY_KEY[key]

    raise KeyError(f"No step-cache adapter is registered for: {candidates!r}")


def register_builtin_step_cache_adapters() -> None:
    """Register built-in model adapters."""
    register_step_cache_adapter("flux1", Flux1StepCacheMissAdapter)
    register_step_cache_adapter("flux2", Flux2StepCacheMissAdapter)
    register_step_cache_adapter("FluxConfigBase", Flux1StepCacheMissAdapter)
    register_step_cache_adapter("Flux2ConfigBase", Flux2StepCacheMissAdapter)
    register_step_cache_adapter(
        "FluxTransformer2DModel", Flux1StepCacheMissAdapter
    )
    register_step_cache_adapter(
        "Flux2Transformer2DModel", Flux2StepCacheMissAdapter
    )
