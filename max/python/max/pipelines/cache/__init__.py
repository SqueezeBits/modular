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

"""Common step-cache utilities for diffusion pipelines."""

from .adapters import StepCacheMissAdapter
from .config import is_step_cache_enabled_from_env
from .controller import StepCacheController, run_step_cache_branch
from .core import compute_can_reuse, step_cache_input_types
from .registry import (
    register_builtin_step_cache_adapters,
    register_step_cache_adapter,
    resolve_step_cache_adapter,
)
from .runner import run_step_cache_cond
from .state import StepCacheRuntime, StepCacheState

# Hydrate default adapter registry.
register_builtin_step_cache_adapters()

__all__ = [
    "StepCacheController",
    "StepCacheMissAdapter",
    "StepCacheRuntime",
    "StepCacheState",
    "compute_can_reuse",
    "is_step_cache_enabled_from_env",
    "register_step_cache_adapter",
    "resolve_step_cache_adapter",
    "run_step_cache_branch",
    "run_step_cache_cond",
    "step_cache_input_types",
]
