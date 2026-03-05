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

"""High-level orchestration for step-cache execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.experimental.tensor import Tensor
from max.graph import TensorType

from .registry import resolve_step_cache_adapter
from .runner import run_step_cache_cond
from .state import StepCacheRuntime, StepCacheState


def run_step_cache_branch(
    *,
    model_or_config: Any,
    adapter_cls: type[Any] | None = None,
    can_use_cache: Tensor,
    output_type: TensorType,
    residual_type: TensorType,
    prev_output: Tensor,
    first_block_residual: Tensor,
    adapter_kwargs: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    """Run cache branch with adapter resolved from the central registry."""
    adapter_type = (
        adapter_cls
        if adapter_cls is not None
        else resolve_step_cache_adapter(model_or_config)
    )
    miss_adapter = adapter_type(**adapter_kwargs)
    return run_step_cache_cond(
        can_use_cache=can_use_cache,
        output_type=output_type,
        residual_type=residual_type,
        prev_output=prev_output,
        first_block_residual=first_block_residual,
        miss_policy=miss_adapter,
    )


class StepCacheController:
    """Pipeline-side step-cache orchestration helper."""

    def __init__(self, runtime: StepCacheRuntime) -> None:
        self.runtime = runtime

    def run_step_cache_step(
        self,
        *,
        context_name: str,
        cache_call: Callable[
            [StepCacheState, Tensor, Tensor], tuple[Tensor, Tensor]
        ],
        no_cache_call: Callable[[], Tensor],
    ) -> Tensor:
        """Execute a denoising step with cache-aware state management."""
        if not self.runtime.enabled:
            return no_cache_call()

        state = self.runtime.get_state(context_name)
        assert self.runtime.step_cache_flag is not None
        assert self.runtime.rdt_tensor is not None
        noise_pred, new_residual = cache_call(
            state,
            self.runtime.step_cache_flag,
            self.runtime.rdt_tensor,
        )
        self.runtime.update(
            context_name,
            prev_residual=new_residual,
            prev_output=noise_pred,
        )
        return noise_pred
