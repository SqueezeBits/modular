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

"""Shared graph runner for step-cache branching."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType, TensorValue

from .adapters.base import StepCacheMissAdapter


def run_step_cache_cond(
    *,
    can_use_cache: Tensor,
    output_type: TensorType,
    residual_type: TensorType,
    prev_output: Tensor,
    first_block_residual: Tensor,
    miss_policy: StepCacheMissAdapter,
) -> tuple[Tensor, Tensor]:
    """Run shared cache branch logic and return ``(output, residual)``."""

    def then_fn(
        _prev_output: Tensor = prev_output,
        _first_block_residual: Tensor = first_block_residual,
    ) -> tuple[TensorValue, TensorValue]:
        return (
            TensorValue(_prev_output),
            TensorValue(_first_block_residual),
        )

    def else_fn(
        _first_block_residual: Tensor = first_block_residual,
        _miss_policy: StepCacheMissAdapter = miss_policy,
    ) -> tuple[TensorValue, TensorValue]:
        out = _miss_policy.compute_output()
        return (
            TensorValue(out),
            TensorValue(_first_block_residual),
        )

    result = F.cond(
        can_use_cache,
        [output_type, residual_type],
        then_fn,
        else_fn,
    )
    return result[0], result[1]
