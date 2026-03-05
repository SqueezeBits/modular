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

"""Runtime cache state helpers for diffusion pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from max.dtype import DType
from max.experimental.tensor import Tensor


@dataclass
class StepCacheState:
    prev_residual: Tensor
    prev_output: Tensor


@dataclass
class StepCacheRuntime:
    """Container for step-cache runtime tensors and per-context states."""

    enabled: bool
    step_cache_flag: Tensor | None
    rdt_tensor: Tensor | None
    states: dict[str, StepCacheState]

    @classmethod
    def disabled(cls) -> StepCacheRuntime:
        """Return a disabled runtime container."""
        return cls(
            enabled=False,
            step_cache_flag=None,
            rdt_tensor=None,
            states={},
        )

    @classmethod
    def create(
        cls,
        *,
        compiled_with_step_cache: bool,
        step_cache_enabled: bool,
        rdt: float,
        context_names: tuple[str, ...],
        device: Any,
        dtype: DType,
        batch_size: int,
        image_seq_len: int,
        inner_dim: int,
        out_dim: int,
    ) -> StepCacheRuntime:
        """Create step-cache runtime tensors and zero-initialized states."""
        if not compiled_with_step_cache:
            return cls.disabled()

        step_cache_flag = Tensor.constant(
            np.array([step_cache_enabled], dtype=np.bool_),
            dtype=DType.bool,
            device=device,
        )
        rdt_tensor = Tensor.constant(
            np.array([rdt], dtype=np.float32),
            dtype=DType.float32,
            device=device,
        )

        states: dict[str, StepCacheState] = {}
        for name in context_names:
            states[name] = StepCacheState(
                prev_residual=Tensor.zeros(
                    (batch_size, image_seq_len, inner_dim),
                    dtype=dtype,
                    device=device,
                ),
                prev_output=Tensor.zeros(
                    (batch_size, image_seq_len, out_dim),
                    dtype=dtype,
                    device=device,
                ),
            )

        return cls(
            enabled=True,
            step_cache_flag=step_cache_flag,
            rdt_tensor=rdt_tensor,
            states=states,
        )

    def get_state(self, context_name: str) -> StepCacheState:
        """Get cache state for the requested context name."""
        return self.states[context_name]

    def update(
        self,
        context_name: str,
        *,
        prev_residual: Tensor,
        prev_output: Tensor,
    ) -> None:
        """Replace cache state for the requested context name."""
        self.states[context_name] = StepCacheState(
            prev_residual=prev_residual,
            prev_output=prev_output,
        )
