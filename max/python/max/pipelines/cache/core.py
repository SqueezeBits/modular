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

"""Core graph utilities for step-cache."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType


def compute_can_reuse(
    intermediate_residual: Tensor,
    prev_intermediate_residual: Tensor | None,
    rdt: Tensor,
) -> Tensor:
    """Return whether previous residual cache is reusable.

    The decision rule is:
    ``mean(abs(curr-prev)) / (mean(abs(prev)) + eps) < rdt``.
    """
    dev = intermediate_residual.device
    if (
        prev_intermediate_residual is None
        or intermediate_residual.shape != prev_intermediate_residual.shape
    ):
        return F.constant(False, DType.bool, device=dev)

    reduced_last_dim_shape = tuple(intermediate_residual.shape[:-1]) + (1,)
    reduced_last_dim_type = TensorType(
        intermediate_residual.dtype,
        shape=reduced_last_dim_shape,
        device=dev,
    )
    mean_diff_rows, mean_prev_rows = F.custom(
        "mo.step_cache.mean_abs_pair_lastdim",
        device=dev,
        values=[intermediate_residual, prev_intermediate_residual],
        out_types=[reduced_last_dim_type, reduced_last_dim_type],
    )
    mean_diff = F.mean(mean_diff_rows, axis=None)
    mean_prev = F.mean(mean_prev_rows, axis=None)
    eps = 1e-9
    relative_diff = mean_diff / (mean_prev + eps)
    pred = relative_diff < F.cast(rdt, relative_diff.dtype)
    # cond predicate must be scalar bool.
    return F.squeeze(pred, 0)


def step_cache_input_types(
    *,
    dtype: DType,
    device: Any,
    inner_dim: int,
    out_dim: int,
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    """Return shared step-cache extra input TensorTypes."""
    prev_residual_type = TensorType(
        dtype,
        shape=["batch_size", "image_seq_len", inner_dim],
        device=device,
    )
    prev_output_type = TensorType(
        dtype,
        shape=["batch_size", "image_seq_len", out_dim],
        device=device,
    )
    cache_enabled_type = TensorType(
        DType.bool,
        shape=[1],
        device=device,
    )
    rdt_type = TensorType(
        DType.float32,
        shape=[1],
        device=device,
    )
    return (
        prev_residual_type,
        prev_output_type,
        cache_enabled_type,
        rdt_type,
    )
