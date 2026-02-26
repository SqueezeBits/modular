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

"""Fused AdaLN modulate kernel: LayerNorm + (1 + scale) * x + shift."""

from __future__ import annotations

from max.graph import DeviceRef, TensorType, TensorValue, ops


def adaln_modulate(
    x: TensorValue,
    scale: TensorValue,
    shift: TensorValue,
    eps: float = 1e-6,
) -> TensorValue:
    """Fused LayerNorm + AdaLN modulation in a single GPU kernel pass.

    Computes: LayerNorm(x) * (1 + scale) + shift

    This fuses the LayerNorm computation and the subsequent elementwise
    modulation into a single kernel, eliminating the intermediate tensor
    write between the two operations.

    Args:
        x: Input tensor of shape [B, S, D].
        scale: Modulation scale tensor of shape [B, 1, D]. Broadcast over S.
        shift: Modulation shift tensor of shape [B, 1, D]. Broadcast over S.
        eps: Epsilon for numerical stability in LayerNorm.

    Returns:
        Modulated tensor of shape [B, S, D] with the same dtype as x.
    """
    epsilon = ops.constant(eps, x.dtype, DeviceRef.CPU())
    return ops.custom(
        "mo.adaln_modulate",
        device=x.device,
        values=[x, scale, shift, epsilon],
        out_types=[TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
    )[0].tensor
