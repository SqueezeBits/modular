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

"""All-to-all emulation using allgather + slice.

Provides Ulysses-style all-to-all for context parallelism:
  - input_alltoall:  sequence-split → head-split (gather sequence, shard heads)
  - output_alltoall: head-split → sequence-split (gather heads, shard sequence)

These are drop-in replacements for a native all-to-all collective.
When a native ops.alltoall becomes available, only the internals of
these functions need to change.
"""

from __future__ import annotations

from max.graph import BufferValue, TensorValue, ops


def input_alltoall(
    xs: list[TensorValue],
    signal_buffers: list[BufferValue],
    num_devices: int,
) -> list[TensorValue]:
    """Ulysses input all-to-all: scatter heads, gather sequence.

    Each device starts with shape [B, S_local, H, D] (full heads, partial seq).
    After this op, each device has [B, S_global, H_local, D] (partial heads, full seq).

    Emulated via allgather(axis=1) to get full sequence on every device,
    then each device slices its assigned head range.

    Args:
        xs: Per-device tensors of shape [B, S_local, H, D].
        signal_buffers: Per-device signal buffers for synchronization.
        num_devices: Number of participating devices.

    Returns:
        Per-device tensors of shape [B, S_global, H_local, D].
    """
    if num_devices <= 1:
        return xs

    # allgather along sequence axis (dim=1):
    # [B, S_local, H, D] per device -> [B, S_global, H, D] on every device
    gathered = ops.allgather(xs, signal_buffers, axis=1)

    # Each device slices its portion of heads (dim=2)
    h_total = gathered[0].shape[2]  # symbolic H
    results = []
    for i in range(num_devices):
        # Compute head range for device i
        # h_local = H // num_devices, start = i * h_local
        h_local = h_total // num_devices
        start = i * h_local
        end = start + h_local
        sliced = gathered[i][:, :, start:end, :]
        results.append(sliced)

    return results


def output_alltoall(
    xs: list[TensorValue],
    signal_buffers: list[BufferValue],
    num_devices: int,
) -> list[TensorValue]:
    """Ulysses output all-to-all: gather heads, scatter sequence.

    Each device starts with shape [B, S_global, H_local, D] (partial heads, full seq).
    After this op, each device has [B, S_local, H, D] (full heads, partial seq).

    Emulated via allgather(axis=2) to get full heads on every device,
    then each device slices its assigned sequence range.

    Args:
        xs: Per-device tensors of shape [B, S_global, H_local, D].
        signal_buffers: Per-device signal buffers for synchronization.
        num_devices: Number of participating devices.

    Returns:
        Per-device tensors of shape [B, S_local, H, D].
    """
    if num_devices <= 1:
        return xs

    # allgather along head axis (dim=2):
    # [B, S_global, H_local, D] per device -> [B, S_global, H, D] on every device
    gathered = ops.allgather(xs, signal_buffers, axis=2)

    # Each device slices its portion of sequence (dim=1)
    s_total = gathered[0].shape[1]  # symbolic S_global
    results = []
    for i in range(num_devices):
        # Compute sequence range for device i
        # s_local = S_global // num_devices, start = i * s_local
        s_local = s_total // num_devices
        start = i * s_local
        end = start + s_local
        sliced = gathered[i][:, start:end, :, :]
        results.append(sliced)

    return results
