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

"""Base adapter contracts for step-cache."""

from __future__ import annotations

from typing import Protocol

from max.experimental.tensor import Tensor


class StepCacheMissAdapter(Protocol):
    """Adapter contract that computes cache-miss outputs."""

    def compute_output(self) -> Tensor:
        """Return model output for cache-miss branch."""
        ...
