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

"""Configuration helpers for step-cache."""

from __future__ import annotations

import os
from collections.abc import Mapping


def is_step_cache_enabled_from_env(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether step-cache should be enabled from env state.

    Uses ``MAX_STEP_CACHE`` and interprets only ``"1"`` as enabled.
    """
    if env is None:
        env = os.environ
    return env.get("MAX_STEP_CACHE", "0") == "1"
