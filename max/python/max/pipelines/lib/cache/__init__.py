# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""First Block Cache (FBC) — 공통 모듈."""

from .mixins import FBCDenoiseMixin, FBCModelMixin
from .first_block_cache import (
    BLOCK_SEQUENCE_NAMES,
    get_block_sequence,
    get_can_use_cache,
)
from .registry import (
    FBCAdapter,
    FBC_REGISTRY,
    TransformerBlockMetadata,
    TransformerModelMetadata,
    build_input_types_fbc,
    get_adapter,
    register,
    register_block,
)
from .transformer_with_fbc import TransformerWithFBC

from .architectures import register_flux

register_flux()

__all__ = [
    "BLOCK_SEQUENCE_NAMES",
    "FBCAdapter",
    "FBC_REGISTRY",
    "FBCDenoiseMixin",
    "FBCModelMixin",
    "TransformerBlockMetadata",
    "TransformerModelMetadata",
    "build_input_types_fbc",
    "get_adapter",
    "get_block_sequence",
    "get_can_use_cache",
    "register",
    "register_block",
    "TransformerWithFBC",
]
