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

"""
First Block Cache (FBC) — 1단계: 공통 유틸리티.

- get_can_use_cache: 캐시 사용 여부 판단 (텐서 연산만 사용, 그래프 유지)
- BLOCK_SEQUENCE_NAMES: 블록 시퀀스로 인식할 자식 이름 (전역 규약 1개)
- get_block_sequence: module에서 블록 시퀀스를 이름 규약으로 획득
"""

import logging
from typing import TYPE_CHECKING

from max import functional as F
from max.dtype import DType
from max.nn.sequential import ModuleList
from max.tensor import Tensor

if TYPE_CHECKING:
    from max.nn import Module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Block discovery (diffusers _ALL_TRANSFORMER_BLOCK_IDENTIFIERS 참고)
# ---------------------------------------------------------------------------

_BLOCK_SEQUENCE_NAMES_SPATIAL = (
    "blocks",
    "transformer_blocks",
    "single_transformer_blocks",
    "layers",
    "visual_transformer_blocks",
)
_BLOCK_SEQUENCE_NAMES_TEMPORAL = ("temporal_transformer_blocks",)
_BLOCK_SEQUENCE_NAMES_CROSS = ("blocks", "transformer_blocks", "layers")

BLOCK_SEQUENCE_NAMES = tuple(
    {
        *_BLOCK_SEQUENCE_NAMES_SPATIAL,
        *_BLOCK_SEQUENCE_NAMES_TEMPORAL,
        *_BLOCK_SEQUENCE_NAMES_CROSS,
    }
)
"""전역 블록 식별자. module.children에서 이 이름이면서 ModuleList인 자식을
블록 시퀀스로 수집. 새 모델 지원 시 여기에 이름만 추가하면 됨."""


# ---------------------------------------------------------------------------
# get_can_use_cache
# ---------------------------------------------------------------------------


def get_can_use_cache(
    intermediate_residual: Tensor,
    prev_intermediate_residual: Tensor | None,
    rdt: float,
) -> Tensor:
    """캐시 사용 여부를 텐서(bool)로 반환. 그래프 컴파일 시 끊김 없음.

    Args:
        intermediate_residual: 현재 첫 블록 residual (hidden states).
        prev_intermediate_residual: 이전 스텝의 residual. None이면 False 반환.
        rdt: Relative difference threshold. mean(|diff|)/mean(|prev|) < rdt 이면 True.

    Returns:
        스칼라 bool 텐서. True = 캐시 사용 가능, False = 전체 계산 필요.
    """
    dev = intermediate_residual.device
    if (
        rdt < 0
        or prev_intermediate_residual is None
        or intermediate_residual.shape != prev_intermediate_residual.shape
    ):
        return F.constant(False, DType.bool, device=dev)

    diff = F.abs(prev_intermediate_residual - intermediate_residual)
    mean_diff = F.mean(diff, axis=None)
    mean_prev = F.mean(F.abs(prev_intermediate_residual), axis=None)
    eps = 1e-9
    relative_diff = mean_diff / (mean_prev + eps)
    pred = relative_diff < rdt
    # mo.if requires rank-0 boolean; F.mean may return shape [1]
    return F.squeeze(pred, axis=0)


# ---------------------------------------------------------------------------
# get_block_sequence
# ---------------------------------------------------------------------------


def get_block_sequence(module: "Module") -> list:
    """module에서 블록 시퀀스를 전역 이름 규약으로 획득.

    children 순서대로 순회하며, BLOCK_SEQUENCE_NAMES에 있는 이름이면서
    ModuleList인 자식의 각 요소를 순서대로 이어 붙인 리스트를 반환.

    예: Flux의 transformer_blocks + single_transformer_blocks → 하나의 flat 리스트.

    Args:
        module: 트랜스포머 모듈 (예: FluxTransformer2DModel).

    Returns:
        [첫블록, ..., 마지막블록] 형태의 블록 리스트.
    """
    blocks: list = []
    for name, submodule in module.children:
        if name not in BLOCK_SEQUENCE_NAMES:
            continue
        if not isinstance(submodule, ModuleList):
            continue
        for block in submodule:
            blocks.append(block)
    return blocks
