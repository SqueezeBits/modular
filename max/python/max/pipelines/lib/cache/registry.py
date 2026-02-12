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
First Block Cache (FBC) — diffusers 스타일 레지스트리.

- TransformerBlockRegistry: 블록 클래스 → 메타데이터 (return_hidden_states_index 등)
- TransformerModelRegistry: 모델 클래스 → embed_fn, tail_fn, input_types_fbc
- 새 모델 지원 시 블록/모델 메타데이터만 등록. 범용 로직은 lib/cache에 집중.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# TransformerBlockMetadata (diffusers TransformerBlockRegistry 참고)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformerBlockMetadata:
    """블록 forward 반환값 인덱스. 범용 블록 실행에 사용."""

    return_hidden_states_index: int
    """출력 튜플에서 hidden_states 위치. Flux: (enc, h) → 1."""

    return_encoder_hidden_states_index: int | None = None
    """출력 튜플에서 encoder_hidden_states 위치. Flux: (enc, h) → 0. 단일 출력이면 None."""


# ---------------------------------------------------------------------------
# TransformerBlockRegistry
# ---------------------------------------------------------------------------

TransformerBlockRegistry: dict[type, TransformerBlockMetadata] = {}
"""블록 클래스 → TransformerBlockMetadata. 새 블록 지원 시 register_block() 호출."""


def register_block(block_cls: type, metadata: TransformerBlockMetadata) -> None:
    """블록 클래스에 메타데이터 등록."""
    TransformerBlockRegistry[block_cls] = metadata
    logger.debug("Registered block metadata for %s", block_cls.__name__)


def get_block_metadata(block: Any) -> TransformerBlockMetadata:
    """블록 인스턴스에서 메타데이터 조회. 미등록 시 ValueError."""
    cls = type(block)
    if cls not in TransformerBlockRegistry:
        raise ValueError(
            f"No block metadata for {cls.__name__}. "
            f"Call register_block({cls.__name__}, ...) in lib/cache/architectures/."
        )
    return TransformerBlockRegistry[cls]


# ---------------------------------------------------------------------------
# 범용 input_types_fbc
# ---------------------------------------------------------------------------

# FBC 공통 규약: transformer.input_types() (7개) + prev_residual, prev_output
# 모델에 inner_dim, patch_size, out_channels(in_channels 대체) 필요
def build_input_types_fbc(transformer: Any) -> tuple[Any, ...]:
    """FBC forward용 input_types. 모델의 input_types() + prev_residual, prev_output.

    transformer에 필요: input_types(), max_dtype, max_device, inner_dim,
    patch_size, out_channels 또는 in_channels.
    """
    from max.graph import TensorType

    base = transformer.input_types()
    m = transformer
    out_dim = m.patch_size * m.patch_size * getattr(m, "out_channels", m.in_channels)
    prev_residual_type = TensorType(
        m.max_dtype,
        shape=["batch_size", "image_seq_len", m.inner_dim],
        device=m.max_device,
    )
    prev_output_type = TensorType(
        m.max_dtype,
        shape=["batch_size", "image_seq_len", out_dim],
        device=m.max_device,
    )
    return (*base, prev_residual_type, prev_output_type)


# ---------------------------------------------------------------------------
# TransformerModelMetadata (embed/tail — MAX 특성상 필요)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformerModelMetadata:
    """모델별 FBC 필수 부분. embed/tail은 구조가 달라 범용화 불가."""

    run_embed: Callable[..., tuple[Any, Any, Any, Any]]
    """(transformer, **forward_kwargs) -> (hidden_states, encoder_hidden_states, temb, image_rotary_emb)"""

    run_tail: Callable[..., Any]
    """(transformer, hidden_states, temb) -> output (최종 proj_out 등)"""

    input_types_fbc: Callable[[Any], tuple[Any, ...]]
    """(transformer) -> FBC forward용 input_types (prev_residual, prev_output 포함)."""


# ---------------------------------------------------------------------------
# TransformerModelRegistry (기존 FBC_REGISTRY 대체)
# ---------------------------------------------------------------------------

TransformerModelRegistry: dict[type, TransformerModelMetadata] = {}
"""모델 클래스 → TransformerModelMetadata."""


def register(transformer_cls: type[T], metadata: TransformerModelMetadata) -> None:
    """모델 클래스에 FBC 메타데이터 등록.

    Args:
        transformer_cls: 트랜스포머 클래스 (예: FluxTransformer2DModel).
        metadata: run_embed, run_tail, input_types_fbc.
    """
    TransformerModelRegistry[transformer_cls] = metadata
    logger.debug("Registered model metadata for %s", transformer_cls.__name__)


def get_adapter(transformer: Any) -> TransformerModelMetadata | None:
    """transformer에 등록된 메타데이터 조회."""
    return TransformerModelRegistry.get(type(transformer))


# ---------------------------------------------------------------------------
# 하위 호환: FBCAdapter (deprecated, TransformerModelMetadata로 통합)
# ---------------------------------------------------------------------------

FBCAdapter = TransformerModelMetadata  # type alias
FBC_REGISTRY = TransformerModelRegistry  # type alias
