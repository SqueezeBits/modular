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
First Block Cache (FBC) — diffusers 스타일 범용 래퍼.

- 블록 발견: get_block_sequence() (이름 규약)
- 블록 실행: TransformerBlockRegistry 메타데이터로 범용 처리
- embed/tail: TransformerModelRegistry (모델별 최소 코드)
- F.cond: MAX 그래프 컴파일 호환
"""

import logging
from collections.abc import Sequence

from max import functional as F
from max.graph import TensorValue
from max.nn import Module
from max.tensor import Tensor

from .first_block_cache import get_block_sequence, get_can_use_cache
from .registry import get_adapter, get_block_metadata

from max.graph.ops import print as debug_print
logger = logging.getLogger(__name__)


def _run_first_block_generic(transformer: Module, h, enc, temb, image_rotary_emb):
    """범용 첫 블록 실행. get_block_sequence + TransformerBlockRegistry 사용."""
    blocks = get_block_sequence(transformer)
    if not blocks:
        raise ValueError(f"No blocks found for {type(transformer).__name__}")
    head_block = blocks[0]
    output = head_block(
        hidden_states=h,
        encoder_hidden_states=enc,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
    )
    meta = get_block_metadata(head_block)
    enc_out = output[meta.return_encoder_hidden_states_index] if meta.return_encoder_hidden_states_index is not None else enc
    h_out = output[meta.return_hidden_states_index]
    return (enc_out, h_out)


def _run_remaining_blocks_and_tail_generic(
    transformer: Module,
    metadata,
    h,
    enc,
    temb,
    image_rotary_emb,
):
    """범용 나머지 블록 + tail 실행."""
    blocks = get_block_sequence(transformer)
    if len(blocks) < 2:
        return metadata.run_tail(transformer, h, temb)
    for block in blocks[1:]:
        output = block(
            hidden_states=h,
            encoder_hidden_states=enc,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        meta = get_block_metadata(block)
        if meta.return_encoder_hidden_states_index is not None:
            enc = output[meta.return_encoder_hidden_states_index]
        h = output[meta.return_hidden_states_index]
    return metadata.run_tail(transformer, h, temb)


class TransformerWithFBC(Module[..., Sequence[Tensor]]):
    """범용 FBC 래퍼. diffusers 스타일: 블록 레지스트리 + 모델 레지스트리."""

    def __init__(self, transformer: Module) -> None:
        super().__init__()
        self.transformer = transformer
        self._metadata = get_adapter(transformer)
        if self._metadata is None:
            raise ValueError(
                f"No FBC metadata registered for {type(transformer).__name__}. "
                "Register in lib/cache/architectures/ (e.g. register_flux)."
            )

    @property
    def config(self):
        """파이프라인 호환."""
        return getattr(self.transformer, "config", None)

    @property
    def devices(self) -> list:
        """파이프라인 호환."""
        return getattr(self.transformer, "devices", None)

    def input_types(self) -> tuple:
        """FBC forward용 input_types."""
        return self._metadata.input_types_fbc(self.transformer)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
        prev_residual: Tensor | None = None,
        prev_output: Tensor | None = None,
        rdt: float = 0.05,
    ) -> tuple[Tensor, Tensor]:
        """FBC forward. F.cond로 캐시 분기."""
        meta = self._metadata
        m = self.transformer

        h, enc, temb, image_rotary_emb = meta.run_embed(
            m,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )
        new_enc, new_h = _run_first_block_generic(m, h, enc, temb, image_rotary_emb)
        first_block_residual = new_h - h

        can_use_cache = get_can_use_cache(
            first_block_residual, prev_residual, rdt
        )

        types = meta.input_types_fbc(m)
        output_type = types[8]
        residual_type = types[7]

        def then_fn():
            return (TensorValue(prev_output), TensorValue(first_block_residual))

        def else_fn():
            out = _run_remaining_blocks_and_tail_generic(
                m, meta, new_h, new_enc, temb, image_rotary_emb
            )
            return (TensorValue(out), TensorValue(first_block_residual))

        result = F.cond(
            can_use_cache,
            [output_type, residual_type],
            then_fn,
            else_fn,
        )
        return (result[0], result[1])
