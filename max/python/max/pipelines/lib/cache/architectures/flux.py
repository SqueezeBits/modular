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
Flux FBC 등록. diffusers _helpers.py의 _register_transformer_blocks_metadata() 스타일.

모델 파일(flux1.py)에는 FBC 코드 없음. 여기서 중앙 집중 등록.
"""

from max import functional as F
from max.tensor import Tensor

from ..registry import (
    TransformerBlockMetadata,
    TransformerModelMetadata,
    build_input_types_fbc,
    register,
    register_block,
)


def _run_embed_flux(
    transformer,
    hidden_states,
    encoder_hidden_states,
    pooled_projections,
    timestep,
    img_ids,
    txt_ids,
    guidance=None,
):
    from max.pipelines.architectures.flux1.layers.embeddings import (
        CombinedTimestepGuidanceTextProjEmbeddings,
        CombinedTimestepTextProjEmbeddings,
    )

    m = transformer
    hidden_states = m.x_embedder(hidden_states)
    timestep = F.cast(timestep, hidden_states.dtype)
    timestep = timestep * 1000.0
    if guidance is not None:
        guidance = F.cast(guidance, hidden_states.dtype) * 1000.0
    if m.guidance_embeds:
        assert isinstance(m.time_text_embed, CombinedTimestepGuidanceTextProjEmbeddings)
        assert isinstance(guidance, Tensor)
        temb = m.time_text_embed(timestep, guidance, pooled_projections)
    else:
        assert isinstance(m.time_text_embed, CombinedTimestepTextProjEmbeddings)
        temb = m.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = m.context_embedder(encoder_hidden_states)
    ids = F.concat((txt_ids, img_ids), axis=0)
    image_rotary_emb = m.pos_embed(ids)
    return (hidden_states, encoder_hidden_states, temb, image_rotary_emb)


def _run_tail_flux(transformer, hidden_states, temb):
    return transformer.proj_out(transformer.norm_out(hidden_states, temb))


def register_flux() -> None:
    """Flux 블록/모델 FBC 등록. diffusers TransformerBlockRegistry 스타일."""
    from max.pipelines.architectures.flux1.flux1 import (
        FluxTransformer2DModel,
        FluxSingleTransformerBlock,
        FluxTransformerBlock,
    )

    # Flux blocks: (encoder_hidden_states, hidden_states) 반환 → index 0=enc, 1=h
    register_block(
        FluxTransformerBlock,
        TransformerBlockMetadata(
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
        ),
    )
    register_block(
        FluxSingleTransformerBlock,
        TransformerBlockMetadata(
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
        ),
    )

    register(
        FluxTransformer2DModel,
        TransformerModelMetadata(
            run_embed=_run_embed_flux,
            run_tail=_run_tail_flux,
            input_types_fbc=build_input_types_fbc,
        ),
    )
