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
FBC Mixins: 모델/파이프라인에서 FBC 래핑·denoise 분기를 재사용.

- FBCModelMixin: ComponentModel에서 FBC 래핑 + use_fbc 설정 + state_dict prefix
- FBCDenoiseMixin: Pipeline에서 use_fbc 분기(7/9인자, 1/2-tuple) 공통 처리
"""

from typing import Any

from max.tensor import Tensor

from .registry import get_adapter
from .transformer_with_fbc import TransformerWithFBC


class FBCModelMixin:
    """ComponentModel에 FBC 래핑 로직 제공. load_model()에서 호출."""

    def _wrap_with_fbc_if_registered(
        self,
        transformer: Any,
    ) -> tuple[Any, bool, str]:
        """어댑터가 등록돼 있고 config.use_fbc이면 TransformerWithFBC로 래핑.

        config.use_fbc=False로 FBC 끄기 가능. use_fbc 없으면 True로 간주.

        Returns:
            (transformer, use_fbc, state_dict_prefix)
            - use_fbc True면 prefix "transformer.", 아니면 "".
        """
        adapter = get_adapter(transformer)
        use_fbc_config = getattr(self.config, "use_fbc", True)
        if adapter is None or not use_fbc_config:
            return transformer, False, ""
        wrapped = TransformerWithFBC(transformer)
        return wrapped, True, "transformer."


class FBCDenoiseMixin:
    """FBC 사용 시 denoise 루프의 7/9인자·1/2-tuple 분기 제공.

    transformer에 use_fbc 속성이 있어야 함 (FBCModelMixin 사용 시 설정됨).
    """

    @property
    def _use_fbc(self) -> bool:
        return getattr(self.transformer, "use_fbc", False)

    def _init_fbc_buffers(
        self,
        batch_size_int: int,
        image_seq_len: int,
        dtype: Any,
        dev: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """FBC용 prev_residual, prev_output, prev_neg_* 초기화. _use_fbc True일 때만 호출."""
        cfg = self.transformer.config
        inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
        out_dim = (
            cfg.patch_size
            * cfg.patch_size
            * (cfg.out_channels or cfg.in_channels)
        )
        prev_residual = Tensor.zeros(
            (batch_size_int, image_seq_len, inner_dim),
            dtype=dtype,
            device=dev,
        )
        prev_output = Tensor.zeros(
            (batch_size_int, image_seq_len, out_dim),
            dtype=dtype,
            device=dev,
        )
        prev_neg_residual = Tensor.zeros(
            (batch_size_int, image_seq_len, inner_dim),
            dtype=dtype,
            device=dev,
        )
        prev_neg_output = Tensor.zeros(
            (batch_size_int, image_seq_len, out_dim),
            dtype=dtype,
            device=dev,
        )
        return prev_residual, prev_output, prev_neg_residual, prev_neg_output

    def _call_transformer_step(
        self,
        latents: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
        prev_residual: Tensor | None = None,
        prev_output: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """한 스텝 transformer 호출. (noise_pred, new_residual) 반환. use_fbc=False면 new_residual=None."""
        if self._use_fbc:
            noise_pred, new_residual = self.transformer(
                latents,
                encoder_hidden_states,
                pooled_projections,
                timestep,
                img_ids,
                txt_ids,
                guidance,
                prev_residual,
                prev_output,
            )
            return noise_pred, new_residual
        (noise_pred,) = self.transformer(
            latents,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )
        return noise_pred, None
