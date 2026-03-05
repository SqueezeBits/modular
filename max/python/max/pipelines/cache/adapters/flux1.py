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

"""Flux1 model-specific step-cache miss adapter."""

from __future__ import annotations

from typing import Any

from max.experimental.tensor import Tensor


class Flux1StepCacheMissAdapter:
    """Compute Flux1 output for cache-miss branch."""

    def __init__(
        self,
        *,
        model: Any,
        new_hidden_states: Tensor,
        new_encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor] | None,
    ) -> None:
        self.model = model
        self.new_hidden_states = new_hidden_states
        self.new_encoder_hidden_states = new_encoder_hidden_states
        self.temb = temb
        self.image_rotary_emb = image_rotary_emb

    def compute_output(self) -> Tensor:
        """Return model output after running remaining blocks."""
        h = self.new_hidden_states
        enc = self.new_encoder_hidden_states

        for rem_block in self.model.transformer_blocks[1:]:
            enc, h = rem_block(
                hidden_states=h,
                encoder_hidden_states=enc,
                temb=self.temb,
                image_rotary_emb=self.image_rotary_emb,
            )

        for single_block in self.model.single_transformer_blocks:
            enc, h = single_block(
                hidden_states=h,
                encoder_hidden_states=enc,
                temb=self.temb,
                image_rotary_emb=self.image_rotary_emb,
            )

        h = self.model.norm_out(h, self.temb)
        return self.model.proj_out(h)
