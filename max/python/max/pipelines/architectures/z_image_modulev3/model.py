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

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from max.driver import Buffer, Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import traced

if TYPE_CHECKING:
    from max.pipelines.lib.interfaces.cache_mixin import DenoisingCacheConfig

from .model_config import ZImageConfig
from .weight_adapters import convert_z_image_transformer_state_dict
from .z_image import ZImageTransformer2DModel


class ZImageTransformerModel(ComponentModel):
    """Component wrapper for the compiled Z-Image transformer graph."""

    model: Callable[..., Any]

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        *,
        cache_config: DenoisingCacheConfig | None = None,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
            cache_config=cache_config,
        )
        self.config = ZImageConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self._cached_default_ids: dict[tuple[int, int, int, str], Tensor] = {}
        self.load_model()

    @traced(message="ZImageTransformerModel.load_model")
    def load_model(self) -> None:
        target_dtype = self.config.dtype
        state_dict = {}
        for key, value in self.weights.items():
            weight = value.data()
            if weight.dtype != target_dtype:
                if weight.dtype.is_float() and target_dtype.is_float():
                    weight = weight.astype(target_dtype)
            state_dict[key] = weight
        state_dict = convert_z_image_transformer_state_dict(state_dict)

        with F.lazy():
            transformer = ZImageTransformer2DModel(
                self.config,
                cache_config=self.cache_config,
            )
            transformer.to(self.devices[0])

        self.model = transformer.compile(
            *transformer.input_types(),
            weights=state_dict,
        )

    def _default_ids(
        self,
        seq_len: int,
        axes: int,
        start_index: int,
        device: Device,
    ) -> Tensor:
        cache_key = (seq_len, axes, start_index, str(device))
        if cache_key in self._cached_default_ids:
            return self._cached_default_ids[cache_key]

        ids = np.zeros((seq_len, axes), dtype=np.int64)
        ids[:, 0] = np.arange(
            start_index, start_index + seq_len, dtype=np.int64
        )
        ids_tensor = Tensor(
            storage=Buffer.from_dlpack(np.ascontiguousarray(ids)).to(device)
        )
        self._cached_default_ids[cache_key] = ids_tensor
        return ids_tensor

    @traced(message="ZImageTransformerModel.__call__")
    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor | None = None,
        txt_ids: Tensor | None = None,
        prev_residual: Tensor | None = None,
        prev_output: Tensor | None = None,
        controlnet_block_samples: Tensor | None = None,
        siglip_feats: Tensor | None = None,
        image_noise_mask: Tensor | None = None,
    ) -> Any:
        if controlnet_block_samples is not None:
            raise NotImplementedError(
                "controlnet_block_samples is not supported in z-image phase 1"
            )
        if siglip_feats is not None or image_noise_mask is not None:
            raise NotImplementedError(
                "Omni(siglip/image_noise_mask) is not supported in z-image phase 1"
            )

        axes = len(self.config.axes_dims)
        txt_len = int(encoder_hidden_states.shape[1])
        img_len = int(hidden_states.shape[1])

        if txt_ids is None:
            txt_ids = self._default_ids(txt_len, axes, 1, hidden_states.device)
        if img_ids is None:
            img_ids = self._default_ids(
                img_len,
                axes,
                txt_len + 1,
                hidden_states.device,
            )

        model_args: tuple[Any, ...] = (
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
        )
        if (
            self.cache_config is not None
            and self.cache_config.first_block_caching
            and prev_residual is not None
            and prev_output is not None
        ):
            model_args = (*model_args, prev_residual, prev_output)
        return self.model(*model_args)
