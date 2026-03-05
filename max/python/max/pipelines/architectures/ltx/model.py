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

from collections.abc import Callable
from typing import Any

from max.driver import Device
from max.experimental import functional as F
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .ltx import LTXVideoTransformer3DModel
from .model_config import LTXConfig
from .weight_adapters import convert_safetensor_state_dict


class LTXTransformer3DModel(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
        )
        self.config = LTXConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {}
        target_dtype = self.config.dtype
        for key, value in self.weights.items():
            weight_data = value.data()
            if (
                weight_data.dtype != target_dtype
                and weight_data.dtype.is_float()
                and target_dtype.is_float()
            ):
                weight_data = weight_data.astype(target_dtype)
            state_dict[key] = weight_data
        state_dict = convert_safetensor_state_dict(state_dict)
        with F.lazy():
            ltx = LTXVideoTransformer3DModel(self.config)
            ltx.to(self.devices[0])
        self.model = ltx.compile(*ltx.input_types(), weights=state_dict)
        return self.model

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        rotary_cos,
        rotary_sin,
    ) -> Any:
        return self.model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
            rotary_cos,
            rotary_sin,
        )
