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

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.engine import Model
from max.experimental import functional as F
from max.experimental.tensor import Tensor, default_dtype
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import T5Config, T5ConfigBase
from .t5 import T5EncoderModel
from .weight_adapters import convert_safetensor_state_dict


class T5Model(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config: T5ConfigBase = T5Config.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Model:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        state_dict = convert_safetensor_state_dict(state_dict)
        target_dtype = self.config.dtype
        for key, value in list(state_dict.items()):
            tensor = Tensor.from_dlpack(value)
            if tensor.dtype != target_dtype:
                state_dict[key] = tensor.cast(target_dtype)
        with F.lazy(), default_dtype(self.config.dtype):
            t5 = T5EncoderModel(self.config)  # type: ignore[arg-type]
            t5.to(self.devices[0])
        self.model: Model = t5.compile(*t5.input_types(), weights=state_dict)  # type: ignore[assignment]
        return self.model

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            input_ids = args[0]
            attention_mask = Tensor.ones(
                list(input_ids.shape),
                dtype=DType.bool,
                device=input_ids.device,
            )
            return self.model(input_ids, attention_mask)

        if len(args) == 2 and not kwargs:
            return self.model(args[0], args[1])

        if len(args) == 1 and set(kwargs.keys()) == {"attention_mask"}:
            return self.model(args[0], kwargs["attention_mask"])

        if not args and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                attention_mask = Tensor.ones(
                    list(input_ids.shape),
                    dtype=DType.bool,
                    device=input_ids.device,
                )
            return self.model(input_ids, attention_mask)

        return self.model(*args, **kwargs)
