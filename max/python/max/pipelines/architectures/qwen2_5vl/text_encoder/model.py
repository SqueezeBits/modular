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

"""Qwen2.5-VL text encoder ComponentModel wrapper for QwenImage pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.driver import Device
from max.experimental import functional as F
from max.graph.weights import Weights
from max.pipelines.architectures.llama3.weight_adapters import (
    LLAMA_SAFETENSOR_MAPPING as QWEN_SAFETENSOR_MAP,
)
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import Qwen25VLTextEncoderConfig
from .qwen25vl import Qwen25VLTextEncoderTransformer


class Qwen25VLTextEncoderModel(ComponentModel):
    """Qwen2.5-VL text encoder ComponentModel wrapper."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = Qwen25VLTextEncoderConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {}
        for key, value in self.weights.items():
            adapted_key = key
            # Strip "model.language_model." prefix first (Qwen2.5-VL stores
            # language model weights under this prefix), then fall back to
            # the generic "model." stripping.
            if adapted_key.startswith("model.language_model."):
                adapted_key = adapted_key[len("model.language_model."):]
            else:
                for before, after in QWEN_SAFETENSOR_MAP.items():
                    adapted_key = adapted_key.replace(before, after)

            state_dict[adapted_key] = value.data()

        with F.lazy():
            model = Qwen25VLTextEncoderTransformer(self.config)
            model.to(self.devices[0])

        self.model = model.compile(*model.input_types(), weights=state_dict)
        return self.model

    def __call__(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        if isinstance(outputs, list):
            return tuple(outputs)
        return outputs
