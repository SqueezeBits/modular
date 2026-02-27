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

"""Qwen3 text encoder ComponentModel wrapper.

This module provides a ComponentModel wrapper for Qwen3 text encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import Qwen3TextEncoderConfig
from .qwen3 import Qwen3TextEncoderTransformer


class Qwen3TextEncoderModel(ComponentModel):
    """Qwen3 text encoder ComponentModel wrapper."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize Qwen3TextEncoderModel.

        Args:
            config: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        super().__init__(config, encoding, devices, weights)
        self.config = Qwen3TextEncoderConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Load and compile the Qwen3 text encoder.

        Returns:
            Compiled model callable.
        """
        state_dict = {}
        for key, value in self.weights.items():
            adapted_key = key
            # Diffusers Qwen text encoder shards are usually nested under `model.*`.
            # The MAX text encoder expects top-level names (`layers.*`, `embed_tokens.*`).
            if adapted_key.startswith("model."):
                adapted_key = adapted_key.removeprefix("model.")
            elif adapted_key.startswith("language_model."):
                adapted_key = adapted_key.removeprefix("language_model.")

            state_dict[adapted_key] = value.data()

        with F.lazy():
            model = Qwen3TextEncoderTransformer(self.config)
            model.to(self.devices[0])

        self.model = model.compile(*model.input_types(), weights=state_dict)
        return self.model

    def __call__(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        hidden_state_index: int | None = None,
    ):
        if tokens.rank == 2:
            if int(tokens.shape[0]) != 1:
                raise ValueError(
                    "Qwen3TextEncoderModel expects batch_size=1 for 2D token input."
                )
            tokens = tokens[0]

        if attention_mask is not None:
            if attention_mask.rank == 2:
                if int(attention_mask.shape[0]) != 1:
                    raise ValueError(
                        "Qwen3TextEncoderModel expects batch_size=1 for 2D attention_mask input."
                    )
                attention_mask = attention_mask[0]

            if int(attention_mask.shape[0]) != int(tokens.shape[0]):
                raise ValueError(
                    "attention_mask length must match tokens length. "
                    f"Got mask={attention_mask.shape[0]}, tokens={tokens.shape[0]}."
                )

            mask_np = np.from_dlpack(attention_mask.cast(DType.bool).to(CPU()))
            if mask_np.ndim != 1:
                raise ValueError(
                    f"attention_mask must be rank-1 after squeeze, got rank={mask_np.ndim}."
                )
            if not np.any(mask_np):
                raise ValueError("attention_mask masks out all tokens.")

            if not np.all(mask_np):
                tokens_np = np.from_dlpack(tokens.to(CPU()))
                tokens_np = tokens_np[mask_np]
                tokens = Tensor.constant(
                    tokens_np.astype(np.int64, copy=False),
                    dtype=DType.int64,
                    device=self.devices[0],
                )

        outputs = self.model(tokens)
        if isinstance(outputs, list):
            outputs = tuple(outputs)

        if hidden_state_index is None:
            return outputs

        if not isinstance(outputs, tuple):
            raise ValueError(
                "`hidden_state_index` requires model outputs to be tuple/list "
                f"of hidden states, got {type(outputs).__name__}."
            )

        num_layers = len(outputs)
        if hidden_state_index < -num_layers or hidden_state_index >= num_layers:
            raise ValueError(
                f"`hidden_state_index` out of range: {hidden_state_index}. "
                f"Valid range is [{-num_layers}, {num_layers - 1}]."
            )

        return outputs[hidden_state_index]
