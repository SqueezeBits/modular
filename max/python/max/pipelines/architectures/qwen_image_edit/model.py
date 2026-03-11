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

"""QwenImage Edit transformer model with lazy graph compilation.

The edit model requires ``zero_cond_t`` support where ``num_noise_tokens``
must be baked into the compiled graph (static slice).  Because the value is
only known at the first ``execute()`` call, the graph is compiled lazily.
"""

from collections.abc import Callable
from typing import Any

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from ..qwen_image.model_config import QwenImageConfig
from ..qwen_image.qwen_image import QwenImageTransformer2DModel


class QwenImageEditTransformerModel(ComponentModel):
    """Edit-specific transformer that lazily compiles the graph.

    On the first ``__call__`` the ``num_noise_tokens`` value is captured and
    the graph is compiled with the correct static split size for
    ``zero_cond_t`` modulation.
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.session = session
        self.config = QwenImageConfig.generate(config, encoding, devices)

        # Prepare weights + nn_model but do NOT compile the graph yet.
        state_dict = {key: value.data() for key, value in self.weights.items()}
        self._nn_model = QwenImageTransformer2DModel(self.config)
        self._nn_model.load_state_dict(
            state_dict, weight_alignment=1, strict=True
        )
        self._state_dict = self._nn_model.state_dict()

        # Compiled models keyed by num_noise_tokens
        self._compiled: dict[int, Model] = {}

    def _get_model(self, num_noise_tokens: int | None) -> Model:
        key = num_noise_tokens or 0
        if key not in self._compiled:
            self._nn_model.num_noise_tokens = num_noise_tokens
            with Graph(
                "qwen_image_edit_transformer",
                input_types=self._nn_model.input_types(),
            ) as graph:
                outputs = self._nn_model(
                    *(value.tensor for value in graph.inputs)
                )
                if isinstance(outputs, tuple):
                    graph.output(*outputs)
                else:
                    graph.output(outputs)
            self._compiled[key] = self.session.load(
                graph,
                weights_registry=self._state_dict,
            )
        return self._compiled[key]

    def load_model(self) -> Callable[..., Any]:
        # Satisfy the interface — actual compilation is lazy.
        return self.__call__

    def __call__(
        self,
        hidden_states: Buffer,
        encoder_hidden_states: Buffer,
        timestep: Buffer,
        img_ids: Buffer,
        txt_ids: Buffer,
        num_noise_tokens: int | None = None,
    ) -> Any:
        model = self._get_model(num_noise_tokens)
        return model.execute(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
        )
