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
from max.experimental.tensor import Tensor
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
import numpy as np

from .flux2 import Flux2Transformer2DModel
from .model_config import Flux2Config
from max.experimental.nn.common_layers.fp8_config_utils import (
    build_dynamic_block_fp8_config,
    build_legacy_scalar_fp8_config,
    validate_fp8_weight_scale_contract,
)
from .weight_adapters import (
    uses_legacy_scalar_fp8_scales,
)


class Flux2TransformerModel(ComponentModel):
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
        self.config = Flux2Config.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        requested_activation_scheme = getattr(
            self.config, "activation_scheme", None
        )
        if (
            requested_activation_scheme == "static"
            and not uses_legacy_scalar_fp8_scales(state_dict)
        ):
            raise ValueError(
                "flux2.transformer: activation_scheme='static' requested, "
                "but the loaded checkpoint does not provide legacy scalar "
                "input_scale/weight_scale tensors."
            )

        if (
            self.encoding == "float8_e4m3fn"
            and self.config.float8_config is None
            and requested_activation_scheme != "static"
        ):
            if any(
                key.endswith(".weight_scale")
                and len(getattr(value, "shape", ())) == 2
                for key, value in state_dict.items()
            ):
                self.config.float8_config = build_dynamic_block_fp8_config(
                    {
                        "quantization_config": {
                            "quant_method": "fp8",
                            "activation_scheme": "dynamic",
                            "weight_block_size": [128, 128],
                        }
                    },
                    component_name="flux2.transformer",
                )
        if (
            requested_activation_scheme is None
            and (
                self.config.float8_config is not None
                or self.encoding == "float8_e4m3fn"
            )
            and uses_legacy_scalar_fp8_scales(state_dict)
        ):
            if hasattr(self.config, "model_copy"):
                self.config = self.config.model_copy(
                    update={
                        "float8_config": build_legacy_scalar_fp8_config(
                            component_name="flux2.transformer"
                        )
                    }
                )
            else:
                self.config.float8_config = build_legacy_scalar_fp8_config(
                    component_name="flux2.transformer"
                )
        # Klein/distilled checkpoints can omit guidance embedder weights.
        has_guidance_embedder = any(
            "time_guidance_embed.guidance_embedder." in k for k in state_dict
        )
        if not has_guidance_embedder and getattr(
            self.config, "guidance_embeds", True
        ):
            if hasattr(self.config, "model_copy"):
                self.config = self.config.model_copy(
                    update={"guidance_embeds": False}
                )
            else:
                self.config.guidance_embeds = False
        if (
            self.config.float8_config is not None
            and self.config.float8_config.weight_scale.block_size is not None
        ):
            validate_fp8_weight_scale_contract(
                state_dict,
                float8_config=self.config.float8_config,
                component_name="flux2.transformer",
            )
        with F.lazy():
            flux = Flux2Transformer2DModel(self.config)
            flux.to(self.devices[0])
            for name, _ in flux.parameters:
                if name.endswith((".input_scale", ".weight_scale")) and name not in state_dict:
                    parameter = getattr(flux, name.split(".")[0], None)
                    shape = None
                    for param_name, tensor in flux.parameters:
                        if param_name == name:
                            shape = tuple(int(dim) for dim in tensor.shape)
                            break
                    if shape is None:
                        raise ValueError(
                            f"could not infer shape for missing scale parameter '{name}'"
                        )
                    default_scale_array = np.ones(
                        shape if shape else (), dtype=np.float32
                    )
                    state_dict[name] = WeightData.from_numpy(
                        default_scale_array, name
                    )
            flux.apply_to_parameters(
                lambda name, tensor: (
                    tensor.cast(state_dict[name].dtype)
                    if name in state_dict and tensor.dtype != state_dict[name].dtype
                    else tensor
                )
            )
        self.model = flux.compile(*flux.input_types(), weights=state_dict)
        return self.model

    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
    ) -> Any:
        return self.model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )
