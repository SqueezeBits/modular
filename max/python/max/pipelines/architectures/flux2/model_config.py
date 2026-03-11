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

from typing import Any, Literal

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.float8_config import Float8Config
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from pydantic import Field

from max.experimental.nn.common_layers.fp8_config_utils import (
    build_dynamic_block_fp8_config,
    build_legacy_scalar_fp8_config,
)


class Flux2ConfigBase(MAXModelConfigBase):
    patch_size: int = 1
    in_channels: int = 128
    out_channels: int | None = None
    num_layers: int = 8
    num_single_layers: int = 48
    attention_head_dim: int = 128
    num_attention_heads: int = 48
    joint_attention_dim: int = 15360
    timestep_guidance_channels: int = 256
    mlp_ratio: float = 3.0
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: int = 2000
    eps: float = 1e-6
    guidance_embeds: bool = True
    """If False (Klein/distilled), no guidance embedder weights are expected."""
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)
    quantization_config: dict[str, Any] | None = None
    activation_scheme: Literal["static", "dynamic"] | None = None
    float8_config: Float8Config | None = None


class Flux2Config(Flux2ConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> Flux2ConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in Flux2ConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
                "quantization_config": config_dict.get("quantization_config"),
            }
        )
        if encoding == "float8_e4m3fn":
            activation_scheme = init_dict.get("activation_scheme")
            if activation_scheme == "static":
                init_dict["float8_config"] = build_legacy_scalar_fp8_config(
                    component_name="flux2.transformer"
                )
            else:
                try:
                    init_dict["float8_config"] = build_dynamic_block_fp8_config(
                        config_dict, component_name="flux2.transformer"
                    )
                except ValueError:
                    # Legacy FP8 checkpoints may not provide quantization metadata.
                    # Keep runtime in metadata-less fallback mode in this case.
                    init_dict["float8_config"] = None
        return Flux2ConfigBase(**init_dict)
