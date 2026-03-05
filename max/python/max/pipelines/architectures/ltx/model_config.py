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
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from pydantic import Field


class LTXConfigBase(MAXModelConfigBase):
    in_channels: int = 128
    out_channels: int = 128
    patch_size: int = 1
    patch_size_t: int = 1
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    cross_attention_dim: int = 2048
    caption_channels: int = 4096
    num_layers: int = 28
    activation_fn: str = "gelu-approximate"
    qk_norm: str = "rms_norm_across_heads"
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6
    attention_bias: bool = True
    attention_out_bias: bool = True
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)


class LTXConfig(LTXConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> LTXConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in LTXConfigBase.__annotations__
        }
        init_dict.update(
            {
                # LTX T2V MVP is bf16-first for memory/perf parity with diffusers.
                "dtype": DType.bfloat16,
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return LTXConfigBase(**init_dict)
