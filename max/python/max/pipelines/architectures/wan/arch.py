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

from dataclasses import dataclass

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import PixelContext
from max.pipelines.lib import (
    PixelGenerationTokenizer,
    SupportedArchitecture,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from typing_extensions import Self

from .pipeline_wan import WanPipeline
from .pipeline_wan_i2v import WanI2VPipeline


@dataclass(kw_only=True)
class WanArchConfig(ArchConfig):
    """Pipeline-level config for Wan (implements ArchConfig; no KV cache)."""

    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        return 0

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        if len(pipeline_config.model.device_specs) != 1:
            raise ValueError("Wan is only supported on a single device")
        return cls(pipeline_config=pipeline_config)


wan_arch = SupportedArchitecture(
    name="WanPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    ],
    pipeline_model=WanPipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
    config=WanArchConfig,
)

wan_i2v_arch = SupportedArchitecture(
    name="WanImageToVideoPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    ],
    pipeline_model=WanI2VPipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
    config=WanArchConfig,
)
