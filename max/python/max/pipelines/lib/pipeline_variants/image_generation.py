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

from __future__ import annotations

from typing import TYPE_CHECKING

from max.interfaces import (
    ImageGenerationInputs,
    ImageGenerationOutput,
    Pipeline,
    RequestID,
)

from ..interfaces import DiffusionPipeline

if TYPE_CHECKING:
    from ..config import PipelineConfig


class ImageGenerationPipeline(
    Pipeline[ImageGenerationInputs, ImageGenerationOutput],
):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        diffusion_pipeline: type[DiffusionPipeline],
    ) -> None:
        self._diffusion_pipeline = diffusion_pipeline(
            pipeline_config,
        )

    def execute(self, inputs: ImageGenerationInputs) -> ImageGenerationOutput:
        outputs = self._diffusion_pipeline(
            prompt=inputs.prompt,
            height=inputs.height,
            width=inputs.width,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=inputs.guidance_scale,
            num_images_per_prompt=inputs.num_images_per_prompt,
        )
        return ImageGenerationOutput(images=outputs.images)

    def release(self, request_id: RequestID) -> None:
        pass
