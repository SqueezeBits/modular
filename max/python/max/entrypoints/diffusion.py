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

from max.interfaces import (
    ImageGenerationInputs,
    ImageGenerationOutput,
    PipelineTask,
)
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig


class DiffusionPipeline:
    def __init__(self, pipeline_config: PipelineConfig) -> None:
        self.pipeline_config = pipeline_config
        _, model_factory = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config,
            task=PipelineTask.IMAGE_GENERATION,
        )
        self.pipeline = model_factory()

    def __call__(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
    ) -> ImageGenerationOutput:
        # TODO: consider all possible diffusion tasks,
        # e.g. T2I, I2I, T2V, I2V, V2V.
        inputs = ImageGenerationInputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )
        pipeline_output: ImageGenerationOutput = self.pipeline.execute(inputs)
        return pipeline_output
