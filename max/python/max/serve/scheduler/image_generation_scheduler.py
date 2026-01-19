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

"""Scheduler for image generation pipelines."""

import logging
import queue
from dataclasses import dataclass

from max.interfaces import (
    ImageGenerationContext,
    ImageGenerationInputs,
    ImageGenerationOutput,
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    Scheduler,
    SchedulerResult,
)
from max.pipelines.lib.pipeline_variants.image_generation import (
    ImageGenerationPipeline,
)
from max.profiler import traced

from .base import SchedulerProgress

logger = logging.getLogger("max.serve")


@dataclass
class ImageGenerationSchedulerConfig:
    """Image generation scheduler configuration."""

    # The maximum number of requests that can be in the batch.
    # For image generation, typically 1 since it's memory intensive.
    max_batch_size: int = 1


class ImageGenerationScheduler(Scheduler):
    """Scheduler for image generation requests.

    This scheduler handles image generation requests one at a time,
    as diffusion models are typically memory-intensive and don't
    benefit from batching in the same way as text generation.
    """

    def __init__(
        self,
        scheduler_config: ImageGenerationSchedulerConfig,
        pipeline: ImageGenerationPipeline,
        request_queue: MAXPullQueue[ImageGenerationContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[ImageGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        offload_queue_draining: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue
        # Note: offload_queue_draining is accepted for API compatibility
        # but not used since image generation is sequential.

    @traced
    def _get_next_request(self) -> ImageGenerationContext | None:
        """Get the next request from the queue."""
        try:
            return self.request_queue.get_nowait()
        except queue.Empty:
            return None

    def run_iteration(self) -> SchedulerProgress:
        """Process one image generation request.

        Returns:
            SchedulerProgress: Indicates whether work was performed.
        """
        request = self._get_next_request()
        if request is None:
            return SchedulerProgress.NO_PROGRESS

        self._execute_request(request)
        return SchedulerProgress.MADE_PROGRESS

    @traced
    def _execute_request(self, request: ImageGenerationContext) -> None:
        """Execute a single image generation request."""
        try:
            # Create inputs from context
            inputs = ImageGenerationInputs(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                num_images_per_prompt=request.num_images_per_prompt,
            )

            # Execute the pipeline
            output: ImageGenerationOutput = self.pipeline.execute(inputs)

            # Send the response
            self.response_queue.put_nowait(
                {request.request_id: SchedulerResult.create(output)}
            )
        except Exception as e:
            logger.error(f"Error executing image generation: {e}")
            # Send cancelled response on error
            self.response_queue.put_nowait(
                {request.request_id: SchedulerResult.cancelled()}
            )
