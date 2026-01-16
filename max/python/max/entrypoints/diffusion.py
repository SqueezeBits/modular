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

"""High-level interface for image generation using diffusion models.

This module provides both programmatic and OpenAI-compatible API access
to diffusion-based image generation pipelines.

Example (Direct API):
    ```python
    from max.entrypoints.diffusion import ImageGenerator
    from max.pipelines import PipelineConfig

    config = PipelineConfig(model="black-forest-labs/FLUX.1-schnell")
    generator = ImageGenerator(config)

    images = generator.generate("A beautiful sunset over mountains")
    images[0].save("output.png")
    ```

Example (OpenAI-compatible API):
    ```python
    from max.entrypoints.diffusion import ImageGenerator
    from max.interfaces import ImageGenerationRequest

    generator = ImageGenerator(config)
    request = ImageGenerationRequest(
        prompt="A beautiful sunset",
        size="1024x1024",
        n=1,
    )
    response = generator.create(request)
    # response.data[0].b64_json contains the base64-encoded image
    ```
"""

from __future__ import annotations

import queue
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import TYPE_CHECKING

import tqdm
from PIL.Image import Image

from max.interfaces import (
    ImageGenerationInputs,
    ImageGenerationOutput,
    ImageGenerationRequest,
    ImageGenerationResponse,
    PipelineTask,
    RequestID,
)
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig

if TYPE_CHECKING:
    from max.pipelines.lib.pipeline_variants.image_generation import ImageGenerationPipeline


# ============================================================================
# Internal Request/Response Types
# ============================================================================


@dataclass
class _ImageRequest:
    """Internal request object for the image generation queue."""

    id: RequestID
    prompts: Sequence[str]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    num_images_per_prompt: int
    use_tqdm: bool


@dataclass
class _ImageResponse:
    """Internal response object from the image generation queue."""

    images: list[Image]


@dataclass
class _ThreadControl:
    """Thread synchronization primitives."""

    ready: Event = field(default_factory=Event)
    cancel: Event = field(default_factory=Event)


# ============================================================================
# Main ImageGenerator Class
# ============================================================================


class ImageGenerator:
    """High-level interface for generating images using diffusion models.

    This class provides a thread-safe interface for image generation with
    support for both direct API calls and OpenAI-compatible request/response.

    The generator runs a background worker thread that processes requests
    from a queue, allowing for concurrent request handling.

    Attributes:
        model_name: The name/path of the loaded model.
        pipeline_config: The configuration for the pipeline.
    """

    # Thread control and communication
    _thread_control: _ThreadControl
    _worker_thread: Thread
    _request_queue: queue.Queue[_ImageRequest]
    _pending_requests: dict[RequestID, queue.Queue[_ImageResponse]]

    # Configuration
    pipeline_config: PipelineConfig
    model_name: str

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        """Initialize the image generator.

        Args:
            pipeline_config: Configuration specifying the model and parameters.
        """
        self.pipeline_config = pipeline_config
        self.model_name = pipeline_config.model_config.model_path

        # Initialize thread control and queues
        self._thread_control = _ThreadControl()
        self._request_queue = queue.Queue()
        self._pending_requests = {}

        # Start background worker
        self._worker_thread = Thread(
            target=_run_worker,
            args=(
                self._thread_control,
                self.pipeline_config,
                self._request_queue,
                self._pending_requests,
            ),
            daemon=True,
        )
        self._worker_thread.start()

        # Wait for worker to be ready
        self._thread_control.ready.wait()

    def __del__(self) -> None:
        """Clean up resources."""
        self._thread_control.cancel.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    # ========================================================================
    # Public API: Direct Generation
    # ========================================================================

    def generate(
        self,
        prompts: str | Sequence[str],
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        use_tqdm: bool = True,
    ) -> list[Image]:
        """Generate images from text prompts.

        This method is thread-safe and can be called from multiple threads.

        Args:
            prompts: Single prompt string or sequence of prompts.
            height: Image height in pixels.
            width: Image width in pixels.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Number of images per prompt.
            use_tqdm: Show progress bar.

        Returns:
            List of generated PIL Images.

        Example:
            ```python
            images = generator.generate(
                "A cat sitting on a couch",
                height=1024,
                width=1024,
                num_inference_steps=30,
            )
            images[0].save("cat.png")
            ```
        """
        # Normalize prompts to sequence
        if isinstance(prompts, str):
            prompts = [prompts]

        # Create internal request
        request = _ImageRequest(
            id=RequestID(),
            prompts=prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            use_tqdm=use_tqdm,
        )

        # Submit request and wait for response
        return self._submit_and_wait(request)

    # ========================================================================
    # Public API: OpenAI-Compatible
    # ========================================================================

    def create(
        self,
        request: ImageGenerationRequest,
    ) -> ImageGenerationResponse:
        """Generate images using OpenAI-compatible request format.

        Args:
            request: OpenAI-compatible image generation request.

        Returns:
            OpenAI-compatible response with base64-encoded images.

        Example:
            ```python
            request = ImageGenerationRequest(
                prompt="A beautiful landscape",
                size="1024x1024",
                n=2,
                response_format="b64_json",
            )
            response = generator.create(request)
            print(f"Generated {len(response.data)} images")
            ```
        """
        # Parse dimensions from size string
        width, height = request.get_dimensions()

        # Generate images
        images = self.generate(
            prompts=request.prompt,
            height=height,
            width=width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.n or 1,
            use_tqdm=False,
        )

        # Convert to OpenAI response format
        output = ImageGenerationOutput(images=images)
        return ImageGenerationResponse.from_pipeline_output(
            output=output,
            response_format=request.response_format,
            output_format=request.get_output_format(),
            prompt=request.prompt,
        )

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _submit_and_wait(self, request: _ImageRequest) -> list[Image]:
        """Submit a request to the queue and wait for response."""
        response_queue: queue.Queue[_ImageResponse] = queue.Queue()
        self._pending_requests[request.id] = response_queue

        try:
            self._request_queue.put_nowait(request)
            response = response_queue.get()
            return response.images
        finally:
            self._pending_requests.pop(request.id, None)

    # ========================================================================
    # Class Methods
    # ========================================================================

    @classmethod
    def from_model(cls, model: str, **kwargs) -> ImageGenerator:
        """Create an ImageGenerator from a model identifier.

        Args:
            model: Model identifier (e.g., "black-forest-labs/FLUX.1-schnell").
            **kwargs: Additional PipelineConfig arguments.

        Returns:
            Configured ImageGenerator instance.

        Example:
            ```python
            generator = ImageGenerator.from_model(
                "black-forest-labs/FLUX.1-schnell"
            )
            ```
        """
        config = PipelineConfig(model=model, **kwargs)
        return cls(config)


# ============================================================================
# Legacy Alias (for backward compatibility)
# ============================================================================

DiffusionPipeline = ImageGenerator


# ============================================================================
# Background Worker
# ============================================================================


def _run_worker(
    thread_control: _ThreadControl,
    pipeline_config: PipelineConfig,
    request_queue: queue.Queue[_ImageRequest],
    pending_requests: Mapping[RequestID, queue.Queue[_ImageResponse]],
) -> None:
    """Background worker that processes image generation requests.

    This function runs in a separate thread and continuously processes
    requests from the queue until cancellation is signaled.
    """
    # Load the pipeline
    _, model_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config,
        task=PipelineTask.IMAGE_GENERATION,
    )
    pipeline: ImageGenerationPipeline = model_factory()

    # Signal that we're ready
    thread_control.ready.set()

    # Main processing loop
    while not thread_control.cancel.is_set():
        try:
            request = request_queue.get(timeout=0.3)
        except queue.Empty:
            continue

        # Process the request
        images = _process_request(pipeline, request)

        # Send response
        if response_queue := pending_requests.get(request.id):
            response_queue.put(_ImageResponse(images=images))


def _process_request(
    pipeline: ImageGenerationPipeline,
    request: _ImageRequest,
) -> list[Image]:
    """Process a single image generation request.

    Args:
        pipeline: The image generation pipeline.
        request: The request to process.

    Returns:
        List of generated images.
    """
    all_images: list[Image] = []

    # Create iterator with optional progress bar
    prompt_iter = request.prompts
    if request.use_tqdm:
        prompt_iter = tqdm.tqdm(prompt_iter, desc="Generating images")

    # Generate images for each prompt
    for prompt in prompt_iter:
        inputs = ImageGenerationInputs(
            prompt=prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images_per_prompt,
        )

        output: ImageGenerationOutput = pipeline.execute(inputs)
        all_images.extend(output.images)

    return all_images
