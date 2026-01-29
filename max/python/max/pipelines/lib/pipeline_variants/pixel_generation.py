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
"""MAX pipeline for pixel generation using diffusion models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Generic

import numpy as np
from max.driver import load_devices
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    PipelineOutputsDict,
    PixelGenerationContextType,
    PixelGenerationInputs,
    PixelGenerationOutput,
    RequestID,
    TokenBuffer,
)

from ..interfaces.diffusion_pipeline import DiffusionPipeline, PixelModelInputs

from .utils import get_weight_paths

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


class PixelGenerationPipeline(
    Pipeline[
        PixelGenerationInputs[PixelGenerationContextType], PixelGenerationOutput
    ],
    Generic[PixelGenerationContextType],
):
    """Pixel generation pipeline for diffusion models."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[DiffusionPipeline],
    ) -> None:
        """Initialize a pixel generation pipeline instance.

        Args:
            pipeline_config: Configuration for the pipeline and runtime behavior.
        """
        from max.engine import InferenceSession  # local import to avoid cycles

        self._pipeline_config = pipeline_config
        model_config = pipeline_config.model
        self._devices = load_devices(pipeline_config.model.device_specs)

        # Initialize Session.
        session = InferenceSession(devices=self._devices)
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Download weights if required and get absolute weight paths.
        weight_paths: list[Path] = get_weight_paths(model_config)

        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=self._devices,
            weight_paths=weight_paths,
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    def execute(
        self,
        inputs: PixelGenerationInputs[PixelGenerationContextType],
    ) -> PipelineOutputsDict[PixelGenerationOutput]:
        model_inputs, flat_batch = self.prepare_batch(inputs.batch)
        if not flat_batch or model_inputs is None:
            return {}

        try:
            model_outputs = self._pipeline_model.execute(
                model_inputs=model_inputs
            )
        except Exception:
            batch_size = len(flat_batch)
            logger.error(
                "Encountered an exception while executing pixel batch: "
                "batch_size=%d, num_images_per_prompt=%s, height=%s, width=%s, "
                "num_inference_steps=%s",
                batch_size,
                model_inputs.num_images_per_prompt,
                model_inputs.height,
                model_inputs.width,
                model_inputs.num_inference_steps,
            )
            raise

        image_list = model_outputs.images
        num_images_per_prompt = model_inputs.num_images_per_prompt
        expected_images = len(flat_batch) * num_images_per_prompt
        if len(image_list) != expected_images:
            raise ValueError(
                "Unexpected number of images returned from pipeline: "
                f"expected {expected_images}, got {len(image_list)}."
            )

        responses: dict[RequestID, PixelGenerationOutput] = {}
        for index, (request_id, _context) in enumerate(flat_batch):
            offset = index * num_images_per_prompt
            # Stack images to preserve batch dimension (NCHW format)
            pixel_data = np.stack(
                image_list[offset : offset + num_images_per_prompt],
                axis=0,
            )
            pixel_data = pixel_data.astype(np.float32, copy=False)
            responses[request_id] = PixelGenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                pixel_data=pixel_data,
            )

        return responses

    def prepare_batch(
        self,
        batch: dict[RequestID, PixelGenerationContextType],
    ) -> tuple[
        PixelModelInputs | None,
        list[tuple[RequestID, PixelGenerationContextType]],
    ]:
        """Prepare batched model inputs for pixel generation execution.

        Converts a batch of PixelContext objects into PixelModelInputs by stacking
        tensors along the batch dimension. All contexts in the batch must have
        compatible dimensions (same height, width, num_inference_steps).

        Args:
            batch: Dictionary mapping request IDs to their PixelContext objects.

        Returns:
            A tuple of:
                - PixelModelInputs | None: Batched inputs ready for model execution,
                  or None if batch is empty.
                - list: Flattened batch as (request_id, context) tuples for
                  response mapping.

        Raises:
            ValueError: If contexts have incompatible dimensions.
        """
        # Handle empty batch
        if not batch:
            return None, []

        # Flatten batch to list of (request_id, context) tuples
        flat_batch = list(batch.items())

        # Extract first context as reference for validation
        _, first_ctx = flat_batch[0]

        # Validate all contexts have compatible dimensions
        for request_id, ctx in flat_batch[1:]:
            if ctx.height != first_ctx.height:
                raise ValueError(
                    f"All requests in batch must have same height. "
                    f"Request {request_id} has height={ctx.height}, "
                    f"expected {first_ctx.height}."
                )
            if ctx.width != first_ctx.width:
                raise ValueError(
                    f"All requests in batch must have same width. "
                    f"Request {request_id} has width={ctx.width}, "
                    f"expected {first_ctx.width}."
                )
            if ctx.num_inference_steps != first_ctx.num_inference_steps:
                raise ValueError(
                    f"All requests in batch must have same num_inference_steps. "
                    f"Request {request_id} has num_inference_steps={ctx.num_inference_steps}, "
                    f"expected {first_ctx.num_inference_steps}."
                )
            if ctx.num_images_per_prompt != first_ctx.num_images_per_prompt:
                raise ValueError(
                    f"All requests in batch must have same num_images_per_prompt. "
                    f"Request {request_id} has num_images_per_prompt={ctx.num_images_per_prompt}, "
                    f"expected {first_ctx.num_images_per_prompt}."
                )

        # Stack latents along batch dimension
        # Each context.latents has shape (num_images_per_prompt, C, H, W)
        batched_latents = np.concatenate(
            [ctx.latents for _, ctx in flat_batch], axis=0
        )

        # Stack latent_image_ids if present
        # Each has shape (seq_len, 3) - same for all contexts with same H, W
        batched_latent_image_ids = first_ctx.latent_image_ids

        # For tokens, TokenBuffer expects 1D arrays. The pipeline internally
        # expands to 2D when needed. For batch_size=1, pass tokens directly.
        # For batch_size>1, we need a different approach (not yet supported).
        if len(flat_batch) == 1:
            batched_tokens = first_ctx.tokens
            batched_tokens_2 = first_ctx.tokens_2
            batched_negative_tokens = first_ctx.negative_tokens
            batched_negative_tokens_2 = first_ctx.negative_tokens_2
        else:
            # For multiple requests, we need to handle token batching differently.
            # The FluxPipeline._prepare_prompt_embeddings expects tokens to be
            # expanded internally. For now, raise an error for batch_size > 1.
            raise NotImplementedError(
                "Batching multiple requests with different prompts is not yet "
                "supported for diffusion models. TokenBuffer requires 1D arrays "
                "but batching would require 2D. Consider processing requests "
                "sequentially or implementing custom token batching logic."
            )

        # Use timesteps and sigmas from first context (same for all with same num_inference_steps)
        timesteps = first_ctx.timesteps
        sigmas = first_ctx.sigmas

        # Build the model inputs
        model_inputs = PixelModelInputs(
            tokens=batched_tokens,
            tokens_2=batched_tokens_2,
            negative_tokens=batched_negative_tokens,
            negative_tokens_2=batched_negative_tokens_2,
            timesteps=timesteps,
            sigmas=sigmas,
            latents=batched_latents,
            latent_image_ids=batched_latent_image_ids,
            height=first_ctx.height,
            width=first_ctx.width,
            num_inference_steps=first_ctx.num_inference_steps,
            guidance_scale=first_ctx.guidance_scale,
            true_cfg_scale=first_ctx.true_cfg_scale,
            num_warmup_steps=first_ctx.num_warmup_steps,
            num_images_per_prompt=first_ctx.num_images_per_prompt,
        )

        return model_inputs, flat_batch

    def release(self, request_id: RequestID) -> None:
        """Release resources associated with a request.

        Args:
            request_id: The request ID to release resources for.
        """
        # TODO: Implement resource cleanup if needed
        pass
