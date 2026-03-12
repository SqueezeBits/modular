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
"""MAX pipeline for pixel generation using diffusion models."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Generic

import numpy as np
from max.driver import load_devices
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    PipelineOutputsDict,
    PixelGenerationContextType,
    PixelGenerationInputs,
    RequestID,
)
from max.interfaces.generation import GenerationOutput
from max.interfaces.request.open_responses import OutputImageContent

from ..interfaces.diffusion_pipeline import (
    DiffusionPipeline,
    PixelModelInputs,
)
from .utils import get_weight_paths

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


class PixelGenerationPipeline(
    Pipeline[
        PixelGenerationInputs[PixelGenerationContextType], GenerationOutput
    ],
    Generic[PixelGenerationContextType],
):
    """Pixel generation pipeline for diffusion models.

    Args:
        pipeline_config: Configuration for the pipeline and runtime behavior.
        pipeline_model: The diffusion pipeline model class to instantiate.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[DiffusionPipeline],
    ) -> None:
        from max.engine import InferenceSession  # local import to avoid cycles

        init_start = time.perf_counter()
        self._pipeline_config = pipeline_config
        model_config = pipeline_config.model
        self._devices = load_devices(pipeline_config.model.device_specs)

        # Initialize Session.
        session_start = time.perf_counter()
        session = InferenceSession(devices=[*self._devices])

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)
        logger.info(
            "Initialized pixel-generation session in %.1f seconds",
            time.perf_counter() - session_start,
        )

        # Download weights if required and get absolute weight paths.
        weight_paths_start = time.perf_counter()
        weight_paths: list[Path] = get_weight_paths(model_config)
        logger.info(
            "Resolved %d pixel-generation weight files in %.1f seconds",
            len(weight_paths),
            time.perf_counter() - weight_paths_start,
        )

        model_start = time.perf_counter()
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=self._devices,
            weight_paths=weight_paths,
        )
        logger.info(
            "Initialized pixel-generation model %s in %.1f seconds",
            pipeline_model.__name__,
            time.perf_counter() - model_start,
        )
        logger.info(
            "Pixel-generation pipeline startup took %.1f seconds",
            time.perf_counter() - init_start,
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    def execute(
        self,
        inputs: PixelGenerationInputs[PixelGenerationContextType],
    ) -> PipelineOutputsDict[GenerationOutput]:
        """Runs the pixel generation pipeline for the given inputs."""
        pixel_outputs = self.execute_images(inputs)

        responses: dict[RequestID, GenerationOutput] = {}
        for request_id, pixel_data in pixel_outputs.items():
            responses[request_id] = GenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                output=[
                    OutputImageContent.from_numpy(img, format="png")
                    for img in pixel_data
                ],
            )

        return responses

    def execute_images(
        self,
        inputs: PixelGenerationInputs[PixelGenerationContextType],
    ) -> dict[RequestID, np.ndarray]:
        """Runs the model and returns uint8 image tensors grouped by request."""
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

        num_images_per_prompt = model_inputs.num_images_per_prompt
        expected_images = len(flat_batch) * num_images_per_prompt
        image_list = self._to_image_list(model_outputs.images, expected_images)

        responses: dict[RequestID, np.ndarray] = {}
        for index, (request_id, _context) in enumerate(flat_batch):
            offset = index * num_images_per_prompt
            responses[request_id] = np.stack(
                image_list[offset : offset + num_images_per_prompt], axis=0
            )
        return responses

    @staticmethod
    def _to_image_list(
        images: np.ndarray | list[np.ndarray],
        expected_images: int,
    ) -> list[np.ndarray]:
        """Normalize model outputs into NHWC uint8 images."""
        if isinstance(images, np.ndarray):
            if images.dtype == np.uint8:
                # Already NHWC uint8 [0, 255] from GPU post-processing.
                image_list = [images[i] for i in range(images.shape[0])]
            else:
                # images shape: (batch_size, H, W, C) or (batch_size, C, H, W)
                if images.ndim == 4 and images.shape[1] in (1, 3, 4):
                    images = np.transpose(images, (0, 2, 3, 1))
                images = np.clip(images * 0.5 + 0.5, 0.0, 1.0)
                images = (images * 255).astype(np.uint8)
                image_list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = [
                (
                    np.clip(
                        np.asarray(img, dtype=np.float32) * 0.5 + 0.5,
                        0.0,
                        1.0,
                    )
                    * 255
                ).astype(np.uint8)
                for img in images
            ]

        if len(image_list) != expected_images:
            raise ValueError(
                "Unexpected number of images returned from pipeline: "
                f"expected {expected_images}, got {len(image_list)}."
            )
        return image_list

    def prepare_batch(
        self,
        batch: dict[RequestID, PixelGenerationContextType],
    ) -> tuple[
        PixelModelInputs | None,
        list[tuple[RequestID, PixelGenerationContextType]],
    ]:
        """Prepare model inputs for pixel generation execution.

        Delegates to the pipeline model for model-specific input preparation.

        Args:
            batch: Dictionary mapping request IDs to their PixelContext objects.

        Returns:
            A tuple of:
                - PixelModelInputs | None: Inputs ready for model execution,
                  or None if batch is empty.
                - list: Flattened batch as (request_id, context) tuples for
                  response mapping.

        Raises:
            ValueError: If batch size is larger than 1 (not yet supported).
        """
        if not batch:
            return None, []

        # Flatten batch to list of (request_id, context) tuples
        flat_batch = list(batch.items())

        if len(flat_batch) > 1:
            raise ValueError(
                "Batching of different requests is not supported yet."
            )

        model_inputs = self._pipeline_model.prepare_inputs(flat_batch[0][1])
        return model_inputs, flat_batch

    def release(self, request_id: RequestID) -> None:
        """Release resources associated with a request.

        Args:
            request_id: The request ID to release resources for.
        """
        pass
