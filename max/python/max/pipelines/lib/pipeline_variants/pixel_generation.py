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

from pathlib import Path
from typing import TYPE_CHECKING, Generic

from max.driver import load_devices
from max.interfaces import (
    Pipeline,
    PipelineOutputsDict,
    PixelGenerationContextType,
    PixelGenerationInputs,
    PixelGenerationOutput,
    RequestID,
)

from ..interfaces.diffusion_pipeline import DiffusionPipeline

from .utils import get_weight_paths

if TYPE_CHECKING:
    from ..config import PipelineConfig


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
        """Execute the pixel generation pipeline.

        Args:
            inputs: Batch of pixel generation contexts.

        Returns:
            Dictionary mapping request IDs to pixel generation outputs.
        """
        # TODO: Implement pixel generation execution
        raise NotImplementedError(
            "PixelGenerationPipeline.execute() is not yet implemented"
        )

    def release(self, request_id: RequestID) -> None:
        """Release resources associated with a request.

        Args:
            request_id: The request ID to release resources for.
        """
        # TODO: Implement resource cleanup if needed
        pass
