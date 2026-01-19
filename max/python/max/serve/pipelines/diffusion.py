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

"""Image generation pipeline for serving diffusion models."""

from __future__ import annotations

import logging
from typing import Any

from typing_extensions import Self


class ImageGeneratorPipeline:
    """Pipeline wrapper for image generation.

    This is a simplified pipeline for image generation that doesn't use
    the same streaming/batching infrastructure as text generation.
    """

    def __init__(
        self,
        model_name: str,
        pipeline_config: Any,
    ) -> None:
        self.model_name = model_name
        self.pipeline_config = pipeline_config
        self._generator: Any | None = None
        self.logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__qualname__
        )

    async def __aenter__(self) -> Self:
        """Initialize the image generator."""
        from max.entrypoints.diffusion import ImageGenerator

        self.logger.info("Loading image generator for model: %s", self.model_name)
        self._generator = ImageGenerator(self.pipeline_config)
        self.logger.info("Image generator loaded successfully")
        return self

    async def __aexit__(
        self, et: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> bool | None:
        """Clean up resources."""
        if self._generator is not None:
            del self._generator
            self._generator = None
        self.logger.info("Image generator pipeline closed: %s", self.model_name)
        return None

    @property
    def generator(self) -> Any:
        """Get the underlying image generator."""
        if self._generator is None:
            raise RuntimeError("Image generator not initialized. Use async with.")
        return self._generator
