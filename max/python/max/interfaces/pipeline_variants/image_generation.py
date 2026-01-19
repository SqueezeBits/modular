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

"""OpenAI-compatible image generation request/response models.

This module provides dataclasses that map to the OpenAI /v1/images/generations
API schema for seamless integration with OpenAI-compatible clients.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from max.interfaces.context import BaseContext
from max.interfaces.pipeline import PipelineInputs
from max.interfaces.request import RequestID
from PIL.Image import Image


@runtime_checkable
class ImageGenerationContext(BaseContext, Protocol):
    """Protocol defining the interface for image generation contexts.

    An ``ImageGenerationContext`` represents model inputs for image generation
    pipelines, managing the state and parameters needed for generating images
    from text prompts using diffusion models.
    """

    @property
    def request_id(self) -> RequestID:
        """Unique identifier for this request."""
        ...

    @property
    def prompt(self) -> str:
        """The text prompt for image generation."""
        ...

    @property
    def height(self) -> int:
        """The height of the generated image in pixels."""
        ...

    @property
    def width(self) -> int:
        """The width of the generated image in pixels."""
        ...

    @property
    def num_inference_steps(self) -> int:
        """Number of denoising steps."""
        ...

    @property
    def guidance_scale(self) -> float:
        """Classifier-free guidance scale."""
        ...

    @property
    def num_images_per_prompt(self) -> int:
        """Number of images to generate per prompt."""
        ...


ImageGenerationContextType = TypeVar(
    "ImageGenerationContextType", bound=ImageGenerationContext
)


# Default image generation parameters
DEFAULT_SIZE = "1024x1024"
DEFAULT_NUM_IMAGES = 1
DEFAULT_RESPONSE_FORMAT = "b64_json"
DEFAULT_QUALITY = "standard"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 3.5


@dataclass(eq=True)
class ImageGenerationInputs(PipelineInputs):
    """Inputs for image-generation pipelines."""

    prompt: str
    negative_prompt: str | None
    true_cfg_scale: float
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    num_images_per_prompt: int


@dataclass(kw_only=True)
class ImageGenerationOutput:
    """Output container for generated images."""

    images: list[Image]
    """List of generated images."""

    @property
    def is_done(self) -> bool:
        """Indicates whether image generation is complete.

        Returns:
            bool: Always True, as image generation is a single-step operation.
        """
        return True


@dataclass
class ImageGenerationRequest:
    """OpenAI-compatible image generation request.

    This maps to the OpenAI /v1/images/generations API schema.
    See: https://platform.openai.com/docs/api-reference/images/create
    """

    # Required field
    prompt: str
    """A text description of the desired image(s). Required."""

    # OpenAI standard fields
    model: str | None = None
    """The model to use for image generation."""

    n: int | None = DEFAULT_NUM_IMAGES
    """The number of images to generate. Must be between 1 and 10."""

    quality: str | None = DEFAULT_QUALITY
    """The quality of the image (e.g., 'standard', 'hd', 'high', 'medium', 'low')."""

    response_format: Literal["url", "b64_json"] | None = DEFAULT_RESPONSE_FORMAT
    """The format in which generated images are returned."""

    size: str | None = DEFAULT_SIZE
    """The size of the generated images (e.g., '1024x1024')."""

    style: str | None = None
    """The style of the generated images (e.g., 'vivid', 'natural')."""

    user: str | None = None
    """A unique identifier representing your end-user."""

    # Extended fields for GPT image models
    background: str | None = None
    """Background transparency ('transparent', 'opaque', 'auto')."""

    moderation: str | None = None
    """Content-moderation level ('low', 'auto')."""

    output_compression: int | None = None
    """Compression level (0-100%)."""

    output_format: str | None = DEFAULT_OUTPUT_FORMAT
    """Output format ('png', 'jpeg', 'webp')."""

    partial_images: int | None = None
    """Number of partial images for streaming (0-3)."""

    stream: bool | None = None
    """Generate in streaming mode."""

    # Extended parameters for diffusion models (not in OpenAI spec)
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    """Number of denoising steps. Extension for diffusion models."""

    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    """Classifier-free guidance scale. Extension for diffusion models."""

    seed: int | None = None
    """Random seed for reproducibility."""

    def to_pipeline_inputs(self) -> ImageGenerationInputs:
        """Convert OpenAI request to pipeline-native inputs."""
        width, height = self.get_dimensions()
        return ImageGenerationInputs(
            prompt=self.prompt,
            height=height,
            width=width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=self.n or 1,
        )

    def get_dimensions(self) -> tuple[int, int]:
        """Parse size string and return (width, height) tuple."""
        size = self.size or DEFAULT_SIZE

        # Handle 'auto' as default
        if size == "auto":
            size = DEFAULT_SIZE

        # Parse WIDTHxHEIGHT format
        try:
            parts = size.lower().split("x")
            if len(parts) == 2:
                width, height = int(parts[0]), int(parts[1])
                if width >= 64 and height >= 64:
                    return width, height
        except (ValueError, IndexError):
            pass

        raise ValueError(
            f"Invalid size '{size}'. Use format 'WIDTHxHEIGHT' (e.g., '1024x1024')."
        )

    def get_output_format(self) -> str:
        """Get the output image format."""
        return self.output_format or DEFAULT_OUTPUT_FORMAT


@dataclass
class ImageData:
    """Individual image data in the response."""

    b64_json: str | None = None
    """The base64-encoded image data."""

    url: str | None = None
    """The URL of the generated image (valid for 60 minutes)."""

    revised_prompt: str | None = None
    """The prompt that was used to generate the image, if revised."""


@dataclass
class InputTokensDetails:
    """Details about input token usage."""

    text_tokens: int = 0
    """Number of text tokens in the input."""

    image_tokens: int = 0
    """Number of image tokens in the input."""


@dataclass
class ImageGenerationUsage:
    """Token usage statistics for image generation."""

    total_tokens: int = 0
    """Total number of tokens used."""

    input_tokens: int = 0
    """Number of input tokens."""

    output_tokens: int = 0
    """Number of output tokens."""

    input_tokens_details: InputTokensDetails | None = None
    """Detailed breakdown of input tokens."""


@dataclass
class ImageGenerationResponse:
    """OpenAI-compatible image generation response.

    This maps to the OpenAI ImagesResponse schema.
    See: https://platform.openai.com/docs/api-reference/images/object
    """

    created: int
    """Unix timestamp when the response was created."""

    data: list[ImageData] = field(default_factory=list)
    """List of generated image data."""

    usage: ImageGenerationUsage | None = None
    """Token usage statistics."""

    @classmethod
    def from_pipeline_output(
        cls,
        output: ImageGenerationOutput,
        response_format: Literal["url", "b64_json"] | None = "b64_json",
        output_format: str = "png",
        prompt: str | None = None,
    ) -> ImageGenerationResponse:
        """Convert pipeline output to OpenAI-compatible response.

        Args:
            output: The raw pipeline output containing PIL images.
            response_format: The desired response format ('url' or 'b64_json').
            output_format: The image format ('png', 'jpeg', 'webp').
            prompt: The original prompt, included as revised_prompt if provided.

        Returns:
            An OpenAI-compatible ImageGenerationResponse.
        """
        data: list[ImageData] = []
        fmt = response_format or "b64_json"

        # Map output_format to PIL format
        pil_format_map = {
            "png": "PNG",
            "jpeg": "JPEG",
            "webp": "WEBP",
        }
        pil_format = pil_format_map.get(output_format.lower(), "PNG")

        for image in output.images:
            image_data = ImageData(revised_prompt=prompt)

            if fmt == "b64_json":
                buffer = io.BytesIO()
                image.save(buffer, format=pil_format)
                buffer.seek(0)
                image_data.b64_json = base64.b64encode(buffer.read()).decode(
                    "utf-8"
                )
            elif fmt == "url":
                raise ValueError(
                    "response_format='url' requires external storage. "
                    "Use 'b64_json' for local image generation."
                )

            data.append(image_data)

        return cls(
            created=int(time.time()),
            data=data,
            usage=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "created": self.created,
            "data": [
                {
                    k: v
                    for k, v in {
                        "b64_json": img.b64_json,
                        "url": img.url,
                        "revised_prompt": img.revised_prompt,
                    }.items()
                    if v is not None
                }
                for img in self.data
            ],
        }

        if self.usage is not None:
            usage_dict: dict[str, Any] = {
                "total_tokens": self.usage.total_tokens,
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            }
            if self.usage.input_tokens_details is not None:
                usage_dict["input_tokens_details"] = {
                    "text_tokens": self.usage.input_tokens_details.text_tokens,
                    "image_tokens": self.usage.input_tokens_details.image_tokens,
                }
            result["usage"] = usage_dict

        return result
