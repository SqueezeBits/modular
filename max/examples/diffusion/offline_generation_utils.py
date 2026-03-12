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

"""Shared helpers for offline diffusion example entrypoints."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from max.driver import DeviceSpec
from max.interfaces import PipelineTask, PixelGenerationInputs, RequestID
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    InputImageContent,
    InputTextContent,
    OpenResponsesRequestBody,
    OutputImageContent,
    UserMessage,
)
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)
from PIL import Image

QWEN_IMAGE_ARCH_NAMES = {
    "QwenImagePipeline",
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
}
QWEN_IMAGE_EDIT_ARCH_NAMES = {
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
}
QWEN_DEFAULT_GUIDANCE_SCALE = 1.0
QWEN_DEFAULT_TRUE_CFG_SCALE = 4.0


def resolve_output_path(output_path: str) -> Path:
    """Resolve relative output paths against the workspace when available."""
    resolved_output_path = Path(output_path)
    if resolved_output_path.is_absolute():
        return resolved_output_path

    if workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        return Path(workspace_dir) / resolved_output_path

    return resolved_output_path


def save_image(image_data: str, output_path: str) -> None:
    """Save a base64-encoded image to disk."""
    resolved_output_path = resolve_output_path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    if resolved_output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
        image.save(resolved_output_path, quality=95, optimize=True)
    else:
        image.save(resolved_output_path)
    print(f"Image saved to: {resolved_output_path}")


def save_generation_output(output: Any, output_path: str) -> list[str]:
    """Persist generated images from a postprocessed diffusion output."""
    if not output.output:
        raise ValueError("No images generated")

    saved_paths: list[str] = []
    for idx, image_content in enumerate(output.output):
        if not isinstance(image_content, OutputImageContent):
            raise TypeError(
                f"Expected OutputImageContent, got {type(image_content)}"
            )

        if len(output.output) > 1:
            base_name, ext = os.path.splitext(output_path)
            resolved_output_path = f"{base_name}_{idx}{ext}"
        else:
            resolved_output_path = output_path

        if image_content.image_data:
            save_image(image_content.image_data, resolved_output_path)
            saved_paths.append(str(resolve_output_path(resolved_output_path)))
        elif image_content.image_url:
            print(f"Image available at URL: {image_content.image_url}")

    return saved_paths


def load_image_as_data_uri(image_path: str | None) -> str | None:
    """Load an image from disk and convert it to a base64 data URI."""
    if image_path is None:
        return None

    resolved_image_path = Path(image_path)
    if not resolved_image_path.exists() and (
        workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY")
    ):
        candidate = Path(workspace_dir) / image_path
        if candidate.exists():
            resolved_image_path = candidate

    image = Image.open(resolved_image_path)
    buffer = BytesIO()
    image_format = image.format or "PNG"
    image.save(buffer, format=image_format)
    image_bytes = buffer.getvalue()
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = f"image/{image_format.lower()}"
    return f"data:{mime_type};base64,{base64_data}"


def load_input_image_data_uris(
    image_paths: list[str] | None,
) -> list[str]:
    """Convert optional input image paths to data URIs."""
    image_data_uris: list[str] = []
    for image_path in image_paths or []:
        if (uri := load_image_as_data_uri(image_path)) is not None:
            image_data_uris.append(uri)
    return image_data_uris


def build_pipeline_and_tokenizer(
    model_path: str,
    *,
    max_length: int | None = None,
    secondary_max_length: int | None = None,
) -> tuple[
    PipelineConfig,
    Any,
    PixelGenerationTokenizer,
    PixelGenerationPipeline[PixelContext],
]:
    """Build one tokenizer/pipeline pair for repeated same-process execution."""
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=model_path,
            device_specs=[DeviceSpec.accelerator()],
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        prefer_module_v3=config.runtime.prefer_module_v3,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, (
        "No matching diffusion architecture found for the provided model."
    )

    has_tokenizer_2 = False
    diffusers_config = config.model.diffusers_config
    resolved_max_length = max_length
    resolved_secondary_max_length = secondary_max_length
    if (
        resolved_max_length is None
        and diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and (components_config.get("tokenizer", None) is not None)
    ):
        resolved_max_length = components_config["tokenizer"]["config_dict"].get(
            "model_max_length", None
        )
        if arch.name in (
            "Flux2Pipeline_ModuleV3",
            "Flux2KleinPipeline_ModuleV3",
        ):
            resolved_max_length = 512
        elif arch.name in QWEN_IMAGE_ARCH_NAMES:
            resolved_max_length = 512
        print(f"Using max length: {resolved_max_length} for tokenizer")

    if (
        resolved_secondary_max_length is None
        and diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and (components_config.get("tokenizer_2", None) is not None)
    ):
        has_tokenizer_2 = True
        resolved_secondary_max_length = components_config["tokenizer_2"][
            "config_dict"
        ].get("model_max_length", None)
        print(
            "Using secondary max length: "
            f"{resolved_secondary_max_length} for tokenizer_2"
        )

    tokenizer = PixelGenerationTokenizer(
        model_path=model_path,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=resolved_max_length,
        subfolder_2="tokenizer_2" if has_tokenizer_2 else None,
        secondary_max_length=(
            resolved_secondary_max_length if has_tokenizer_2 else None
        ),
    )

    if not issubclass(arch.pipeline_model, DiffusionPipeline):
        raise TypeError(
            "Selected architecture does not implement DiffusionPipeline: "
            f"{arch.pipeline_model}"
        )

    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=cast(type[DiffusionPipeline], arch.pipeline_model),
    )
    return config, arch, tokenizer, pipeline


def resolve_guidance_scales(
    *,
    arch_name: str,
    negative_prompt: str | None,
    guidance_scale: float | None,
    true_cfg_scale: float | None,
) -> tuple[float, float]:
    """Resolve diffusion guidance defaults for the selected architecture."""
    is_qwen_image_family = arch_name in QWEN_IMAGE_ARCH_NAMES

    resolved_guidance_scale = guidance_scale
    if resolved_guidance_scale is None:
        resolved_guidance_scale = (
            QWEN_DEFAULT_GUIDANCE_SCALE if is_qwen_image_family else 3.5
        )

    resolved_true_cfg_scale = true_cfg_scale
    if resolved_true_cfg_scale is None:
        if is_qwen_image_family and negative_prompt is not None:
            resolved_true_cfg_scale = QWEN_DEFAULT_TRUE_CFG_SCALE
        else:
            resolved_true_cfg_scale = 1.0

    return resolved_guidance_scale, resolved_true_cfg_scale


def build_generation_request(
    *,
    arch_name: str,
    model_path: str,
    prompt: str,
    negative_prompt: str | None,
    width: int | None,
    height: int | None,
    num_inference_steps: int,
    guidance_scale: float | None,
    true_cfg_scale: float | None,
    seed: int | None,
    input_image_data_uris: list[str] | None = None,
) -> tuple[OpenResponsesRequest, float, float]:
    """Build one OpenResponsesRequest for a single diffusion run."""
    resolved_guidance_scale, resolved_true_cfg_scale = resolve_guidance_scales(
        arch_name=arch_name,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
    )

    if input_image_data_uris:
        content: list[InputImageContent | InputTextContent] = [
            InputImageContent(type="input_image", image_url=uri)
            for uri in input_image_data_uris
        ]
        content.append(InputTextContent(type="input_text", text=prompt))
        request_body = OpenResponsesRequestBody(
            model=model_path,
            input=[UserMessage(role="user", content=content)],
            seed=seed,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    steps=num_inference_steps,
                    guidance_scale=resolved_guidance_scale,
                    true_cfg_scale=resolved_true_cfg_scale,
                )
            ),
        )
    else:
        request_body = OpenResponsesRequestBody(
            model=model_path,
            input=prompt,
            seed=seed,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    steps=num_inference_steps,
                    guidance_scale=resolved_guidance_scale,
                    true_cfg_scale=resolved_true_cfg_scale,
                )
            ),
        )

    return (
        OpenResponsesRequest(request_id=RequestID(), body=request_body),
        resolved_guidance_scale,
        resolved_true_cfg_scale,
    )


async def build_context_and_inputs(
    tokenizer: PixelGenerationTokenizer,
    request: OpenResponsesRequest,
) -> tuple[PixelContext, PixelGenerationInputs[PixelContext]]:
    """Convert a request into a PixelContext and single-item pipeline input."""
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )
    return context, inputs


async def postprocess_output(
    tokenizer: PixelGenerationTokenizer,
    outputs: dict[RequestID, Any],
    context: PixelContext,
) -> Any:
    """Postprocess one pipeline output for the given request context."""
    return await tokenizer.postprocess(outputs[context.request_id])
