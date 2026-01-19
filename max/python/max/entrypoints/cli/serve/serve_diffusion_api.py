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

"""API server for diffusion image generation.

This module provides an OpenAI-compatible API server for image generation
using diffusion models (e.g., FLUX.1).

The server implements the /v1/images/generations endpoint following the
OpenAI API specification.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvloop
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from max.entrypoints.diffusion import ImageGenerator
from max.interfaces import ImageGenerationRequest
from max.pipelines import PipelineConfig
from max.profiler import Tracer
from max.serve.api_server import validate_port_is_free
from max.serve.config import Settings
from uvicorn import Config, Server

logger = logging.getLogger("max.serve.diffusion")


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    settings: Settings,
    pipeline_config: PipelineConfig,
) -> AsyncGenerator[None]:
    """Manage the lifecycle of the diffusion server."""
    logger.info("Starting diffusion image generation server...")

    # Initialize the diffusion pipeline
    try:
        pipeline = ImageGenerator(pipeline_config)
        app.state.pipeline = pipeline
        app.state.pipeline_config = pipeline_config
        app.state.settings = settings
    except Exception:
        logger.exception("Failed to initialize diffusion pipeline")
        raise

    logger.info(
        f"\n\n{'*' * 80}\n\n"
        f"{'Image generation server ready on http://' + settings.host + ':' + str(settings.port) + ' (Press CTRL+C to quit)'.center(80)}\n\n"
        f"{'*' * 80}\n"
    )

    yield

    logger.info("Shutting down diffusion server...")


def create_diffusion_app(
    settings: Settings,
    pipeline_config: PipelineConfig,
) -> FastAPI:
    """Create the FastAPI application for diffusion serving."""

    @asynccontextmanager
    async def lifespan_wrap(app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            async with lifespan(app, settings, pipeline_config):
                yield
        except Exception:
            logger.exception("Server exception, shutting down...")
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)

    app = FastAPI(title="MAX Serve - Image Generation", lifespan=lifespan_wrap)

    # Health check
    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/v1/health")
    async def v1_health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # Version endpoint
    @app.get("/version")
    async def version() -> JSONResponse:
        from importlib.metadata import PackageNotFoundError, version

        try:
            package_version = version("max")
            return JSONResponse({"version": package_version})
        except PackageNotFoundError:
            return JSONResponse({"version": "unknown"})

    # OpenAI-compatible image generation endpoint
    @app.post("/v1/images/generations")
    async def create_image(request: Request) -> JSONResponse:
        """Generate images from a text prompt (OpenAI-compatible).

        Request body follows the OpenAI /v1/images/generations schema:
        - prompt (required): A text description of the desired image(s)
        - model: The model to use for image generation
        - n: Number of images to generate (1-10)
        - quality: Image quality ('standard', 'hd', 'high', 'medium', 'low')
        - response_format: 'url' or 'b64_json'
        - size: Image size (e.g., '1024x1024')
        - style: Image style ('vivid', 'natural')
        - user: End-user identifier
        - background: Background transparency ('transparent', 'opaque', 'auto')
        - output_format: Output format ('png', 'jpeg', 'webp')
        - num_inference_steps: Number of denoising steps (extension)
        - guidance_scale: Classifier-free guidance scale (extension)
        - seed: Random seed for reproducibility (extension)

        Returns:
            OpenAI-compatible ImagesResponse with created timestamp and
            image data (b64_json or url).
        """
        try:
            # Parse request body
            body = await request.json()

            # Validate required field
            if "prompt" not in body:
                raise ValueError("'prompt' is a required field")

            # Get pipeline from app state
            pipeline: ImageGenerator = request.app.state.pipeline

            # Build internal request from OpenAI-compatible fields
            internal_request = ImageGenerationRequest(
                prompt=body["prompt"],
                model=body.get("model"),
                n=body.get("n", 1),
                quality=body.get("quality", "standard"),
                response_format=body.get("response_format", "b64_json"),
                size=body.get("size", "1024x1024"),
                style=body.get("style"),
                user=body.get("user"),
                background=body.get("background"),
                moderation=body.get("moderation"),
                output_compression=body.get("output_compression"),
                output_format=body.get("output_format", "png"),
                partial_images=body.get("partial_images"),
                stream=body.get("stream"),
                # Extension parameters for diffusion models
                num_inference_steps=body.get("num_inference_steps", 50),
                guidance_scale=body.get("guidance_scale", 3.5),
                seed=body.get("seed"),
            )

            logger.debug(
                "Processing image generation request: prompt=%r, size=%s, n=%d",
                internal_request.prompt[:50] if len(internal_request.prompt) > 50 else internal_request.prompt,
                internal_request.size,
                internal_request.n or 1,
            )

            # Generate images
            response = pipeline.generate(internal_request)

            # Return OpenAI-compatible response
            return JSONResponse(response.to_dict())

        except ValueError as e:
            logger.warning("Invalid request: %s", str(e))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Image generation failed")
            raise HTTPException(
                status_code=500, detail="Image generation failed"
            ) from e

    # Model info endpoint
    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        """List available models."""
        pipeline: ImageGenerator = request.app.state.pipeline
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": pipeline.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "modular",
                    }
                ],
            }
        )

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str, request: Request) -> JSONResponse:
        """Get model information."""
        pipeline: ImageGenerator = request.app.state.pipeline

        # Check if the model_id matches
        if model_id == pipeline.model_name or model_id in pipeline.model_name:
            return JSONResponse(
                {
                    "id": pipeline.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "modular",
                }
            )

        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return app


def serve_diffusion_api_server(
    settings: Settings,
    pipeline_config: PipelineConfig,
) -> None:
    """Start the diffusion API server.

    Args:
        settings: Server settings (port, host, etc.).
        pipeline_config: Configuration for the diffusion pipeline.
    """
    # Create the FastAPI app
    app = create_diffusion_app(settings, pipeline_config)

    # Configure uvicorn
    config = Config(
        app=app,
        log_config=None,
        loop="uvloop",
        host=settings.host,
        port=settings.port,
        timeout_graceful_shutdown=5,
    )

    # Validate port before loading models
    validate_port_is_free(settings.port)

    server = Server(config)

    with Tracer("diffusion_api_server"):
        uvloop.run(server.serve())
