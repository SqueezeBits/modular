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

"""Example: Using OpenAI-compatible API for image generation.

This example demonstrates how to use the ImageGenerationRequest and
ImageGenerationResponse classes for OpenAI-compatible image generation.

Usage:
    python openai_api_example.py --seed 42 --prompt "A futuristic city skyline at sunset with flying cars" --model-path "black-forest-labs/FLUX.1-dev"
"""

import argparse
import base64
import os
from pathlib import Path

from max.entrypoints.diffusion import ImageGenerator
from max.interfaces import ImageGenerationRequest
from max.pipelines import PipelineConfig


def main() -> None:
    # Configure random seed for reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A futuristic city skyline at sunset with flying cars",
    )
    parser.add_argument(
        "--model-path", type=str, default="black-forest-labs/FLUX.1-dev"
    )
    args = parser.parse_args()
    seed = args.seed
    os.environ["SEED"] = str(seed)
    model_path = args.model_path

    # Initialize the generator
    pipeline_config = PipelineConfig(model_path=model_path)
    generator = ImageGenerator(pipeline_config)

    print(f"Model loaded: {generator.model_name}")
    print(f"Seed: {os.getenv('SEED', 'not set')}")

    # Create an OpenAI-compatible request
    request = ImageGenerationRequest(
        prompt="A futuristic city skyline at sunset with flying cars",
        size="1024x1024",
        n=1,
        quality="standard",
        response_format="b64_json",
        output_format="png",
        # Diffusion-specific parameters
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=seed,
    )

    print(f"Generating image with prompt: {request.prompt}")
    print(f"Size: {request.size}, Steps: {request.num_inference_steps}")

    # Generate using OpenAI-compatible API (create method)
    response = generator.create(request)

    print(f"Response created at: {response.created}")
    print(f"Number of images: {len(response.data)}")

    # Save the generated image
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_data in enumerate(response.data):
        if image_data.b64_json:
            # Decode base64 and save
            image_bytes = base64.b64decode(image_data.b64_json)
            output_path = output_dir / f"openai_api_output_{i}.png"
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            print(f"Image saved to: {output_path}")

        if image_data.revised_prompt:
            print(f"Revised prompt: {image_data.revised_prompt}")


if __name__ == "__main__":
    main()
