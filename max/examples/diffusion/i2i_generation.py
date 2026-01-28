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

import argparse
from pathlib import Path

from PIL import Image

from max.entrypoints.diffusion import DiffusionPipeline
from max.experimental.realization_context import set_seed
from max.pipelines import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="black-forest-labs/FLUX.2-dev"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-image", type=str, required=True, help="Path to input image for I2I"
    )
    args = parser.parse_args()

    model_path = args.model_path
    set_seed(args.seed)
    pipeline_config = PipelineConfig(model_path=model_path)
    pipe = DiffusionPipeline(pipeline_config)

    # Load input image
    input_image = Image.open(args.input_image)
    print(f"Loaded input image: {args.input_image} ({input_image.size})")

    prompt = "Change the word Hello world to SqueezeBits"
    print(f"Prompt: {prompt}")

    result = pipe(
        prompt=prompt,
        image=input_image,  # I2I input
        height=512,
        width=512,
        num_inference_steps=28,
        guidance_scale=4.0,
    )

    images = result.images

    output_path = Path("output_i2i.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
