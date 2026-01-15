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

import os
from pathlib import Path

from max.entrypoints.diffusion import DiffusionPipeline
from max.pipelines import PipelineConfig


def main() -> None:
    model_path = "black-forest-labs/FLUX.1-dev"
    use_torch_randn = True
    if use_torch_randn:
        seed = 42
        os.environ["USE_TORCH_RANDN"] = "1"
        os.environ["SEED"] = str(seed)
    pipeline_config = PipelineConfig(model_path=model_path)
    pipe = DiffusionPipeline(pipeline_config)

    prompt = "A cat holding a sign that says hello world"
    print(f"Prompt: {prompt}")

    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
    )

    images = result.images

    output_path = Path("output.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
