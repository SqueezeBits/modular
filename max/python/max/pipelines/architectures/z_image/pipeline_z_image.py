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

"""Buffer-based diffusion pipeline for Z-Image (Graph API / ModuleV2).

Inherits the full execution logic from the ModuleV3 pipeline, overriding
only the transformer component (Graph API model) and the ``run_transformer``
method to omit caching arguments not supported by the Graph API model.
"""

from typing import Any

from ..autoencoders import AutoencoderKLModel
from ..qwen3.text_encoder import Qwen3TextEncoderZImageModel
from ..z_image_modulev3.pipeline_z_image import (
    ZImagePipeline as ZImagePipelineV3,
)
from .model import ZImageTransformerModel


class ZImagePipeline(ZImagePipelineV3):
    """Z-Image pipeline using the Graph API transformer model.

    Reuses the full V3 pipeline execute logic (prompt encoding, latent
    preprocessing, denoising loop, decode) but swaps in the Graph API
    compiled transformer model.
    """

    transformer: ZImageTransformerModel  # type: ignore[assignment]

    components = {
        "vae": AutoencoderKLModel,
        "text_encoder": Qwen3TextEncoderZImageModel,
        "transformer": ZImageTransformerModel,
    }

    def run_transformer(
        self,
        cache_state: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        """Call the Graph API transformer without caching arguments.

        Wraps Buffer results as experimental Tensors for compatibility with
        V3 pipeline operations (F.mul, F.sub, etc.) used in CFG and
        scheduler steps.
        """
        from max.experimental.tensor import Tensor as ExpTensor

        result = self.transformer(
            kwargs["latents"],
            kwargs["prompt_embeds"],
            kwargs["timestep"],
            kwargs["img_ids"],
            kwargs["txt_ids"],
        )
        # Model.execute() returns list[Buffer]; wrap as Tensors.
        if isinstance(result, (list, tuple)):
            return tuple(
                ExpTensor(storage=r) if not isinstance(r, ExpTensor) else r
                for r in result
            )
        return (result,)
