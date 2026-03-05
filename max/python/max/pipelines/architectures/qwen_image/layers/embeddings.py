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

"""Embeddings for QwenImage transformer: timestep projection and 3D RoPE."""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module

from max.pipelines.architectures.flux2.layers.embeddings import (
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)


class QwenImageTimestepProjEmbeddings(Module[[Tensor], Tensor]):
    """Timestep-only projection embeddings (no guidance embedding).

    Unlike Flux2 which combines timestep + guidance, QwenImage only uses timestep
    since guidance_embeds=False.
    """

    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 3072,
        bias: bool = False,
    ):
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )

    def forward(self, timestep: Tensor) -> Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.cast(timestep.dtype)
        )
        return timesteps_emb


class QwenImagePosEmbed(Module[[Tensor], tuple[Tensor, Tensor]]):
    """3D Rotary Position Embeddings for QwenImage.

    Uses axes_dims_rope = (16, 56, 56) for (T, H, W) dimensions,
    compared to Flux2's 4D (32, 32, 32, 32).
    """

    theta: int
    axes_dim: tuple[int, ...]

    def __init__(self, theta: int, axes_dim: tuple[int, ...]):
        self.theta = theta
        self.axes_dim = tuple(axes_dim)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Compute rotary position embeddings from position IDs.

        Args:
            ids: Position IDs of shape [S, len(axes_dim)] (3D: T, H, W).

        Returns:
            Tuple of (cos, sin) tensors of shape [S, sum(axes_dim)] for RoPE.
        """
        cos_out = []
        sin_out = []

        pos = ids.cast(DType.float32) if ids.dtype != DType.float32 else ids

        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                use_real=True,
                repeat_interleave_real=True,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = F.concat(cos_out, axis=-1)
        freqs_sin = F.concat(sin_out, axis=-1)

        return freqs_cos, freqs_sin
