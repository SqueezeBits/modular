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

"""Rotary Position Embedding for Mistral3 text encoder.

This implementation uses non-interleaved format where the first half of head_dim
contains real parts and the second half contains imaginary parts, which is the
format expected by Mistral models.
"""

from collections.abc import Iterable
from functools import cached_property

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim
from max.nn import Module
from max.tensor import Tensor


class RotaryEmbedding(Module[..., Tensor]):
    """Rotary Position Embedding for Mistral3 (non-interleaved format).

    This module computes and applies rotary position embeddings in the
    non-interleaved format used by Mistral models.
    """

    dim: int
    n_heads: int
    theta: float
    max_seq_len: int
    head_dim: int
    device: Device
    _freqs_cis: Tensor | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
    ) -> None:
        """Initialize RotaryEmbedding.

        Args:
            dim: Hidden dimension of the model.
            n_heads: Number of attention heads.
            theta: Base frequency for position encoding.
            max_seq_len: Maximum sequence length.
            device: Device to place tensors on.
            head_dim: Dimension per head (defaults to dim // n_heads).
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.device = device

    def _compute_inv_freqs(self) -> Tensor:
        """Computes inv_freqs for n // 2 rotation blocks to be used by RoPE.

        Returns:
            a 1D tensor of thetas of shape [head_dim // 2]
        """
        n = self.head_dim

        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
        # Calculate theta for n/2 blocks: theta_for_block_i = theta ** (-2i/n) where n is dim for each head.
        iota = F.arange(0, n, step=2, dtype=DType.float64, device=self.device)
        inv_freq = F.cast(1.0 / (self.theta ** (iota / n)), DType.float32)

        return inv_freq

    def freqs_cis_base(self) -> Tensor:
        """Compute frequency tensor for complex exponentials.

        Returns:
            Frequency tensor with shape [max_seq_len * 2, head_dim // 2, 2].
        """
        if self._freqs_cis is None:
            inv_freqs = self._compute_inv_freqs()

            # Position ids [0, 1, ..., max_seq_len*2)
            t = F.arange(
                0, self.max_seq_len * 2, device=self.device, dtype=DType.float32
            )

            # Outer product: [max_seq_len*2, head_dim // 2]
            freqs = F.outer(t, inv_freqs)

            # Stack cos and sin: [max_seq_len*2, head_dim // 2, 2]
            self._freqs_cis = F.stack([F.cos(freqs), F.sin(freqs)], axis=-1)

        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    @cached_property
    def freqs_cis(self) -> Tensor:
        """Get flattened frequency tensor.

        Returns:
            Frequency tensor with shape [max_seq_len * 2, head_dim].
        """
        freqs = self.freqs_cis_base()
        d1, d2, d3 = freqs.shape  # (max_seq_len * 2, head_dim // 2, 2)
        new_f_shape = [d1, d2 * d3]  # (max_seq_len * 2, head_dim)
        self._freqs_cis = F.reshape(freqs, new_f_shape)
        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    @property
    def local_parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Override to exclude freqs_cis from parameters."""
        return []

    def forward(
        self,
        x: Tensor,
        start_pos: Dim | None = None,
        seq_len: Dim | None = None,
    ) -> Tensor:
        """Apply rotary positional embeddings to input tensor.

        Uses non-interleaved format where first half of head_dim is real part
        and second half is imaginary part.

        Args:
            x: Input tensor with shape [batch, seq_len, n_heads, head_dim]
               or [seq_len, n_heads, head_dim].
            start_pos: Starting position (default 0).
            seq_len: Sequence length (default from input shape).

        Returns:
            Tensor with RoPE applied, same shape as input.
        """
        v = x

        # Non-interleaved format: first half is real, second half is imaginary
        head_dim = v.shape[-1]
        half_dim = head_dim // 2
        x_re = v[..., :half_dim]
        x_im = v[..., half_dim:head_dim]

        if start_pos is None:
            start_pos = Dim(0)
        if seq_len is None:
            if len(v.shape) == 4:
                seq_len = v.shape[1]
            else:
                seq_len = v.shape[0]

        freqs_cis_sliced = self.freqs_cis[start_pos : start_pos + seq_len]
        # Handle optimized case that flattens freqs_cis.
        # This is needed so naive llama3 can still use Llama3RotaryEmbedding with correct freqs_cis.
        if len(freqs_cis_sliced.shape) == 2:
            d0, d1 = freqs_cis_sliced.shape
            freqs_cis_sliced = F.reshape(freqs_cis_sliced, (d0, d1 // 2, 2))

        freqs_cis_sliced = F.cast(freqs_cis_sliced, v.dtype)

        # Broadcast for attention heads
        # Handle both 3D and 4D input shapes
        if len(v.shape) == 4:
            # [B, S, H, D] -> broadcast [1, S, 1, half_dim]
            freqs_cis_bcast = F.unsqueeze(F.unsqueeze(freqs_cis_sliced, 1), 0)
        else:
            # [S, H, D] -> broadcast [S, 1, half_dim]
            freqs_cis_bcast = F.unsqueeze(freqs_cis_sliced, 1)

        freqs_re = freqs_cis_bcast[..., 0]
        freqs_im = freqs_cis_bcast[..., 1]

        rope_re = (x_re * freqs_re) - (x_im * freqs_im)
        rope_im = (x_re * freqs_im) + (x_im * freqs_re)

        rope_result = F.concat((rope_re, rope_im), axis=-1)

        return F.cast(F.reshape(rope_result, v.shape), v.dtype)
