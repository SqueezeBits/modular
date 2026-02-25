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

from __future__ import annotations

from typing import Literal

from max import random
from max.dtype import DType
from max.graph import Dim, DimLike, TensorValue
from max import functional as F
from max.nn import Module
from max.nn.legacy.float8_config import Float8Config
from max.nn.legacy.kernels import (
    dynamic_scaled_matmul,
    quantize_dynamic_scaled_float8,
)
from max.tensor import Tensor


class FP8Linear(Module[[Tensor], Tensor]):
    """Tensor-API Linear with optional dynamic FP8 execution path.

    When `float8_config` is set and a loaded weight is float8, this layer expects
    `<name>.weight_scale` to be present and executes:
      dynamic quantize(x) -> dynamic_scaled_matmul(...)
    Otherwise it falls back to BF16/FP32 matmul.
    """

    weight: Tensor
    bias: Tensor | Literal[0]
    weight_scale: Tensor | None
    _weight_scale_cache: dict[str, Tensor]

    def __init__(
        self,
        in_dim: DimLike,
        out_dim: DimLike,
        *,
        bias: bool = True,
        float8_config: Float8Config | None = None,
        weight_dtype: DType | None = None,
        bias_dtype: DType | None = None,
    ):
        self.weight = random.normal([out_dim, in_dim], dtype=weight_dtype)
        self.bias = (
            random.normal([out_dim], dtype=bias_dtype or weight_dtype)
            if bias
            else 0
        )
        self.float8_config = float8_config

        # In FP8 mode, this must be replaced by a checkpoint tensor.
        if float8_config is not None:
            assert float8_config.weight_scale.block_size is not None
            block_n, block_k = float8_config.weight_scale.block_size
            out_blocks = (int(out_dim) + block_n - 1) // block_n
            in_blocks = (int(in_dim) + block_k - 1) // block_k
            self.weight_scale = Tensor.ones(
                [out_blocks, in_blocks], dtype=DType.float32
            )
        else:
            self.weight_scale = None
        # Cache per-device scale tensors to avoid repeated H2D copies.
        self._weight_scale_cache = {}

    def _weight_scale_on(self, device: object) -> Tensor:
        if self.weight_scale is None:
            raise ValueError("weight_scale is required for FP8 execution path.")
        key = str(device)
        cached = self._weight_scale_cache.get(key)
        if cached is None:
            cached = self.weight_scale.to(device)
            self._weight_scale_cache[key] = cached
        return cached

    @property
    def in_dim(self) -> Dim:
        return self.weight.shape[1]

    @property
    def out_dim(self) -> Dim:
        return self.weight.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        # Legacy FP8 kernels currently expect rank-2 input; flatten and restore.
        orig_shape = x.shape
        is_rank2 = len(orig_shape) == 2
        if not is_rank2:
            leading_dim = orig_shape[0]
            for d in orig_shape[1:-1]:
                leading_dim = leading_dim * d
            x_2d = F.reshape(x, (leading_dim, self.in_dim))
        else:
            x_2d = x

        if (
            self.float8_config is not None
            and self.weight_scale is not None
            and self.weight.dtype.is_float8()
        ):
            x_q, x_scales = quantize_dynamic_scaled_float8(
                TensorValue(x_2d),
                self.float8_config.input_scale,
                self.float8_config.weight_scale,
                scales_type=self.weight_scale.dtype,
                group_size_or_per_token=(
                    self.float8_config.weight_scale.block_size[1]
                    if self.float8_config.weight_scale.block_size is not None
                    else -1
                ),
                out_type=self.weight.dtype,
            )
            out = dynamic_scaled_matmul(
                x_q,
                TensorValue(self.weight),
                x_scales,
                TensorValue(self._weight_scale_on(x_q.device)),
                self.float8_config.input_scale,
                self.float8_config.weight_scale,
                out_type=DType.bfloat16,
            )
            if isinstance(self.bias, Tensor):
                out = out + TensorValue(self.bias).to(out.device)
            out_t = Tensor.from_graph_value(out)
            if not is_rank2:
                out_t = F.reshape(out_t, (*orig_shape[:-1], self.out_dim))
            return out_t

        weight = self.weight
        bias = self.bias
        # Metadata-less FP8 checkpoints: compute in input dtype fallback.
        if weight.dtype.is_float8() and x_2d.dtype != weight.dtype:
            weight = weight.cast(x_2d.dtype)
            if isinstance(bias, Tensor):
                bias = bias.cast(x_2d.dtype)
        out = x_2d @ weight.T + bias
        if not is_rank2:
            out = F.reshape(out, (*orig_shape[:-1], self.out_dim))
        return out
