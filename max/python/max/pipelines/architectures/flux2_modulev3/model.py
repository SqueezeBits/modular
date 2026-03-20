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

import ctypes
import struct
from collections.abc import Callable
from typing import Any

from max.driver import Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.fp8_config_utils import (
    build_dynamic_block_fp8_config,
    build_legacy_scalar_fp8_config,
    validate_fp8_weight_scale_contract,
)
from max.experimental.tensor import Tensor
from max.graph.shape import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import traced

from .flux2 import Flux2Transformer2DModel
from .model_config import Flux2Config
from .nvfp4_weight_adapter import convert_nvfp4_state_dict
from .weight_adapters import uses_legacy_scalar_fp8_scales

# Mapping from stacked QKV key infixes to the split (Q, K, V) infixes.
_STACKED_QKV_INFIXES = {
    ".attn.qkv_proj.": (".attn.to_q.", ".attn.to_k.", ".attn.to_v."),
    ".attn.add_qkv_proj.": (
        ".attn.add_q_proj.",
        ".attn.add_k_proj.",
        ".attn.add_v_proj.",
    ),
}


class Flux2TransformerModel(ComponentModel):
    model: Callable[..., Any] | None

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
        )
        self.config = Flux2Config.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    @traced(message="Flux2TransformerModel.load_model")
    def load_model(self) -> None:
        state_dict = {key: value.data() for key, value in self.weights.items()}

        # Convert BFL single-file NVFP4 naming to MAX parameter naming.
        if getattr(self.config, "quant_config", None) is not None:
            state_dict = convert_nvfp4_state_dict(state_dict)

        # Detect stacked (fused) QKV weights and split into separate Q/K/V
        # so the model always sees the split format.
        stacked_qkv = any(
            ".attn.qkv_proj." in k or ".attn.add_qkv_proj." in k
            for k in state_dict
        )
        if stacked_qkv:
            state_dict = self._split_stacked_qkv(state_dict)

        # FP8 auto-detection: if encoding is float8_e4m3fn and float8_config
        # wasn't already set by initialize_from_config, inspect the state dict.
        requested_activation_scheme = getattr(
            self.config, "activation_scheme", None
        )
        if self.encoding == "float8_e4m3fn" and getattr(
            self.config, "float8_config", None
        ) is None:
            if requested_activation_scheme != "static" and any(
                key.endswith(".weight_scale")
                and len(getattr(value, "shape", ())) == 2
                for key, value in state_dict.items()
            ):
                self.config = self.config.model_copy(
                    update={
                        "float8_config": build_dynamic_block_fp8_config(
                            {
                                "quantization_config": {
                                    "quant_method": "fp8",
                                    "activation_scheme": "dynamic",
                                    "weight_block_size": [128, 128],
                                }
                            },
                            component_name="flux2.transformer",
                        )
                    }
                )
            elif uses_legacy_scalar_fp8_scales(state_dict):
                self.config = self.config.model_copy(
                    update={
                        "float8_config": build_legacy_scalar_fp8_config(
                            component_name="flux2.transformer"
                        )
                    }
                )

        # Validate FP8 weight scale contract if blockwise FP8.
        fp8_cfg = getattr(self.config, "float8_config", None)
        if (
            fp8_cfg is not None
            and fp8_cfg.weight_scale.block_size is not None
        ):
            validate_fp8_weight_scale_contract(
                state_dict,
                float8_config=fp8_cfg,
                component_name="flux2.transformer",
            )

        # Inject default FP8 scale tensors for any missing scale parameters.
        if fp8_cfg is not None:
            with F.lazy():
                _flux_tmp = Flux2Transformer2DModel(self.config)
                for param_name, tensor in _flux_tmp.parameters:
                    if (
                        param_name.endswith((".input_scale", ".weight_scale"))
                        and param_name not in state_dict
                    ):
                        shape = tuple(int(d) for d in tensor.shape)
                        shape_list = list(shape) if shape else []
                        n = 1
                        for d in shape_list:
                            n *= d
                        raw = struct.pack(f"<{n}f", *([1.0] * n))
                        buf = Buffer(DType.float32, shape_list)
                        ctypes.memmove(buf._data_ptr(), raw, len(raw))
                        state_dict[param_name] = WeightData(
                            data=buf,
                            name=param_name,
                            dtype=DType.float32,
                            shape=shape_list,
                        )

        self._state_dict = state_dict

        # Klein/distilled checkpoints can omit guidance embedder weights.
        has_guidance_embedder = any(
            "time_guidance_embed.guidance_embedder." in k or "guidance_in." in k
            for k in state_dict
        )
        if not has_guidance_embedder and getattr(
            self.config, "guidance_embeds", True
        ):
            if hasattr(self.config, "model_copy"):
                self.config = self.config.model_copy(
                    update={"guidance_embeds": False}
                )
            else:
                self.config.guidance_embeds = False
        with F.lazy():
            flux = Flux2Transformer2DModel(self.config)
            flux.to(self.devices[0])
        self._flux_model = flux
        self._standard_model: Callable[..., Any] | None = None
        self._step_cache_model: Callable[..., Any] | None = None
        self.model = None

    @staticmethod
    def _split_stacked_qkv(
        state_dict: dict[str, WeightData],
    ) -> dict[str, WeightData]:
        """Split fused QKV weights into separate Q, K, V entries."""
        out: dict[str, WeightData] = {}
        for key, value in state_dict.items():
            matched = False
            for stacked, (q, k, v) in _STACKED_QKV_INFIXES.items():
                if stacked not in key:
                    continue
                matched = True
                if key.endswith((".weight", ".weight_scale")):
                    buf = value.to_buffer()
                    chunk = buf.shape[0] // 3
                    for infix, i in zip([q, k, v], range(3), strict=False):
                        split_name = key.replace(stacked, infix)
                        split_buf = buf[i * chunk : (i + 1) * chunk, :]
                        out[split_name] = WeightData(
                            split_buf,
                            split_name,
                            value.dtype,
                            Shape(split_buf.shape),
                        )
                elif key.endswith((".weight_scale_2", ".input_scale")):
                    # Per-tensor scales are shared across Q/K/V.
                    for infix in (q, k, v):
                        out[key.replace(stacked, infix)] = value
                break
            if not matched:
                out[key] = value
        return out

    @traced(message="Flux2TransformerModel.use_standard_model")
    def use_standard_model(self) -> None:
        if self._standard_model is None:
            self._flux_model._step_cache_enabled = False
            self._standard_model = self._flux_model.compile(
                *self._flux_model.input_types(step_cache_enabled=False),
                weights=self._state_dict,
            )
        if self.model is self._step_cache_model:
            self._step_cache_model = None
        self.model = self._standard_model

    @traced(message="Flux2TransformerModel.use_step_cache_model")
    def use_step_cache_model(self, rdt: float = 0.05) -> None:
        if self._step_cache_model is None:
            assert self._flux_model is not None
            self._flux_model._step_cache_enabled = True
            self._flux_model._rdt_value = rdt
            self._step_cache_model = self._flux_model.compile(
                *self._flux_model.input_types(step_cache_enabled=True),
                weights=self._state_dict,
            )
        if self.model is self._standard_model:
            self._standard_model = None
        self.model = self._step_cache_model

    @traced(message="Flux2TransformerModel.__call__")
    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
        prev_residual: Tensor | None = None,
        prev_output: Tensor | None = None,
    ) -> Any:
        args: tuple[Any, ...] = (
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )
        if prev_residual is not None:
            args = (*args, prev_residual, prev_output)
        if self.model is None:
            raise RuntimeError(
                "Model not compiled. Call use_standard_model() or use_step_cache_model() first."
            )
        return self.model(*args)
