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

"""Mistral3 text encoder for Flux2 pipeline."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
from max.driver import Device, DeviceSpec
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, TensorType, TensorValue
from max.graph.weights import Weights
from max.nn import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    SupportedEncoding,
)
from max.pipelines.lib.hf_utils import download_weight_files
from max.pipelines.lib.interfaces.max_model import MaxModel
from transformers import AutoConfig

from .model import Mistral3Model
from .tokenizer import Mistral3Tokenizer


class Mistral3TextEncoderModel(MaxModel):
    """Mistral3 text encoder wrapper implementing MaxModel interface.

    This class wraps Mistral3Model to function as a text encoder for Flux2 pipeline.
    It uses the full Mistral3 text generation infrastructure internally but exposes
    a simpler interface that returns hidden states from all layers.

    Note: Although text encoding is a single forward pass operation and doesn't
    actually use KV cache for multi-step generation, the compiled model graph
    requires KV cache inputs as part of its interface. We allocate minimal KV
    cache to satisfy the graph requirements.
    """

    config_name = "config.json"

    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize Mistral3TextEncoderModel.

        Args:
            config: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        super().__init__(config, encoding, devices, weights)

        # Extract model path from config
        self._text_encoder_path = config.get("text_encoder_path") or config.get(
            "model_path"
        )
        if not self._text_encoder_path:
            raise ValueError(
                "model_path or text_encoder_path must be provided in config"
            )

        # For PipelineConfig, use the text_encoder_path
        self._model_path = self._text_encoder_path

        # Text encoder uses single forward pass, so minimal KV cache is sufficient
        self.device_memory_utilization = config.get(
            "device_memory_utilization", 0.3
        )

        # Lazy initialization attributes (set in load_model)
        self._mistral_model: Mistral3Model | None = None
        self._session: InferenceSession | None = None
        self._tokenizer: Mistral3Tokenizer | None = None
        self._pipeline_config: PipelineConfig | None = None

        # Load model during initialization
        self.load_model()

    def load_model(self) -> Model:
        """Load pretrained model weights and compile the model graph.

        Returns:
            Compiled Model instance.
        """
        # Convert Device objects to DeviceSpec objects for PipelineConfig
        device_specs = []
        for device in self.devices:
            if device.label == "cpu":
                device_specs.append(DeviceSpec.cpu(id=device.id))
            elif device.label == "gpu":
                device_specs.append(DeviceSpec.accelerator(id=device.id))
            else:
                device_specs.append(
                    DeviceSpec(id=device.id, device_type=device.label)
                )

        self._pipeline_config = PipelineConfig(
            model_path=self._model_path,
            return_hidden_states=ReturnHiddenStates.ALL_LAYERS,
            device_specs=device_specs,
        )

        # Set minimal device_memory_utilization for KV cache (text encoder only needs single pass)
        # This is critical to avoid OOM when loading other models (transformer, VAE)
        self._pipeline_config.model._kv_cache.device_memory_utilization = (
            self.device_memory_utilization  # 0.3 = 30%
        )

        # Create inference session with Device objects (not DeviceSpec)
        self._session = InferenceSession(devices=self.devices)

        # Perform memory estimation to set _available_cache_memory
        from max.pipelines.lib.memory_estimation import MemoryEstimator

        MemoryEstimator.estimate_memory_footprint(
            self._pipeline_config,
            Mistral3Model,
            self._pipeline_config.model,
            self.devices,
        )

        # Load AutoConfig from pretrained path for Mistral3Model
        huggingface_config = AutoConfig.from_pretrained(self._text_encoder_path)

        # Get weight adapter from mistral3_arch
        from max.graph.weights import WeightsFormat
        from max.pipelines.architectures.mistral3.arch import mistral3_arch

        adapter = mistral3_arch.weight_adapters.get(WeightsFormat.safetensors, None)

        # Create Mistral3Model with return_hidden_states=ALL_LAYERS
        self._mistral_model = Mistral3Model(
            pipeline_config=self._pipeline_config,
            session=self._session,
            huggingface_config=huggingface_config,
            encoding=self.encoding,
            devices=self.devices,
            kv_cache_config=self._pipeline_config.model.kv_cache,
            weights=self.weights,
            adapter=adapter,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.ALL_LAYERS,
        )

        self._tokenizer = Mistral3Tokenizer(
            model_path=self._model_path,
            pipeline_config=self._pipeline_config,
        )

        # Return the compiled model (for MaxModel interface compatibility)
        return self._mistral_model.model

    def __call__(
        self,
        input_ids: TensorValue | np.ndarray,
        attention_mask: TensorValue | None = None,
        position_ids: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Apply Mistral3 text encoder forward pass.

        Args:
            input_ids: Input token IDs as numpy array (preferred) or MAX TensorValue.
                       Passing numpy directly avoids unnecessary GPU->CPU transfer.
            attention_mask: Attention mask (not used, kept for compatibility).
            position_ids: Position IDs (not used, kept for compatibility).

        Returns:
            Tuple of hidden states from all layers as MAX TensorValues.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._mistral_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert input_ids to numpy if needed
        # Prefer numpy input to avoid GPU->CPU transfer overhead
        if isinstance(input_ids, np.ndarray):
            input_ids_np = input_ids
        elif hasattr(input_ids, "to_numpy"):
            # V2 Buffer Tensor
            input_ids_np = input_ids.to_numpy()
        else:
            # Fallback: try to convert to numpy
            input_ids_np = np.asarray(input_ids)

        input_ids_list = (
            input_ids_np.flatten().tolist()
            if input_ids_np.ndim > 1
            else input_ids_np.tolist()
        )

        # Create text generation request
        from max.interfaces import (
            RequestID,
            SamplingParams,
            SamplingParamsInput,
            TextGenerationRequest,
        )

        sampling_params = SamplingParams.from_input_and_generation_config(
            SamplingParamsInput(max_new_tokens=1),
            sampling_params_defaults=self._pipeline_config.model.sampling_params_defaults,
        )

        request = TextGenerationRequest(
            request_id=RequestID(),
            model_name=self._model_path or "",
            prompt=input_ids_list,
            sampling_params=sampling_params,
        )

        # Create context using tokenizer
        context = asyncio.run(self._tokenizer.new_context(request))

        num_steps = 1
        request_id = context.request_id

        try:
            # Claim and allocate KV cache
            self._mistral_model.kv_manager.claim(request_id, replica_idx=0)
            self._mistral_model.kv_manager.alloc(context, num_steps=num_steps)

            kv_cache_inputs_list = (
                self._mistral_model.kv_manager.get_runtime_inputs(
                    [context], num_steps=num_steps
                )
            )
            kv_cache_inputs = kv_cache_inputs_list[0]

            model_inputs = self._mistral_model.prepare_initial_token_inputs(
                replica_batches=[[context]],
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=1,
            )

            model_outputs = self._mistral_model.execute(model_inputs=model_inputs)

            if model_outputs.hidden_states is None:
                raise RuntimeError(
                    "Model did not return hidden states. "
                    "Check return_hidden_states configuration."
                )

            return model_outputs.hidden_states

        finally:
            # IMPORTANT: Release KV cache to prevent memory leak
            self._mistral_model.kv_manager.release(request_id)

    @property
    def session(self) -> InferenceSession:
        """Return the InferenceSession instance.

        Returns:
            InferenceSession: The compiled inference session.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._session
