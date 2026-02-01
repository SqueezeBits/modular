# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Mistral3 text encoder for Flux2 pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from max.driver import Device, DeviceSpec
from max.engine import InferenceSession
from max.graph import TensorValue
from max.graph.weights import Weights
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    PipelineConfig,
    SupportedEncoding,
)
from max.pipelines.lib.interfaces.component_model import ComponentModel
from transformers import AutoConfig

from .model import Mistral3Model
from .tokenizer import Mistral3Tokenizer


class Mistral3TextEncoderModel(ComponentModel):
    """Mistral3 text encoder wrapper implementing ComponentModel interface.

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
        
        # Store root_model_path for tokenizer (needs HuggingFace repo ID, not local path)
        self._root_model_path = config.get("root_model_path")

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

    def load_model(self) -> Callable[..., Any]:
        """Load pretrained model weights and compile the model graph.

        Returns:
            Compiled model callable (Model instance).
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
        self._pipeline_config.model.kv_cache.device_memory_utilization = (
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

        # For tokenizer, determine the correct path.
        # It could be in the text_encoder directory (Mistral3) or the root directory (Flux2).
        tokenizer_path = self._text_encoder_path
        
        # Check if tokenizer config exists in text_encoder path
        import os
        from pathlib import Path
        
        is_local_path = os.path.exists(self._text_encoder_path) or Path(self._text_encoder_path).exists()
        
        if is_local_path:
             # Check for common tokenizer files
            has_tokenizer = False
            for f in ["tokenizer.json", "tokenizer_config.json"]:
                if (Path(self._text_encoder_path) / f).exists():
                    has_tokenizer = True
                    break
            
            # If not found in text_encoder path, fallback to root/tokenizer if available
            if not has_tokenizer and self._root_model_path:
                 root_tokenizer_path = Path(self._root_model_path) / "tokenizer"
                 if root_tokenizer_path.exists():
                     pass  # use root_tokenizer_path
                     tokenizer_path = str(root_tokenizer_path)

        self._tokenizer = Mistral3Tokenizer(
            model_path=tokenizer_path,
            pipeline_config=self._pipeline_config,
            root_model_path=self._root_model_path,
        )

        # Return the compiled model (for ComponentModel interface compatibility)
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

        # Convert to Python list
        # tolist() already returns a plain Python list, no need for additional list() call
        if input_ids_np.ndim > 1:
            input_ids_list = input_ids_np.flatten().tolist()
        else:
            input_ids_list = input_ids_np.tolist()

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
        # Handle both sync and async contexts
        # If we're in an async context, use ThreadPoolExecutor to run in a new event loop
        # This avoids the "asyncio.run() cannot be called from a running event loop" error
        try:
            # Check if we're in an async context
            asyncio.get_running_loop()
            # We're in an async context, run in a new thread with new event loop
            loop = asyncio.new_event_loop()
            with ThreadPoolExecutor() as pool:
                fut = pool.submit(loop.run_until_complete, self._tokenizer.new_context(request))
                context = fut.result()
            loop.close()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            context = asyncio.run(self._tokenizer.new_context(request))

        num_steps = 1
        request_id = context.request_id

        replica_idx = 0
        try:
            # Claim and allocate KV cache
            self._mistral_model.kv_manager.claim(request_id, replica_idx=replica_idx)
            self._mistral_model.kv_manager.alloc(context, replica_idx=replica_idx, num_steps=num_steps)

            # get_runtime_inputs expects per-replica batches: Sequence[Sequence[TextGenerationContext]]
            # For single replica (replica_idx=0), we pass [[context]]
            kv_cache_inputs_list = (
                self._mistral_model.kv_manager.get_runtime_inputs(
                    [[context]], num_steps=num_steps
                )
            )
            kv_cache_inputs = kv_cache_inputs_list[0]

            model_inputs = self._mistral_model.prepare_initial_token_inputs(
                replica_batches=[[context]],
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=1,
            )

            model_outputs = self._mistral_model.execute(model_inputs=model_inputs)

            # Debug: Check the actual model_outputs structure
            # The model.execute() returns a tuple from the compiled graph
            # We need to check if hidden states are actually in the tuple
            if model_outputs.hidden_states is None:
                # Check if the underlying model.execute() returned multiple outputs
                # by inspecting the internal model_outputs tuple
                import logging
                logger = logging.getLogger("max.pipelines")
                
                # Access the raw model outputs to see what was actually returned
                # model_outputs is a ModelOutputs object, but we need to check
                # what the underlying model.execute() returned
                logger.warning(
                    f"Model did not return hidden states. "
                    f"return_logits={self._mistral_model.return_logits}, "
                    f"return_hidden_states={self._mistral_model.return_hidden_states}, "
                    f"model_outputs type: {type(model_outputs)}, "
                    f"has hidden_states attr: {hasattr(model_outputs, 'hidden_states')}"
                )
                
                # Try to access the raw outputs from the model
                # The issue might be that the graph wasn't built with return_hidden_states
                # Let's check if we can access the internal model outputs
                raise RuntimeError(
                    "Model did not return hidden states. "
                    "Check return_hidden_states configuration. "
                    f"return_logits={self._mistral_model.return_logits}, "
                    f"return_hidden_states={self._mistral_model.return_hidden_states}. "
                    "The model graph may not have been compiled with return_hidden_states=ALL_LAYERS."
                )

            return model_outputs.hidden_states

        finally:
            # IMPORTANT: Release KV cache to prevent memory leak
            self._mistral_model.kv_manager.release(request_id, replica_idx=replica_idx)

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
