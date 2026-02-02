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

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from max.driver import Buffer, Device, DeviceSpec
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
    WeightsFormat,
)
from max.interfaces import RequestID, TokenBuffer
from max.kv_cache import PagedKVCacheManager, load_kv_manager
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    PipelineConfig,
    SupportedEncoding,
)
from max.pipelines.lib.interfaces.component_model import ComponentModel
from transformers import AutoConfig

from ..mistral.mistral import Mistral
from ..mistral.model_config import MistralConfig
from .arch import mistral3_arch

logger = logging.getLogger(__name__)


class Mistral3TextEncoderModel(ComponentModel):
    """Mistral3 text encoder wrapper implementing ComponentModel interface.

    This class builds and compiles a Mistral3 model graph directly for text encoding,
    without depending on PipelineModel or Mistral3Tokenizer. It exposes a simple
    interface that returns hidden states from all layers.

    Note: Although text encoding is a single forward pass operation and doesn't
    actually use KV cache for multi-step generation, the compiled model graph
    requires KV cache inputs as part of its interface. We allocate minimal KV
    cache to satisfy the graph requirements.
    """

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

        # Lazy initialization attributes (set in load_model)
        self._model: Model | None = None
        self._session: InferenceSession | None = None
        self._kv_manager: PagedKVCacheManager | None = None
        self._config = config

        # Load model during initialization
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Load pretrained model weights and compile the model graph.

        This method builds and compiles the Mistral3 graph directly without
        depending on PipelineModel. It initializes the KV cache manager
        directly for use during inference.

        Returns:
            Compiled model callable (Model instance).
        """
        self._session = InferenceSession(devices=self.devices)

        device_specs = [
            DeviceSpec.cpu(id=d.id) if d.label == "cpu"
            else DeviceSpec.accelerator(id=d.id) if d.label == "gpu"
            else DeviceSpec(id=d.id, device_type=d.label)
            for d in self.devices
        ]
        self._pipeline_config = PipelineConfig(
            model_path=self._text_encoder_path,
            return_hidden_states=ReturnHiddenStates.ALL_LAYERS,
            device_specs=device_specs,
        )
        self._huggingface_config = AutoConfig.from_pretrained(
            self._text_encoder_path
        )
        text_config = self._huggingface_config.text_config

        # TODO: initialize KVCacheParams and kv_manager properly without hardcoded values.
        self._kv_params = KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=self._config["text_config"]["num_key_value_heads"],
            head_dim=self._config["text_config"]["head_dim"],
            num_layers=self._config["text_config"]["num_hidden_layers"],
            devices=[DeviceRef.from_device(d) for d in self.devices],
        )
        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_batch_size=512,
            max_seq_len=131072,
            session=self._session,
            available_cache_memory=37769707520,
        )

        self._model = self._build_and_load_model(text_config)

        return self._model

    def _get_state_dict(
        self, adapter: WeightsAdapter | None
    ) -> dict[str, WeightData]:
        """Get the state dict from weights, optionally applying adapter."""
        if adapter:
            state_dict = adapter(
                dict(self.weights.items()),
                huggingface_config=self._huggingface_config.text_config,
                pipeline_config=self._pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        return state_dict

    def _build_and_load_model(
        self,
        text_config: AutoConfig,
    ) -> Model:
        """Build and load the Mistral model graph.

        Args:
            text_config: HuggingFace text model configuration.
            adapter: Optional weight adapter.

        Returns:
            Compiled Model instance.
        """
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are currently supported."
            )

        adapter = mistral3_arch.weight_adapters.get(
            WeightsFormat.safetensors, None
        )

        # Get state dict
        state_dict = self._get_state_dict(adapter)

        # Initialize model config
        mistral_config = MistralConfig.initialize_from_config(
            self._pipeline_config, text_config
        )
        mistral_config.return_logits = ReturnLogits.LAST_TOKEN
        mistral_config.return_hidden_states = ReturnHiddenStates.ALL_LAYERS

        # Build graph inputs
        from max.dtype import DType

        device_ref = DeviceRef.from_device(self.devices[0])
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        # Get KV cache input types
        kv_inputs = self._kv_params.get_symbolic_inputs()

        graph_inputs = (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *kv_inputs[0],
        )

        # Build the neural network model
        nn_model = Mistral(mistral_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=False,
        )

        # Build the graph
        timer = CompilationTimer("text_encoder_model")
        with Graph("mistral3_text_encoder", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collection = PagedCacheValues(
                kv_blocks=kv_cache_inputs[0].buffer,
                cache_lengths=kv_cache_inputs[1].tensor,
                lookup_table=kv_cache_inputs[2].tensor,
                max_lengths=kv_cache_inputs[3].tensor,
            )
            outputs = nn_model(
                tokens.tensor,
                kv_collection,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)

        timer.mark_build_complete()
        model = self._session.load(graph, weights_registry=nn_model.state_dict())
        timer.done()

        return model

    def __call__(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
        position_ids: np.ndarray | None = None,
    ) -> tuple[Buffer, ...]:
        """Apply Mistral3 text encoder forward pass.

        Args:
            input_ids: Input token IDs as numpy array.
            attention_mask: Attention mask (not used, kept for compatibility).
            position_ids: Position IDs (not used, kept for compatibility).

        Returns:
            Tuple of hidden states from all layers as MAX Buffers.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None or self._kv_manager is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure input is a numpy array with correct dtype, flattened to 1D
        arr = np.asarray(input_ids, dtype=np.int64).ravel()

        # Create TextContext directly without using tokenizer
        request_id = RequestID()
        token_buffer = TokenBuffer(arr)
        context = TextContext(
            request_id=request_id,
            max_length=131072,
            tokens=token_buffer,
            eos_token_ids=set(),
            model_name=self._text_encoder_path or "",
        )

        replica_idx = 0
        num_steps = 1
        try:
            # Claim and allocate KV cache directly
            self._kv_manager.claim(request_id, replica_idx=replica_idx)
            self._kv_manager.alloc(
                context, replica_idx=replica_idx, num_steps=num_steps
            )

            # Get KV cache runtime inputs directly
            kv_cache_inputs_list = self._kv_manager.get_runtime_inputs(
                [[context]], num_steps=num_steps
            )
            kv_cache_inputs = kv_cache_inputs_list[0]

            # Prepare model inputs directly
            input_row_offsets = Buffer.from_numpy(
                np.cumsum(
                    [0] + [context.tokens.active_length],
                    dtype=np.uint32,
                )
            ).to(self.devices[0])

            next_tokens_batch = Buffer.from_numpy(
                context.tokens.active
            ).to(self.devices[0])

            return_n_logits = Buffer.from_numpy(
                np.array([1], dtype=np.int64)
            )

            # Execute the model
            model_outputs = self._model.execute(
                next_tokens_batch,
                input_row_offsets,
                return_n_logits,
                *kv_cache_inputs,
            )

            # First output is logits, rest are hidden states
            if len(model_outputs) <= 1:
                logger.warning(
                    "Model did not return hidden states; "
                    "graph may not be compiled with return_hidden_states=ALL_LAYERS"
                )
                raise RuntimeError(
                    "Model did not return hidden states. "
                    "Ensure the model graph was compiled with "
                    "return_hidden_states=ALL_LAYERS."
                )

            # Return hidden states (skip the first output which is logits)
            hidden_states = tuple(model_outputs[1:])
            return hidden_states
        finally:
            self._kv_manager.release(request_id, replica_idx=replica_idx)

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
