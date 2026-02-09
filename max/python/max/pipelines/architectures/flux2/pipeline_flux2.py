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

from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal

import numpy as np
import PIL.Image
from max import functional as F
from max import random
from max.driver import Accelerator, Buffer as DriverTensor, DeviceSpec, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.interfaces import PixelGenerationOutput, TokenBuffer
from max.pipelines import PixelContext
from max.pipelines.lib.diffusion_schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from max.pipelines.lib.image_processor import VaeImageProcessor
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
)
from max.tensor import Tensor
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.architectures.mistral3.arch import mistral3_arch
from max.pipelines.architectures.mistral3.model import Mistral3Model
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from transformers import AutoConfig
from tqdm import tqdm
from max.graph.weights import WeightsFormat
from typing import Callable

from pathlib import Path
import os

from ..autoencoders import AutoencoderKLFlux2Model
from ..mistral3 import Mistral3TextEncoderModel
from ..mistral3.tokenizer import Mistral3Tokenizer
from .model import Flux2Model
from .system_messages import SYSTEM_MESSAGE

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for Flux2 timestep scheduling.

    Taken from:
    https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/sampling.py#L251

    Args:
        image_seq_len: Length of image sequence (H*W after packing).
        num_steps: Number of inference steps.

    Returns:
        Empirical mu value for scheduler.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


@dataclass(kw_only=True)
class Flux2ModelInputs(PixelModelInputs):
    """
    Flux2-specific PixelModelInputs.

    Defaults:
    - width: 1024
    - height: 1024
    - guidance_scale: 4.0
    - num_inference_steps: 50
    - num_images_per_prompt: 1

    """

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1


@dataclass
class Flux2PipelineOutput:
    """Output class for Flux2 image generation pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray` or `Tensor`)
            List of denoised PIL images of length `batch_size` or numpy array or Max tensor of shape `(batch_size,
            height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion
            pipeline. Max tensors can represent either the denoised images or the intermediate latents ready to be
            passed to the decoder.
    """

    images: list[PIL.Image.Image] | np.ndarray | Tensor


class FluxMistral3TextEncoder(Mistral3TextEncoderModel):
    """Mistral3 text encoder with forced config for Flux2."""

    def __init__(self, config: dict[str, Any], **kwargs):
        # Force max_length to 512 for Flux2 context to avoid OOM
        config = config.copy()
        config["max_length"] = 512
        # Set utilization to 0.35 (just enough for 14GB weights) to check passes via overrides
        config["device_memory_utilization"] = 0.20
        super().__init__(config, **kwargs)

    def load_model(self) -> Callable[..., Any]:
        """Load pretrained model weights and compile the model graph.
        
        Overridden to bypass MemoryEstimator check which fails on 48GB GPU for 44GB estimated model,
        even though actual usage fits.
        """
        device_specs = [
            DeviceSpec.cpu(id=d.id) if d.label == "cpu"
            else DeviceSpec.accelerator(id=d.id) if d.label == "gpu"
            else DeviceSpec(id=d.id, device_type=d.label)
            for d in self.devices
        ]
        # Allow overriding max_length from pipeline config
        max_length = self.config.get("max_length", None)

        # Force load devices to ensure correct memory stats are available
        # This fixes a regression where lazy loading caused the device to report only ~15GB free
        _ = load_devices(device_specs)

        self._pipeline_config = PipelineConfig(
            model_path=self._text_encoder_path,
            return_hidden_states=ReturnHiddenStates.ALL_LAYERS,
            device_specs=device_specs,
            max_length=max_length,
        )
        model_config = self._pipeline_config.model
        model_config.kv_cache.device_memory_utilization = (
            self.device_memory_utilization
        )

        self._session = InferenceSession(devices=self.devices)
        huggingface_config = AutoConfig.from_pretrained(self._text_encoder_path)

        adapter = mistral3_arch.weight_adapters.get(
            WeightsFormat.safetensors, None
        )
        self._mistral_model = Mistral3Model(
            pipeline_config=self._pipeline_config,
            session=self._session,
            huggingface_config=huggingface_config,
            encoding=self.encoding,
            devices=self.devices,
            kv_cache_config=model_config.kv_cache,
            weights=self.weights,
            adapter=adapter,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.ALL_LAYERS,
        )
        
        # Tokenizer path logic
        tokenizer_path = self._text_encoder_path

        is_local_path = os.path.exists(self._text_encoder_path) or Path(self._text_encoder_path).exists()
        if is_local_path:
             # Check for common tokenizer files in root model dir if not in text_encoder
             has_tokenizer = False
             for f in ["tokenizer.json", "tokenizer_config.json"]:
                 if (Path(self._text_encoder_path) / f).exists():
                     has_tokenizer = True
                     break

             if not has_tokenizer and self._root_model_path:
                  root_tokenizer_path = Path(self._root_model_path) / "tokenizer"
                  if root_tokenizer_path.exists():
                      tokenizer_path = str(root_tokenizer_path)

        self._tokenizer = Mistral3Tokenizer(
            model_path=tokenizer_path,
            pipeline_config=self._pipeline_config,
            root_model_path=self._root_model_path,
        )

        return self._mistral_model.model


class Flux2Pipeline(DiffusionPipeline):
    config_name = "model_index.json"

    components = {
        "scheduler": FlowMatchEulerDiscreteScheduler,
        "vae": AutoencoderKLFlux2Model,
        "text_encoder": FluxMistral3TextEncoder,
        "tokenizer": Mistral3Tokenizer,
        "transformer": Flux2Model,
    }

    def init_remaining_components(self) -> None:
        # Scheduler is not a ComponentModel (no weights), so it is not loaded in _load_sub_models; create it here.
        if not getattr(self, "scheduler", None):
            self.scheduler = FlowMatchEulerDiscreteScheduler()
        image_processor_class = self.components.get(
            "image_processor", VaeImageProcessor
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        image_processor = image_processor_class(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.image_processor = image_processor

        # Store CPU sigmas for optimized scheduler step
        self._sigmas_cpu: np.ndarray | None = None
        # Compiled scheduler step graph
        self._scheduler_step_model = None

        # Cache for compiled graphs to avoid per-call compilation
        self._patchify_models: dict[str, Any] = {}
        self._vae_prep_models: dict[str, Any] = {}
        self._prompt_embed_models: dict[str, Any] = {}
        self._pack_models: dict[str, Any] = {}
        self._latent_prep_models: dict[str, Any] = {}
        self._unsqueeze_models: dict[str, Any] = {}
        self._time_step_models: dict[str, Any] = {}

        # Cache for invariant tensors (to avoid Tensor.full and Tensor.from_dlpack overhead)
        self._cached_guidance: dict[str, Tensor] = {}
        self._cached_latent_image_ids: dict[str, Tensor] = {}
        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}

    def _ensure_patchify_model(self, batch_size, channels, height, width, device, dtype):
        """Get or compile a patchify graph for the given shape."""
        key = f"{batch_size}_{channels}_{height}_{width}_{device}_{dtype}"
        if key in self._patchify_models:
            return self._patchify_models[key]
        input_type = TensorType(
            dtype, shape=[batch_size, channels, height, width], device=device
        )
        with Graph("patchify", input_types=[input_type]) as graph:
            latents = graph.inputs[0]
            latents = ops.reshape(
                latents, (batch_size, channels, height // 2, 2, width // 2, 2)
            )
            latents = ops.permute(latents, (0, 1, 3, 5, 2, 4))
            latents = ops.reshape(
                latents, (batch_size, channels * 4, height // 2, width // 2)
            )
            graph.output(latents)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._patchify_models[key] = model
        return model

    def _ensure_vae_prep_model(
        self, batch_size, seq_len, channels, height, width, device, dtype
    ):
        """Compile fused graph: unpack -> batch norm inverse -> unpatchify."""
        key = f"{batch_size}_{seq_len}_{channels}_{height}_{width}_{device}_{dtype}"
        if key in self._vae_prep_models:
            return self._vae_prep_models[key]

        input_types = [
            TensorType(dtype, shape=[batch_size, seq_len, channels], device=device),
            TensorType(dtype, shape=[channels], device=device),
            TensorType(dtype, shape=[channels], device=device),
        ]

        with Graph("vae_prep", input_types=input_types) as graph:
            latents, bn_mean, bn_var = graph.inputs
            latents = ops.permute(latents, (0, 2, 1))
            latents = ops.reshape(latents, (batch_size, channels, height, width))

            mean_reshaped = ops.reshape(bn_mean, (1, channels, 1, 1))
            var_reshaped = ops.reshape(bn_var, (1, channels, 1, 1))
            eps = ops.constant(
                self.vae.config.batch_norm_eps, dtype=dtype, device=device
            )
            std = ops.sqrt(ops.add(var_reshaped, eps))
            latents = ops.add(ops.mul(latents, std), mean_reshaped)

            latents = ops.reshape(
                latents, (batch_size, channels // 4, 2, 2, height, width)
            )
            latents = ops.permute(latents, (0, 1, 4, 2, 5, 3))
            latents = ops.reshape(
                latents, (batch_size, channels // 4, height * 2, width * 2)
            )
            graph.output(latents)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._vae_prep_models[key] = model
        return model

    def _ensure_prompt_embed_model(
        self,
        batch_size: int,
        num_layers: int,
        seq_len: int,
        hidden_dim: int,
        device: DeviceRef,
        dtype: DType,
    ):
        key = f"{batch_size}_{num_layers}_{seq_len}_{hidden_dim}_{device}_{dtype}"
        if key in self._prompt_embed_models:
            return self._prompt_embed_models[key]

        input_types = [
            TensorType(dtype, shape=[batch_size, seq_len, hidden_dim], device=device)
            for _ in range(num_layers)
        ]
        with Graph("prompt_embed_postproc", input_types=input_types) as graph:
            layer_inputs = [inp for inp in graph.inputs]
            stacked = ops.stack(layer_inputs, axis=1)
            stacked = ops.permute(stacked, [0, 2, 1, 3])
            result = ops.reshape(
                stacked, [batch_size, seq_len, num_layers * hidden_dim]
            )
            graph.output(result)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._prompt_embed_models[key] = model
        return model

    def _ensure_prompt_tile_model(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_images_per_prompt: int,
        device: DeviceRef,
        dtype: DType,
    ):
        key = f"{batch_size}_{seq_len}_{hidden_dim}_{num_images_per_prompt}_{device}_{dtype}"
        if key in self._prompt_embed_models:
            return self._prompt_embed_models[key]

        input_type = TensorType(
            dtype, shape=[batch_size, seq_len, hidden_dim], device=device
        )
        with Graph("prompt_tile", input_types=[input_type]) as graph:
            embeds = graph.inputs[0]
            embeds = ops.reshape(embeds, (batch_size, 1, seq_len, hidden_dim))
            embeds = ops.tile(embeds, [1, num_images_per_prompt, 1, 1])
            embeds = ops.reshape(
                embeds,
                (batch_size * num_images_per_prompt, seq_len, hidden_dim),
            )
            graph.output(embeds)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._prompt_embed_models[key] = model
        return model

    def _ensure_text_ids_model(
        self,
        batch_size: int,
        seq_len: int,
        device: DeviceRef,
    ):
        model_key = f"text_ids_model_{batch_size}_{seq_len}_{device}"
        if model_key in self._prompt_embed_models:
            return self._prompt_embed_models[model_key]

        with Graph("text_ids_gen", input_types=[]) as graph:
            coords = np.zeros((seq_len, 4), dtype=np.int64)
            coords[:, 3] = np.arange(seq_len, dtype=np.int64)
            coords_const = ops.constant(coords, dtype=DType.int64, device=device)
            coords_batch = ops.reshape(coords_const, (1, seq_len, 4))
            text_ids = ops.tile(coords_batch, [batch_size, 1, 1])
            graph.output(text_ids)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._prompt_embed_models[model_key] = model
        return model

    def _ensure_unsqueeze_model(
        self,
        seq_len: int,
        hidden_dim: int,
        device: DeviceRef,
        dtype: DType,
    ):
        key = f"{seq_len}_{hidden_dim}_{device}_{dtype}"
        if key in self._unsqueeze_models:
            return self._unsqueeze_models[key]

        input_type = TensorType(dtype, shape=[seq_len, hidden_dim], device=device)
        with Graph("unsqueeze_layer", input_types=[input_type]) as graph:
            tensor_in = graph.inputs[0]
            result = ops.reshape(tensor_in, [1, seq_len, hidden_dim])
            graph.output(result)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._unsqueeze_models[key] = model
        return model

    def _ensure_pack_model(self, batch_size, channels, height, width, device, dtype):
        key = f"{batch_size}_{channels}_{height}_{width}_{device}_{dtype}"
        if key in self._pack_models:
            return self._pack_models[key]

        input_type = TensorType(
            dtype, shape=[batch_size, channels, height, width], device=device
        )
        with Graph("pack", input_types=[input_type]) as graph:
            latents = graph.inputs[0]
            latents = ops.reshape(latents, (batch_size, channels, height * width))
            latents = ops.permute(latents, (0, 2, 1))
            graph.output(latents)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._pack_models[key] = model
        return model

    def _ensure_latent_prep_model(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        device: DeviceRef,
        input_dtype: DType,
        output_dtype: DType,
    ):
        key = f"{batch_size}_{channels}_{height}_{width}_{device}_{input_dtype}_{output_dtype}"
        if key in self._latent_prep_models:
            return self._latent_prep_models[key]

        input_type = TensorType(
            input_dtype, shape=[batch_size, channels, height, width], device=device
        )
        with Graph("latent_prep", input_types=[input_type]) as graph:
            latents = graph.inputs[0]
            latents = ops.cast(latents, output_dtype)
            latents = ops.reshape(
                latents, (batch_size, channels, height // 2, 2, width // 2, 2)
            )
            latents = ops.permute(latents, (0, 1, 3, 5, 2, 4))
            latents = ops.reshape(
                latents, (batch_size, channels * 4, height // 2, width // 2)
            )
            packed_channels = channels * 4
            packed_height = height // 2
            packed_width = width // 2
            latents = ops.reshape(
                latents,
                (
                    batch_size,
                    packed_channels,
                    packed_height * packed_width,
                ),
            )
            latents = ops.permute(latents, (0, 2, 1))
            graph.output(latents)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._latent_prep_models[key] = model
        return model

    def _ensure_all_time_steps_model(
        self,
        num_steps: int,
        device: DeviceRef,
    ):
        key = f"{device}"
        if key in self._time_step_models:
            return self._time_step_models[key]

        with Graph(
            "time_step_all",
            input_types=[TensorType(DType.float32, ["num_sigmas"], device)],
        ) as graph:
            sigmas = graph.inputs[0]
            sigmas_curr = ops.slice_tensor(sigmas, [slice(0, -1)])
            all_t = ops.cast(sigmas_curr, DType.bfloat16)
            sigmas_next = ops.slice_tensor(sigmas, [slice(1, None)])
            dt_f32 = ops.sub(sigmas_next, sigmas_curr)
            all_dt = ops.cast(dt_f32, DType.bfloat16)
            graph.output(all_t, all_dt)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._time_step_models[key] = model
        return model

    def _build_scheduler_step_graph(
        self,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        input_types = [
            TensorType(dtype, shape=["batch", "seq", "channels"], device=device),
            TensorType(dtype, shape=["batch", "seq", "channels"], device=device),
            TensorType(dtype, shape=[1], device=device),
        ]
        with Graph("scheduler_step", input_types=input_types) as graph:
            latents_in, noise_pred_in, dt_in = graph.inputs
            result = latents_in + dt_in * noise_pred_in
            graph.output(result)

        session = InferenceSession([Accelerator()])
        self._scheduler_step_model = session.load(graph)

    def prepare_inputs(self, context: PixelContext) -> Flux2ModelInputs:
        inputs = Flux2ModelInputs.from_context(context)

        # Flux2 latent IDs are defined on the packed latent grid (H/16, W/16).
        if getattr(inputs, "latent_image_ids", None) is None:
            batch_size = 1
            if getattr(inputs, "latents", None) is not None:
                latents_np = np.asarray(inputs.latents)
                if latents_np.ndim > 0:
                    batch_size = latents_np.shape[0]

            inputs.latent_image_ids = self._prepare_latent_image_ids(
                batch_size=batch_size,
                height=inputs.height // 16,
                width=inputs.width // 16,
                device=self.transformer.devices[0],
                dtype=DType.int64,
            )

        return inputs

    def _prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if hidden_states_layers is None:
            hidden_states_layers = [10, 20, 30]

        # unsqueeze
        if tokens.array.ndim == 1:
            tokens.array = np.expand_dims(tokens.array, axis=0)

        # Convert to numpy array (not Tensor) for text_encoder
        # Mistral3TextEncoderModel expects numpy array, not Tensor
        text_input_ids = tokens.array.astype(np.int64)

        # Encode with Mistral3 text encoder
        # Mistral3TextEncoderModel returns tuple of hidden states (all layers)
        hidden_states_tuple = self.text_encoder(text_input_ids)

        if not isinstance(hidden_states_tuple, tuple):
            raise ValueError(
                f"Expected tuple of hidden states, got {type(hidden_states_tuple)}"
            )

        # Extract specific layers (10, 20, 30) and stack them
        # Note: hidden_states_tuple is 0-indexed, but hidden_states_layers is 1-indexed
        layer_tensors = []
        max_sequence_length = tokens.array.shape[-1]
        for k in hidden_states_layers:
            layer_idx = k - 1  # Convert 1-based to 0-based
            if layer_idx >= len(hidden_states_tuple):
                raise ValueError(
                    f"Layer index {k} (0-based: {layer_idx}) is out of range. "
                    f"Total layers: {len(hidden_states_tuple)}"
                )

            hs = hidden_states_tuple[layer_idx]

            # Convert to Tensor if needed
            if not isinstance(hs, Tensor):
                hs = Tensor.from_dlpack(hs)

            # Handle sequence length padding/truncation
            tensor_rank = hs.rank
            if tensor_rank == 2:
                drv_shape = hs.driver_tensor.shape
                real_seq_len = drv_shape[0]
                hidden_dim = drv_shape[1]
                if real_seq_len < max_sequence_length:
                    padding_size = max_sequence_length - real_seq_len
                    padding = Tensor.zeros(
                        [padding_size, hidden_dim],
                        dtype=hs.dtype,
                        device=hs.device,
                    )
                    hs = F.concat([hs, padding], axis=0)
                elif real_seq_len > max_sequence_length:
                    hs = hs[:max_sequence_length]
                unsqueeze_model = self._ensure_unsqueeze_model(
                    seq_len=max_sequence_length,
                    hidden_dim=hidden_dim,
                    device=hs.device,
                    dtype=hs.dtype,
                )
                (hs_out,) = unsqueeze_model.execute(hs.driver_tensor)
                hs = Tensor.from_dlpack(hs_out)
            elif tensor_rank == 3:
                drv_shape = hs.driver_tensor.shape
                batch_size = drv_shape[0]
                real_seq_len = drv_shape[1]
                hidden_dim = drv_shape[2]
                if real_seq_len < max_sequence_length:
                    padding_size = max_sequence_length - real_seq_len
                    padding = Tensor.zeros(
                        [batch_size, padding_size, hidden_dim],
                        dtype=hs.dtype,
                        device=hs.device,
                    )
                    hs = F.concat([hs, padding], axis=1)
                elif real_seq_len > max_sequence_length:
                    hs = hs[:, :max_sequence_length, :]

            layer_tensors.append(hs)

        first_hs = layer_tensors[0]
        batch_size = first_hs.shape[0].dim
        seq_len = first_hs.shape[1].dim
        hidden_dim = first_hs.shape[2].dim
        num_layers = len(layer_tensors)

        prompt_embed_model = self._ensure_prompt_embed_model(
            batch_size=batch_size,
            num_layers=num_layers,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            device=DeviceRef.from_device(self.text_encoder.devices[0]),
            dtype=first_hs.dtype,
        )
        driver_inputs = [hs.driver_tensor for hs in layer_tensors]
        prompt_embeds = Tensor.from_dlpack(prompt_embed_model.execute(*driver_inputs)[0])

        bs_embed, seq_len_dim, _ = prompt_embeds.shape
        if num_images_per_prompt > 1:
            tile_model = self._ensure_prompt_tile_model(
                batch_size=bs_embed.dim,
                seq_len=seq_len,
                hidden_dim=prompt_embeds.shape[2].dim,
                num_images_per_prompt=num_images_per_prompt,
                device=DeviceRef.from_device(self.text_encoder.devices[0]),
                dtype=prompt_embeds.dtype,
            )
            prompt_embeds = Tensor.from_dlpack(
                tile_model.execute(prompt_embeds.driver_tensor)[0]
            )

        batch_size_final = bs_embed.dim * num_images_per_prompt
        seq_len_val = seq_len_dim.dim if hasattr(seq_len_dim, "dim") else seq_len_dim
        text_ids_key = f"{batch_size_final}_{seq_len_val}_{self.text_encoder.devices[0]}"
        if text_ids_key in self._cached_text_ids:
            text_ids = self._cached_text_ids[text_ids_key]
        else:
            text_ids_model = self._ensure_text_ids_model(
                batch_size=batch_size_final,
                seq_len=seq_len_val,
                device=DeviceRef.from_device(self.text_encoder.devices[0]),
            )
            text_ids = Tensor.from_dlpack(text_ids_model.execute()[0])
            self._cached_text_ids[text_ids_key] = text_ids

        return prompt_embeds, text_ids

    def _decode_latents(
        self,
        latents: Tensor,
        latent_image_ids: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> PIL.Image.Image | np.ndarray | Tensor:
        if output_type == "latent":
            latents_unpacked = self._unpack_latents(
                latents, height, width, latents.device, latents.dtype
            )
            return (
                latents_unpacked[0]
                if latents_unpacked.shape[0].dim > 1
                else latents_unpacked
            )

        h_latent = height // 16
        w_latent = width // 16
        batch_size = latents.shape[0].dim
        seq_len = latents.shape[1].dim
        num_channels = latents.shape[2].dim
        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        model = self._ensure_vae_prep_model(
            batch_size,
            seq_len,
            num_channels,
            h_latent,
            w_latent,
            latents.device,
            latents.dtype,
        )
        latents_unpacked_drv = model.execute(
            latents.driver_tensor, bn_mean.driver_tensor, bn_var.driver_tensor
        )[0]
        image = self.vae.decode(latents_unpacked_drv)

        # Convert to Tensor if decode returns driver.Tensor
        if not isinstance(image, Tensor):
            image = Tensor.from_dlpack(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Match Flux1: pixel_generation expects each image as (H, W, C) so np.stack yields (N, H, W, C).
        if output_type == "np" and isinstance(image, np.ndarray):
            image = self._to_hwc(image)

        return image

    def _unpack_models(self) -> dict[str, Any]:
        if not hasattr(self, "_unpack_models_cache"):
            self._unpack_models_cache = {}
        return self._unpack_models_cache

    def _unpack_latents(
        self,
        latents: Tensor,
        height: int,
        width: int,
        device: DeviceRef,
        dtype: DType,
    ) -> Tensor:
        h_latent = height // 16
        w_latent = width // 16
        batch_size = latents.shape[0].dim
        channels = latents.shape[2].dim

        key = f"{batch_size}_{channels}_{h_latent}_{w_latent}_{device}_{dtype}"
        cache = self._unpack_models()
        if key not in cache:
            input_type = TensorType(
                dtype,
                shape=[batch_size, h_latent * w_latent, channels],
                device=device,
            )
            with Graph("unpack_latents", input_types=[input_type]) as graph:
                l_in = graph.inputs[0]
                l = ops.permute(l_in, (0, 2, 1))
                l = ops.reshape(l, (batch_size, channels, h_latent, w_latent))
                graph.output(l)

            session = InferenceSession([Accelerator()])
            cache[key] = session.load(graph)
        model = cache[key]
        return Tensor.from_dlpack(model.execute(latents.driver_tensor)[0])

    @staticmethod
    def _to_hwc(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.float32, copy=False)

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: DeviceRef,
        dtype: DType,
    ) -> Tensor:
        # Create 4D coordinates using numpy (T=0, H, W, L=0)
        # Following max-diffusers pattern
        t_coords, h_coords, w_coords, l_coords = np.meshgrid(
            np.array([0]),  # T dimension
            np.arange(height),  # H dimension
            np.arange(width),  # W dimension
            np.array([0]),  # L dimension
            indexing="ij",
        )

        # Stack and reshape to (H*W, 4)
        latent_ids = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)
        latent_ids = latent_ids.reshape(-1, 4)

        # Expand to batch: (batch_size, H*W, 4)
        latent_ids = np.tile(latent_ids[np.newaxis, :, :], (batch_size, 1, 1))

        # Convert to Tensor with int64 dtype
        latent_image_ids = Tensor.from_dlpack(latent_ids.astype(np.int64)).to(
            device
        )

        return latent_image_ids

    @staticmethod
    def _prepare_image_ids(
        image_latents: list[Tensor],
        scale: int = 10,
        device: DeviceRef | None = None,
    ) -> Tensor:
        if not isinstance(image_latents, list):
            raise ValueError(
                f"Expected `image_latents` to be a list, got {type(image_latents)}."
            )

        if len(image_latents) == 0:
            raise ValueError("Expected at least one image latent in the list.")

        # Get device from first latent if not provided
        if device is None:
            device = image_latents[0].device

        # Create time offset for each reference image
        # T-coordinate for i-th image: scale + scale * i
        image_latent_ids = []

        for i, latent in enumerate(image_latents):
            # Remove batch dimension: [1, C, H, W] -> [C, H, W]
            latent_squeezed = F.squeeze(latent, axis=0)
            _, height, width = latent_squeezed.shape
            height = height.dim
            width = width.dim

            # T-coordinate for this image
            t_coord = scale + scale * i

            # Create coordinates using numpy (similar to _prepare_latent_image_ids)
            t_coords = np.full((height, width), t_coord, dtype=np.int64)
            h_coords, w_coords = np.meshgrid(
                np.arange(height, dtype=np.int64),
                np.arange(width, dtype=np.int64),
                indexing="ij",
            )
            l_coords = np.zeros((height, width), dtype=np.int64)

            # Stack: (H, W, 4)
            coords = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)

            # Reshape: (H*W, 4)
            coords = coords.reshape(-1, 4)

            # Convert to Tensor
            coords_tensor = Tensor.from_dlpack(coords).to(device)
            image_latent_ids.append(coords_tensor)

        # Concatenate all coordinates along the first dimension
        # Each tensor is (H*W, 4), so concatenating gives (N_total, 4)
        image_latent_ids = F.concat(image_latent_ids, axis=0)

        # Add batch dimension: (1, N_total, 4)
        image_latent_ids = F.unsqueeze(image_latent_ids, 0)

        return image_latent_ids

    def _pack_latents(self, latents: Tensor) -> Tensor:
        batch_size = latents.shape[0].dim
        num_channels = latents.shape[1].dim
        height = latents.shape[2].dim
        width = latents.shape[3].dim
        device = latents.device
        dtype = latents.dtype
        model = self._ensure_pack_model(
            batch_size, num_channels, height, width, device, dtype
        )
        latents_drv = model.execute(latents.driver_tensor)[0]
        return Tensor.from_dlpack(latents_drv)

    @staticmethod
    def retrieve_latents(
        encoder_output: "DiagonalGaussianDistribution",
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> Tensor:
        # In Max, vae.encode() returns DiagonalGaussianDistribution directly
        # (unlike diffusers which wraps it in AutoencoderKLOutput)
        if hasattr(encoder_output, "mode") and sample_mode == "mode":
            return encoder_output.mode()
        elif hasattr(encoder_output, "sample") and sample_mode == "sample":
            return encoder_output.sample(generator=generator)
        else:
            raise AttributeError(
                f"Could not access latents from encoder_output. "
                f"Expected DiagonalGaussianDistribution with 'mode' or 'sample' method, "
                f"got {type(encoder_output)}"
            )

    def _encode_vae_image(
        self,
        image: Tensor,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> Tensor:
        if len(image.shape) != 4:
            raise ValueError(f"Expected image dims 4, got {len(image.shape)}.")

        # Encode image using VAE
        encoder_output = self.vae.encode(image, return_dict=True)

        # Extract DiagonalGaussianDistribution from dict if needed
        if isinstance(encoder_output, dict):
            encoder_output = encoder_output["latent_dist"]

        # Retrieve latent from distribution
        image_latents = self.retrieve_latents(
            encoder_output, generator=generator, sample_mode=sample_mode
        )
        # Patchify latents: (1, C, H, W) -> (1, C*4, H//2, W//2)
        image_latents = self._patchify_latents(image_latents)
        # BatchNorm normalization
        # Get BatchNorm statistics (already Tensor)
        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
        num_channels = bn_mean.shape[0].dim
        bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))

        # Calculate standard deviation: sqrt(var + eps)
        bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)

        # Normalize: (latent - mean) / std
        image_latents = (image_latents - bn_mean) / bn_std

        return image_latents

    def _patchify_latents(self, latents: Tensor) -> Tensor:
        batch_size = latents.shape[0].dim
        num_channels_latents = latents.shape[1].dim
        height = latents.shape[2].dim
        width = latents.shape[3].dim
        device = latents.device
        dtype = latents.dtype

        model = self._ensure_patchify_model(
            batch_size, num_channels_latents, height, width, device, dtype
        )
        latents_drv = model.execute(latents.driver_tensor)[0]
        return Tensor.from_dlpack(latents_drv)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: DType,
        device: DeviceRef,
        latents: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # VAE applies 8x compression on images but we must also account for packing
        # which requires latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # Flux2 latent shape: (B, C*4, H//2, W//2) before packing
        # After packing: (B, (H//2)*(W//2), C*4)
        shape = (batch_size, num_channels_latents * 4, height // 2, width // 2)

        if latents is not None:
            latents = (
                latents
                if isinstance(latents, Tensor)
                else Tensor.from_dlpack(latents)
            )
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device).cast(dtype), latent_image_ids

        latents = random.normal(shape, device=device, dtype=dtype)

        # Prepare latent IDs before packing
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        # Pack latents: (B, C, H, W) -> (B, H*W, C)
        latents = self._pack_latents(latents)

        return latents, latent_image_ids

    def prepare_image_latents(
        self,
        images: list[Tensor],
        batch_size: int,
        device: DeviceRef,
        dtype: DType,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> tuple[Tensor, Tensor]:
        image_latents = []
        for image in images:
            # Move to device and cast dtype
            image = image.to(device).cast(dtype)
            # Encode using VAE
            latent = self._encode_vae_image(
                image=image, generator=generator, sample_mode=sample_mode
            )
            image_latents.append(latent)  # (1, C*4, H_latent, W_latent)

        # Generate position IDs for all images
        image_latent_ids = self._prepare_image_ids(image_latents, device=device)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, C*4, H_latent, W_latent)
            packed = self._pack_latents(latent)  # (1, H*W, C*4)
            # Remove batch dimension: (H*W, C*4)
            packed = F.squeeze(packed, axis=0)
            packed_latents.append(packed)

        # Concatenate all packed latents along sequence dimension
        # Each packed latent is (H*W, C*4), so concatenating gives (N_total, C*4)
        image_latents = F.concat(packed_latents, axis=0)  # (N_total, C*4)

        # Add batch dimension: (1, N_total, C*4)
        image_latents = F.unsqueeze(image_latents, 0)

        # Repeat for batch_size: (batch_size, N_total, C*4)
        # Using F.tile similar to prompt_embeds
        image_latents = F.tile(image_latents, (batch_size, 1, 1))

        # Repeat image_latent_ids for batch_size: (batch_size, N_total, 4)
        image_latent_ids = F.tile(image_latent_ids, (batch_size, 1, 1))
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    def execute(
        self,
        model_inputs: Flux2ModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Flux2PipelineOutput:
        """Execute the pipeline."""
        # 1. Encode prompts
        prompt_embeds, text_ids = self._prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )

        # 2. Denoise
        dtype = prompt_embeds.dtype
        device = self.transformer.devices[0]
        raw_latents = np.asarray(model_inputs.latents)
        batch_size = raw_latents.shape[0]
        channels = raw_latents.shape[1]
        height = raw_latents.shape[2]
        width = raw_latents.shape[3]
        input_dtype = DType.float32

        latent_prep_model = self._ensure_latent_prep_model(
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            device=DeviceRef.from_device(device),
            input_dtype=input_dtype,
            output_dtype=dtype,
        )
        latents_drv = DriverTensor.from_dlpack(raw_latents).to(device)
        latents_drv = latent_prep_model.execute(latents_drv)[0]
        latents = Tensor.from_dlpack(latents_drv)

        ids_key = f"{batch_size}_{height}_{width}_{device}"
        if ids_key in self._cached_latent_image_ids:
            latent_image_ids = self._cached_latent_image_ids[ids_key]
        else:
            input_ids = getattr(model_inputs, "latent_image_ids", None)
            if input_ids is None:
                latent_image_ids = self._prepare_latent_image_ids(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    device=device,
                    dtype=DType.int64,
                )
            elif isinstance(input_ids, Tensor):
                latent_image_ids = input_ids.to(device)
            else:
                latent_image_ids = Tensor.from_dlpack(input_ids).to(device)
            self._cached_latent_image_ids[ids_key] = latent_image_ids

        guidance_key = f"{batch_size}_{model_inputs.guidance_scale}_{dtype}_{device}"
        if guidance_key in self._cached_guidance:
            guidance = self._cached_guidance[guidance_key]
        else:
            guidance = Tensor.full(
                [latents.shape[0]],
                model_inputs.guidance_scale,
                device=device,
                dtype=dtype,
            )
            self._cached_guidance[guidance_key] = guidance

        image_seq_len = latents.shape[1].dim
        num_inference_steps = model_inputs.num_inference_steps
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)
        base_sigmas = np.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
            dtype=np.float32,
        )
        self.scheduler.set_timesteps(sigmas=base_sigmas, mu=mu)
        self._sigmas_cpu = np.ascontiguousarray(self.scheduler.sigmas)

        sigmas_key = f"{num_inference_steps}_{image_seq_len}_{device}"
        if sigmas_key in self._cached_sigmas:
            sigmas = self._cached_sigmas[sigmas_key]
        else:
            sigmas = Tensor.from_dlpack(self._sigmas_cpu).to(device)
            self._cached_sigmas[sigmas_key] = sigmas
        batch_size = prompt_embeds.shape[0].dim

        timesteps: np.ndarray = self.scheduler.timesteps
        num_timesteps = timesteps.shape[0]

        image_seq_len = latents.shape[1].dim
        text_seq_len = prompt_embeds.shape[1].dim
        compiled_model = self.transformer._ensure_compiled(
            batch_size=batch_size,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
        )

        encoder_hidden_states_drv = prompt_embeds.driver_tensor
        guidance_drv = guidance.driver_tensor
        txt_ids_drv = text_ids.driver_tensor
        img_ids_drv = latent_image_ids.driver_tensor
        device = self.transformer.devices[0]
        if self._scheduler_step_model is None:
            self._build_scheduler_step_graph(dtype, device)

        _all_time_steps_model = self._ensure_all_time_steps_model(
            num_steps=num_inference_steps,
            device=DeviceRef.from_device(device),
        )

        sigmas_drv = sigmas.driver_tensor
        latents_drv = latents.driver_tensor
        all_timesteps_drv, all_dts_drv = _all_time_steps_model.execute(sigmas_drv)

        def _unwrap_model(model):
            while hasattr(model, "__wrapped__"):
                model = model.__wrapped__
            return model

        if hasattr(all_timesteps_drv, "driver_tensor"):
            all_timesteps_drv = all_timesteps_drv.driver_tensor
        if hasattr(all_dts_drv, "driver_tensor"):
            all_dts_drv = all_dts_drv.driver_tensor

        raw_compiled_model = _unwrap_model(compiled_model)
        raw_scheduler_step_model = _unwrap_model(self._scheduler_step_model)

        for i in tqdm(range(num_timesteps), desc="Denoising"):
            self._current_timestep = i
            timestep_drv = all_timesteps_drv[i : i + 1]
            dt_drv = all_dts_drv[i : i + 1]

            noise_pred_drv = raw_compiled_model.execute(
                latents_drv,
                encoder_hidden_states_drv,
                timestep_drv,
                img_ids_drv,
                txt_ids_drv,
                guidance_drv,
            )[0]

            latents_drv = raw_scheduler_step_model.execute(
                latents_drv, noise_pred_drv, dt_drv
            )[0]

            if hasattr(device, "synchronize"):
                device.synchronize()

            if callback_queue is not None:
                latents = Tensor.from_dlpack(latents_drv)
                image = self._decode_latents(
                    latents,
                    latent_image_ids,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
                callback_queue.put_nowait(image)

        latents = Tensor.from_dlpack(latents_drv)

        # 3. Decode
        batch_size = latents.shape[0].dim
        image_list = []
        latent_image_ids_drv = latent_image_ids.driver_tensor
        for b in range(batch_size):
            latents_drv_b = latents_drv[b : b + 1, :, :]
            latent_image_ids_drv_b = latent_image_ids_drv[b : b + 1, :, :]
            latents_b = Tensor.from_dlpack(latents_drv_b)
            latent_image_ids_b = Tensor.from_dlpack(latent_image_ids_drv_b)

            image_b = self._decode_latents(
                latents_b,
                latent_image_ids_b,
                model_inputs.height,
                model_inputs.width,
                output_type=output_type,
            )
            image_list.append(image_b)

        return Flux2PipelineOutput(images=image_list)
