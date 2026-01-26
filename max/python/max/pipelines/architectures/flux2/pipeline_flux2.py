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

"""Flux2 image generation pipeline."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
from max.driver import Buffer as Tensor
from max.dtype import DType
from max.experimental import Tensor as Tensor_v3
from max.experimental import functional as F
from max.experimental import random
from max.graph import DeviceRef
from max.pipelines.lib.diffusion_schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from max.pipelines.lib.image_processor import (
    PipelineImageInput,
    VaeImageProcessor,
)
from max.pipelines.lib.interfaces.diffusion_pipeline import (
    DiffusionPipeline,
)
from tqdm import tqdm

from ..autoencoders import AutoencoderKLFlux2Model
from ..mistral3 import Mistral3TextEncoderModel
from ..mistral3.tokenizer import Mistral3Tokenizer
from .model import Flux2Model
from .system_messages import SYSTEM_MESSAGE


def format_input(
    prompts: list[str],
    system_message: str = SYSTEM_MESSAGE,
    images: list[PIL.Image.Image] | list[list[PIL.Image.Image]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Format a batch of text prompts into the conversation format expected by apply_chat_template.

    Optionally, add images to the input.

    Adapted from:
    https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/text_encoder.py#L68

    Args:
        prompts: List of text prompts.
        system_message: System message to use (default: SYSTEM_MESSAGE).
        images: Optional list of images to add to the input.

    Returns:
        List of conversations, where each conversation is a list of message dicts.
    """
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    if images is None or len(images) == 0:
        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]
    else:
        assert len(images) == len(
            prompts
        ), "Number of images must match number of prompts"
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
            ]
            for _ in cleaned_txt
        ]

        for i, (el, img_list) in enumerate(zip(messages, images)):
            # optionally add the images per batch element.
            if img_list is not None:
                el.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_obj}
                            for image_obj in img_list
                        ],
                    }
                )
            # add the text.
            el.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": cleaned_txt[i]}],
                }
            )

        return messages


def retrieve_timesteps(
    scheduler: Any,
    num_inference_steps: int | None = None,
    device: str | DeviceRef | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, int]:
    r"""Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.

    Handles custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `DeviceRef`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
        **kwargs (`Any`, *optional*):
            Additional arguments to pass to the scheduler's `set_timesteps` method.

    Returns:
        `tuple[Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = int(timesteps.shape[0])
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = int(timesteps.shape[0])
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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


class Flux2Pipeline(DiffusionPipeline):
    """Flux2 image generation pipeline."""

    config_name = "model_index.json"

    components = {
        "scheduler": FlowMatchEulerDiscreteScheduler,
        "vae": AutoencoderKLFlux2Model,
        "text_encoder": Mistral3TextEncoderModel,
        "tokenizer": Mistral3Tokenizer,
        "transformer": Flux2Model,
    }

    def init_remaining_components(self) -> None:
        """Initialize remaining pipeline components."""
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

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: DeviceRef | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Tensor | None = None,
        max_sequence_length: int = 512,
        lora_scale: float | None = None,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`DeviceRef`):
                Max device
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            prompt_embeds (`Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            max_sequence_length (`int`, defaults to 512): Maximum sequence length to use with the `prompt`.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            hidden_states_layers (`List[int]`, *optional*, defaults to [10, 20, 30]):
                List of layer indices (1-based) to extract hidden states from. For Flux2, layers 10, 20, 30 are stacked.

        Returns:
            Tuple of (prompt_embeds, text_ids) where:
            - prompt_embeds: Text embeddings of shape [B, seq_len, 3*hidden_dim] (stacked from 3 layers)
            - text_ids: Text position IDs of shape [seq_len, 4]
        """
        if hidden_states_layers is None:
            hidden_states_layers = [10, 20, 30]

        if lora_scale is not None and isinstance(self, Flux2Pipeline):
            self._lora_scale = lora_scale

            if self.text_encoder is not None and hasattr(
                self.text_encoder, "set_lora_scale"
            ):
                self.text_encoder.set_lora_scale(lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            # Format prompt using Flux2 chat template
            messages_batch = format_input(
                prompts=prompt, system_message=SYSTEM_MESSAGE
            )

            # Use HuggingFace tokenizer's apply_chat_template
            # Access the delegate tokenizer from Mistral3Tokenizer
            hf_tokenizer = self.tokenizer.delegate
            inputs = hf_tokenizer.apply_chat_template(
                messages_batch,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
                return_length=False,
                return_overflowing_tokens=False,
            )

            # Extract real tokens only (using attention mask)
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs.get("attention_mask", None)
            attention_mask = (
                attention_mask[0]
                if attention_mask is not None
                else [1] * len(input_ids)
            )

            # Filter to keep only real tokens (where mask == 1)
            real_token_ids = [
                token_id
                for token_id, mask in zip(input_ids, attention_mask)
                if mask == 1
            ]
            text_input_ids = np.array([real_token_ids], dtype=np.int64)

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
            for k in hidden_states_layers:
                layer_idx = k - 1  # Convert 1-based to 0-based
                if layer_idx >= len(hidden_states_tuple):
                    raise ValueError(
                        f"Layer index {k} (0-based: {layer_idx}) is out of range. "
                        f"Total layers: {len(hidden_states_tuple)}"
                    )

                hs = hidden_states_tuple[layer_idx]

                # Convert to Tensor_v3 if needed
                # model_outputs.hidden_states returns tuple of TensorValue (V2 Tensor)
                # Following max-diffusers pattern: Tensor_v3.from_dlpack(hs)
                if not isinstance(hs, Tensor_v3):
                    # TensorValue (V2 Tensor) - convert to Tensor_v3
                    # TensorValue can be directly converted using from_dlpack
                    hs = Tensor_v3.from_dlpack(hs)

                # Handle sequence length padding/truncation
                if hs.rank == 2:
                    # Shape: [seq_len, hidden_dim]
                    real_seq_len = hs.shape[0].dim
                    hidden_dim = hs.shape[1].dim
                    if real_seq_len < max_sequence_length:
                        padding_size = max_sequence_length - real_seq_len
                        padding = Tensor_v3.zeros(
                            [padding_size, hidden_dim],
                            dtype=hs.dtype,
                            device=hs.device,
                        )
                        hs = F.concat([hs, padding], axis=0)
                    elif real_seq_len > max_sequence_length:
                        hs = hs[:max_sequence_length]

                    # Reshape to [1, seq_len, hidden_dim]
                    hs = F.reshape(hs, [1, max_sequence_length, hidden_dim])
                elif hs.rank == 3:
                    # Shape: [batch, seq_len, hidden_dim]
                    batch_size = hs.shape[0].dim
                    real_seq_len = hs.shape[1].dim
                    hidden_dim = hs.shape[2].dim
                    if real_seq_len < max_sequence_length:
                        padding_size = max_sequence_length - real_seq_len
                        padding = Tensor_v3.zeros(
                            [batch_size, padding_size, hidden_dim],
                            dtype=hs.dtype,
                            device=hs.device,
                        )
                        hs = F.concat([hs, padding], axis=1)
                    elif real_seq_len > max_sequence_length:
                        hs = hs[:, :max_sequence_length, :]

                layer_tensors.append(hs)

            # Stack layers: [1, 3, seq_len, hidden_dim]
            stacked = F.stack(layer_tensors, axis=1)

            # Permute to [1, seq_len, 3, hidden_dim]
            stacked = F.permute(stacked, [0, 2, 1, 3])

            # Reshape to [1, seq_len, 3*hidden_dim]
            batch_size = stacked.shape[0].dim
            seq_len = stacked.shape[1].dim
            num_layers = stacked.shape[2].dim
            hidden_dim = stacked.shape[3].dim
            prompt_embeds = F.reshape(
                stacked, [batch_size, seq_len, num_layers * hidden_dim]
            )

            # Ensure correct device and dtype
            prompt_embeds = prompt_embeds.to(device).cast(
                prompt_embeds.dtype if hasattr(prompt_embeds, "dtype") else DType.bfloat16
            )

        bs_embed, seq_len, _ = prompt_embeds.shape

        # Tile for multiple images per prompt
        prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.reshape(
            (bs_embed.dim * num_images_per_prompt, seq_len, -1)
        )

        # Prepare text position IDs (4D for Flux2)
        # Flux2 uses 4D position IDs: [batch_size, seq_len, 4]
        # Following max-diffusers pattern: (T=0, H=0, W=0, L=[0..seq_len-1])
        batch_size_final = bs_embed.dim * num_images_per_prompt
        text_ids = self._prepare_text_ids(
            batch_size=batch_size_final,
            seq_len=seq_len.dim if hasattr(seq_len, 'dim') else seq_len,
            device=device,
        )

        return prompt_embeds, text_ids

    @staticmethod
    def _prepare_text_ids(
        batch_size: int,
        seq_len: int,
        device: DeviceRef,
    ) -> Tensor_v3:
        """Prepare 4D text position IDs (T=0, H=0, W=0, L=[0..seq_len-1]).

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            device: Device to place tensors on.

        Returns:
            Text position IDs of shape [batch_size, seq_len, 4].
        """
        # Create 4D coordinates: (T=0, H=0, W=0, L=[0..seq_len-1])
        coords = np.stack(
            [
                np.zeros(seq_len, dtype=np.int64),  # T dimension
                np.zeros(seq_len, dtype=np.int64),  # H dimension
                np.zeros(seq_len, dtype=np.int64),  # W dimension
                np.arange(seq_len, dtype=np.int64),  # L dimension
            ],
            axis=-1,
        )  # (seq_len, 4)

        # Expand to batch (batch_size, seq_len, 4)
        text_ids = np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        text_ids = Tensor_v3.from_dlpack(text_ids).to(device)
        return text_ids

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: DeviceRef,
        dtype: DType,
    ) -> Tensor_v3:
        """Prepare latent image position IDs for Flux2 (4D).

        Args:
            batch_size: Batch size.
            height: Latent height.
            width: Latent width.
            device: Device to place tensors on.
            dtype: Data type for tensors (ignored, always int64).

        Returns:
            Image position IDs of shape [batch_size, height*width, 4].
        """
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

        # Convert to Tensor_v3 with int64 dtype
        latent_image_ids = Tensor_v3.from_dlpack(latent_ids.astype(np.int64)).to(
            device
        )

        return latent_image_ids

    @staticmethod
    def _pack_latents(latents: Tensor_v3) -> Tensor_v3:
        """Pack latents: (B, C, H, W) -> (B, H*W, C).

        Args:
            latents: Latent tensor of shape (B, C, H, W).

        Returns:
            Packed latents of shape (B, H*W, C).
        """
        batch_size = latents.shape[0].dim
        num_channels = latents.shape[1].dim
        height = latents.shape[2].dim
        width = latents.shape[3].dim
        latents = F.reshape(latents, (batch_size, num_channels, height * width))
        latents = F.permute(latents, (0, 2, 1))
        return latents

    @staticmethod
    def _unpack_latents_with_ids(x: Tensor_v3, x_ids: Tensor_v3) -> Tensor_v3:
        """Using position ids to scatter tokens into place.

        Args:
            x: Latent tensor of shape [B, seq_len, C].
            x_ids: Position IDs tensor of shape [B, seq_len, 4].

        Returns:
            Unpacked latents of shape [B, C, H, W].
        """
        batch_size = x.shape[0].dim
        seq_len = x.shape[1].dim
        ch = x.shape[2].dim

        # Get h_ids and w_ids from position tensor (columns 1 and 2)
        h_ids = x_ids[:, :, 1].cast(DType.int64)  # [B, seq_len]
        w_ids = x_ids[:, :, 2].cast(DType.int64)  # [B, seq_len]

        # Calculate H and W from max indices + 1
        h = int(h_ids.max().item()) + 1
        w = int(w_ids.max().item()) + 1

        flat_ids = h_ids * w + w_ids

        # Create output tensor and scatter data into place
        x_list = []
        for b in range(batch_size):
            data_b = x[b]  # [seq_len, C]
            flat_ids_b = flat_ids[b]  # [seq_len]

            # Initialize output with zeros
            out = Tensor_v3.zeros([h * w, ch], dtype=x.dtype, device=x.device)

            # Scatter: out[flat_ids[i], :] = data[i, :] for each i
            indices = F.reshape(flat_ids_b, [seq_len, 1]).cast(DType.int64)
            out = F.scatter_nd(out, data_b, indices)

            # Reshape from (H * W, C) to (C, H, W)
            out = F.reshape(out, [h, w, ch])
            out = F.permute(out, (2, 0, 1))  # [C, H, W]
            x_list.append(out)

        # Stack batches
        result = F.stack(x_list, axis=0)  # [B, C, H, W]
        return result

    @staticmethod
    def _unpatchify_latents(latents: Tensor_v3) -> Tensor_v3:
        """Unpatchify latents from (B, C, H, W) to (B, C//4, H*2, W*2).

        Args:
            latents: Patchified latents of shape [B, C, H, W].

        Returns:
            Unpatchified latents of shape [B, C//4, H*2, W*2].
        """
        batch_size = latents.shape[0].dim
        num_channels_latents = latents.shape[1].dim
        height = latents.shape[2].dim
        width = latents.shape[3].dim

        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents // 4, 2, 2, height, width),
        )
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents // 4, height * 2, width * 2),
        )
        return latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: DType,
        device: DeviceRef,
        latents: Tensor_v3 | None = None,
    ) -> tuple[Tensor_v3, Tensor_v3]:
        """Prepare latents for the Flux2 pipeline.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of latent channels (before packing).
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type for the latents.
            device: The device to run on.
            latents: Pre-generated latents.

        Returns:
            Tuple of latents and latent image ids.
        """
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
                if isinstance(latents, Tensor_v3)
                else Tensor_v3.from_dlpack(latents)
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

    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        latents: Tensor | None = None,
        prompt_embeds: Tensor | None = None,
        negative_prompt_embeds: Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        max_sequence_length: int = 512,
        hidden_states_layers: list[int] | None = None,
    ):
        r"""Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guidance scale is enabled by setting `guidance_scale` > 1.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            latents (`Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux2.Flux2PipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            hidden_states_layers (`List[int]`, *optional*, defaults to [10, 20, 30]):
                List of layer indices (1-based) to extract hidden states from. For Flux2, layers 10, 20, 30 are stacked.

        Returns:
            [`~pipelines.flux2.Flux2PipelineOutput`] or `tuple`: [`~pipelines.flux2.Flux2PipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        height = height or getattr(
            self.transformer.config, "sample_size", 1024
        ) * self.vae_scale_factor
        width = width or getattr(
            self.transformer.config, "sample_size", 1024
        ) * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device()

        lora_scale = None  # Flux2 may not use LoRA in the same way
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # Encode prompts
        (
            prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
            hidden_states_layers=hidden_states_layers,
        )

        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
                hidden_states_layers=hidden_states_layers,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler, "use_flow_sigmas")
            and self.scheduler.use_flow_sigmas
        ):
            sigmas = None
        image_seq_len = latents.shape[1].dim
        mu = compute_empirical_mu(
            image_seq_len=image_seq_len,
            num_steps=num_inference_steps,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        self._num_timesteps = timesteps.shape[0]

        # Handle guidance
        # Flux2 always uses guidance embeddings
        guidance = Tensor_v3.full(
            [latents.shape[0].dim],
            guidance_scale,
            device=device,
            dtype=prompt_embeds.dtype,
        )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        batch_size = latents.shape[0].dim
        for i in tqdm(range(self._num_timesteps), desc="Denoising"):
            if self._interrupt:
                continue

            t = timesteps[i]
            self._current_timestep = t

            # Convert timestep to V2 Tensor (Buffer) for compiled Model
            # Note: Compiled Model can accept Tensor_v3 (auto-converts), but we convert
            # timestep to V2 for consistency with Flux1 pattern
            # Must cast to prompt_embeds.dtype (bfloat16) to match compiled model input type
            timestep_np = np.full((batch_size,), t, dtype=np.float32) / 1000.0
            timestep = (
                Tensor_v3.from_dlpack(timestep_np)
                .to(prompt_embeds.device)
                .cast(prompt_embeds.dtype)
            )

            # Flux2 transformer call (no pooled_prompt_embeds)
            # Compiled Model accepts Tensor_v3 and auto-converts to V2 internally
            noise_pred = self.transformer(
                latents,
                prompt_embeds,
                timestep,
                latent_image_ids,
                text_ids,
                guidance,
            )[0]

            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    latents,
                    negative_prompt_embeds,
                    timestep,
                    latent_image_ids,
                    negative_text_ids,
                    guidance,
                )[0]
                noise_pred = neg_noise_pred + true_cfg_scale * (
                    noise_pred - neg_noise_pred
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(
                    self, i, t, callback_kwargs
                )

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop(
                    "prompt_embeds", prompt_embeds
                )

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            # Convert to Tensor_v3 if needed (latents may be driver.Tensor after scheduler.step)
            if not isinstance(latents, Tensor_v3):
                latents = Tensor_v3.from_dlpack(latents)

            # Unpack latents using position IDs (Flux2 specific)
            latents_v3 = self._unpack_latents_with_ids(latents, latent_image_ids)

            # Apply BatchNorm inverse transform (Flux2 specific)
            # Flux2 uses BatchNorm statistics instead of scaling_factor/shift_factor
            # VAE weights and bn stats are already bfloat16 - no cast needed
            bn_mean = self.vae.bn.running_mean
            bn_var = self.vae.bn.running_var

            num_channels = bn_mean.shape[0].dim
            bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
            bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))
            bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)

            latents_v3 = latents_v3 * bn_std + bn_mean

            # Unpatchify latents: (B, C, H, W) -> (B, C//4, H*2, W*2)
            latents_v3 = self._unpatchify_latents(latents_v3)

            # VAE decode (weights and graph are bfloat16, no dtype conversion needed)
            image = self.vae.decode(latents_v3.driver_tensor)

            # Convert to Tensor_v3 if decode returns driver.Tensor
            if not isinstance(image, Tensor_v3):
                image = Tensor_v3.from_dlpack(image)

            image = self.image_processor.postprocess(
                image, output_type=output_type
            )

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
