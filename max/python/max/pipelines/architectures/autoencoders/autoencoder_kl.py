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

from collections.abc import Callable
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.profiler import traced

from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLConfig
from .vae import Decoder, Encoder


class AutoencoderKL(Module[[Tensor, Tensor | None], Tensor]):
    r"""A VAE model with KL loss for encoding images into latents and decoding latent representations into images."""

    def __init__(
        self,
        config: AutoencoderKLConfig,
    ) -> None:
        """Initialize VAE AutoencoderKL model.

        Args:
            config: Autoencoder configuration containing channel sizes, block
                structure, normalization settings, and device/dtype information.
        """
        super().__init__()
        self.encoder = Encoder(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            down_block_types=tuple(config.down_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            double_z=True,
            mid_block_add_attention=config.mid_block_add_attention,
            use_quant_conv=config.use_quant_conv,
            device=config.device,
            dtype=config.dtype,
        )
        self.decoder = Decoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            norm_type="group",
            mid_block_add_attention=config.mid_block_add_attention,
            use_post_quant_conv=config.use_post_quant_conv,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, z: Tensor, temb: Tensor | None = None) -> Tensor:
        """Apply AutoencoderKL forward pass (decoding only).

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].
            temb: Optional time embedding tensor.

        Returns:
            Decoded image tensor of shape [N, C_out, H, W].
        """
        return self.decoder(z, temb)


class AutoencoderKLModel(BaseAutoencoderModel):
    """ComponentModel wrapper for AutoencoderKL.

    This class provides the ComponentModel interface for AutoencoderKL,
    handling configuration, weight loading, and model compilation.
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        **kwargs: Any,
    ) -> None:
        """Initialize AutoencoderKLModel.

        Args:
            config: Model configuration dictionary.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
            **kwargs: Additional keyword arguments forwarded to ComponentModel.
        """
        super().__init__(
            config=config,
            encoding=encoding,
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLConfig,
            autoencoder_class=AutoencoderKL,
            **kwargs,
        )

    @traced(message="AutoencoderKLModel.build_fused_decode")
    def build_fused_decode(self, device: Device) -> Callable[..., Any]:
        """Build fused unpack + latent denorm + decoder + uint8 conversion.

        Accepts packed latents ``(B, S, C)`` plus shape-carrier tensors whose
        lengths encode packed spatial dims ``half_h`` and ``half_w``.
        """
        dtype = self.config.dtype
        device_ref = DeviceRef.from_device(device)

        fused_weights: dict[str, Any] = {}
        for key, value in self.weights.items():
            adapted_key = key
            while adapted_key.startswith(("vae.", "model.")):
                if adapted_key.startswith("vae."):
                    adapted_key = adapted_key.removeprefix("vae.")
                    continue
                adapted_key = adapted_key.removeprefix("model.")

            weight_data = value.data()
            if weight_data.dtype != dtype:
                if weight_data.dtype.is_float() and dtype.is_float():
                    weight_data = weight_data.astype(dtype)

            if adapted_key.startswith("decoder."):
                fused_weights[adapted_key] = weight_data
            elif adapted_key.startswith("post_quant_conv."):
                fused_weights[f"decoder.{adapted_key}"] = weight_data

        with F.lazy():
            autoencoder = AutoencoderKL(self.config)
            fused = _PostprocessAndDecodeKL(
                decoder=autoencoder.decoder,
                scaling_factor=float(self.config.scaling_factor),
                shift_factor=float(self.config.shift_factor or 0.0),
                device=device_ref,
                dtype=dtype,
            )
            fused.to(device)
            self._fused_decode = fused.compile(
                *fused.input_types(), weights=fused_weights
            )

        return self._fused_decode


class _PostprocessAndDecodeKL(Module[..., Tensor]):
    """Fused postprocess + decode for standard AutoencoderKL pipelines."""

    def __init__(
        self,
        decoder: Decoder,
        scaling_factor: float,
        shift_factor: float,
        *,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self._device = device
        self._dtype = dtype

    def forward(
        self,
        latents_bsc: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> Tensor:
        batch = latents_bsc.shape[0]
        c = latents_bsc.shape[2]
        half_h = h_carrier.shape[0]
        half_w = w_carrier.shape[0]

        # Assert seq == half_h * half_w for symbolic reshape validation.
        latents_bsc = F.rebind(latents_bsc, [batch, half_h * half_w, c])
        latents = F.reshape(latents_bsc, (batch, half_h, half_w, c))
        latents = F.rebind(latents, [batch, half_h, half_w, (c // 4) * 4])
        latents = F.reshape(latents, (batch, half_h, half_w, 2, 2, c // 4))
        latents = F.permute(latents, (0, 5, 1, 3, 2, 4))
        latents = F.reshape(latents, (batch, c // 4, half_h * 2, half_w * 2))
        latents = (latents / self.scaling_factor) + self.shift_factor

        decoded = self.decoder(latents, None)
        decoded = F.permute(decoded, (0, 2, 3, 1))
        decoded = decoded * 0.5 + 0.5
        decoded = F.max(decoded, 0.0)
        decoded = F.min(decoded, 1.0)
        decoded = decoded * 255.0
        return F.transfer_to(F.cast(decoded, DType.uint8), DeviceRef.CPU())

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch", "seq", "channels"],
                device=self._device,
            ),
            TensorType(
                DType.float32,
                shape=["half_h"],
                device=self._device,
            ),
            TensorType(
                DType.float32,
                shape=["half_w"],
                device=self._device,
            ),
        )
