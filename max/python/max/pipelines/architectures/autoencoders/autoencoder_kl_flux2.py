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

from typing import ClassVar, Optional

from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.nn.module_v3 import Module
from max.pipelines.lib import SupportedEncoding

from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLFlux2Config
from .vae import Decoder


class AutoencoderKLFlux2(Module[[Tensor], Tensor]):
    r"""A VAE model with KL loss for Flux2, encoding images into latents and decoding latent representations into images using module_v3.

    This is similar to AutoencoderKL but uses Flux2-specific configuration
    with 32 latent channels (vs 4 for Flux1) and supports BatchNorm statistics
    for latent patchification.
    """

    def __init__(
        self,
        config: AutoencoderKLFlux2Config,
    ) -> None:
        """Initialize VAE AutoencoderKLFlux2 model.

        Args:
            config: AutoencoderKLFlux2 configuration containing channel sizes, block
                structure, normalization settings, BatchNorm parameters, and device/dtype information.
        """
        super().__init__()
        self.decoder = Decoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
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
        """Apply AutoencoderKLFlux2 forward pass (decoding only).

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].
            temb: Optional time embedding tensor.

        Returns:
            Decoded image tensor of shape [N, C_out, H, W].
        """
        return self.decoder(z, temb)


class BatchNormStats:
    """Container for BatchNorm statistics.

    This class provides a simple interface to access BatchNorm running statistics
    (mean and variance) for Flux2's latent patchification process.
    """

    def __init__(self, running_mean: Tensor, running_var: Tensor) -> None:
        """Initialize BatchNormStats.

        Args:
            running_mean: Running mean tensor.
            running_var: Running variance tensor.
        """
        self.running_mean = running_mean
        self.running_var = running_var


class AutoencoderKLFlux2Model(BaseAutoencoderModel):
    """MaxModel wrapper for AutoencoderKLFlux2.

    This class provides the MaxModel interface for AutoencoderKLFlux2, handling
    configuration, weight loading, model compilation, and BatchNorm statistics
    for Flux2's latent patchification.
    """

    config_name: ClassVar[str] = AutoencoderKLFlux2Config.config_name

    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize AutoencoderKLFlux2Model.

        Args:
            config: Model configuration dictionary.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        # Initialize BatchNorm statistics BEFORE super().__init__()
        # because super().__init__() calls load_model() which sets these values
        self.bn_running_mean: Optional[Tensor] = None
        self.bn_running_var: Optional[Tensor] = None

        super().__init__(
            config=config,
            encoding=encoding,
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLFlux2Config,
            autoencoder_class=AutoencoderKLFlux2,
        )

    def load_model(self) -> None:
        """Load and compile the decoder model with BatchNorm statistics.

        Extracts decoder weights and BatchNorm statistics (running_mean, running_var)
        from the full model weights and compiles the decoder for inference.
        
        Following max-diffusers pattern: all weights are converted to bfloat16.
        """
        from max.graph import DeviceRef
        
        # Helper functions to convert weights to target dtype (bfloat16)
        # Following max-diffusers pattern: to_bf16() and to_bf16_gpu()
        target_dtype = self.config.dtype
        
        def to_bf16(data) -> Tensor:
            """Convert weight data to bfloat16 Tensor."""
            return Tensor.from_dlpack(data).cast(target_dtype)
        
        def to_bf16_gpu(data) -> Tensor:
            """Convert weight data to bfloat16 Tensor on target device."""
            return Tensor.from_dlpack(data).to(self.devices[0]).cast(target_dtype)
        
        def to_bf16_dlpack(data):
            """Convert weight data to bfloat16 and return as DLPackArray for compile()."""
            tensor = Tensor.from_dlpack(data).cast(target_dtype)
            return tensor.driver_tensor

        state_dict = {}

        for key, value in self.weights.items():
            weight_data = value.data().data  # Get DLPackArray from WeightData
            
            if key.startswith("decoder."):
                # Remove "decoder." prefix for decoder weights
                # Convert to bfloat16 following max-diffusers pattern
                # compile() expects DLPackArray, so convert Tensor back to driver_tensor
                state_dict[key.removeprefix("decoder.")] = to_bf16_dlpack(weight_data)
            elif key.startswith("post_quant_conv."):
                # Keep post_quant_conv prefix as-is
                # compile() expects DLPackArray, so convert Tensor back to driver_tensor
                state_dict[key] = to_bf16_dlpack(weight_data)
            elif key == "bn.running_mean" or key == "latent_bn.running_mean":
                # Load BatchNorm running mean as Tensor and convert to bfloat16
                self.bn_running_mean = to_bf16_gpu(weight_data)
            elif key == "bn.running_var" or key == "latent_bn.running_var":
                # Load BatchNorm running variance as Tensor and convert to bfloat16
                self.bn_running_var = to_bf16_gpu(weight_data)
            # Note: encoder weights are filtered out (not included in state_dict)

        # Compile decoder
        with F.lazy():
            autoencoder = self.autoencoder_class(self.config)
            autoencoder.decoder.to(self.devices[0])

        self.model = autoencoder.decoder.compile(
            *autoencoder.decoder.input_types(), weights=state_dict
        )

    @property
    def bn(self) -> BatchNormStats:
        """Property to access BatchNorm statistics, compatible with diffusers API.

        This returns a simple object with running_mean and running_var attributes
        for compatibility with pipeline code that accesses self.vae.bn.running_mean.
        The statistics are returned as MAX Tensors.

        Returns:
            BatchNormStats: Object containing running_mean and running_var.

        Raises:
            ValueError: If BatchNorm statistics are not loaded.
        """
        if self.bn_running_mean is None or self.bn_running_var is None:
            raise ValueError(
                "BatchNorm statistics (running_mean, running_var) not loaded. "
                "Make sure the model weights contain 'bn.running_mean' and 'bn.running_var'."
            )

        return BatchNormStats(self.bn_running_mean, self.bn_running_var)
