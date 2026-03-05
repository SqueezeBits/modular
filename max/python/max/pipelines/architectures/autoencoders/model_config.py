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

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from pydantic import Field


class AutoencoderKLConfigBase(MAXModelConfigBase):
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: list[str] = Field(default_factory=list, max_length=4)
    up_block_types: list[str] = Field(default_factory=list, max_length=4)
    block_out_channels: list[int] = Field(default_factory=list, max_length=4)
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    scaling_factor: float = 0.18215
    shift_factor: float | None = None
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    force_upcast: bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    device: DeviceRef = Field(default_factory=DeviceRef.CPU)
    dtype: DType = DType.bfloat16


class AutoencoderKLConfig(AutoencoderKLConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLConfig":
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return AutoencoderKLConfig(**init_dict)


class AutoencoderKLFlux2Config(AutoencoderKLConfigBase):
    patch_size: tuple[int, int] = (2, 2)
    batch_norm_eps: float = 1e-4
    batch_norm_momentum: float = 0.1
    latent_channels: int = 32  # Flux2 uses 32 channels, Flux1 uses 4

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLFlux2Config":
        """Generate AutoencoderKLFlux2Config from dictionary.

        Args:
            config_dict: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.

        Returns:
            AutoencoderKLFlux2Config instance.
        """
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLConfigBase.__annotations__
        }
        # Add Flux2-specific parameters if present
        flux2_params = ["patch_size", "batch_norm_eps", "batch_norm_momentum"]
        for param in flux2_params:
            if param in config_dict:
                init_dict[param] = config_dict[param]
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return AutoencoderKLFlux2Config(**init_dict)


class AutoencoderKLLTXVideoConfig(AutoencoderKLConfigBase):
    """Configuration for the LTX video VAE component."""

    latent_channels: int = 128
    patch_size: int = 4
    patch_size_t: int = 1
    layers_per_block: tuple[int, ...] = (4, 3, 3, 3, 4)
    decoder_layers_per_block: tuple[int, ...] | None = None
    spatio_temporal_scaling: tuple[bool, ...] = (True, True, True, False)
    decoder_spatio_temporal_scaling: tuple[bool, ...] | None = None
    decoder_inject_noise: tuple[bool, ...] | None = None
    upsample_residual: tuple[bool, ...] | None = None
    upsample_factor: tuple[int, ...] | None = None
    decoder_block_out_channels: tuple[int, ...] | None = None
    resnet_norm_eps: float = 1e-6
    encoder_causal: bool = True
    decoder_causal: bool = False
    spatial_compression_ratio: int | None = None
    temporal_compression_ratio: int | None = None
    timestep_conditioning: bool = False
    sample_height: int = 512
    sample_width: int = 704
    sample_num_frames: int = 161

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLLTXVideoConfig":
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLConfigBase.__annotations__
            or key in AutoencoderKLLTXVideoConfig.__annotations__
        }

        # Normalize list inputs to tuples for strict pydantic typing.
        if "latents_mean" in init_dict and isinstance(
            init_dict["latents_mean"], list
        ):
            init_dict["latents_mean"] = tuple(init_dict["latents_mean"])
        if "latents_std" in init_dict and isinstance(
            init_dict["latents_std"], list
        ):
            init_dict["latents_std"] = tuple(init_dict["latents_std"])

        if "layers_per_block" in init_dict and isinstance(
            init_dict["layers_per_block"], list
        ):
            init_dict["layers_per_block"] = tuple(init_dict["layers_per_block"])

        if "spatio_temporal_scaling" in init_dict and isinstance(
            init_dict["spatio_temporal_scaling"], list
        ):
            init_dict["spatio_temporal_scaling"] = tuple(
                bool(x) for x in init_dict["spatio_temporal_scaling"]
            )

        if "decoder_layers_per_block" in init_dict and isinstance(
            init_dict["decoder_layers_per_block"], list
        ):
            init_dict["decoder_layers_per_block"] = tuple(
                init_dict["decoder_layers_per_block"]
            )

        if "decoder_spatio_temporal_scaling" in init_dict and isinstance(
            init_dict["decoder_spatio_temporal_scaling"], list
        ):
            init_dict["decoder_spatio_temporal_scaling"] = tuple(
                bool(x) for x in init_dict["decoder_spatio_temporal_scaling"]
            )

        if "decoder_inject_noise" in init_dict and isinstance(
            init_dict["decoder_inject_noise"], list
        ):
            init_dict["decoder_inject_noise"] = tuple(
                bool(x) for x in init_dict["decoder_inject_noise"]
            )

        if "upsample_residual" in init_dict and isinstance(
            init_dict["upsample_residual"], list
        ):
            init_dict["upsample_residual"] = tuple(
                bool(x) for x in init_dict["upsample_residual"]
            )

        if "upsample_factor" in init_dict and isinstance(
            init_dict["upsample_factor"], list
        ):
            init_dict["upsample_factor"] = tuple(
                int(x) for x in init_dict["upsample_factor"]
            )

        if "decoder_block_out_channels" in init_dict and isinstance(
            init_dict["decoder_block_out_channels"], list
        ):
            init_dict["decoder_block_out_channels"] = tuple(
                int(x) for x in init_dict["decoder_block_out_channels"]
            )

        block_out_channels = init_dict.get("block_out_channels", [128, 256, 512, 512])
        num_blocks = len(block_out_channels)

        layers_per_block = init_dict.get(
            "layers_per_block", (4, 3, 3, 3, 4)
        )
        if isinstance(layers_per_block, int):
            layers_per_block = tuple([layers_per_block] * (num_blocks + 1))
        init_dict["layers_per_block"] = tuple(int(x) for x in layers_per_block)

        if init_dict.get("decoder_layers_per_block") is None:
            init_dict["decoder_layers_per_block"] = init_dict["layers_per_block"]

        if init_dict.get("decoder_block_out_channels") is not None:
            init_dict["block_out_channels"] = list(
                init_dict["decoder_block_out_channels"]
            )
            block_out_channels = init_dict["block_out_channels"]
            num_blocks = len(block_out_channels)

        if init_dict.get("decoder_spatio_temporal_scaling") is None:
            init_dict["decoder_spatio_temporal_scaling"] = init_dict.get(
                "spatio_temporal_scaling",
                (True, True, True, False),
            )

        if init_dict.get("decoder_inject_noise") is None:
            init_dict["decoder_inject_noise"] = tuple(
                False for _ in range(len(init_dict["layers_per_block"]))
            )

        if init_dict.get("upsample_residual") is None:
            init_dict["upsample_residual"] = tuple(
                False for _ in range(num_blocks)
            )

        if init_dict.get("upsample_factor") is None:
            init_dict["upsample_factor"] = tuple(1 for _ in range(num_blocks))

        if init_dict.get("spatial_compression_ratio") is None:
            init_dict["spatial_compression_ratio"] = int(
                init_dict.get("patch_size", 4)
            ) * (2 ** sum(bool(x) for x in init_dict["spatio_temporal_scaling"]))

        if init_dict.get("temporal_compression_ratio") is None:
            init_dict["temporal_compression_ratio"] = int(
                init_dict.get("patch_size_t", 1)
            ) * (2 ** sum(bool(x) for x in init_dict["spatio_temporal_scaling"]))

        init_dict.update(
            {
                # LTX video VAE is currently tuned for bf16 execution in MAX.
                "dtype": DType.bfloat16,
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return AutoencoderKLLTXVideoConfig(**init_dict)
