from dataclasses import dataclass

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding


@dataclass
class AutoencoderKLConfigBase(MAXModelConfigBase):
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: tuple[str] = ("DownEncoderBlock2D",)
    up_block_types: tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: tuple[int] = (64,)
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    scaling_factor: float = 0.18215
    shift_factor: float | None = None
    latents_mean: tuple[float] | None = None
    latents_std: tuple[float] | None = None
    force_upcast: bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    device: DeviceRef = DeviceRef.CPU()
    dtype: DType = DType.bfloat16


@dataclass
class AutoencoderKLConfig(AutoencoderKLConfigBase):
    config_name = "config.json"

    @staticmethod
    def generate(
        config_dict: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
    ):
        init_dict = {
            key: value for key, value in config_dict.items() if key in AutoencoderKLConfigBase.__annotations__
        }
        init_dict.update({
            "dtype": encoding.dtype,
            "device": DeviceRef.from_device(devices[0]),
        })
        return AutoencoderKLConfigBase(
            **init_dict
        )
