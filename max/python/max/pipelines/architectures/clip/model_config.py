from dataclasses import dataclass

from max.graph import DeviceRef
from max.dtype import DType
from max.pipelines.lib import MAXModelConfigBase
from max.pipelines.lib import SupportedEncoding
from max.driver import Device

@dataclass
class ClipConfigBase(MAXModelConfigBase):
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu",
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    dtype: DType = DType.bfloat16
    device: DeviceRef = DeviceRef.GPU()


@dataclass
class ClipConfig(ClipConfigBase):
    config_name = "config.json"

    @staticmethod
    def generate(
        config_dict: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
    ):
        init_dict = {
            key: value for key, value in config_dict.items() if key in ClipConfigBase.__annotations__
        }
        init_dict.update({
            "dtype": encoding.dtype,
            "device": DeviceRef.from_device(devices[0]),
        })
        return ClipConfigBase(
            **init_dict
        )
