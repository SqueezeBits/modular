from dataclasses import dataclass

from max.graph import DeviceRef
from max.dtype import DType
from max.pipelines.lib import MAXModelConfigBase
from max.pipelines.lib import SupportedEncoding
from max.driver import Device


@dataclass
class T5ConfigBase(MAXModelConfigBase):
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int | None = None
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    classifier_dropout: float = 0.0
    device: DeviceRef = DeviceRef.GPU()
    dtype: DType = DType.bfloat16


@dataclass
class T5Config(T5ConfigBase):
    config_name = "config.json"

    @staticmethod
    def generate(
        config_dict: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
    ):
        init_dict = {
            key: value for key, value in config_dict.items() if key in T5ConfigBase.__annotations__
        }
        init_dict.update({
            "dtype": encoding.dtype,
            "device": DeviceRef.from_device(devices[0]),
        })
        return T5ConfigBase(
            **init_dict
        )
