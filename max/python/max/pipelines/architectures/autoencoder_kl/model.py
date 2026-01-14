from max.driver import CPU, Accelerator, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.base_model import BaseModel

from .autoencoder_kl import AutoencoderKL
from .model_config import AutoencoderKLConfig


class AutoencoderKLModel(BaseModel):
    config_name = AutoencoderKLConfig.config_name
    
    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights
        )
        self.config = AutoencoderKLConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> None:
        autoencoder_kl = AutoencoderKL(self.config)

        if self.config.device.is_cpu():
            session = InferenceSession([CPU()])
        else:
            session = InferenceSession([Accelerator()])

        self.load_decoder(session, autoencoder_kl)

    def load_decoder(self, session: InferenceSession, autoencoder_kl: AutoencoderKL) -> Model:
        state_dict = {
            key: value.data() for key, value in self.weights.items()
            if not key.startswith("encoder.")
        }
        autoencoder_kl.load_state_dict(state_dict)
        with Graph(
            "autoencoder_kl_decoder", input_types=autoencoder_kl.decoder.input_types()
        ) as graph:
            outputs = autoencoder_kl.decoder(*graph.inputs)
            graph.output(outputs)
            compiled_graph = graph
        self.decode_session = session.load(
            compiled_graph, weights_registry=autoencoder_kl.state_dict()
        )
    
    def decode(self, *args, **kwargs):
        return self.decode_session.execute(
            *args,
            **kwargs
        )
