from max.driver import Accelerator, CPU, Device
from max.graph import Graph
from max.graph.weights import Weights
from max.engine import InferenceSession, Model
from max.pipelines.lib.interfaces.base_model import BaseModel
from max.pipelines.lib import SupportedEncoding

from .model_config import FluxConfig
from .flux1 import FluxTransformer2DModel


class Flux1Model(BaseModel):
    config_name = FluxConfig.config_name
    
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
            weights,
        )
        self.config = FluxConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Model:
        flux = FluxTransformer2DModel(self.config)

        if self.config.device.is_cpu():
            session = InferenceSession([CPU()])
        else:
            session = InferenceSession([Accelerator()])
        state_dict = {
            key: value.data() for key, value in self.weights.items()
        }
        flux.load_state_dict(state_dict)
        with Graph(
            "flux_transformer_2d_model", input_types=flux.input_types()
        ) as graph:
            outputs = flux(
                *graph.inputs,
                joint_attention_kwargs={},
                controlnet_block_samples=None,
                controlnet_single_block_samples=None,
                return_dict=False,
                controlnet_blocks_repeat=False,
            )
            graph.output(*outputs)
            compiled_graph = graph
        self.session = session.load(
            compiled_graph, weights_registry=flux.state_dict()
        )
    
    def __call__(self, *args, **kwargs):
        return self.session.execute(
            *args,
            **kwargs
        )
