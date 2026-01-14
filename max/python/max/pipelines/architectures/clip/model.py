from max.driver import Accelerator, CPU
from max.graph import Graph
from max.graph.weights import Weights
from max.engine import InferenceSession, Model
from max.pipelines.lib.interfaces.base_model import BaseModel
from max.pipelines.lib import SupportedEncoding
from max.driver import Device

from .model_config import ClipConfig
from .clip import CLIPTextModel


class ClipModel(BaseModel):
    config_name = ClipConfig.config_name
    
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
        self.config = ClipConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Model:
        clip = CLIPTextModel(self.config)

        if self.config.device.is_cpu():
            session = InferenceSession([CPU()])
        else:
            session = InferenceSession([Accelerator()])
        state_dict = {
            key: value.data() for key, value in self.weights.items()
        }
        clip.load_state_dict(state_dict)
        with Graph(
            "clip_text_model", input_types=clip.input_types()
        ) as graph:
            outputs = clip(
                *graph.inputs,
                attention_mask=None,
                position_ids=None,
            )
            graph.output(*outputs)
            compiled_graph = graph
        self.session = session.load(
            compiled_graph, weights_registry=clip.state_dict()
        )
    
    def __call__(self, *args, **kwargs):
        return self.session.execute(
            *args,
            **kwargs
        )
