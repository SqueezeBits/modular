from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from max.driver import Device
from max.engine import Model
from max.graph.weights import Weights

if TYPE_CHECKING:
    from max.pipelines.lib import SupportedEncoding

class BaseModel(ABC):
    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights
    ) -> None:
        self.config = config
        self.encoding = encoding
        self.devices = devices
        self.weights = weights
    
    @abstractmethod
    def load_model(self) -> Model:
        ...
