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
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .clip import CLIPTextModel, CLIPVisionModel
from .model_config import ClipConfig


class ClipModel(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
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
        self.config = ClipConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        with F.lazy():
            clip = CLIPTextModel(self.config)
            clip.to(self.devices[0])
        self.model = clip.compile(*clip.input_types(), weights=state_dict)
        return self.model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ClipVisionModel(ComponentModel):
    """MAX-native CLIP Vision Model (ViT).

    Replaces the PyTorch CLIPImageEncoderBridge with a compiled MAX graph.
    Handles image preprocessing (resize, center crop, normalize) internally.
    """

    # CLIPImageProcessor constants
    CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        config: dict[str, Any],
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
        self.config = ClipConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        with F.lazy():
            clip = CLIPVisionModel(self.config)
            clip.to(self.devices[0])
        self.model = clip.compile(*clip.input_types(), weights=state_dict)
        return self.model

    def preprocess_image(self, image: Any) -> Any:
        """Preprocess a PIL image for CLIP vision encoding.

        Replicates CLIPImageProcessor: resize shortest edge to 224,
        center crop to 224x224, rescale to [0,1], normalize.

        Args:
            image: PIL Image.

        Returns:
            numpy array of shape [1, 3, 224, 224] in model dtype.
        """
        import numpy as np
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert("RGB")
        size = self.config.image_size

        # Resize shortest edge to `size` (bicubic)
        w, h = image.size
        if h < w:
            new_h = size
            new_w = int(w * size / h + 0.5)
        else:
            new_w = size
            new_h = int(h * size / w + 0.5)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # Center crop to size x size
        w, h = image.size
        left = (w - size) // 2
        top = (h - size) // 2
        image = image.crop((left, top, left + size, top + size))

        # To float32, rescale to [0, 1], normalize
        pixels = np.array(image, dtype=np.float32) / 255.0
        mean = np.array(self.CLIP_IMAGE_MEAN, dtype=np.float32)
        std = np.array(self.CLIP_IMAGE_STD, dtype=np.float32)
        pixels = (pixels - mean) / std

        # [H, W, 3] -> [1, 3, H, W]
        pixels = pixels.transpose(2, 0, 1)[np.newaxis]
        return pixels

    def encode(self, image: Any) -> Any:
        """Encode an image to CLIP features.

        Args:
            image: PIL Image or numpy array.

        Returns:
            Buffer with shape [1, 257, 1280] in model dtype.
        """
        import numpy as np
        from max.driver import Buffer
        from max.pipelines.lib.bfloat16_utils import (
            float32_to_bfloat16_as_uint16,
        )

        pixels = self.preprocess_image(image)

        # Convert to model dtype buffer
        if self.config.dtype == DType.bfloat16:
            u16 = float32_to_bfloat16_as_uint16(pixels)
            pixel_buf = (
                Buffer.from_numpy(u16)
                .to(self.devices[0])
                .view(dtype=DType.bfloat16, shape=pixels.shape)
            )
        else:
            pixel_buf = Buffer.from_numpy(pixels).to(self.devices[0])

        result = self.model(pixel_buf)
        if isinstance(result, tuple):
            result = result[0]
        # Extract Buffer from Tensor if needed
        if hasattr(result, "storage"):
            return result.storage
        return result

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
