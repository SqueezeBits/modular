# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Flux2 architecture definition."""

from max.graph.weights import WeightsFormat
from max.interfaces import BaseContext, PipelineTask
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .pipeline_flux2 import Flux2Pipeline

flux2_arch = SupportedArchitecture(
    name="Flux2Pipeline",
    task=PipelineTask.IMAGE_GENERATION,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: []},
    example_repo_ids=[
        "black-forest-labs/FLUX.2-dev",
    ],
    pipeline_model=Flux2Pipeline,
    tokenizer=TextTokenizer,
    context_type=BaseContext,
    default_weights_format=WeightsFormat.safetensors,
)
