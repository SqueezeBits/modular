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

"""Weight key remapping for Qwen2.5-VL text encoder.

HuggingFace Qwen2.5-VL weights use the same naming pattern as Llama/Qwen models.
We reuse the LLAMA_SAFETENSOR_MAPPING to remap weight keys.
"""

# The Qwen2.5-VL text encoder weights from HuggingFace use keys like:
#   model.embed_tokens.weight
#   model.layers.0.self_attn.q_proj.weight
#   model.layers.0.self_attn.q_proj.bias
#   model.layers.0.self_attn.k_proj.weight
#   model.layers.0.self_attn.k_proj.bias
#   model.layers.0.self_attn.v_proj.weight
#   model.layers.0.self_attn.v_proj.bias
#   model.layers.0.self_attn.o_proj.weight
#   model.layers.0.mlp.gate_proj.weight
#   model.layers.0.mlp.up_proj.weight
#   model.layers.0.mlp.down_proj.weight
#   model.layers.0.input_layernorm.weight
#   model.layers.0.post_attention_layernorm.weight
#
# The LLAMA_SAFETENSOR_MAPPING handles the "model." prefix stripping
# which maps to our module attribute names.
