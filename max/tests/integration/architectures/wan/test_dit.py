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

from __future__ import annotations

import torch
from max.pipelines.architectures.wan.model_config import WanDiTConfigBase
from max.pipelines.architectures.wan.pipeline_wan import WanPipeline
from max.experimental.tensor import Tensor
from torch.utils.dlpack import from_dlpack


def test_wan_dit_config_defaults_match_diffusers_signature() -> None:
    config = WanDiTConfigBase()
    assert config.patch_size == (1, 2, 2)
    assert config.num_attention_heads == 40
    assert config.attention_head_dim == 128
    assert config.in_channels == 16
    assert config.out_channels == 16
    assert config.text_dim == 4096
    assert config.freq_dim == 256
    assert config.ffn_dim == 13824
    assert config.num_layers == 40
    assert config.cross_attn_norm is True
    assert config.qk_norm == "rms_norm_across_heads"
    assert config.eps == 1e-6
    assert config.image_dim is None
    assert config.added_kv_proj_dim is None
    assert config.rope_max_seq_len == 1024
    assert config.pos_embed_seq_len is None


def test_boundary_timestep_and_stage_selection() -> None:
    boundary = WanPipeline.compute_boundary_timestep(0.875, 1000)
    assert boundary == 875.0
    assert WanPipeline.use_low_noise_transformer(900.0, boundary) is False
    assert WanPipeline.use_low_noise_transformer(800.0, boundary) is True
    assert WanPipeline.use_low_noise_transformer(100.0, None) is False


@torch.no_grad()
def test_get_t5_prompt_embeds_from_hidden_matches_diffusers_logic() -> None:
    hidden_states = torch.tensor(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ],
            [
                [5.0, 50.0],
                [6.0, 60.0],
                [7.0, 70.0],
                [8.0, 80.0],
            ],
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.int64,
    )
    num_videos_per_prompt = 2
    max_sequence_length = 4

    expected = hidden_states * attention_mask.unsqueeze(-1).to(torch.float32)
    expected = expected.repeat(1, num_videos_per_prompt, 1).view(
        hidden_states.shape[0] * num_videos_per_prompt,
        max_sequence_length,
        hidden_states.shape[-1],
    )

    actual = WanPipeline.get_t5_prompt_embeds_from_hidden(
        hidden_states=Tensor.from_dlpack(hidden_states),
        attention_mask=Tensor.from_dlpack(attention_mask),
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    actual_torch = from_dlpack(actual).to(torch.float32)
    torch.testing.assert_close(actual_torch, expected, rtol=0.0, atol=0.0)


@torch.no_grad()
def test_get_t5_prompt_embeds_uses_prefix_seq_len_like_diffusers() -> None:
    # Diffusers uses seq_len = attention_mask.sum(), then takes prefix [:seq_len].
    hidden_states = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32
    )
    attention_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.int64)

    expected = torch.tensor([[[1.0], [2.0], [0.0], [0.0]]], dtype=torch.float32)
    actual = WanPipeline.get_t5_prompt_embeds_from_hidden(
        hidden_states=Tensor.from_dlpack(hidden_states),
        attention_mask=Tensor.from_dlpack(attention_mask),
        num_videos_per_prompt=1,
        max_sequence_length=4,
    )
    actual_torch = from_dlpack(actual).to(torch.float32)
    torch.testing.assert_close(actual_torch, expected, rtol=0.0, atol=0.0)
