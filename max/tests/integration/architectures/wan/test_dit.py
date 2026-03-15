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

import numpy as np
import torch
from max.pipelines.architectures.wan.model_config import WanConfigBase
from max.pipelines.architectures.wan.pipeline_wan import (
    WanPipeline,
    _WanTensorUniPCScheduler,
)
from max.pipelines.lib.diffusion_schedulers import UniPCMultistepScheduler
from max.driver import Buffer


def test_wan_dit_config_defaults_match_diffusers_signature() -> None:
    config = WanConfigBase()
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


def test_compute_video_latent_shape_matches_wan_tokenizer_shape() -> None:
    assert WanPipeline.compute_video_latent_shape(
        batch_size=1,
        z_dim=16,
        num_frames=81,
        height=720,
        width=1280,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
    ) == (1, 16, 21, 90, 160)

    assert WanPipeline.compute_video_latent_shape(
        batch_size=2,
        z_dim=16,
        num_frames=82,
        height=722,
        width=1287,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
    ) == (2, 16, 22, 90, 160)


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
        hidden_states=Buffer.from_numpy(hidden_states.numpy().copy()),
        attention_mask=Buffer.from_numpy(attention_mask.numpy().astype(np.int64)),
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    actual_torch = torch.from_numpy(np.from_dlpack(actual)).to(torch.float32)
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
        hidden_states=Buffer.from_numpy(hidden_states.numpy().copy()),
        attention_mask=Buffer.from_numpy(attention_mask.numpy().astype(np.int64)),
        num_videos_per_prompt=1,
        max_sequence_length=4,
    )
    actual_torch = torch.from_numpy(np.from_dlpack(actual)).to(torch.float32)
    torch.testing.assert_close(actual_torch, expected, rtol=0.0, atol=0.0)


@torch.no_grad()
def test_tensor_unipc_scheduler_matches_numpy_scheduler() -> None:
    scheduler_np = UniPCMultistepScheduler(use_flow_sigmas=True, flow_shift=3.0)
    scheduler_np.set_timesteps(4)

    scheduler_tensor = UniPCMultistepScheduler(
        use_flow_sigmas=True, flow_shift=3.0
    )
    scheduler_tensor.set_timesteps(4)
    coefficients = _WanTensorUniPCScheduler._build_step_coefficients(
        scheduler_tensor
    )

    torch.manual_seed(2026)
    sample = torch.randn(1, 2, 3, dtype=torch.float32).numpy()
    model_outputs: list[torch.Tensor | None] = [None] * scheduler_tensor.solver_order
    last_sample: torch.Tensor | None = None

    assert scheduler_np.timesteps is not None
    for idx, timestep in enumerate(scheduler_np.timesteps):
        coeffs = coefficients[idx]
        model_output = torch.randn(1, 2, 3, dtype=torch.float32).numpy()
        expected = scheduler_np.step(
            model_output,
            int(timestep),
            sample,
        )

        sample_t = torch.from_numpy(sample).to(torch.float64)
        model_output_t = torch.from_numpy(model_output).to(torch.float64)
        converted = sample_t - coeffs.sigma * model_output_t

        previous_model_output = model_outputs[-1]
        older_model_output = model_outputs[-2] if len(model_outputs) > 1 else None

        corrected_sample = sample_t
        if coeffs.corrector_order == 1:
            assert last_sample is not None
            assert previous_model_output is not None
            corrected_sample = (
                coeffs.corrector_sample_scale * last_sample
                + coeffs.corrector_m0_scale * previous_model_output
                + coeffs.corrector_mt_scale * converted
            )
        elif coeffs.corrector_order == 2:
            assert last_sample is not None
            assert previous_model_output is not None
            assert older_model_output is not None
            corrected_sample = (
                coeffs.corrector_sample_scale * last_sample
                + coeffs.corrector_m0_scale * previous_model_output
                + coeffs.corrector_m1_scale * older_model_output
                + coeffs.corrector_mt_scale * converted
            )

        for output_idx in range(len(model_outputs) - 1):
            model_outputs[output_idx] = model_outputs[output_idx + 1]
        model_outputs[-1] = converted

        actual_torch = (
            coeffs.predictor_sample_scale * corrected_sample
            + coeffs.predictor_m0_scale * converted
        )
        if coeffs.predictor_order == 2:
            history_model_output = model_outputs[0]
            assert history_model_output is not None
            actual_torch = (
                actual_torch
                + coeffs.predictor_m1_scale * history_model_output
            )
        actual_torch = actual_torch.to(torch.float32)
        expected_torch = torch.from_numpy(expected).to(torch.float32)
        torch.testing.assert_close(
            actual_torch,
            expected_torch,
            rtol=1e-5,
            atol=1e-5,
        )
        last_sample = corrected_sample
        sample = expected.astype(np.float32)
