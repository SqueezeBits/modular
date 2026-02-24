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
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanCausalConv3d as HFWanCausalConv3d,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Conv3d
from max.pipelines.architectures.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWanModel,
    WanCausalConv3d,
    WanRMSNorm,
)
from max.pipelines.architectures.autoencoders.model_config import (
    AutoencoderKLWanConfigBase,
)
from max.pipelines.architectures.wan.pipeline_wan import WanPipeline
from max.tensor import Tensor
from torch.utils.dlpack import from_dlpack


@torch.no_grad()
def test_denormalize_vae_latents_matches_diffusers_formula() -> None:
    torch.manual_seed(7)
    z_dim = 16
    latents = torch.randn(2, z_dim, 3, 5, 7, dtype=torch.float32)
    latents_mean = [float(x) for x in torch.linspace(-1.0, 1.0, z_dim)]
    latents_std = [float(x) for x in torch.linspace(0.5, 2.0, z_dim)]

    latents_mean_t = torch.tensor(latents_mean, dtype=torch.float32).view(
        1, z_dim, 1, 1, 1
    )
    latents_std_t = torch.tensor(latents_std, dtype=torch.float32).view(
        1, z_dim, 1, 1, 1
    )
    latents_recip_std = 1.0 / latents_std_t
    expected = latents / latents_recip_std + latents_mean_t

    actual = WanPipeline.denormalize_vae_latents(
        latents=Tensor.from_dlpack(latents),
        latents_mean=latents_mean,
        latents_std=latents_std,
        z_dim=z_dim,
    )
    actual_torch = from_dlpack(actual).to(torch.float32)

    torch.testing.assert_close(expected, actual_torch, rtol=1e-5, atol=1e-6)


def test_wan_vae_temporal_upsample_formula() -> None:
    # 2 input frames with temporal scale 4 => (2 - 1) * 4 + 1 = 5 frames.
    x = torch.tensor([[[[[0.0]], [[10.0]]]]], dtype=torch.float32).numpy()
    out = AutoencoderKLWanModel.upsample_temporal_linear(
        x, scale_factor_temporal=4
    )
    expected = torch.tensor(
        [0.0, 2.5, 5.0, 7.5, 10.0], dtype=torch.float32
    ).view(1, 1, 5, 1, 1)
    torch.testing.assert_close(torch.from_numpy(out), expected)


@torch.no_grad()
def test_wan_rmsnorm_preserves_rank_and_shape() -> None:
    x = torch.randn(1, 384, 5, 32, 32, dtype=torch.float32)
    norm = WanRMSNorm(
        dim=384,
        channel_first=True,
        images=False,
        dtype=DType.float32,
        device=DeviceRef.CPU(),
    )

    y = norm(Tensor.from_dlpack(x.contiguous()))
    y_torch = from_dlpack(y).to(torch.float32)

    assert y_torch.ndim == x.ndim
    assert tuple(y_torch.shape) == tuple(x.shape)


@torch.no_grad()
def test_conv3d_matches_torch() -> None:
    torch.manual_seed(123)

    x = torch.randn(2, 4, 3, 5, 7, dtype=torch.float32)
    torch_conv = torch.nn.Conv3d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    expected = torch_conv(x)

    max_conv = Conv3d(
        kernel_size=3,
        in_channels=4,
        out_channels=6,
        dtype=DType.float32,
        stride=1,
        padding=1,
        has_bias=True,
        permute=True,
        device=DeviceRef.CPU(),
    )
    max_conv.weight = Tensor.from_dlpack(
        torch_conv.weight.detach().contiguous()
    )
    max_conv.bias = Tensor.from_dlpack(torch_conv.bias.detach().contiguous())

    actual = max_conv(Tensor.from_dlpack(x.contiguous()))
    actual_torch = from_dlpack(actual).to(torch.float32)

    torch.testing.assert_close(actual_torch, expected, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_wan_vae_config_defaults_match_diffusers() -> None:
    cfg = AutoencoderKLWanConfigBase()
    assert cfg.base_dim == 96
    assert cfg.z_dim == 16
    assert cfg.dim_mult == (1, 2, 4, 4)
    assert cfg.num_res_blocks == 2
    assert cfg.temperal_downsample == (False, True, True)


@torch.no_grad()
def test_wan_causal_conv3d_matches_diffusers_reference() -> None:
    torch.manual_seed(2026)

    hf_conv = HFWanCausalConv3d(
        in_channels=4,
        out_channels=6,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
    ).to(torch.float32)
    hf_conv.eval()

    max_conv = WanCausalConv3d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
        dtype=DType.float32,
        device=DeviceRef.CPU(),
    )

    max_conv.weight = Tensor.from_dlpack(
        hf_conv.weight.detach().permute(2, 3, 4, 1, 0).contiguous()
    )
    max_conv.bias = Tensor.from_dlpack(hf_conv.bias.detach().contiguous())

    hidden_states = torch.randn(1, 4, 3, 5, 7, dtype=torch.float32)
    hf_out = hf_conv(hidden_states).to(torch.float32)
    max_out = max_conv(Tensor.from_dlpack(hidden_states.contiguous()))
    max_out_torch = from_dlpack(max_out).to(torch.float32)

    abs_diff = (hf_out - max_out_torch).abs()
    mean_abs_diff = float(abs_diff.mean().item())
    max_abs_diff = float(abs_diff.max().item())

    assert mean_abs_diff < 1e-5, (
        f"mean abs diff too high: {mean_abs_diff:.6f}, "
        f"max abs diff: {max_abs_diff:.6f}"
    )
    assert max_abs_diff < 1e-4, (
        f"max abs diff too high: {max_abs_diff:.6f}, "
        f"mean abs diff: {mean_abs_diff:.6f}"
    )
