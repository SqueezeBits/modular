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
from max.pipelines.architectures.autoencoders.autoencoder_kl_wan import (
    WanCausalConv3d,
    WanRMSNorm,
)
from max.pipelines.architectures.autoencoders.model_config import (
    AutoencoderKLWanConfigBase,
)
from max.pipelines.architectures.wan.pipeline_wan import WanPipeline
from max.experimental.tensor import Tensor
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


@torch.no_grad()
def test_wan_rmsnorm_parameter_layout() -> None:
    norm = WanRMSNorm(
        dim=384,
        channel_first=True,
        images=False,
        dtype=DType.float32,
        device=DeviceRef.CPU(),
    )
    assert tuple(norm.gamma.shape) == (384, 1, 1, 1)


@torch.no_grad()
def test_wan_vae_config_defaults_match_diffusers() -> None:
    cfg = AutoencoderKLWanConfigBase()
    assert cfg.base_dim == 96
    assert cfg.z_dim == 16
    assert cfg.dim_mult == (1, 2, 4, 4)
    assert cfg.num_res_blocks == 2
    assert cfg.temperal_downsample == (False, True, True)


@torch.no_grad()
def test_wan_causal_conv3d_uses_qrscf_weight_layout() -> None:
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

    max_conv.load_state_dict(
        {
            "weight": Tensor.from_dlpack(
                hf_conv.weight.detach().permute(2, 3, 4, 1, 0).contiguous()
            ),
            "bias": Tensor.from_dlpack(hf_conv.bias.detach().contiguous()),
        }
    )
    assert tuple(max_conv.filter.shape) == (3, 3, 3, 4, 6)
