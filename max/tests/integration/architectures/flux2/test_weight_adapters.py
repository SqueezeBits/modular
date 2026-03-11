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
from max.driver import Buffer
from max.graph.weights import WeightData
from max.pipelines.architectures.flux2.weight_adapters import (
    adapt_bflabs_flux2_transformer_weights,
    convert_safetensor_state_dict,
)
from safetensors import safe_open
from safetensors.torch import save_file


def _as_numpy(weight: WeightData) -> np.ndarray:
    return np.asarray(Buffer.from_dlpack(weight.data).to_numpy())


def _write_minimal_bflabs_fp8_checkpoint(src) -> None:
    packed_qkv = torch.full((6, 2), 2.0, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    single_linear = torch.full((6, 2), 1.5, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    save_file(
        {
            "img_in.weight": torch.zeros((2, 2), dtype=torch.float32),
            "txt_in.weight": torch.zeros((2, 2), dtype=torch.float32),
            "time_in.in_layer.weight": torch.zeros((2, 2), dtype=torch.float32),
            "time_in.out_layer.weight": torch.zeros((2, 2), dtype=torch.float32),
            "double_stream_modulation_img.lin.weight": torch.zeros(
                (2, 2), dtype=torch.float32
            ),
            "double_stream_modulation_txt.lin.weight": torch.zeros(
                (2, 2), dtype=torch.float32
            ),
            "single_stream_modulation.lin.weight": torch.zeros(
                (2, 2), dtype=torch.float32
            ),
            "final_layer.adaLN_modulation.1.weight": torch.zeros(
                (2, 2), dtype=torch.float32
            ),
            "final_layer.linear.weight": torch.zeros(
                (2, 2), dtype=torch.float32
            ),
            "double_blocks.0.img_attn.qkv.weight": packed_qkv,
            "double_blocks.0.img_attn.qkv.input_scale": torch.tensor(
                0.25, dtype=torch.float32
            ),
            "double_blocks.0.img_attn.qkv.weight_scale": torch.tensor(
                0.001, dtype=torch.float32
            ),
            "single_blocks.0.linear1.weight": single_linear,
            "single_blocks.0.linear1.input_scale": torch.tensor(
                0.5, dtype=torch.float32
            ),
            "single_blocks.0.linear1.weight_scale": torch.tensor(
                0.002, dtype=torch.float32
            ),
            "single_blocks.0.linear2.weight": torch.zeros(
                (2, 2), dtype=torch.float32
            ),
        },
        str(src),
    )


def test_convert_legacy_scalar_scales_keeps_weight_scale_direct() -> None:
    state_dict = {
        "img_in.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32), "img_in.weight"
        ),
        "txt_in.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32), "txt_in.weight"
        ),
        "time_in.in_layer.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32), "time_in.in_layer.weight"
        ),
        "time_in.out_layer.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32), "time_in.out_layer.weight"
        ),
        "double_stream_modulation_img.lin.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32),
            "double_stream_modulation_img.lin.weight",
        ),
        "double_stream_modulation_txt.lin.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32),
            "double_stream_modulation_txt.lin.weight",
        ),
        "single_stream_modulation.lin.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32),
            "single_stream_modulation.lin.weight",
        ),
        "final_layer.adaLN_modulation.1.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32),
            "final_layer.adaLN_modulation.1.weight",
        ),
        "final_layer.linear.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32), "final_layer.linear.weight"
        ),
        "double_blocks.0.img_attn.qkv.weight": WeightData.from_numpy(
            np.zeros((6, 2), dtype=np.float32),
            "double_blocks.0.img_attn.qkv.weight",
        ),
        "double_blocks.0.img_attn.qkv.input_scale": WeightData.from_numpy(
            np.array(0.25, dtype=np.float32),
            "double_blocks.0.img_attn.qkv.input_scale",
        ),
        "double_blocks.0.img_attn.qkv.weight_scale": WeightData.from_numpy(
            np.array(0.001, dtype=np.float32),
            "double_blocks.0.img_attn.qkv.weight_scale",
        ),
        "single_blocks.0.linear1.weight": WeightData.from_numpy(
            np.zeros((6, 2), dtype=np.float32),
            "single_blocks.0.linear1.weight",
        ),
        "single_blocks.0.linear1.input_scale": WeightData.from_numpy(
            np.array(0.5, dtype=np.float32),
            "single_blocks.0.linear1.input_scale",
        ),
        "single_blocks.0.linear1.weight_scale": WeightData.from_numpy(
            np.array(0.002, dtype=np.float32),
            "single_blocks.0.linear1.weight_scale",
        ),
        "single_blocks.0.linear2.weight": WeightData.from_numpy(
            np.zeros((2, 2), dtype=np.float32),
            "single_blocks.0.linear2.weight",
        ),
    }

    converted = convert_safetensor_state_dict(state_dict)

    q_input_scale = converted["transformer_blocks.0.attn.to_q.input_scale"]
    q_weight_scale = converted["transformer_blocks.0.attn.to_q.weight_scale"]
    single_input_scale = converted[
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.input_scale"
    ]
    single_weight_scale = converted[
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight_scale"
    ]

    np.testing.assert_allclose(_as_numpy(q_input_scale), np.array(4.0))
    np.testing.assert_allclose(_as_numpy(q_weight_scale), np.array(0.001))
    np.testing.assert_allclose(_as_numpy(single_input_scale), np.array(2.0))
    np.testing.assert_allclose(_as_numpy(single_weight_scale), np.array(0.002))


def test_adapt_bflabs_flux2_transformer_weights_persists_adapted_file(
    tmp_path,
) -> None:
    src = tmp_path / "flux-2-klein-4b-fp8.safetensors"
    _write_minimal_bflabs_fp8_checkpoint(src)

    adapted = adapt_bflabs_flux2_transformer_weights(src)

    assert adapted == tmp_path / "flux-2-klein-4b-fp8.max.safetensors"
    assert adapted.exists()
    assert adapt_bflabs_flux2_transformer_weights(src) == adapted

    with safe_open(str(adapted), framework="pt", device="cpu") as f:
        assert "transformer_blocks.0.attn.to_q.weight" in f.keys()
        assert "img_in.weight" not in f.keys()
        assert "transformer_blocks.0.attn.to_q.input_scale" not in f.keys()
        weight_scale = f.get_tensor("transformer_blocks.0.attn.to_q.weight_scale")
        assert weight_scale.shape == (1, 1)
        assert weight_scale.dtype == torch.float32
        assert weight_scale.item() > 0.0
        assert (
            f.get_tensor("transformer_blocks.0.attn.to_q.weight").dtype
            == torch.float8_e4m3fn
        )
        assert f.metadata()["max_flux2_adapted_format"] == "dynamic_block_fp8_v1"


def test_adapt_bflabs_flux2_transformer_weights_persists_static_adapted_file(
    tmp_path,
) -> None:
    src = tmp_path / "flux-2-klein-4b-fp8.safetensors"
    _write_minimal_bflabs_fp8_checkpoint(src)

    adapted = adapt_bflabs_flux2_transformer_weights(
        src, activation_scheme="static"
    )

    assert adapted == tmp_path / "flux-2-klein-4b-fp8.max.static.safetensors"
    assert adapted.exists()
    assert (
        adapt_bflabs_flux2_transformer_weights(
            src, activation_scheme="static"
        )
        == adapted
    )

    with safe_open(str(adapted), framework="pt", device="cpu") as f:
        assert "transformer_blocks.0.attn.to_q.input_scale" in f.keys()
        np.testing.assert_allclose(
            f.get_tensor("transformer_blocks.0.attn.to_q.input_scale").item(),
            4.0,
        )
        np.testing.assert_allclose(
            f.get_tensor("transformer_blocks.0.attn.to_q.weight_scale").item(),
            0.001,
        )
        assert f.metadata()["max_flux2_adapted_format"] == "legacy_scalar_static_v1"
