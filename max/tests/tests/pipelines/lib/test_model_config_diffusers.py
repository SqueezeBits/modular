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
"""Tests for MAXModelConfig.diffusers_config property."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from max.pipelines.lib import MAXModelConfig
from max.pipelines.lib.diffusers_config import DiffusersConfig
from max.pipelines.lib.pipeline_variants.utils import get_weight_paths


def create_diffusers_model_structure(
    tmp_path: Path,
    pipeline_class: str = "FluxPipeline",
    with_weights: bool = False,
) -> Path:
    """Create a mock diffusers model directory structure."""
    model_index = {
        "_class_name": pipeline_class,
        "_diffusers_version": "0.30.0",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(model_index))
    (tmp_path / "transformer").mkdir(exist_ok=True)
    (tmp_path / "vae").mkdir(exist_ok=True)

    if with_weights:
        # Add weight files to components
        transformer_weight = tmp_path / "transformer" / "model.safetensors"
        transformer_weight.write_bytes(b"fake transformer weights")
        vae_weight = tmp_path / "vae" / "diffusion_pytorch_model.safetensors"
        vae_weight.write_bytes(b"fake vae weights")

    return tmp_path


def create_transformers_model_structure(tmp_path: Path) -> Path:
    """Create a mock transformers model directory structure."""
    config = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


class TestMAXModelConfigDiffusersProperty:
    """Tests for MAXModelConfig.diffusers_config property."""

    def test_diffusers_config_returns_none_for_transformers_model(
        self, tmp_path: Path
    ) -> None:
        """Test that diffusers_config returns None for transformers-style model."""
        create_transformers_model_structure(tmp_path)

        model_config = MAXModelConfig(model_path=str(tmp_path))
        assert model_config.diffusers_config is None

    def test_diffusers_config_returns_config_for_diffusers_model(
        self, tmp_path: Path
    ) -> None:
        """Test that diffusers_config returns DiffusersConfig for diffusers model."""
        create_diffusers_model_structure(tmp_path)

        model_config = MAXModelConfig(model_path=str(tmp_path))
        diffusers_config = model_config.diffusers_config

        assert diffusers_config is not None
        assert isinstance(diffusers_config, DiffusersConfig)
        assert diffusers_config.pipeline_class == "FluxPipeline"

    def test_diffusers_config_raises_for_invalid_json(
        self, tmp_path: Path
    ) -> None:
        """Test that diffusers_config raises ValueError for invalid JSON."""
        (tmp_path / "model_index.json").write_text("not valid json {{{")

        model_config = MAXModelConfig(model_path=str(tmp_path))
        with pytest.raises(ValueError, match=r"Invalid model_index\.json"):
            _ = model_config.diffusers_config

    def test_get_weight_paths_for_diffusers_model(self, tmp_path: Path) -> None:
        """Test that get_weight_paths returns correct paths for diffusers models.

        This test simulates what happens during resolution: weight_path is
        populated from diffusers_config.all_weight_paths.
        """
        create_diffusers_model_structure(tmp_path, with_weights=True)

        model_config = MAXModelConfig(model_path=str(tmp_path))

        # Verify diffusers_config populates all_weight_paths
        assert model_config.diffusers_config is not None
        all_weights = model_config.diffusers_config.all_weight_paths
        assert len(all_weights) == 2

        # Simulate what _resolve_weight_path does for diffusers models:
        # populate weight_path from diffusers_config.all_weight_paths
        model_config.weight_path = all_weights

        # Verify get_weight_paths returns the correct paths
        weight_paths = get_weight_paths(model_config)
        assert len(weight_paths) == 2

        # Verify the paths exist and are correct
        weight_names = {p.name for p in weight_paths}
        assert "model.safetensors" in weight_names
        assert "diffusion_pytorch_model.safetensors" in weight_names

        # Verify all paths exist on disk
        for path in weight_paths:
            assert path.exists(), f"Weight path {path} should exist"
