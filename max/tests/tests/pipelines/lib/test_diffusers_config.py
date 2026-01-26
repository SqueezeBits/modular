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
"""Tests for DiffusersConfig and DiffusersComponentConfig."""

import json
from pathlib import Path

from max.pipelines.lib.diffusers_config import (
    DiffusersComponentConfig,
    DiffusersConfig,
)


class TestDiffusersComponentConfig:
    """Tests for DiffusersComponentConfig."""

    def test_from_subfolder_with_config(self, tmp_path: Path) -> None:
        """Test loading a component with config.json."""
        component_dir = tmp_path / "transformer"
        component_dir.mkdir()

        config_content = {
            "_class_name": "FluxTransformer2DModel",
            "num_attention_heads": 24,
        }
        config_path = component_dir / "config.json"
        config_path.write_text(json.dumps(config_content))

        weight_file = component_dir / "model.safetensors"
        weight_file.write_bytes(b"fake weights")

        component = DiffusersComponentConfig.from_subfolder(
            name="transformer",
            subfolder=component_dir,
            library="diffusers",
            class_name="FluxTransformer2DModel",
        )

        assert component.name == "transformer"
        assert component.library == "diffusers"
        assert component.class_name == "FluxTransformer2DModel"
        assert component.config_path == config_path
        assert component.config_dict == config_content
        assert len(component.weight_paths) == 1
        assert component.has_weights is True


class TestDiffusersConfig:
    """Tests for DiffusersConfig."""

    def test_from_model_path(self, tmp_path: Path) -> None:
        """Test loading a diffusers config from a local path."""
        model_index = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0.dev0",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        (tmp_path / "model_index.json").write_text(json.dumps(model_index))

        for component_name in [
            "scheduler",
            "text_encoder",
            "transformer",
            "vae",
        ]:
            component_dir = tmp_path / component_name
            component_dir.mkdir()
            config_file = (
                "scheduler_config.json"
                if component_name == "scheduler"
                else "config.json"
            )
            (component_dir / config_file).write_text(
                json.dumps({"_class_name": f"{component_name}_class"})
            )

        config = DiffusersConfig.from_model_path(tmp_path)

        assert config.pipeline_class == "FluxPipeline"
        assert config.diffusers_version == "0.30.0.dev0"
        assert config.model_path == tmp_path
        assert len(config.components) == 4
        assert config.components["transformer"].library == "diffusers"
        assert (
            config.components["transformer"].class_name
            == "FluxTransformer2DModel"
        )

    def test_get_weight_paths(self, tmp_path: Path) -> None:
        """Test getting weight paths for a component."""
        model_index = {
            "_class_name": "TestPipeline",
            "transformer": ["diffusers", "TestModel"],
        }
        (tmp_path / "model_index.json").write_text(json.dumps(model_index))

        component_dir = tmp_path / "transformer"
        component_dir.mkdir()
        weight_file = component_dir / "model.safetensors"
        weight_file.write_bytes(b"fake weights")

        config = DiffusersConfig.from_model_path(tmp_path)

        assert config.get_weight_paths("transformer") == [weight_file]
        assert config.get_weight_paths("missing") == []

    def test_all_weight_paths(self, tmp_path: Path) -> None:
        """Test getting all weight paths from all components."""
        model_index = {
            "_class_name": "TestPipeline",
            "transformer": ["diffusers", "TestTransformer"],
            "vae": ["diffusers", "TestVAE"],
            "scheduler": ["diffusers", "TestScheduler"],
        }
        (tmp_path / "model_index.json").write_text(json.dumps(model_index))

        # Create transformer with weights
        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        transformer_weight = transformer_dir / "model.safetensors"
        transformer_weight.write_bytes(b"transformer weights")

        # Create vae with weights
        vae_dir = tmp_path / "vae"
        vae_dir.mkdir()
        vae_weight = vae_dir / "diffusion_pytorch_model.safetensors"
        vae_weight.write_bytes(b"vae weights")

        # Create scheduler without weights (schedulers don't have weights)
        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir()
        (scheduler_dir / "scheduler_config.json").write_text("{}")

        config = DiffusersConfig.from_model_path(tmp_path)

        all_weights = config.all_weight_paths
        assert len(all_weights) == 2
        assert transformer_weight in all_weights
        assert vae_weight in all_weights
