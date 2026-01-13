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

"""Pipeline utilities for MAX-optimized pipelines."""
from __future__ import annotations

from abc import ABC, abstractmethod
import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import huggingface_hub
import requests
from huggingface_hub.utils import OfflineModeIsEnabled
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from requests.exceptions import HTTPError
from tqdm import tqdm

from .configuration_utils import ConfigMixin
from ..config_enums import RepoType

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger(__name__)


class DiffusionPipeline(ABC, ConfigMixin):
    config_name = "model_index.json"

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        device: DeviceRef = DeviceRef.GPU(),
        dtype: DType = DType.bfloat16,
        **kwargs: Any,
    ) -> "DiffusionPipeline":
        """Load a pipeline from a pretrained model.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier.
            device: Device to run the pipeline on.
            dtype: Data type for the pipeline.
            **kwargs: Additional arguments.

        Returns:
            The loaded pipeline.
        """
        # 1. Download checkpoints if required
        # NOTE: In contrast to TextGenerationPipeline where each files,
        # such as configs and weights, are downloaded individually,
        # DiffusionPipeline downloads the entire snapshot at once,
        # since it normally contains multiple components.
        pretrained_model_name_or_path = pipeline_config.model_config.huggingface_model_repo.repo_id
        if pipeline_config.model_config.huggingface_model_repo.repo_type == RepoType.online:
            cached_folder = self.download(
                pretrained_model_name_or_path,
                force_download=pipeline_config.model_config.force_download,
                revision=pipeline_config.model_config.huggingface_model_revision,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        # 2. Load pipeline configuration
        config_dict = self.load_config(cached_folder)
        init_dict = self.extract_init_dict(config_dict)

        # 3. Load sub models
        loaded_sub_models = self.load_sub_models(
            cached_folder,
            init_dict,
            device=device,
            dtype=dtype,
        )
        for name, model in loaded_sub_models.items():
            setattr(self, name, model)
        
        self.init_remainig_components()
    
    @abstractmethod
    def init_remainig_components(self):
        pass

    def load_sub_models(
        self,
        pretrained_model_name_or_path: str | os.PathLike,
        init_dict: dict,
        device: DeviceRef = DeviceRef.GPU(),
        dtype: DType = DType.bfloat16,
    ) -> dict:
        """Load sub-models for the pipeline.

        Args:
            pretrained_model_name_or_path: Path to pretrained model.
            init_dict: Dictionary containing the init parameters.
            device: Device to load the models on.
            dtype: Data type for the models.

        Returns:
            Dictionary containing the loaded sub-models.
        """
        loaded_sub_models = {}
        for name in tqdm(init_dict.keys(), desc="Loading sub models"):
            component_class = self.components[name]
            component_path = os.path.join(pretrained_model_name_or_path, name)
            if "tokenizer" in name:
                # NOTE: Currently, we are using tokenizers from transformers.
                # It might be replaced with TextTokenizer in Max,
                # when this repository is merged into the main Max repository.
                loaded_sub_models[name] = component_class.from_pretrained(
                    component_path
                )
                continue

            config = component_class.load_config(component_path)
            init_config = component_class.extract_init_dict(config)
            init_config.update(
                {
                    "device": device,
                    "dtype": dtype,
                }
            )
            if (
                "pretrained_model_name_or_path"
                in component_class._get_init_keys(component_class)
            ):
                init_config["pretrained_model_name_or_path"] = (
                    component_path
                )
            loaded_sub_models[name] = component_class(**init_config)

        return loaded_sub_models

    def download(
        self,
        pretrained_model_name: str | os.PathLike,
        force_download: bool = False,
        revision: str | None = None,
    ) -> str:
        """Download the pipeline components from the Hugging Face Hub.

        Args:
            pretrained_model_name: Model identifier.
            cache_dir: Cache directory.
            force_download: Whether to force download.
            revision: Model revision.

        Returns:
            Path to the downloaded model folder.
        """
        try:
            info = huggingface_hub.model_info(
                pretrained_model_name, revision=revision
            )
        except (HTTPError, OfflineModeIsEnabled, requests.ConnectionError) as e:
            logger.warning(
                f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache."
            )
            model_info_call_error = (
                e  # save error to reraise it if model is not cached locally
            )

        config_file = huggingface_hub.hf_hub_download(
            pretrained_model_name,
            self.config_name,
            revision=revision,
            force_download=force_download,
        )
        config_dict = self._dict_from_json_file(config_file)
        ignore_filenames = config_dict.pop("_ignore_files", [])

        filenames = {sibling.rfilename for sibling in info.siblings}
        filenames = set(filenames) - set(ignore_filenames)

        ignore_patterns = [
            "*.bin",
            "*.msgpack",
            "*.onnx",
            "*.pb",
            "*.bin.index.*json",
            "*.msgpack.index.*json",
            "*.onnx.index.*json",
            "*.pb.index.*json",
        ]
        components = self._get_init_keys(self)

        allow_patterns = [f"{k}/*" for k in components]
        allow_patterns += [
            "scheduler_config.json",
            "config.json",
            self.config_name,
        ]
        re_ignore_pattern = [
            re.compile(fnmatch.translate(p)) for p in ignore_patterns
        ]
        re_allow_pattern = [
            re.compile(fnmatch.translate(p)) for p in allow_patterns
        ]

        expected_files = [
            f
            for f in filenames
            if not any(p.match(f) for p in re_ignore_pattern)
        ]
        expected_files = [
            f
            for f in expected_files
            if any(p.match(f) for p in re_allow_pattern)
        ]

        snapshot_folder = Path(config_file).parent
        pipeline_is_cached = all(
            (snapshot_folder / f).is_file() for f in expected_files
        )

        if pipeline_is_cached and not force_download:
            # if the pipeline is cached, we can directly return it
            # else call snapshot_download
            return snapshot_folder

        user_agent = {"pipeline_class": cls.__name__}

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = huggingface_hub.snapshot_download(
                pretrained_model_name,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )

            return cached_folder

        except FileNotFoundError:
            # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
            # This can happen in two cases:
            # 1. If the user passed `local_files_only=True`                    => we raise the error directly
            # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
            if model_info_call_error is None:
                # 1. user passed `local_files_only=True`
                raise
            else:
                # 2. we forced `local_files_only=True` when `model_info` failed
                raise OSError(
                    f"Cannot load model {pretrained_model_name}: model is not cached locally and an error occurred"
                    " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                    " above."
                ) from model_info_call_error
    
    def finalize_pipeline_config(self) -> None:
        pass

    def _execution_device(self) -> DeviceRef:
        r"""Returns the device on which the pipeline's models will be executed.

        This property checks pipeline components to determine the execution device.
        It supports MAX models (with DeviceRef device attribute).
        Similar structure to diffusers' _execution_device but returns DeviceRef instead of DeviceRef.

        Returns:
            DeviceRef: The execution device (GPU if available, otherwise CPU).
        """
        # Check MAX models - prioritize GPU
        # Similar to diffusers' _execution_device but for MAX models (not torch.nn.Module)
        sub_models = {k: getattr(self, k) for k in self.components}
        for name, model in sub_models.items():
            exclude_from_cpu_offload = getattr(
                self, "_exclude_from_cpu_offload", set()
            )
            if name in exclude_from_cpu_offload:
                continue

            if hasattr(model, "device") and isinstance(model.device, DeviceRef):
                return model.device

        if hasattr(self, "device"):
            try:
                device = self.device
                if isinstance(device, DeviceRef):
                    return device
            except Exception:
                pass

        return DeviceRef.CPU()
