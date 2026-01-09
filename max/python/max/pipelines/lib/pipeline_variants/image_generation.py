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

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import (
    hf_hub_download,
    model_info,
    snapshot_download,
)
from huggingface_hub.utils import OfflineModeIsEnabled
from max.dtype import DType
from max.graph import DeviceRef
from requests.exceptions import HTTPError
from tqdm import tqdm

from .configuration_utils import ConfigMixin

logger = logging.getLogger(__name__)


class DiffusionPipeline(ConfigMixin):
    config_name = "model_index.json"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
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
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        # 1. Download checkpoints if required
        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                custom_pipeline=custom_pipeline,
                **kwargs,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        # 2. Load pipeline configuration
        config_dict = cls.load_config(cached_folder)
        init_dict = cls.extract_init_dict(config_dict)

        # 3. Load sub models
        loaded_sub_models = cls.load_sub_models(
            cached_folder,
            init_dict,
            device=device,
            dtype=dtype,
        )

        # 4. Instantiate the pipeline
        pipeline = cls(loaded_sub_models)

        return pipeline

    @classmethod
    def load_sub_models(
        cls,
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
            component_class = cls.components[name]
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
                    pretrained_model_name_or_path
                )
            loaded_sub_models[name] = component_class(**init_config)

        return loaded_sub_models

    @classmethod
    def download(
        cls,
        pretrained_model_name: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        proxies: dict | None = None,
        token: str | None = None,
        revision: str | None = None,
        use_safetensors: bool | None = None,
        custom_pipeline: str | None = None,
    ) -> str:
        """Download the pipeline components from the Hugging Face Hub.

        Args:
            pretrained_model_name: Model identifier.
            cache_dir: Cache directory.
            force_download: Whether to force download.
            proxies: Proxies.
            token: Authentication token.
            revision: Model revision.
            use_safetensors: Whether to use safetensors.
            custom_pipeline: Custom pipeline.

        Returns:
            Path to the downloaded model folder.
        """
        # NOTE: For simplicity, this download method is not exactly
        # the same as diffusers' download method.
        # It might be replaced with Max's download method,
        # when this repository is merged into the main Max repository.
        use_safetensors = (
            use_safetensors if use_safetensors is not None else True
        )

        try:
            info = model_info(
                pretrained_model_name, token=token, revision=revision
            )
        except (HTTPError, OfflineModeIsEnabled, requests.ConnectionError) as e:
            logger.warning(
                f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache."
            )
            model_info_call_error = (
                e  # save error to reraise it if model is not cached locally
            )

        config_file = hf_hub_download(
            pretrained_model_name,
            cls.config_name,
            cache_dir=cache_dir,
            revision=revision,
            proxies=proxies,
            force_download=force_download,
            token=token,
        )
        config_dict = cls._dict_from_json_file(config_file)
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
        components = cls._get_init_keys(cls)

        allow_patterns = [f"{k}/*" for k in components]
        allow_patterns += [
            "scheduler_config.json",
            "config.json",
            cls.config_name,
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
        if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
            user_agent["custom_pipeline"] = custom_pipeline

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = snapshot_download(
                pretrained_model_name,
                cache_dir=cache_dir,
                proxies=proxies,
                token=token,
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
