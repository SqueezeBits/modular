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

"""Mistral-specific tokenizer implementation."""

from __future__ import annotations

import json
import logging
import os

import huggingface_hub
from max.pipelines.lib import TextTokenizer, try_to_load_from_cache
from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")


class Mistral3Tokenizer(TextTokenizer):
    """Mistral-specific tokenizer that corrects the chat template.

    This class only overrides __init__ to correct the chat template, while inheriting
    all other methods from TextTokenizer.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        pipeline_config: PipelineConfig | None = None,
        **unused_kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            revision=revision,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )

        self._load_and_set_chat_template(
            revision=revision, pipeline_config=pipeline_config
        )

    def _load_and_set_chat_template(
        self,
        revision: str | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        """Load chat template from chat_template.json or chat_template.jinja file and set it on the tokenizer."""

        if revision is None:
            # Prefer revision from pipeline config when not explicitly provided.
            model_cfg = getattr(pipeline_config, "model", None)
            candidate = getattr(model_cfg, "huggingface_model_revision", None)
            revision = (
                candidate if isinstance(candidate, str) and candidate else None
            )
        revision = revision or "main"

        # Try both chat_template.json and chat_template.jinja
        template_files = [
            ("chat_template.json", False),
            ("chat_template.jinja", True),
        ]

        template_file_path = None
        is_jinja_file = False

        for filename, is_jinja in template_files:
            # Try to load from cache first
            cached_path = try_to_load_from_cache(
                repo_id=self.model_path,
                filename=filename,
                revision=revision,
            )

            # Check if the result is a valid path
            if cached_path and isinstance(cached_path, (str, os.PathLike)):
                template_file_path = cached_path
                is_jinja_file = is_jinja
                break

        # If not in cache, try to download
        if not template_file_path:
            for filename, is_jinja in template_files:
                try:
                    template_file_path = huggingface_hub.hf_hub_download(
                        repo_id=self.model_path,
                        filename=filename,
                        revision=revision,
                    )
                    is_jinja_file = is_jinja
                    logger.info(f"Successfully downloaded {filename}")
                    break
                except Exception:
                    continue

        # If neither file was found, use tokenizer's default if available
        if not template_file_path:
            if hasattr(self.delegate, "chat_template") and self.delegate.chat_template:
                logger.info(
                    f"Neither chat_template.json nor chat_template.jinja found, "
                    f"using tokenizer's default chat template for {self.model_path}"
                )
                return
            else:
                raise RuntimeError(
                    f"Failed to find 'chat_template.json' or 'chat_template.jinja' "
                    f"from model repo '{self.model_path}' at revision '{revision}'"
                )

        # Load and set the chat template
        try:
            with open(template_file_path, encoding="utf-8") as f:
                if is_jinja_file:
                    chat_template = f.read().strip()
                    logger.info(f"Loaded chat template from {template_file_path} (Jinja format)")
                else:
                    template_data = json.load(f)
                    chat_template = template_data.get("chat_template")
                    if not chat_template:
                        raise KeyError(
                            f"No 'chat_template' key found in {template_file_path} for model {self.model_path}"
                        )

            self.delegate.chat_template = chat_template
            logger.info(f"Loaded custom chat template from {template_file_path}")

        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load chat template from {template_file_path}: {e}") from e
