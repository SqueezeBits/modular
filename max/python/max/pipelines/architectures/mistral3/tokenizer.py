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

"""Mistral-specific tokenizer implementation."""

from __future__ import annotations

import json
import logging

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
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
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        root_model_path: str | None = None,
        **unused_kwargs,
    ) -> None:
        # For Flux2 models, tokenizer is in root_model_path/tokenizer subdirectory
        if root_model_path:
            from transformers import AutoTokenizer
            self.model_path = root_model_path
            self.delegate = AutoTokenizer.from_pretrained(
                root_model_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                model_max_length=max_length,
                subfolder="tokenizer",
            )
            self.max_length = max_length or self.delegate.model_max_length
            self._custom_template_provided = False
            self._enable_llama_whitespace_fix = False
            self._llama_whitespace_fix_dummy_token_id = None
            self._llama_whitespace_fix_dummy_token_len = None
            eos_token_id = self.delegate.eos_token_id
            self._default_eos_token_ids = set([eos_token_id] if eos_token_id is not None else [])
            self._context_validators = []
            if pipeline_config:
                huggingface_config = pipeline_config.model.huggingface_config
                if eos_token_id := getattr(huggingface_config, "eos_token_id", None):
                    if isinstance(eos_token_id, int):
                        self._default_eos_token_ids.add(eos_token_id)
                    elif isinstance(eos_token_id, list):
                        self._default_eos_token_ids.update(eos_token_id)
            self._root_model_path = root_model_path
        else:
            super().__init__(
                model_path=model_path,
                pipeline_config=pipeline_config,
                revision=revision,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                trust_remote_code=trust_remote_code,
            )
            self._root_model_path = None

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

        # For Flux2 models, chat_template is in root_model_path/tokenizer
        search_repo = self._root_model_path if self._root_model_path else self.model_path
        tokenizer_prefix = "tokenizer/" if self._root_model_path else ""

        # Try chat_template.json first, then chat_template.jinja
        template_file_path = None
        for filename_suffix in [".json", ".jinja"]:
            search_filename = f"{tokenizer_prefix}chat_template{filename_suffix}"

            # Try to load from cache first
            template_file_path = try_to_load_from_cache(
                repo_id=search_repo,
                filename=search_filename,
                revision=revision,
            )

            # Check if file was found in cache
            # try_to_load_from_cache returns object() singleton when file doesn't exist
            if not template_file_path or not isinstance(template_file_path, str):
                logger.info(
                    f"{search_filename} not in cache, attempting to download..."
                )
                try:
                    template_file_path = huggingface_hub.hf_hub_download(
                        repo_id=search_repo,
                        filename=search_filename,
                        revision=revision,
                    )
                    logger.info(f"Successfully downloaded {search_filename}")
                    break  # Found the file, exit the loop
                except EntryNotFoundError:
                    # File doesn't exist, try next format
                    logger.debug(
                        f"{search_filename} not found, trying alternative format..."
                    )
                    template_file_path = None
                    continue
                except Exception as e:
                    # Other errors should be raised
                    raise RuntimeError(
                        f"Failed to download '{search_filename}' from model repo '{search_repo}' "
                        f"at revision '{revision}': {e}"
                    ) from e
            else:
                # Found in cache
                break

        # If no template file found, use tokenizer's default
        if not template_file_path:
            logger.warning(
                f"Neither chat_template.json nor chat_template.jinja found in {search_repo}. "
                f"Using tokenizer's default chat template."
            )
            return

        # Load and set the chat template
        try:
            with open(template_file_path, encoding="utf-8") as f:
                template_content = f.read()

            # Try to parse as JSON and extract chat_template if present
            try:
                template_data = json.loads(template_content)
                if isinstance(template_data, dict) and "chat_template" in template_data:
                    chat_template = template_data["chat_template"]
                    logger.info(
                        f"Loaded chat_template from JSON in {template_file_path} "
                        f"({len(chat_template)} characters)"
                    )
                else:
                    # JSON but no chat_template key, use entire content
                    chat_template = template_content
                    logger.info(
                        f"Loaded chat template from {template_file_path} "
                        f"({len(template_content)} characters, JSON without chat_template key)"
                    )
            except json.JSONDecodeError:
                # Not valid JSON (e.g., .jinja file), use entire content as template
                chat_template = template_content
                logger.info(
                    f"Loaded chat template from {template_file_path} "
                    f"({len(template_content)} characters, non-JSON format)"
                )

            if not chat_template:
                raise ValueError(
                    f"Empty chat template loaded from {template_file_path}"
                )

            self.delegate.chat_template = chat_template
            logger.info(
                f"Successfully set chat template on tokenizer from {template_file_path}"
            )

        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(
                f"Failed to read chat template from {template_file_path}: {e}"
            ) from e