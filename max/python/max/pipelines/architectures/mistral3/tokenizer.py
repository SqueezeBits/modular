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

import huggingface_hub
from max.pipelines.lib import TextTokenizer, try_to_load_from_cache
from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")


class Mistral3Tokenizer(TextTokenizer):
    """Mistral-specific tokenizer that corrects the chat template.

    This class only overrides __init__ to correct the chat template, while inheriting
    all other methods from TextTokenizer.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        pipeline_config: PipelineConfig | None = None,
        **kwargs,
    ) -> "Mistral3Tokenizer":
        """Load a Mistral3Tokenizer from a pretrained model path.

        This is a convenience method for compatibility with HuggingFace tokenizers.

        Args:
            model_path: Path to the model directory.
            pipeline_config: Optional pipeline configuration.
            **kwargs: Additional keyword arguments passed to __init__.

        Returns:
            Mistral3Tokenizer instance.
        """
        return cls(
            model_path=model_path,
            pipeline_config=pipeline_config,
            **kwargs,
        )

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig | None = None,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            pipeline_config=pipeline_config,
            revision=revision,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )

        self._load_and_set_chat_template(
            revision=revision, pipeline_config=pipeline_config
        )

    def __call__(self, *args, **kwargs):
        """Make the tokenizer callable by delegating to the underlying HuggingFace tokenizer.

        This enables using the tokenizer like a HuggingFace tokenizer:
            tokenizer(text, padding="max_length", max_length=512, ...)

        Args:
            *args: Positional arguments passed to the delegate tokenizer.
            **kwargs: Keyword arguments passed to the delegate tokenizer.

        Returns:
            The output from the delegate tokenizer (typically BatchEncoding).
        """
        return self.delegate(*args, **kwargs)

    def _load_and_set_chat_template(
        self,
        revision: str | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        """Load chat template from chat_template.json or chat_template.jinja file and set it on the tokenizer."""
        import os

        if revision is None:
            # Prefer revision from pipeline config when not explicitly provided.
            model_cfg = getattr(pipeline_config, "model", None)
            candidate = getattr(model_cfg, "huggingface_model_revision", None)
            revision = (
                candidate if isinstance(candidate, str) and candidate else None
            )
        revision = revision or "main"

        # Check if model_path is a local path
        is_local_path = os.path.exists(self.model_path)
        
        template_file_path = None
        is_jinja_file = False
        
        # Try both chat_template.json and chat_template.jinja
        template_files = [
            ("chat_template.json", False),
            ("chat_template.jinja", True),
        ]
        
        if is_local_path:
            # For local paths, check files directly
            for filename, is_jinja in template_files:
                local_file = os.path.join(self.model_path, filename)
                if os.path.exists(local_file):
                    template_file_path = local_file
                    is_jinja_file = is_jinja
                    break
        else:
            # For HuggingFace repos, try cache first then download
            for filename, is_jinja in template_files:
                cached_path = try_to_load_from_cache(
                    repo_id=self.model_path,
                    filename=filename,
                    revision=revision,
                )
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
        
        # If no template file found, use tokenizer's default if available
        if not template_file_path:
            if hasattr(self.delegate, "chat_template") and self.delegate.chat_template:
                logger.info(
                    f"No chat template file found, using tokenizer's default for {self.model_path}"
                )
                return
            else:
                raise RuntimeError(
                    f"Failed to find 'chat_template.json' or 'chat_template.jinja' "
                    f"in model path '{self.model_path}'"
                )

        # Load and set the chat template
        try:
            with open(template_file_path, encoding="utf-8") as f:
                if is_jinja_file:
                    chat_template = f.read().strip()
                else:
                    template_data = json.load(f)
                    chat_template = template_data.get("chat_template")

            if not chat_template:
                raise KeyError(
                    f"No 'chat_template' key found in {template_file_path} for model {self.model_path}"
                )

            self.delegate.chat_template = chat_template
            logger.info(
                f"Loaded custom chat template from {template_file_path}"
            )

        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to load chat template from {template_file_path}: {e}"
            ) from e
