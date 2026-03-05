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

import dataclasses
import logging
import math
import os
import random
import urllib.error
import urllib.request
from collections.abc import Sequence

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import override

from .local import LocalBenchmarkDataset
from .types import (
    PixelGenerationImageOptions,
    PixelGenerationSampledRequest,
    RequestSamples,
)

logger = logging.getLogger(__name__)


class PixelBenchmarkDataset(LocalBenchmarkDataset):
    """Base class for text-to-image benchmark datasets."""

    def _build_image_options(
        self,
        *,
        image_width: int | None = None,
        image_height: int | None = None,
        image_steps: int | None = None,
        image_guidance_scale: float | None = None,
        image_negative_prompt: str | None = None,
        image_seed: int | None = None,
    ) -> PixelGenerationImageOptions | None:
        options = PixelGenerationImageOptions(
            width=image_width,
            height=image_height,
            steps=image_steps,
            guidance_scale=image_guidance_scale,
            negative_prompt=image_negative_prompt,
            seed=image_seed,
        )
        if all(value is None for value in dataclasses.asdict(options).values()):
            return None
        return options

    def _build_request(
        self,
        prompt: str,
        image_options: PixelGenerationImageOptions | None,
    ) -> PixelGenerationSampledRequest:
        return PixelGenerationSampledRequest(
            prompt_formatted=prompt,
            prompt_len=0,
            output_len=None,
            encoded_images=[],
            ignore_eos=True,
            image_options=image_options,
        )


class SyntheticPixelBenchmarkDataset(PixelBenchmarkDataset):
    @override
    def fetch(self) -> None:
        """Fetch Synthetic Pixel dataset.

        Synthetic pixel prompts are generated in-memory and do not require a
        local file.
        """
        pass

    @override
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase | None,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        image_options = self._build_image_options(
            image_width=kwargs.get("image_width"),
            image_height=kwargs.get("image_height"),
            image_steps=kwargs.get("image_steps"),
            image_guidance_scale=kwargs.get("image_guidance_scale"),
            image_negative_prompt=kwargs.get("image_negative_prompt"),
            image_seed=kwargs.get("image_seed"),
        )

        requests = [
            self._build_request(
                prompt=f"Random prompt {idx} for benchmarking pixel generation pipelines",
                image_options=image_options,
            )
            for idx in range(num_requests)
        ]
        return RequestSamples(requests=requests)


class TextFilePixelBenchmarkDataset(PixelBenchmarkDataset):
    """Pixel benchmark dataset that loads one prompt per line from a text file."""

    _prompts: list[str]

    @override
    def fetch(self) -> None:
        """Validate dataset path and load non-empty prompts from the text file."""
        super().fetch()
        assert self.dataset_path is not None, (
            "dataset_path must be provided for TextFilePixelBenchmarkDataset"
        )
        with open(self.dataset_path, encoding="utf-8") as txt_file:
            prompts = [line.strip() for line in txt_file if line.strip()]
        if not prompts:
            raise ValueError(
                "Text file dataset is empty. Provide at least one non-empty line."
            )
        self._prompts = prompts

    @override
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase | None,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        del tokenizer, output_lengths
        image_options = self._build_image_options(
            image_width=kwargs.get("image_width"),
            image_height=kwargs.get("image_height"),
            image_steps=kwargs.get("image_steps"),
            image_guidance_scale=kwargs.get("image_guidance_scale"),
            image_negative_prompt=kwargs.get("image_negative_prompt"),
            image_seed=kwargs.get("image_seed"),
        )

        if shuffle:
            sampled_prompts = random.choices(self._prompts, k=num_requests)
        else:
            repeats = math.ceil(num_requests / len(self._prompts))
            sampled_prompts = (self._prompts * repeats)[:num_requests]

        requests = [
            self._build_request(prompt=prompt, image_options=image_options)
            for prompt in sampled_prompts
        ]
        return RequestSamples(requests=requests)


class VBenchPixelBenchmarkDataset(TextFilePixelBenchmarkDataset):
    """Pixel benchmark dataset that uses VBench subject consistency prompts."""

    T2I_PROMPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/"
        "prompts_per_dimension/subject_consistency.txt"
    )
    DEFAULT_CACHE_FILENAME = "vbench_subject_consistency.txt"

    @override
    def fetch(self) -> None:
        if self.dataset_path is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "max", "benchmark"
            )
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, self.DEFAULT_CACHE_FILENAME)
            if not os.path.exists(cache_path):
                logger.info(
                    "Downloading VBench text-to-image prompts to %s", cache_path
                )
                try:
                    with urllib.request.urlopen(self.T2I_PROMPT_URL) as response:
                        prompt_bytes = response.read()
                except urllib.error.URLError as e:
                    raise ValueError(
                        "Failed to download VBench prompts. "
                        "Provide --dataset-path with a local prompt txt file."
                    ) from e
                with open(cache_path, "wb") as cache_file:
                    cache_file.write(prompt_bytes)
            self.dataset_path = cache_path
        super().fetch()
