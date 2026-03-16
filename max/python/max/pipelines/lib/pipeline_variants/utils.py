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

"""Provides utility functions for computing allowed generation steps in pipeline variants."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    LogProbabilities,
    RequestID,
    TextGenerationContextType,
    TextGenerationOutput,
)
from transformers import AutoConfig

from ..hf_utils import download_weight_files

if TYPE_CHECKING:
    from ..config.model_config import MAXModelConfig

logger = logging.getLogger("max.pipelines")


def calculate_num_steps(
    context: TextGenerationContextType,
    num_steps: int,
    max_seq_len: int,
) -> int:
    """Compute the number of generation steps allowed for a context.

    The value is clamped by the remaining capacity with respect to
    the model's configured ``max_seq_len``.

    Args:
        context: The context whose sequence length constraints apply.
        num_steps: Desired number of steps to attempt.
        max_seq_len: The maximum allowed sequence length for the model.

    Returns:
        The number of steps to execute for this context (>= 1).

    Raises:
        ValueError: If the current request length is already >= ``max_seq_len``.
    """
    num_available_steps = context.compute_num_available_steps(max_seq_len)

    if num_available_steps <= 0:
        raise ValueError(
            f"Request {context.request_id} length ({len(context.tokens)}) is larger than or equal to the configured max_length ({max_seq_len})"
        )

    return min(num_available_steps, num_steps)


def update_context_and_prepare_responses(
    generated_tokens_host: npt.NDArray[np.int32],
    flat_batch: list[TextGenerationContextType],
    num_steps: int,
    batch_log_probabilities: list[list[LogProbabilities | None]] | None = None,
    enable_log_probs: bool = False,
    overwrite_future: bool = False,
) -> dict[RequestID, TextGenerationOutput]:
    """Updates context objects and prepares response objects after generation.

    Args:
        generated_tokens_host: Array of generated tokens on the host, indexed
            as [batch, step].
        flat_batch: List of generation contexts, one per request, matching
            batch dimension.
        num_steps: Number of generation steps to process for each context.
        batch_log_probabilities: List of per-step log probability outputs (or
            None), each entry is a list per batch for that step.
        enable_log_probs: Whether to include log probability data in outputs.
        overwrite_future: Whether to overwrite future tokens in the context.

    Returns:
        A dictionary mapping request IDs to their respective generation outputs.
    """
    res: dict[RequestID, TextGenerationOutput] = {}
    for batch_index, context in enumerate(flat_batch):
        for step in range(num_steps):
            # Convert to a Python scalar to improve serialization performance.
            next_token = int(generated_tokens_host[batch_index, step])

            # Get Log probs if needed.
            log_probs: LogProbabilities | None = None
            if enable_log_probs:
                assert batch_log_probabilities is not None
                if step < len(batch_log_probabilities):
                    log_probs_for_step = batch_log_probabilities[step]
                    if log_probs_for_step and batch_index < len(
                        log_probs_for_step
                    ):
                        log_probs = log_probs_for_step[batch_index]

            if overwrite_future:
                # If generated_length is still 0, then there is no placeholder
                # future token. This is possible due to chunked prefill or preemption.
                if context.tokens.generated_length:
                    context.realize_future_token(
                        new_token=next_token, log_probabilities=log_probs
                    )
            else:
                context.update(
                    new_token=next_token, log_probabilities=log_probs
                )

            if context.is_done:
                break

        # Only add the output if there are tokens to return.
        # It is possible that there are no generated tokens due to chunked prefill.
        output = context.to_generation_output()
        if output.tokens:
            res[context.request_id] = output

    return res


def get_rope_theta(config: AutoConfig) -> float:
    """Gets rope_theta from a HuggingFace config, compatible with transformers v4 and v5.

    Transformers v5 moved rope_theta into config.rope_parameters["rope_theta"].
    This function checks rope_parameters first, then falls back to config.rope_theta.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return rope_params["rope_theta"]

    return config.rope_theta


def get_eos_tokens(hf_config: AutoConfig, eos_token_id: int) -> set[int]:
    """Returns the set of end-of-sequence token IDs from config or fallback.

    Args:
        hf_config: HuggingFace model configuration.
        eos_token_id: Default EOS token id when not present in config.

    Returns:
        Set of EOS token ids to use for generation.
    """
    # Expand eos tokens if more are provided in pipeline_config
    if "eos_token_id" not in hf_config:
        return set([eos_token_id])

    hf_eos_tokens = hf_config.eos_token_id
    if isinstance(hf_eos_tokens, int):
        if hf_eos_tokens != eos_token_id:
            msg = f"eos_token_id provided in huggingface config ({hf_eos_tokens}), does not match provided eos_token_id ({eos_token_id}), using provided eos_token_id"
            logger.warning(msg)
        return set([hf_eos_tokens])
    elif isinstance(hf_eos_tokens, list):
        if eos_token_id in hf_eos_tokens:
            return set(hf_eos_tokens)
        else:
            return set([eos_token_id])
    else:
        msg = f"eos_token_id in huggingface_config is neither int or list: {hf_eos_tokens}"
        logger.warning(msg)
        return set([eos_token_id])


def get_weight_paths(model_config: MAXModelConfig) -> list[Path]:
    """Resolves local paths or downloads weight files for the model config.

    Args:
        model_config: Model configuration containing weight repo and paths.

    Returns:
        List of paths to weight files (local or downloaded).
    """
    weight_repo = model_config.huggingface_weight_repo
    if weight_repo.repo_type == "online":
        # Download weight files if not existent.
        weight_paths = download_weight_files(
            huggingface_model_id=weight_repo.repo_id,
            filenames=[str(x) for x in model_config.weight_path],
            revision=model_config.huggingface_weight_revision,
            force_download=model_config.force_download,
        )
        return _maybe_adapt_flux2_klein_fp8_weight_paths(
            model_config, weight_paths
        )
    else:
        # Use the resolved repo_id (which points to local cache in offline mode)
        local_path = Path(weight_repo.repo_id)
        weight_paths = [local_path / x for x in model_config.weight_path]
        return _maybe_adapt_flux2_klein_fp8_weight_paths(
            model_config, weight_paths
        )


def _maybe_adapt_flux2_klein_fp8_weight_paths(
    model_config: MAXModelConfig, weight_paths: list[Path]
) -> list[Path]:
    model_path = model_config.model_path.lower()
    weight_repo_id = model_config.huggingface_weight_repo_id.lower()

    if (
        "flux.2-klein" not in model_path
        and "flux.2-klein" not in weight_repo_id
    ):
        return weight_paths

    if not any("fp8" in path.name.lower() for path in weight_paths):
        return weight_paths

    if not all(path.suffix == ".safetensors" for path in weight_paths):
        return weight_paths

    if any(len(p.parts) > 1 for p in model_config.weight_path):
        return weight_paths

    from copy import deepcopy

    from max.pipelines.architectures.flux2_modulev3.weight_adapters import (
        adapt_bflabs_flux2_transformer_weights,
    )

    activation_scheme = model_config.fp8_activation_scheme or "static"
    if activation_scheme != "static":
        return [
            adapt_bflabs_flux2_transformer_weights(
                path, activation_scheme=activation_scheme
            )
            for path in weight_paths
        ]

    if len(weight_paths) != 1:
        raise ValueError(
            "Expected a single flat safetensors file for FLUX.2 Klein FP8 "
            f"static loading, got {len(weight_paths)} paths."
        )

    # Adapt the transformer weights once; result is cached on disk so
    # subsequent runs skip the heavy PyTorch conversion entirely.
    adapted_path = adapt_bflabs_flux2_transformer_weights(
        weight_paths[0], activation_scheme="static"
    )

    # Patch diffusers_config in memory to inject the FP8 metadata needed
    # by the pipeline's encoding-upgrade logic.  No files are written.
    diffusers_config = model_config.diffusers_config
    if diffusers_config is None:
        raise ValueError(
            "diffusers_config is required for FLUX.2 Klein FP8 static loading."
        )
    patched = deepcopy(diffusers_config)
    transformer_comp = (
        patched.setdefault("components", {}).setdefault("transformer", {})
    )
    transformer_cfg = transformer_comp.setdefault("config_dict", {})
    quant_cfg = dict(transformer_cfg.get("quantization_config") or {})
    quant_cfg["quant_method"] = "fp8"
    quant_cfg["activation_scheme"] = "static"
    transformer_cfg["quantization_config"] = quant_cfg
    transformer_cfg["activation_scheme"] = "static"
    model_config._diffusers_config = patched

    # Expose only the adapted transformer file as a flat weight path.
    # Flux2KleinPipeline.unprefixed_weight_component = "transformer" routes
    # it to the transformer component.  VAE and text-encoder weights are
    # downloaded from the original HF repo via the normal fallback.
    model_config.weight_path = [Path(adapted_path.name)]

    return [adapted_path]
