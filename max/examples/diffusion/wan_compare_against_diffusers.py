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

"""Compare reduced Wan outputs against diffusers and report L2 norms.

This utility runs a reduced-cost Wan comparison against diffusers and the
MAX-native implementation, then reports L2-style metrics for:

- text encoder prompt embeddings
- selected DiT noise predictions (high-noise and low-noise stages)
- final raw / denormalized latents
- VAE output from the same diffusers denormalized latents
- end-to-end decoded output

Example:
    ./bazelw run //max/examples/diffusion:wan_compare_against_diffusers -- \
      --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
      --prompt "Two cats boxing on a stage." \
      --negative-prompt "low quality" \
      --num-inference-steps 4 \
      --num-frames 5 \
      --height 480 \
      --width 832 \
      --output-dir /tmp/wan_compare
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from transformers import AutoTokenizer, UMT5EncoderModel
from max.driver import CPU, DeviceSpec, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.tensor import Tensor
from max.pipelines import MAXModelConfig, PipelineConfig
from max.pipelines.architectures.wan.pipeline_wan import WanPipeline as MaxWanPipeline
from max.pipelines.lib.pipeline_variants.utils import get_weight_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Wan MAX-native intermediates against diffusers."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "diffusers", "max", "compare"),
        default="all",
        help="Which stage to run. 'all' orchestrates subprocesses.",
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Hugging Face model id.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="low quality")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for json/npy comparison artifacts.",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Persist compared tensors as .npy files.",
    )
    return parser.parse_args()


def _np_float32(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32).cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _max_tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    cpu_tensor = tensor.cast(DType.float32).to(CPU())
    return np.from_dlpack(cpu_tensor)


def _max_buffer_to_numpy(buffer: Any) -> np.ndarray:
    return _max_tensor_to_numpy(Tensor.from_dlpack(buffer))


def _l2_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32)
    act = np.asarray(actual, dtype=np.float32)
    diff = act - ref
    ref_norm = float(np.linalg.norm(ref.reshape(-1)))
    diff_norm = float(np.linalg.norm(diff.reshape(-1)))
    denom = max(ref_norm, 1e-12)
    return {
        "shape": list(ref.shape),
        "l2_norm": diff_norm,
        "relative_l2_norm": diff_norm / denom,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
    }


def _save_array(
    output_dir: Path, name: str, value: np.ndarray, save_arrays: bool
) -> None:
    if not save_arrays:
        return
    np.save(output_dir / f"{name}.npy", np.asarray(value))


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "outputs" / timestamp


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return _default_output_dir()


def _save_reference(prefix: str, data: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "selected_step_indices": list(data["selected_step_indices"]),
        "stage_by_step": dict(data["stage_by_step"]),
    }
    for key in (
        "positive_input_ids",
        "positive_attention_mask",
        "negative_input_ids",
        "negative_attention_mask",
        "prompt_embeds",
        "negative_prompt_embeds",
        "scheduler_timesteps",
        "initial_latents",
        "raw_final_latents",
        "denorm_latents",
        "vae_output",
    ):
        np.save(output_dir / f"{prefix}_{key}.npy", np.asarray(data[key]))

    for step_key, tensors in data["step_debug"].items():
        for tensor_name, value in tensors.items():
            np.save(
                output_dir / f"{prefix}_{step_key}_{tensor_name}.npy",
                np.asarray(value),
            )

    (output_dir / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def _load_reference(prefix: str, output_dir: Path) -> dict[str, Any]:
    metadata = json.loads(
        (output_dir / f"{prefix}_metadata.json").read_text()
    )
    data: dict[str, Any] = {
        "selected_step_indices": metadata["selected_step_indices"],
        "stage_by_step": metadata["stage_by_step"],
        "step_debug": {},
    }
    for key in (
        "positive_input_ids",
        "positive_attention_mask",
        "negative_input_ids",
        "negative_attention_mask",
        "prompt_embeds",
        "negative_prompt_embeds",
        "scheduler_timesteps",
        "initial_latents",
        "raw_final_latents",
        "denorm_latents",
        "vae_output",
    ):
        data[key] = np.load(output_dir / f"{prefix}_{key}.npy")

    for step_idx in data["selected_step_indices"]:
        step_key = f"step_{step_idx}"
        data["step_debug"][step_key] = {}
        for tensor_name in ("cond", "uncond", "guided", "latents_in", "timestep"):
            data["step_debug"][step_key][tensor_name] = np.load(
                output_dir / f"{prefix}_{step_key}_{tensor_name}.npy"
            )
    return data


def _basic_clean(text: str) -> str:
    return html.unescape(html.unescape(text)).strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _diffusers_prompt_batch(prompt: str) -> list[str]:
    return [_whitespace_clean(_basic_clean(prompt))]


def _tokenize_for_diffusers(
    tokenizer: Any,
    prompt: str,
    *,
    max_sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    text_inputs = tokenizer(
        _diffusers_prompt_batch(prompt),
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return (
        text_inputs.input_ids.cpu().numpy().astype(np.int64, copy=False),
        text_inputs.attention_mask.cpu().numpy().astype(np.int64, copy=False),
    )


def _prepare_timestep_tensor_diffusers(
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    expand_timesteps: bool,
) -> torch.Tensor:
    if expand_timesteps:
        mask = torch.ones(latents.shape, dtype=torch.float32, device=latents.device)
        temp_ts = (mask[0][0][:, ::2, ::2] * timestep).flatten()
        return temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
    return timestep.expand(latents.shape[0])


def _run_diffusers_reference(args: argparse.Namespace) -> dict[str, Any]:
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=args.model,
            device_specs=[DeviceSpec.accelerator()],
        )
    )
    diffusers_config = config.model.diffusers_config or {}
    components_cfg = diffusers_config.get("components", {})
    scheduler_cfg = components_cfg.get("scheduler", {}).get("config_dict", {})
    boundary_ratio = diffusers_config.get("boundary_ratio")
    transformer_cfg = components_cfg.get("transformer", {}).get(
        "config_dict", {}
    )
    expand_timesteps = bool(transformer_cfg.get("expand_timesteps", False))
    vae_cfg = components_cfg.get("vae", {}).get("config_dict", {})
    vae_scale_factor_temporal = int(vae_cfg.get("scale_factor_temporal", 4))
    vae_scale_factor_spatial = int(vae_cfg.get("scale_factor_spatial", 8))

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    pos_ids_np, pos_mask_np = _tokenize_for_diffusers(
        tokenizer, args.prompt, max_sequence_length=args.max_sequence_length
    )
    neg_ids_np, neg_mask_np = _tokenize_for_diffusers(
        tokenizer, args.negative_prompt, max_sequence_length=args.max_sequence_length
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    ).to(device)
    text_encoder.eval()

    pos_ids_t = torch.from_numpy(pos_ids_np).to(device=device, dtype=torch.long)
    pos_mask_t = torch.from_numpy(pos_mask_np).to(device=device, dtype=torch.long)
    neg_ids_t = torch.from_numpy(neg_ids_np).to(device=device, dtype=torch.long)
    neg_mask_t = torch.from_numpy(neg_mask_np).to(device=device, dtype=torch.long)

    with torch.no_grad():
        prompt_hidden = text_encoder(
            input_ids=pos_ids_t,
            attention_mask=pos_mask_t,
        ).last_hidden_state
        negative_hidden = text_encoder(
            input_ids=neg_ids_t,
            attention_mask=neg_mask_t,
        ).last_hidden_state

    prompt_embeds = prompt_hidden.to(dtype=torch.bfloat16, device=device)
    negative_prompt_embeds = negative_hidden.to(
        dtype=torch.bfloat16, device=device
    )
    pos_seq_lens = pos_mask_t.gt(0).sum(dim=1).long()
    neg_seq_lens = neg_mask_t.gt(0).sum(dim=1).long()
    prompt_embeds = torch.stack(
        [
            torch.cat(
                [
                    row[:seq_len],
                    row.new_zeros(
                        args.max_sequence_length - seq_len,
                        row.size(1),
                    ),
                ]
            )
            for row, seq_len in zip(prompt_embeds, pos_seq_lens, strict=False)
        ],
        dim=0,
    )
    negative_prompt_embeds = torch.stack(
        [
            torch.cat(
                [
                    row[:seq_len],
                    row.new_zeros(
                        args.max_sequence_length - seq_len,
                        row.size(1),
                    ),
                ]
            )
            for row, seq_len in zip(
                negative_prompt_embeds, neg_seq_lens, strict=False
            )
        ],
        dim=0,
    )

    del text_encoder, pos_ids_t, pos_mask_t, neg_ids_t, neg_mask_t
    gc.collect()
    torch.cuda.empty_cache()

    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.model, subfolder="scheduler"
    )
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    num_latent_frames = (
        (args.num_frames - 1) // vae_scale_factor_temporal + 1
    )
    latents = torch.randn(
        1,
        int(transformer_cfg.get("in_channels", 16)),
        num_latent_frames,
        args.height // vae_scale_factor_spatial,
        args.width // vae_scale_factor_spatial,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    initial_latents = _np_float32(latents)

    if boundary_ratio is not None:
        boundary_timestep = (
            float(boundary_ratio) * scheduler.config.num_train_timesteps
        )
    else:
        boundary_timestep = None

    first_low_noise_idx: int | None = None
    selected_step_indices = [0]
    if boundary_timestep is not None:
        for idx, t in enumerate(timesteps):
            if float(t.item()) < boundary_timestep:
                first_low_noise_idx = idx
                if idx not in selected_step_indices:
                    selected_step_indices.append(idx)
                break

    step_debug: dict[str, dict[str, np.ndarray]] = {}
    stage_by_step: dict[str, str] = {}

    low_noise_start_idx = (
        first_low_noise_idx if first_low_noise_idx is not None else len(timesteps)
    )

    def _run_phase(
        model: Any,
        step_indices: range,
        *,
        stage: str,
        guidance_scale: float,
    ) -> torch.Tensor:
        nonlocal latents
        for step_idx in step_indices:
            timestep = timesteps[step_idx]
            current_model = model

            latent_model_input = latents.to(torch.bfloat16)
            timestep_tensor = _prepare_timestep_tensor_diffusers(
                latents, timestep, expand_timesteps=expand_timesteps
            )

            with current_model.cache_context("cond"):
                noise_cond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

            with current_model.cache_context("uncond"):
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

            noise_guided = noise_uncond + guidance_scale * (
                noise_cond - noise_uncond
            )

            if step_idx in selected_step_indices:
                step_key = f"step_{step_idx}"
                stage_by_step[step_key] = stage
                step_debug[step_key] = {
                    "cond": _np_float32(noise_cond),
                    "uncond": _np_float32(noise_uncond),
                    "guided": _np_float32(noise_guided),
                    "latents_in": _np_float32(latents),
                    "timestep": np.asarray(
                        [float(timestep.item())], dtype=np.float32
                    ),
                }

            latents = scheduler.step(
                noise_guided, timestep, latents, return_dict=False
            )[0]
        return latents

    transformer = WanTransformer3DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to(device)
    transformer.eval()
    _run_phase(
        transformer,
        range(0, low_noise_start_idx),
        stage="high_noise",
        guidance_scale=args.guidance_scale,
    )

    del transformer
    gc.collect()
    torch.cuda.empty_cache()

    if low_noise_start_idx < len(timesteps):
        transformer_2 = WanTransformer3DModel.from_pretrained(
            args.model,
            subfolder="transformer_2",
            torch_dtype=torch.bfloat16,
        ).to(device)
        transformer_2.eval()
        _run_phase(
            transformer_2,
            range(low_noise_start_idx, len(timesteps)),
            stage="low_noise",
            guidance_scale=args.guidance_scale_2,
        )
        del transformer_2
        gc.collect()
        torch.cuda.empty_cache()

    raw_final_latents = _np_float32(latents)

    vae = AutoencoderKLWan.from_pretrained(
        args.model,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval()

    latents_for_vae = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents_for_vae.device, latents_for_vae.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents_for_vae.device, latents_for_vae.dtype)
    )
    denorm_latents = latents_for_vae / latents_std + latents_mean
    vae_output = vae.decode(denorm_latents, return_dict=False)[0]
    vae_output = vae_output[
        :,
        :,
        : args.num_frames,
        : args.height,
        : args.width,
    ]

    result = {
        "positive_input_ids": pos_ids_np,
        "positive_attention_mask": pos_mask_np,
        "negative_input_ids": neg_ids_np,
        "negative_attention_mask": neg_mask_np,
        "prompt_embeds": _np_float32(prompt_embeds),
        "negative_prompt_embeds": _np_float32(negative_prompt_embeds),
        "scheduler_timesteps": _np_float32(timesteps),
        "initial_latents": initial_latents,
        "selected_step_indices": selected_step_indices,
        "stage_by_step": stage_by_step,
        "step_debug": step_debug,
        "raw_final_latents": raw_final_latents,
        "denorm_latents": _np_float32(denorm_latents),
        "vae_output": _np_float32(vae_output),
    }

    del (
        tokenizer,
        scheduler,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        latents_for_vae,
        latents_mean,
        latents_std,
        denorm_latents,
        vae_output,
        vae,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _build_max_pipeline(model_id: str) -> MaxWanPipeline:
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=model_id,
            device_specs=[DeviceSpec.accelerator()],
        )
    )
    devices = load_devices(config.model.device_specs)
    session = InferenceSession(devices=devices)
    config.configure_session(session)
    weight_paths = get_weight_paths(config.model)
    return MaxWanPipeline(
        pipeline_config=config,
        session=session,
        devices=devices,
        weight_paths=weight_paths,
    )


def _max_prompt_embeds_from_ids(
    pipe: MaxWanPipeline,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    text_input_ids = Tensor.constant(
        input_ids,
        dtype=DType.int64,
        device=pipe.text_encoder.devices[0],
    )
    text_attention_mask = Tensor.constant(
        attention_mask,
        dtype=DType.int64,
        device=pipe.text_encoder.devices[0],
    )
    hidden_states = pipe.text_encoder(text_input_ids, text_attention_mask)
    prompt_embeds = pipe.get_t5_prompt_embeds_from_hidden(
        hidden_states=hidden_states,
        attention_mask=text_attention_mask,
        num_videos_per_prompt=1,
        max_sequence_length=input_ids.shape[-1],
    )
    return _max_tensor_to_numpy(prompt_embeds)


def _guidance_tensor(pipe: MaxWanPipeline, value: float) -> Tensor:
    return Tensor.full(
        [1],
        value,
        dtype=pipe.transformer.config.dtype,
        device=pipe.transformer.devices[0],
    )


def _run_max_reference(
    args: argparse.Namespace, diffusers_ref: Mapping[str, Any]
) -> dict[str, Any]:
    pipe = _build_max_pipeline(args.model)
    device = pipe.transformer.devices[0]
    cpu = CPU()

    prompt_embeds_np = _max_prompt_embeds_from_ids(
        pipe,
        diffusers_ref["positive_input_ids"],
        diffusers_ref["positive_attention_mask"],
    )
    negative_prompt_embeds_np = _max_prompt_embeds_from_ids(
        pipe,
        diffusers_ref["negative_input_ids"],
        diffusers_ref["negative_attention_mask"],
    )

    prompt_embeds = Tensor.from_dlpack(
        np.ascontiguousarray(prompt_embeds_np, dtype=np.float32)
    ).cast(pipe.transformer.config.dtype).to(device)
    negative_prompt_embeds = Tensor.from_dlpack(
        np.ascontiguousarray(negative_prompt_embeds_np, dtype=np.float32)
    ).cast(pipe.transformer.config.dtype).to(device)

    latents_np = np.ascontiguousarray(
        diffusers_ref["initial_latents"], dtype=np.float32
    )
    latents = Tensor.from_dlpack(latents_np).to(device)

    pipe._scheduler.set_timesteps(args.num_inference_steps)
    scheduler_timesteps = pipe._scheduler.timesteps
    assert scheduler_timesteps is not None

    rope_cos, rope_sin = pipe.transformer.compute_rope(
        num_frames=int(latents.shape[2]),
        height=int(latents.shape[3]),
        width=int(latents.shape[4]),
    )
    batched_timesteps = pipe._get_batched_timesteps(
        scheduler_timesteps=scheduler_timesteps,
        batch_size=int(latents.shape[0]),
        device=device,
    )
    p_t, p_h, p_w = pipe.transformer.config.patch_size
    spatial_shape = pipe._get_spatial_shape(
        int(latents.shape[2]) // p_t,
        int(latents.shape[3]) // p_h,
        int(latents.shape[4]) // p_w,
        device,
    )

    boundary_timestep = pipe.compute_boundary_timestep(
        pipe.boundary_ratio, pipe.num_train_timesteps
    )
    guidance_high = _guidance_tensor(pipe, args.guidance_scale)
    guidance_low = _guidance_tensor(
        pipe, args.guidance_scale_2 if args.guidance_scale_2 is not None else args.guidance_scale
    )

    step_debug: dict[str, dict[str, np.ndarray]] = {}
    stage_by_step: dict[str, str] = {}
    selected_step_indices = list(diffusers_ref["selected_step_indices"])

    prompt_embeds_buf = prompt_embeds.driver_tensor
    negative_prompt_embeds_buf = negative_prompt_embeds.driver_tensor

    for step_idx, _ in enumerate(scheduler_timesteps):
        timestep_value = float(int(scheduler_timesteps[step_idx]))
        if pipe.use_low_noise_transformer(timestep_value, boundary_timestep):
            transformer_model = pipe.transformer_2
            guidance_tensor = guidance_low
            stage = "low_noise"
        else:
            transformer_model = pipe.transformer
            guidance_tensor = guidance_high
            stage = "high_noise"

        assert transformer_model is not None
        latent_model_input = latents.cast(DType.bfloat16).driver_tensor
        noise_cond_buf = transformer_model(
            latent_model_input,
            batched_timesteps[step_idx],
            prompt_embeds_buf,
            rope_cos,
            rope_sin,
            spatial_shape,
        )
        noise_uncond_buf = transformer_model(
            latent_model_input,
            batched_timesteps[step_idx],
            negative_prompt_embeds_buf,
            rope_cos,
            rope_sin,
            spatial_shape,
        )
        noise_cond = Tensor.from_dlpack(noise_cond_buf)
        noise_uncond = Tensor.from_dlpack(noise_uncond_buf)
        noise_guided = pipe._guidance_model(
            noise_cond, noise_uncond, guidance_tensor
        )

        if step_idx in selected_step_indices:
            step_key = f"step_{step_idx}"
            stage_by_step[step_key] = stage
            step_debug[step_key] = {
                "cond": _max_tensor_to_numpy(noise_cond),
                "uncond": _max_tensor_to_numpy(noise_uncond),
                "guided": _max_tensor_to_numpy(noise_guided),
                "latents_in": np.asarray(latents_np, dtype=np.float32),
                "timestep": np.asarray([timestep_value], dtype=np.float32),
            }

        noise_np = np.from_dlpack(noise_guided.cast(DType.float32).to(cpu))
        latents_np = pipe._scheduler.step(
            noise_np, int(scheduler_timesteps[step_idx]), latents_np
        )
        latents = (
            Tensor.from_dlpack(np.ascontiguousarray(latents_np, dtype=np.float32))
            .cast(DType.bfloat16)
            .to(device)
        )

    raw_final_latents = np.asarray(latents_np, dtype=np.float32)
    raw_final_tensor = Tensor.from_dlpack(
        np.ascontiguousarray(raw_final_latents, dtype=np.float32)
    ).to(device)
    denorm_latents = pipe.denormalize_vae_latents(
        latents=raw_final_tensor,
        latents_mean=list(pipe.vae.config.latents_mean),
        latents_std=list(pipe.vae.config.latents_std),
        z_dim=int(pipe.vae.config.z_dim),
    )
    denorm_latents_np = _max_tensor_to_numpy(denorm_latents)

    diffusers_denorm_tensor = (
        Tensor.from_dlpack(
            np.ascontiguousarray(diffusers_ref["denorm_latents"], dtype=np.float32)
        )
        .cast(DType.bfloat16)
        .to(device)
    )
    pipe.vae.load_model()
    max_vae_same_input = pipe.vae.decode_5d(diffusers_denorm_tensor)
    max_vae_same_input_np = _max_tensor_to_numpy(max_vae_same_input)[
        :,
        :,
        : args.num_frames,
        : args.height,
        : args.width,
    ]

    max_own_vae_output = pipe.vae.decode_5d(
        denorm_latents.cast(pipe.transformer.config.dtype)
    )
    max_own_vae_output_np = _max_tensor_to_numpy(max_own_vae_output)[
        :,
        :,
        : args.num_frames,
        : args.height,
        : args.width,
    ]

    return {
        "prompt_embeds": prompt_embeds_np,
        "negative_prompt_embeds": negative_prompt_embeds_np,
        "scheduler_timesteps": np.asarray(
            scheduler_timesteps, dtype=np.float32
        ),
        "selected_step_indices": selected_step_indices,
        "stage_by_step": stage_by_step,
        "step_debug": step_debug,
        "raw_final_latents": raw_final_latents,
        "denorm_latents": denorm_latents_np,
        "vae_output_same_input": max_vae_same_input_np,
        "vae_output_own_latents": max_own_vae_output_np,
    }


def _compare_and_save(
    args: argparse.Namespace,
    diffusers_ref: Mapping[str, Any],
    max_ref: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "settings": {
            "model": args.model,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2,
            "seed": args.seed,
            "max_sequence_length": args.max_sequence_length,
        },
        "metrics": {},
    }

    pairs = {
        "text_encoder.prompt_embeds": (
            diffusers_ref["prompt_embeds"],
            max_ref["prompt_embeds"],
        ),
        "text_encoder.negative_prompt_embeds": (
            diffusers_ref["negative_prompt_embeds"],
            max_ref["negative_prompt_embeds"],
        ),
        "scheduler.timesteps": (
            diffusers_ref["scheduler_timesteps"],
            max_ref["scheduler_timesteps"],
        ),
        "pipeline.final_raw_latents": (
            diffusers_ref["raw_final_latents"],
            max_ref["raw_final_latents"],
        ),
        "pipeline.final_denorm_latents": (
            diffusers_ref["denorm_latents"],
            max_ref["denorm_latents"],
        ),
        "vae.same_diffusers_input": (
            diffusers_ref["vae_output"],
            max_ref["vae_output_same_input"],
        ),
        "pipeline.final_decoded_output": (
            diffusers_ref["vae_output"],
            max_ref["vae_output_own_latents"],
        ),
    }

    for name, (reference, actual) in pairs.items():
        report["metrics"][name] = _l2_metrics(reference, actual)
        _save_array(output_dir, f"diffusers_{name.replace('.', '_')}", reference, args.save_arrays)
        _save_array(output_dir, f"max_{name.replace('.', '_')}", actual, args.save_arrays)

    for step_key, diff_step in diffusers_ref["step_debug"].items():
        max_step = max_ref["step_debug"][step_key]
        for tensor_name in ("cond", "uncond", "guided"):
            metric_name = f"dit.{step_key}.{tensor_name}"
            report["metrics"][metric_name] = _l2_metrics(
                diff_step[tensor_name],
                max_step[tensor_name],
            )
            report.setdefault("step_stage", {})[step_key] = diffusers_ref[
                "stage_by_step"
            ][step_key]

    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved comparison report to {report_path}")

    print("\nWan diffusers vs MAX comparison (L2-based)")
    print("=========================================")
    for name, metrics in sorted(report["metrics"].items()):
        print(
            f"{name}: l2={metrics['l2_norm']:.6f} "
            f"rel_l2={metrics['relative_l2_norm']:.6e} "
            f"max_abs={metrics['max_abs']:.6f}"
        )

    return report


def main() -> int:
    args = parse_args()
    diffusers_ref = _run_diffusers_reference(args)
    max_ref = _run_max_reference(args, diffusers_ref)
    _compare_and_save(args, diffusers_ref, max_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
