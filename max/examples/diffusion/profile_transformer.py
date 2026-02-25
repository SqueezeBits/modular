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

"""Profile the transformer component of a Flux diffusion pipeline.

Runs a single (framework, input-shape) combination and dumps a Chrome-trace
JSON file.  Run six times with different arguments to cover all combinations:

    # Diffusers (activate conda env first: conda activate diffusers)
    python max/examples/diffusion/profile_transformer.py \
        --model black-forest-labs/FLUX.1-dev \
        --framework diffusers --input-shape flux1 \
        --output traces/transformer_diffusers_flux1.json

    # MAX (use bazel so MAX packages are on the path)
    ./bazelw run //max/examples/diffusion:profile_transformer -- \
        --model black-forest-labs/FLUX.2-dev \
        --framework max --input-shape flux2-t2i \
        --output traces/transformer_max_flux2-t2i.json
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

# ---------------------------------------------------------------------------
# Constants for 1024×1024 image generation
# ---------------------------------------------------------------------------
IMAGE_SIZE = 1024
VAE_SCALE_FACTOR = 8
PATCH_FACTOR = 2
LATENT_H = IMAGE_SIZE // (VAE_SCALE_FACTOR * PATCH_FACTOR)  # 64
LATENT_W = LATENT_H  # 64
IMAGE_SEQ_LEN = LATENT_H * LATENT_W  # 4096
TEXT_SEQ_LEN = 512

# Flux1 model dimensions
FLUX1_IN_CHANNELS = 64  # 16 VAE channels × 4 (2×2 patch)
FLUX1_JOINT_ATTENTION_DIM = 4096  # T5-XXL
FLUX1_POOLED_PROJECTION_DIM = 768  # CLIP

# Flux2 model dimensions
FLUX2_IN_CHANNELS = 128  # 32 VAE channels × 4 (2×2 patch)
FLUX2_JOINT_ATTENTION_DIM = 15360  # Mistral3 (3 layers × 5120)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the transformer component of a Flux diffusion pipeline "
            "for a single (framework, input-shape) combination."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "HuggingFace model identifier "
            "(e.g. black-forest-labs/FLUX.1-dev or FLUX.2-dev)."
        ),
    )
    parser.add_argument(
        "--framework",
        required=True,
        choices=["diffusers", "max"],
        help="Inference framework to use.",
    )
    parser.add_argument(
        "--input-shape",
        required=True,
        choices=["flux1", "flux2-t2i", "flux2-i2i"],
        help=(
            "Input shape variant: "
            "flux1 (FLUX.1 text-to-image), "
            "flux2-t2i (FLUX.2 text-to-image), "
            "flux2-i2i (FLUX.2 image-to-image with concatenated noise+image latents)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output Chrome trace JSON file path. "
            "Defaults to transformer_{framework}_{input_shape}.json in the "
            "current directory."
        ),
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=3,
        help="Number of warmup iterations before profiling (default: 3).",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of profiled iterations captured in the trace (default: 3).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Position ID helpers
# ---------------------------------------------------------------------------


def _make_flux1_img_ids_np() -> np.ndarray:
    """Flux1 image IDs: (image_seq_len, 3) with (T=0, H, W) coordinates."""
    img_ids = np.zeros((IMAGE_SEQ_LEN, 3), dtype=np.float32)
    h_coords = np.arange(LATENT_H).repeat(LATENT_W)
    w_coords = np.tile(np.arange(LATENT_W), LATENT_H)
    img_ids[:, 1] = h_coords
    img_ids[:, 2] = w_coords
    return img_ids


def _make_flux2_img_ids_t2i_np() -> np.ndarray:
    """Flux2 t2i image IDs: (1, image_seq_len, 4) with (T=10, H, W, L=0)."""
    img_ids = np.zeros((1, IMAGE_SEQ_LEN, 4), dtype=np.int64)
    h_coords = np.arange(LATENT_H).repeat(LATENT_W)
    w_coords = np.tile(np.arange(LATENT_W), LATENT_H)
    img_ids[0, :, 0] = 10
    img_ids[0, :, 1] = h_coords
    img_ids[0, :, 2] = w_coords
    return img_ids


def _make_flux2_img_ids_i2i_np() -> np.ndarray:
    """Flux2 i2i image IDs: (1, 2*image_seq_len, 4) — noise (T=10) + image (T=20)."""
    total = IMAGE_SEQ_LEN * 2
    img_ids = np.zeros((1, total, 4), dtype=np.int64)
    h_coords = np.arange(LATENT_H).repeat(LATENT_W)
    w_coords = np.tile(np.arange(LATENT_W), LATENT_H)
    img_ids[0, :IMAGE_SEQ_LEN, 0] = 10
    img_ids[0, :IMAGE_SEQ_LEN, 1] = h_coords
    img_ids[0, :IMAGE_SEQ_LEN, 2] = w_coords
    img_ids[0, IMAGE_SEQ_LEN:, 0] = 20
    img_ids[0, IMAGE_SEQ_LEN:, 1] = h_coords
    img_ids[0, IMAGE_SEQ_LEN:, 2] = w_coords
    return img_ids


def _make_flux2_txt_ids_np() -> np.ndarray:
    """Flux2 text IDs: (1, text_seq_len, 4) with (T=0, H=0, W=0, L=position)."""
    txt_ids = np.zeros((1, TEXT_SEQ_LEN, 4), dtype=np.int64)
    txt_ids[0, :, 3] = np.arange(TEXT_SEQ_LEN)
    return txt_ids


# ---------------------------------------------------------------------------
# Diffusers framework
# ---------------------------------------------------------------------------


def prepare_diffusers_inputs(
    input_shape: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create dummy torch tensor inputs for the diffusers transformer."""
    dtype = torch.bfloat16

    if input_shape == "flux1":
        return {
            "hidden_states": torch.randn(
                1, IMAGE_SEQ_LEN, FLUX1_IN_CHANNELS, device=device, dtype=dtype
            ),
            "encoder_hidden_states": torch.randn(
                1,
                TEXT_SEQ_LEN,
                FLUX1_JOINT_ATTENTION_DIM,
                device=device,
                dtype=dtype,
            ),
            "pooled_projections": torch.randn(
                1, FLUX1_POOLED_PROJECTION_DIM, device=device, dtype=dtype
            ),
            "timestep": torch.full((1,), 500.0, device=device, dtype=dtype),
            "img_ids": torch.from_numpy(_make_flux1_img_ids_np()).to(
                device=device, dtype=dtype
            ),
            "txt_ids": torch.zeros(TEXT_SEQ_LEN, 3, device=device, dtype=dtype),
            "guidance": torch.full((1,), 3.5, device=device, dtype=dtype),
        }

    seq_len = IMAGE_SEQ_LEN if input_shape == "flux2-t2i" else IMAGE_SEQ_LEN * 2
    img_ids_np = (
        _make_flux2_img_ids_t2i_np()
        if input_shape == "flux2-t2i"
        else _make_flux2_img_ids_i2i_np()
    )
    return {
        "hidden_states": torch.randn(
            1, seq_len, FLUX2_IN_CHANNELS, device=device, dtype=dtype
        ),
        "encoder_hidden_states": torch.randn(
            1,
            TEXT_SEQ_LEN,
            FLUX2_JOINT_ATTENTION_DIM,
            device=device,
            dtype=dtype,
        ),
        "timestep": torch.full((1,), 0.5, device=device, dtype=dtype),
        "img_ids": torch.from_numpy(img_ids_np).to(device=device),
        "txt_ids": torch.from_numpy(_make_flux2_txt_ids_np()).to(device=device),
        "guidance": torch.full((1,), 3.5, device=device, dtype=dtype),
    }


def _load_diffusers_pipeline(model: str) -> Any:
    """Load a diffusers DiffusionPipeline, keeping only the transformer on GPU."""
    from diffusers import DiffusionPipeline

    print(f"  Loading diffusers pipeline: {model}")
    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
    )
    # Only move the transformer to GPU; keep everything else on CPU
    pipe.transformer = pipe.transformer.to("cuda")
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    print(f"  Transformer type: {type(pipe.transformer).__name__}")
    print(
        f"  Transformer parameters: "
        f"{sum(p.numel() for p in pipe.transformer.parameters()) / 1e9:.2f}B"
    )
    return pipe


def _run_diffusers(
    transformer: Any,
    input_shape: str,
    num_warmups: int,
    num_iterations: int,
    trace_path: str,
) -> list[float]:
    """Run warmup + profiled iterations for a diffusers transformer.

    Returns:
        List of per-iteration latencies in milliseconds (warmup runs).
    """
    import inspect

    device = torch.device("cuda")
    inputs = prepare_diffusers_inputs(input_shape, device)

    # After torch.compile, use _orig_mod to get the real forward signature.
    orig_mod = getattr(transformer, "_orig_mod", transformer)
    sig = inspect.signature(orig_mod.forward)
    call_kwargs = {k: v for k, v in inputs.items() if k in sig.parameters}
    call_kwargs["return_dict"] = False

    print("  Input shapes:")
    for k, v in inputs.items():
        print(f"    {k}: {tuple(v.shape)} ({v.dtype})")

    warmup_times: list[float] = []
    print(f"  Warmup ({num_warmups} iters):")
    for i in range(num_warmups):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            transformer(**call_kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        warmup_times.append(elapsed_ms)
        print(f"    iter {i}: {elapsed_ms:.2f} ms")

    print(f"  Profiling ({num_iterations} iters)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_iterations):
            with torch.no_grad():
                transformer(**call_kwargs)
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"  Trace saved to: {trace_path}")
    return warmup_times


# ---------------------------------------------------------------------------
# MAX framework
# ---------------------------------------------------------------------------


def prepare_max_inputs(input_shape: str, device: Any) -> list[Any]:
    """Create dummy MAX tensor inputs for the transformer."""
    from max.dtype import DType
    from max.tensor import Tensor

    dtype = DType.bfloat16

    def _randn(shape: list[int], dt: DType) -> Tensor:
        arr = np.random.randn(*shape).astype(np.float32)
        return Tensor.from_dlpack(arr).to(device).cast(dt)

    def _full(shape: list[int], value: float, dt: DType) -> Tensor:
        return Tensor.full(shape, value, dtype=dt, device=device)

    def _from_np(arr: np.ndarray) -> Tensor:
        return Tensor.from_dlpack(arr).to(device)

    if input_shape == "flux1":
        return [
            _randn([1, IMAGE_SEQ_LEN, FLUX1_IN_CHANNELS], dtype),
            _randn([1, TEXT_SEQ_LEN, FLUX1_JOINT_ATTENTION_DIM], dtype),
            _randn([1, FLUX1_POOLED_PROJECTION_DIM], dtype),
            _full([1], 500.0, DType.float32),
            _from_np(_make_flux1_img_ids_np()).cast(dtype),
            _from_np(np.zeros((TEXT_SEQ_LEN, 3), dtype=np.float32)).cast(dtype),
            _full([1], 3.5, dtype),
        ]

    seq_len = IMAGE_SEQ_LEN if input_shape == "flux2-t2i" else IMAGE_SEQ_LEN * 2
    img_ids_np = (
        _make_flux2_img_ids_t2i_np()
        if input_shape == "flux2-t2i"
        else _make_flux2_img_ids_i2i_np()
    )
    return [
        _randn([1, seq_len, FLUX2_IN_CHANNELS], dtype),
        _randn([1, TEXT_SEQ_LEN, FLUX2_JOINT_ATTENTION_DIM], dtype),
        _full([1], 0.5, dtype),
        _from_np(img_ids_np),
        _from_np(_make_flux2_txt_ids_np()),
        _full([1], 3.5, dtype),
    ]


def _load_max_pipeline(model: str) -> tuple[Any, Any, Any]:
    """Initialize a MAX pipeline and return (pixel_pipeline, transformer, device)."""
    from typing import cast

    from max.driver import DeviceSpec
    from max.interfaces import PipelineTask
    from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
    from max.pipelines.core import PixelContext
    from max.pipelines.lib.interfaces import (
        DiffusionPipeline as MaxDiffusionPipeline,
    )
    from max.pipelines.lib.pipeline_variants.pixel_generation import (
        PixelGenerationPipeline,
    )

    print(f"  Loading MAX pipeline: {model}")
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=model,
            device_specs=[DeviceSpec.accelerator()],
        ),
        use_legacy_module=False,
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        use_legacy_module=config.use_legacy_module,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, "No matching diffusion architecture found."

    pipeline_model = cast(type[MaxDiffusionPipeline], arch.pipeline_model)
    pixel_pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )
    inner: Any = pixel_pipeline._pipeline_model
    print(f"  Architecture: {arch.name}")
    print(f"  Transformer type: {type(inner.transformer).__name__}")
    return pixel_pipeline, inner.transformer, inner.devices[0]


def _run_max(
    transformer: Any,
    device: Any,
    input_shape: str,
    num_warmups: int,
    num_iterations: int,
    trace_path: str,
) -> list[float]:
    """Run warmup + profiled iterations for a MAX transformer.

    Returns:
        List of per-iteration latencies in milliseconds (warmup runs).
    """
    inputs = prepare_max_inputs(input_shape, device)

    print("  Input shapes:")
    for i, t in enumerate(inputs):
        print(f"    arg[{i}]: {list(t.shape)} ({t.dtype})")

    def _sync() -> None:
        if hasattr(device, "synchronize"):
            device.synchronize()

    warmup_times: list[float] = []
    print(f"  Warmup ({num_warmups} iters):")
    for i in range(num_warmups):
        _sync()
        t0 = time.perf_counter()
        transformer(*inputs)
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        warmup_times.append(elapsed_ms)
        print(f"    iter {i}: {elapsed_ms:.2f} ms")

    print(f"  Profiling ({num_iterations} iters)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_iterations):
            transformer(*inputs)
            _sync()

    prof.export_chrome_trace(trace_path)
    print(f"  Trace saved to: {trace_path}")
    return warmup_times


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _free_gpu() -> None:
    """Best-effort GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"  GPU memory after cleanup: "
            f"{allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved"
        )


def _print_warmup_summary(warmup_times: list[float]) -> None:
    if not warmup_times:
        return
    parts = [f"iter {i}: {t:.2f} ms" for i, t in enumerate(warmup_times)]
    print(f"\nWarmup: {',  '.join(parts)}")
    stable = warmup_times[1:] if len(warmup_times) > 1 else warmup_times
    print(f"Stable mean (iter 1+): {sum(stable) / len(stable):.2f} ms")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resolve output path
    input_shape_safe = args.input_shape  # e.g. "flux2-t2i"
    output = args.output or (
        f"transformer_{args.framework}_{input_shape_safe}.json"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"Model       : {args.model}")
    print(f"Framework   : {args.framework}")
    print(f"Input shape : {args.input_shape}")
    print(f"Warmups     : {args.num_warmups}")
    print(f"Iterations  : {args.num_iterations}")
    print(f"Output      : {output}")
    print()

    try:
        if args.framework == "diffusers":
            pipe = _load_diffusers_pipeline(args.model)
            warmup_times = _run_diffusers(
                pipe.transformer,
                args.input_shape,
                args.num_warmups,
                args.num_iterations,
                output,
            )
            pipe.transformer.to("cpu")
            del pipe
        else:  # max
            pixel_pipeline, transformer, device = _load_max_pipeline(args.model)
            warmup_times = _run_max(
                transformer,
                device,
                args.input_shape,
                args.num_warmups,
                args.num_iterations,
                output,
            )
            del pixel_pipeline, transformer

        _free_gpu()
        _print_warmup_summary(warmup_times)
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    raise SystemExit(main())
