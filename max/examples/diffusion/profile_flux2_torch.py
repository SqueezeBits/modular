#!/usr/bin/env python3
"""Profile Flux2 pipeline using torch.profiler to capture kernel-level traces."""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from max.driver import DeviceSpec
from max.interfaces import (
    PixelGenerationInputs,
    PixelGenerationRequest,
    RequestID,
)
from max.pipelines import PipelineConfig
from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2Pipeline
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)


async def main():
    parser = argparse.ArgumentParser(description="Profile Flux2 pipeline with torch.profiler")
    parser.add_argument("--model", type=str, required=True, help="Path to FLUX.2 model")
    parser.add_argument("--prompt", type=str, default="A cat in a garden", help="Prompt for generation")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--output-trace", type=str, default="flux2_profile", help="Output trace directory")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before profiling")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    
    # Setup pipeline config (matching simple_offline_generation.py)
    device_specs = [DeviceSpec.accelerator()]
    config = PipelineConfig(
        model_path=args.model,
        device_specs=device_specs,
        use_legacy_module=False,
    )
    
    # Tokenizer
    tokenizer = PixelGenerationTokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=512,
    )
    
    # Create pipeline (matching simple_offline_generation.py pattern)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=Flux2Pipeline,
    )
    
    # Warmup
    print(f"Running {args.warmup} warmup iteration(s)...")
    for w in range(args.warmup):
        warmup_request = PixelGenerationRequest(
            request_id=RequestID(),
            model_name=args.model,
            prompt="test",
            height=1024,
            width=1024,
            num_inference_steps=args.steps,
            guidance_scale=3.5,
            seed=42,
        )
        warmup_context = await tokenizer.new_context(warmup_request)
        warmup_inputs = PixelGenerationInputs[PixelContext](
            batch={warmup_context.request_id: warmup_context}
        )
        pipeline.execute(warmup_inputs)
        print(f"Warmup {w+1}/{args.warmup} complete")
    
    # Profile with torch.profiler
    print(f"Profiling with torch.profiler (steps={args.steps})...")
    
    # Create profiled request
    request = PixelGenerationRequest(
        request_id=RequestID(),
        model_name=args.model,
        prompt=args.prompt,
        height=1024,
        width=1024,
        num_inference_steps=args.steps,
        guidance_scale=3.5,
        seed=42,
    )
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(args.output_trace),
    ) as prof:
        t0 = time.time()
        output = pipeline.execute(inputs)
        t1 = time.time()
    
    print(f"\nTotal generation time: {(t1 - t0) * 1000:.2f} ms")
    
    # Print summary
    print("\n=== CUDA Kernel Summary (top 30 by CUDA time) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    
    print(f"\nTrace saved to: {args.output_trace}/")
    print("View with: tensorboard --logdir=" + args.output_trace)


if __name__ == "__main__":
    asyncio.run(main())
