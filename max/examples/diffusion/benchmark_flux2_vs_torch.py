# mypy: ignore-errors
import argparse
import time
import torch
import numpy as np
import warnings
import asyncio
from diffusers import Flux2Pipeline as DiffusersFlux2Pipeline
from max.pipelines import PipelineConfig
from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2Pipeline as MaxFlux2Pipeline
from max.driver import DeviceSpec
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.pipeline_variants.pixel_generation import PixelGenerationPipeline
from max.interfaces import PixelGenerationRequest, RequestID, PixelGenerationInputs
from max.pipelines.core import PixelContext

# Suppress warnings
warnings.filterwarnings("ignore")

def benchmark_diffusers(model_path, num_steps=1, num_warmup=3, num_runs=5):
    print(f"\n--- Benchmarking Diffusers (torch.compile) ---")
    print(f"Loading model from {model_path}...")
    
    try:
        # Load pipeline
        pipe = DiffusersFlux2Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            revision="refs/pr/1", 
        ).to("cuda")
        
        # Optimize with torch.compile
        print("Compiling transformer with torch.compile...")
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        
        # Warmup
        print("Warming up...")
        for _ in range(num_warmup):
            _ = pipe(
                prompt="A cat",
                num_inference_steps=num_steps,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512, 
                output_type="pil"
            )
            
        # Benchmark
        latencies = []
        print(f"Running {num_runs} compiled runs...")
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = pipe(
                prompt="A cat",
                num_inference_steps=num_steps,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                output_type="pil"
            )
            torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000) # ms
            print(f"Run {i+1}: {latencies[-1]:.2f} ms")
            
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        print(f"Diffusers Average Latency: {avg_latency:.2f} ms")
        print(f"Diffusers Median Latency: {median_latency:.2f} ms")
        return median_latency
    except Exception as e:
        print(f"Diffusers benchmark failed: {e}")
        return float('inf')

async def setup_and_benchmark_max(model_path, num_steps=1, num_warmup=3, num_runs=5):
    print(f"\n--- Benchmarking MAX ---")
    print(f"Loading model from {model_path}...")
    
    # Config
    config = PipelineConfig(
        model_path=model_path,
        device_specs=[DeviceSpec.accelerator()],
        use_legacy_module=False,
    )
    
    # Tokenizer
    tokenizer = PixelGenerationTokenizer(
        model_path=model_path,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=512,
    )
    
    # Pipeline
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=MaxFlux2Pipeline,
    )
    
    # Request
    request = PixelGenerationRequest(
        request_id=RequestID(),
        model_name="benchmark",
        prompt="A cat",
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        guidance_scale=3.5,
    )
    
    # Context
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )
    
    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        _ = pipeline.execute(inputs)
        
    # Benchmark
    latencies = []
    print(f"Running {num_runs} runs...")
    for i in range(num_runs):
        start_time = time.time()
        _ = pipeline.execute(inputs)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)
        print(f"Run {i+1}: {latencies[-1]:.2f} ms")

    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    print(f"MAX Average Latency: {avg_latency:.2f} ms")
    print(f"MAX Median Latency: {median_latency:.2f} ms")
    return median_latency

def benchmark_max(model_path, num_steps=1, num_warmup=3, num_runs=5):
    return asyncio.run(setup_and_benchmark_max(model_path, num_steps, num_warmup, num_runs))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to FLUX.2 model")
    parser.add_argument("--steps", type=int, default=1, help="Number of steps")
    parser.add_argument("--framework", type=str, choices=["diffusers", "max", "both"], default="both", help="Framework to benchmark")
    args = parser.parse_args()
    
    diffusers_time = 0
    max_time = 0
    
    if args.framework in ["diffusers", "both"]:
        diffusers_time = benchmark_diffusers(args.model, num_steps=args.steps)
        # Clear memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    if args.framework in ["max", "both"]:
        max_time = benchmark_max(args.model, num_steps=args.steps)
        
    print(f"\nSummary:")
    if args.framework in ["diffusers", "both"]:
        print(f"Diffusers: {diffusers_time:.2f} ms")
    if args.framework in ["max", "both"]:
        print(f"MAX:       {max_time:.2f} ms")
    if args.framework == "both" and diffusers_time > 0 and max_time > 0:
        print(f"Speedup:   {diffusers_time / max_time:.2fx}")
