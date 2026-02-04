
# mypy: ignore-errors
import argparse
import time
import os
import numpy as np
import torch # Pre-load torch to avoid symbol errors
from max.driver import DeviceSpec, CPU
from max.pipelines import PipelineConfig
from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2Pipeline
from max.tensor import Tensor
from max.dtype import DType

from max.pipelines.lib.pipeline_variants.pixel_generation import PixelGenerationPipeline
from max.pipelines import PixelContext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    args = parser.parse_args()

    model_path = os.path.normpath(args.model)
    print(f"Loading VAE from {model_path}...")
    
    # 1. Load Pipeline
    config = PipelineConfig(
        model_path=model_path,
        device_specs=[DeviceSpec.accelerator()],
        use_legacy_module=False,
    )
    
    # Use Wrapper to handle init
    pipeline_wrapper = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=Flux2Pipeline,
    )
    pipeline = pipeline_wrapper._pipeline_model
    
    # Ensure VAE is present
    if not hasattr(pipeline, "vae"):
        raise RuntimeError("VAE not found in pipeline!")

    print("VAE loaded.")
    dev = pipeline.vae.devices[0]
    print(f"Device: {dev}")

    # 2. Prepare Inputs
    # Flux2 VAE: 32 channels. 
    # For 1024x1024 image -> 128x128 latents (8x downsample)
    H, W = 128, 128
    C = 32
    print(f"Preparing inputs (1x{C}x{H}x{W})...")
    latents_np = np.random.randn(1, C, H, W).astype(np.float32)
    dtype = DType.bfloat16
    
    # Convert to MAX Tensor
    latents = Tensor.from_dlpack(latents_np).to(dev).cast(dtype)
    latents_drv = latents.driver_tensor
    
    # 3. Warmup
    print(f"Compiling/Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        _ = pipeline.vae.decode(latents_drv)
    # Sync after warmup
    _ = np.array(Tensor.from_dlpack(pipeline.vae.decode(latents_drv)).to(CPU()))
    print("Warmup complete.")
    
    # 4. Benchmark
    print(f"Running Benchmark ({args.iterations} iters)...")
    latencies = []
    
    for i in range(args.iterations):
        # Synchronize before start
        # (Technically usually not needed if we measure submission + sync at end, 
        # but to be clean let's ensure previous work is done)
        
        start = time.perf_counter()
        res_drv = pipeline.vae.decode(latents_drv)
        
        # Synchronize
        # VAE output is DriverTensor (or whatever decode returns, likely Tensor if wrapped, but pipeline.vae.decode might be raw Module)
        # pipeline.vae is an AutoencoderKLFlux2 which calls super().forward -> returns Tensor
        # Let's check type
        if not isinstance(res_drv, Tensor):
             res_t = Tensor.from_dlpack(res_drv)
        else:
             res_t = res_drv
             
        # Force download to host to ensure completion
        _ = np.array(res_t.to(CPU()))
        end = time.perf_counter()
        
        latency = (end - start) * 1000
        latencies.append(latency)
        print(f"Iter {i}: {latency:.2f} ms")

    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    print(f"\nResults for 1024x1024 (Latent {C}x{H}x{W}):")
    print(f"Avg VAE Latency:    {avg_latency:.2f} ms")
    print(f"Median VAE Latency: {median_latency:.2f} ms")

if __name__ == "__main__":
    main()
