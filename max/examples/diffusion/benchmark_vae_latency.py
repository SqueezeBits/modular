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
    # Shape: (1, 32, 128, 128) based on error message
    print("Preparing inputs (1x32x128x128)...")
    latents_np = np.random.randn(1, 32, 128, 128).astype(np.float32)
    # Convert to bfloat16 if needed (default for GPU)
    # Check VAE config dtype or just assume bfloat16
    dtype = DType.bfloat16
    
    latents = Tensor.from_dlpack(latents_np).to(dev).cast(dtype)
    
    # 3. Warmup (Compilation)
    print("Compiling VAE (Warmup)...")
    t0 = time.time()
    # Note: pipeline.vae.decode expects driver_tensor
    # Check signature: def decode(self, sample: Any) -> Any:
    # It likely unwraps or expects driver tensor. 
    # In pipeline_flux2.py: self.vae.decode(latents_unpacked.driver_tensor)
    _ = pipeline.vae.decode(latents.driver_tensor)
    t1 = time.time()
    print(f"Compilation/Warmup took: {t1 - t0:.4f}s")
    
    # 4. Benchmark
    print("Running Benchmark (5 iters)...")
    latencies = []
    for i in range(5):
        start = time.time()
        _ = pipeline.vae.decode(latents.driver_tensor)
        # We should synchronize to measure execution time properly?
        # Tensor operations are async. 
        # But .decode returns a value. If it returns Tensor, we might need to materialize it?
        # pipeline_flux2.py converts result to Tensor via from_dlpack if not Tensor.
        # But here we call vae.decode directly. It returns DriverTensor?
        # If it returns DriverTensor, execution is enqueued.
        # To synchronize, we can copy to host.
        end = time.time() # This is just submission time if async!
        latencies.append(end - start)
    
    # However, to be sure, let's materialize result
    print("Benchmarking with synchronization...")
    latencies = []
    for i in range(5):
        start = time.time()
        res = pipeline.vae.decode(latents.driver_tensor)
        # Materialize to force sync
        # If res is DriverTensor, wrap in Tensor
        if not isinstance(res, Tensor):
             res_t = Tensor.from_dlpack(res)
        else:
             res_t = res
        _ = res_t.to(CPU()) 
        # Actually Tensor.to_numpy() or .item() forces sync.
        _ = np.array(res_t) 
        end = time.time()
        print(f"Iter {i}: {end - start:.4f}s")
        latencies.append(end - start)

    print(f"Avg VAE Latency: {np.mean(latencies):.4f}s")

if __name__ == "__main__":
    main()
