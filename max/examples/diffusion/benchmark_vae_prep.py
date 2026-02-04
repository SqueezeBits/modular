
# mypy: ignore-errors
import argparse
import time
import numpy as np
import torch
from max.driver import Accelerator, Device, DeviceSpec, CPU
from max.dtype import DType
from max.tensor import Tensor
from max.graph import Graph, TensorType, ops, DeviceRef, DeviceKind
from max.engine import InferenceSession

def benchmark_vae_prep(iterations=100, warmup=10):
    device_id = 0
    device_ref = DeviceRef(DeviceKind.GPU, device_id)
    accelerator = Accelerator(device_id)
    device = accelerator
    dtype = DType.bfloat16
    
    # Dimensions for 1024x1024
    height = 1024
    width = 1024
    h_latent = height // 16 # 64
    w_latent = width // 16 # 64
    seq_len = h_latent * w_latent # 4096
    
    latent_channels = 32
    packed_channels = latent_channels * 4 # 128 (Flux2 patchify)
    batch_size = 1
    
    bn_eps = 1e-4
    
    print(f"Benchmarking VAE Prep Graph:")
    print(f"  Shape: 1x{seq_len}x{packed_channels} -> 1x32x128x128")
    print(f"  Device: {device}")
    
    # 1. Build Graph
    input_types = [
         TensorType(dtype, shape=[batch_size, seq_len, packed_channels], device=device_ref),
         TensorType(dtype, shape=[packed_channels], device=device_ref), # bn_mean
         TensorType(dtype, shape=[packed_channels], device=device_ref), # bn_var
    ]
    
    with Graph("vae_prep", input_types=input_types) as graph:
        latents, bn_mean, bn_var = graph.inputs
        
        # 1. Fast Unpack
        latents = ops.permute(latents, (0, 2, 1))
        latents = ops.reshape(latents, (batch_size, packed_channels, h_latent, w_latent))
        
        # 2. BatchNorm
        mean_reshaped = ops.reshape(bn_mean, (1, packed_channels, 1, 1))
        var_reshaped = ops.reshape(bn_var, (1, packed_channels, 1, 1))
        
        eps = ops.constant(bn_eps, dtype=dtype, device=device_ref)
        std = ops.sqrt(ops.add(var_reshaped, eps))
        
        latents = ops.add(ops.mul(latents, std), mean_reshaped)
        
        # 3. Unpatchify (B, C, H, W) -> (B, C//4, H*2, W*2)
        # 128 -> 32, 64 -> 128
        latents = ops.reshape(latents, (batch_size, packed_channels // 4, 2, 2, h_latent, w_latent))
        latents = ops.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = ops.reshape(latents, (batch_size, packed_channels // 4, h_latent * 2, w_latent * 2))
        
        graph.output(latents)

    session = InferenceSession([accelerator])
    model = session.load(graph)
    
    # 2. Prepare Inputs
    latents_np = np.random.randn(batch_size, seq_len, packed_channels).astype(np.float32)
    bn_mean_np = np.random.randn(packed_channels).astype(np.float32)
    bn_var_np = np.random.randn(packed_channels).astype(np.float32)
    
    latents_t = Tensor.from_dlpack(latents_np).to(device).cast(dtype)
    bn_mean_t = Tensor.from_dlpack(bn_mean_np).to(device).cast(dtype)
    bn_var_t = Tensor.from_dlpack(bn_var_np).to(device).cast(dtype)
    
    latents_drv = latents_t.driver_tensor
    bn_mean_drv = bn_mean_t.driver_tensor
    bn_var_drv = bn_var_t.driver_tensor
    
    # 3. Warmup
    print("Warmup...")
    for _ in range(warmup):
        _ = model.execute(latents_drv, bn_mean_drv, bn_var_drv)
    
    # 4. Benchmark Execution
    print(f"Running {iterations} iterations...")
    acc_time = 0.0
    latencies = []
    
    for i in range(iterations):
        # Sync before
        torch.cuda.synchronize() # Use torch to sync (hacky but works if sharing device)
        # Or better: ensure previous result is done. 
        # But for pure dispatch latency, we just time the call + optional sync
        
        t0 = time.perf_counter()
        
        res = model.execute(latents_drv, bn_mean_drv, bn_var_drv)[0]
        
        # We want to measure completion time
        # Force sync by copying to CPU
        _ = Tensor.from_dlpack(res).to(CPU())
        
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000
        latencies.append(lat)
        
    avg_latency = np.mean(latencies)
    print(f"Avg Prep Graph Latency: {avg_latency:.4f} ms")
    print(f"Median Prep Graph Latency: {np.median(latencies):.4f} ms")
    print(f"Min Prep Graph Latency: {np.min(latencies):.4f} ms")
    print(f"Max Prep Graph Latency: {np.max(latencies):.4f} ms")

if __name__ == "__main__":
    benchmark_vae_prep()
