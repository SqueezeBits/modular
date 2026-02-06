#!/usr/bin/env python3
"""Benchmark GroupNorm fusion patterns found in Flux2 VAE."""

import argparse
import time

import numpy as np
from max.tensor import Tensor
from max.driver import load_devices, DeviceSpec, Device
from max.graph import Graph, TensorType, DeviceRef, ops
from max.engine import InferenceSession
from max import functional as F
from max.dtype import DType

from max.driver import CPU

def manual_conv2d(x, weight, bias=None, stride=1, padding=1):
    # Input x is NCHW. ops.conv2d requires NHWC.
    x_nhwc = F.permute(x, (0, 2, 3, 1))
    
    # Weight is [Co, Ci, H, W]. ops.conv2d requires RSCF [H, W, Ci, Co].
    w_rscf = F.permute(weight, (2, 3, 1, 0))
    
    out = ops.conv2d(
        x_nhwc, w_rscf, 
        bias=bias,
        stride=(stride, stride), 
        padding=(padding, padding, padding, padding), 
        dilation=(1, 1), 
        groups=1
    )
    
    # Output is NHWC. Convert back to NCHW.
    return F.permute(out, (0, 3, 1, 2))

def manual_group_norm(x, num_groups, weight, bias, eps=1e-6):
    # Implementation copied from max.nn.norm.group_norm.group_norm
    # Because F.group_norm does not exist directly.
    return F.custom(
        "group_norm",
        x.device,
        [
            x,
            weight.to(x.device),
            bias.to(x.device),
            F.constant(eps, dtype=x.dtype, device=CPU()),
            F.constant(num_groups, dtype=DType.int32, device=CPU())
        ],
        [x.type]
    )[0]

def benchmark_pattern(
    name: str,
    create_graph_fn,
    input_shapes: dict,
    device: Device,
    dtype: DType,
    num_warmup: int = 10,
    num_iter: int = 50,
):
    print(f"\nBenchmarking {name}...")
    
    inputs = {}
    for k, shape in input_shapes.items():
        arr = np.random.randn(*shape).astype(np.float32)
        t = Tensor.from_dlpack(arr).to(DeviceRef.from_device(device)).cast(dtype)
        inputs[k] = t.driver_tensor

    session = InferenceSession([device])
    try:
        model = create_graph_fn(session, inputs, dtype)
    except Exception as e:
        print(f"Failed to compile {name}: {e}")
        import traceback
        traceback.print_exc()
        return

    print("  Running warmup...")
    input_list = [inputs[k] for k in input_shapes.keys()]
    
    for _ in range(num_warmup):
        outputs = model.execute(*input_list)
        for o in outputs:
            o.to_numpy()

    print(f"  Running {num_iter} iterations...")
    latencies = []
    for _ in range(num_iter):
        t0 = time.time()
        outputs = model.execute(*input_list)
        for o in outputs:
            o.to_numpy()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

    avg_lat = np.mean(latencies)
    print(f"  Result: {avg_lat:.3f} ms")
    return avg_lat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    
    print("Initializing device...")
    devices = load_devices([DeviceSpec.accelerator()])
    device = devices[0]
    dtype = DType.float32 # VAE usually runs in float32? Or float16? 
                          # Flux VAE is float32/bfloat16. Let's use float32 for safety first, or bfloat16 if Flux uses it.
                          # Profiler trace showed GEMM kernels, likely bf16 or fp16. 
                          # Let's use float32 for now to avoid cast issues, or align with Flux.
    dtype = DType.bfloat16 # Using bfloat16 to match Flux Transformers
    
    C = args.channels
    H = args.height
    W = args.width
    G = args.groups
    B = 1
    
    print(f"Config: Shape=[{B}, {C}, {H}, {W}], Groups={G}, DType={dtype}")

    # 1. ResNet Pattern: GroupNorm -> SiLU -> Conv2d
    def create_resnet(session, inputs, dtype):
        input_types = [
            TensorType(dtype, shape=inputs["x"].shape, device=inputs["x"].device),
            TensorType(dtype, shape=inputs["gn_scale"].shape, device=inputs["x"].device),
            TensorType(dtype, shape=inputs["gn_bias"].shape, device=inputs["x"].device),
            TensorType(dtype, shape=inputs["conv_w"].shape, device=inputs["x"].device),
            TensorType(dtype, shape=inputs["conv_b"].shape, device=inputs["x"].device),
        ]
        
        with Graph("resnet_pattern", input_types=input_types) as graph:
            x = graph.inputs[0]
            gn_scale = graph.inputs[1]
            gn_bias = graph.inputs[2]
            conv_w = graph.inputs[3]
            conv_b = graph.inputs[4]
            
            # GroupNorm
            # manual_group_norm(input, num_groups, weight, bias, eps=1e-05)
            h = manual_group_norm(x, G, gn_scale, gn_bias, 1e-6)
            
            # SiLU
            h = F.silu(h)
            
            # Conv2d
            # Standard ResNet 3x3 Conv, padding=1
            h = manual_conv2d(h, conv_w, conv_b, stride=1, padding=1)
            
            graph.output(h.cast(DType.float32))
            
        return session.load(graph)

    # 2. Attention Pattern: GroupNorm -> (Reshape/Permute) -> Linear
    def create_attn(session, inputs, dtype):
        input_types = [
            TensorType(dtype, shape=inputs["x"].shape, device=inputs["x"].device),
            TensorType(dtype, shape=inputs["gn_scale"].shape, device=inputs["x"].device),
            TensorType(dtype, shape=inputs["gn_bias"].shape, device=inputs["x"].device),
            TensorType(dtype, shape=inputs["lin_w"].shape, device=inputs["x"].device),
            TensorType(dtype, shape=inputs["lin_b"].shape, device=inputs["x"].device),
        ]
        
        with Graph("attn_pattern", input_types=input_types) as graph:
            x = graph.inputs[0]
            gn_scale = graph.inputs[1]
            gn_bias = graph.inputs[2]
            lin_w = graph.inputs[3]
            lin_b = graph.inputs[4]
            
            # GroupNorm
            h = manual_group_norm(x, G, gn_scale, gn_bias, 1e-6)
            
            # Reshape/Permute (to N, L, C)
            # N, C, H, W -> N, C, H*W -> N, H*W, C
            N_dim = h.shape[0]
            C_dim = h.shape[1]
            HW_dim = h.shape[2] * h.shape[3]
            
            h = F.reshape(h, [N_dim, C_dim, HW_dim])
            h = F.permute(h, [0, 2, 1])
            
            # Linear (GEMM)
            h = h @ lin_w.T + lin_b
            
            graph.output(h.cast(DType.float32))
            
        return session.load(graph)



    # Run ResNet Benchmark
    benchmark_pattern(
        "ResNet Pattern (GroupNorm -> SiLU -> Conv)",
        create_resnet,
        {
            "x": [B, C, H, W],
            "gn_scale": [C],
            "gn_bias": [C],
            "conv_w": [C, C, 3, 3], # Out, In, K, K
            "conv_b": [C]
        },
        device,
        dtype
    )

    # Run Attention Benchmark
    benchmark_pattern(
        "Attention Pattern (GroupNorm -> Reshape -> Linear)",
        create_attn,
        {
            "x": [B, C, H, W],
            "gn_scale": [C],
            "gn_bias": [C],
            "lin_w": [C, C], # Out, In
            "lin_b": [C]
        },
        device,
        dtype
    )

if __name__ == "__main__":
    main()
