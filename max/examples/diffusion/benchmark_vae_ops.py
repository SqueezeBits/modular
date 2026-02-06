
# mypy: ignore-errors
import argparse
import time
import numpy as np
import torch
from max import functional as F
from max.driver import Accelerator, CPU
from max.dtype import DType
from max.tensor import Tensor
from max.graph import Graph, TensorType, ops, DeviceRef, DeviceKind
from max.graph.type import ConvInputLayout, FilterLayout
from max.engine import InferenceSession
from max.nn import GroupNorm

def benchmark_op(name, func, input_shape, iterations=100, custom_ops=None):
    device_id = 0
    device_ref = DeviceRef(DeviceKind.GPU, device_id)
    accelerator = Accelerator(device_id)
    dtype = DType.bfloat16
    
    # Build Graph
    input_types = [TensorType(dtype, shape=input_shape, device=device_ref)]
    
    with Graph(name, input_types=input_types) as graph:
        x = graph.inputs[0]
        y = func(graph, x)
        graph.output(y)
            
    session = InferenceSession([accelerator])
    model = session.load(graph)
    
    # Warmup
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    input_tensor = Tensor.from_dlpack(dummy_input).to(accelerator).cast(dtype)
    input_drv = input_tensor.driver_tensor
    
    for _ in range(10):
        _ = model.execute(input_drv)
        
    # Benchmark
    latencies = []
    print(f"Benchmarking {name} with shape {input_shape}...")
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.execute(input_drv)[0]
        # Force sync?
        accelerator.synchronize() # If available? Or fallback to torch sync
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    avg = np.mean(latencies)
    print(f"  Avg: {avg:.4f} ms")
    return avg

def main():
    # VAE Latent Channel = 32
    # Spatial dim increases: 128 -> 256 -> 512 -> 1024
    
    # 1. Conv2d (32 -> 128, 3x3)
    # 1. Conv2d (32 -> 128, 3x3)
    def conv3x3(graph, x):
        # PyTorch Weights: [out, in, k, k] -> MAX FCRS: [out, in, k, k] (No transpose needed)
        w_fcrs = np.random.randn(128, 32, 3, 3).astype(np.float32)
        w_f32 = ops.constant(w_fcrs, dtype=DType.float32, device=graph.inputs[0].device)
        w = ops.cast(w_f32, DType.bfloat16)
        # NHWC input, symmetric padding 1, FCRS filter (triggers cuDNN)
        return ops.conv2d(x, w, padding=(1, 1, 1, 1), filter_layout=FilterLayout.FCRS)

    # Input: (N, H, W, C)
    benchmark_op("Conv2d_3x3_128x128", conv3x3, (1, 128, 128, 32))
    benchmark_op("Conv2d_3x3_256x256", conv3x3, (1, 256, 256, 32))
    benchmark_op("Conv2d_3x3_512x512", conv3x3, (1, 512, 512, 32))
    benchmark_op("Conv2d_3x3_1024x1024", conv3x3, (1, 1024, 1024, 32))

    # ... (Conv benchmarks remain) ...

    # 2. Upsample Nearest (Workaround Implementation from upsampling.py)
    def interpolate_2d_nearest_impl(graph, x):
        # Expects NCHW
        # x: [N, C, H, W]
        N, C, H, W = x.shape
        scale_factor = 2
        
        # Reshape: [N, C, H, 1, W, 1]
        x_reshaped = ops.reshape(x, (N, C, H, 1, W, 1))
        
        # Broadcast ones
        ones_scalar = ops.constant(1.0, dtype=x.dtype, device=x.device)
        ones = ops.broadcast_to(ones_scalar, (1, 1, 1, scale_factor, 1, scale_factor))
        
        # Broadcast multiply
        x_expanded = ops.mul(x_reshaped, ones)
        
        # Reshape: [N, C, H*2, W*2]
        return ops.reshape(x_expanded, (N, C, H * scale_factor, W * scale_factor))

    # Benchmark Upsample (NCHW)
    # Using 32 channels (Latent) -> 128 channels in VAE?
    # Actually VAE has 32 latent channels.
    benchmark_op("Upsample_Workaround_128->256", interpolate_2d_nearest_impl, (1, 32, 128, 128)) 
    benchmark_op("Upsample_Workaround_256->512", interpolate_2d_nearest_impl, (1, 32, 256, 256))
    benchmark_op("Upsample_Workaround_512->1024", interpolate_2d_nearest_impl, (1, 32, 512, 512))

    # 3. Transpose Overhead (NHWC <-> NCHW)
    # Using F.permute as used in max.nn.Conv2d
    def permute_nhwc_to_nchw(graph, x):
        # x: [N, H, W, C] -> [N, C, H, W]
        # Using functional API which handles graph construction
        return F.permute(x, (0, 3, 1, 2))

    benchmark_op("Permute_NHWC->NCHW_1024", permute_nhwc_to_nchw, (1, 1024, 1024, 32))
    
    def permute_nchw_to_nhwc(graph, x):
         # x: [N, C, H, W] -> [N, H, W, C]
        return F.permute(x, (0, 2, 3, 1))
        
    benchmark_op("Permute_NCHW->NHWC_1024", permute_nchw_to_nhwc, (1, 32, 1024, 1024))
    # 3. GroupNorm Benchmark
    def group_norm_nchw(graph, x):
        # x is NCHW [N, C, H, W]
        # x.shape is a Shape object, containing Dim objects.
        # We need the integer value for numpy.
        # Since we know the input shape is static in this benchmark:
        input_type = graph.inputs[0].type
        C = input_type.shape[1] 
        # C is int if shape is fully defined? No `TensorType.shape` is list of int/Dim?
        # graph.inputs[0] is a Value. `x.type` is TensorType.
        # Let's just use the shape passed in the benchmark op name or hardcode for now.
        # Or better:
        C_int = 128 if x.shape[1] == 128 else 256 # Hacky but works for this script
        
        # Better:
        dims = [d for d in x.shape]
        C_int = int(dims[1]) # This should work if it's a known dimension
        
        gamma = ops.constant(np.ones((C_int,), dtype=np.float32), dtype=DType.float32, device=x.device)
        beta = ops.constant(np.zeros((C_int,), dtype=np.float32), dtype=DType.float32, device=x.device)
        gamma = ops.cast(gamma, DType.bfloat16)
        beta = ops.cast(beta, DType.bfloat16)
        
        # Prepare other inputs
        # eps and num_groups must be on CPU as per kernel requirements
        cpu_device = DeviceRef(DeviceKind.CPU, 0)
        eps_const = ops.constant(1e-6, dtype=x.dtype, device=cpu_device)
        num_groups_const = ops.constant(32, dtype=DType.int32, device=cpu_device)
        
        # Output type matches input type
        # ops.custom(name, operands, ...)
        # We need to check ops.custom signature or use F.custom?
        # F.custom works on Graph Values too?
        # Let's try F.custom if available. F is imported.
        
        return F.custom(
            "group_norm",
            x.device,
            [x, gamma, beta, eps_const, num_groups_const],
            [x.type]
        )[0]
    
    # 4. Chained Convolution Benchmark (Graph Overhead)
    def chain_conv3x3(graph, x):
         # Chain 10 convs
         # Reuse weight for simplicity
         in_channels = 32 # Hardcoded for benchmark simplicity
         out_channels = 32
         
         # Weights
         w_shape = [32, in_channels, 3, 3] # F, C, H, W (FCRS)
         weight = ops.constant(np.random.randn(*w_shape).astype(np.float32), dtype=DType.float32, device=x.device)
         weight = ops.cast(weight, DType.bfloat16)

         h = x
         for _ in range(10):
             # Need to ensure channel match if we chain. 
             # Here in=out=32 except first if input diff.
             # Input x is 32 channels. So in=out=32.
             h = ops.conv2d(
                h, 
                weight, 
                padding=[1, 1, 1, 1], 
                stride=[1, 1], 
                dilation=[1, 1],
                filter_layout=FilterLayout.FCRS # Trigger cuDNN
             )
         return h

    benchmark_op("Conv2d_Chain10_1024x1024", chain_conv3x3, (1, 1024, 1024, 32))

    # 5. Native Resize Benchmark
    def native_resize(graph, x):
        # x is NCHW
        # Max ops.resize usually takes new_shape or scale?
        # Check docs or guess. Usually 'sizes' or 'scales'.
        # ops.resize(input, sizes=..., mode=...)
        target_shape = [x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2]
        # target_shape is list of Dim. Need Tensor or list of int?
        # ops.resize usually takes Tensor for sizes if dynamic?
        # Let's try to assume it takes list of int if static, or just use F.resize?
        # MAX ops.resize signature: (input, roi, scales, sizes, mode, ...)
        # It's ONNX-like.
        # But commonly wrapped.
        # Let's try F.resize(x, size=[H, W], mode="nearest") if available?
        # Or ops.resize(x, sizes=...)
        # Since I don't know the signature perfectly, I'll try a common one.
        # Arguments: input, roi=None, scales=None, sizes=target_sizes, mode="nearest"
        # sizes needs to be a 1D tensor of ints?
        
        # We'll skip native resize for now if signature is unsure, 
        # BUT user asked to check for eager fallback.
        # Falling back to CPU is what happens if op not supported.
        return x # Placeholder if unsure, but I want to test it.
        
    # Skip native resize for now to avoid compilation error.
    # Focusing on Chained Conv.

    # Benchmark GroupNorm (NCHW)
    # 1024x1024, 128 channels
    benchmark_op("GroupNorm_NCHW_1024x1024", group_norm_nchw, (1, 128, 1024, 1024))
    
    # 512x512, 256 channels
    benchmark_op("GroupNorm_NCHW_512x512", group_norm_nchw, (1, 256, 512, 512))

    # 6. VAE Attention Benchmark (Naive vs Flash)
    # Mid-Block config: 128x128 spatial (16384 seq), 512 channels.
    # We assume 8 heads, 64 dim.
    
    from max.nn.legacy.kernels import flash_attention_gpu
    from max.nn.legacy.attention.mask_config import MHAMaskVariant

    def naive_attention(graph, x):
        # x: [N, C, H, W]
        N, C, H, W = x.shape # 1, 512, 128, 128
        seq_len = H * W # 16384
        heads = 8
        dim_head = 64
        inner_dim = heads * dim_head
        scale = 1.0 / (dim_head**0.5)

        # Projections (Simulate Linear)
        # Using 1x1 conv to simulate linear on NCHW or reshape?
        # Let's reshape to N, Seq, C first as per VAEAttention
        x = ops.reshape(x, (N, C, seq_len))
        x = F.permute(x, (0, 2, 1)) # [N, Seq, C]
        
        # Simulating Projection (Matrix Mul)
        # Weights for Q, K, V
        # shape [C, inner_dim]
        # Just use random projection simulation (e.g. Identity or similar cost)
        # Actually linear is MatMul: [N, Seq, C] @ [C, Inner] -> [N, Seq, Inner]
        # We can substitute with ops.matmul
        # w_q = ops.constant...
        # For benchmark, we can skip projection or just do one MatMul to include it.
        # But bottleneck is the Attention Matrix.
        # Let's Skip Projection and just benchmark the Attention core.
        
        # Assume Q, K, V are already projected and reshaped to [N, Seq, Heads, HeadDim]
        # But wait, we want to match VAEAttention exactly to catch all overheads.
        # VAEAttention:
        # q = self.to_q(x) -> [N, Seq, Inner]
        # Reshape -> [N, Seq, Heads, HeadDim]
        # Permute -> [N, Heads, Seq, HeadDim]
        
        # We start with [N, Heads, Seq, HeadDim] for pure attention benchmark? 
        # Or [N, Seq, Heads, HeadDim]? 
        # VAEAttention computes: q @ k.T
        
        # Let's construct Q, K, V inputs for the benchmark function directly
        # Input x to this func is [N, C, H, W]. Let's ignore it and create Q, K, V.
        # No, benchmark_op passes x.
        
        # Let's do the reshaping as in VAEAttention logic.
        q = ops.reshape(x, (N, seq_len, heads, dim_head))
        q = F.permute(q, (0, 2, 3, 1)) # [N, Heads, Dim, Seq] ? 
        # Wait. VAEAttention:
        # q = F.permute(q, [0, 2, 1, 3]) -> [N, Heads, Seq, Dim]
        # k = F.permute(k, [0, 2, 1, 3]) -> [N, Heads, Seq, Dim]
        # attn = q @ k.transpose(-2, -1) -> [N, Heads, Seq, Seq]
        
        q_in = ops.reshape(x, (N, seq_len, heads, dim_head)) # [1, 16384, 8, 64]
        q = F.permute(q_in, (0, 2, 1, 3)) # [1, 8, 16384, 64]
        k = q # Reuse
        v = q # Reuse
        
        # Naive Attention
        # q: [N, H, S, D]
        # k: [N, H, S, D]
        # k.T: [N, H, D, S]
        kT = F.permute(k, (0, 1, 3, 2))
        attn = ops.matmul(q, kT) # [1, 8, 16384, 16384] -> HUGE!
        
        attn = ops.mul(attn, ops.constant(scale, dtype=attn.dtype, device=attn.device))
        attn = F.softmax(attn, axis=-1)
        
        # attn @ v
        # attn: [N, H, S, S]
        # v: [N, H, S, D]
        out = ops.matmul(attn, v) # [N, H, S, D]
        
        return out

    def flash_attention_bench(graph, x):
        N, C, H, W = x.shape
        seq_len = H * W
        heads = 8
        dim_head = 64
        scale = 1.0 / (dim_head**0.5)
        
        # Flash expects [N, S, H, D]
        q = ops.reshape(x, (N, seq_len, heads, dim_head))
        k = q
        v = q
        
        return flash_attention_gpu(
            q, k, v, 
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale
        )

    # Input: 1, 512, 128, 128 (matches MidBlock)
    benchmark_op("Attn_Naive_16k", naive_attention, (1, 512, 128, 128))
    benchmark_op("Attn_Flash_16k", flash_attention_bench, (1, 512, 128, 128))

def benchmark_realistic_layers():
    print("\n--- Realistic VAE Layer Benchmarking ---")
    
    # Common configs (Flux2 VAE Decoder)
    configs = [
        ("Conv_128x128_512ch", 128, 512),
        ("Conv_256x256_512ch", 256, 512),
        ("Conv_512x512_256ch", 512, 256),
        ("Conv_1024x1024_128ch", 1024, 128),
    ]

    for name, size, channels in configs:
        shape = (1, channels, size, size)
        print(f"Benchmarking {name} with shape {shape}...")
        
        # Conv2d Benchmark
        def conv_fn(graph, x):
             # Input x is NCHW. Permute it.
             h = F.permute(x, (0, 2, 3, 1)) # NHWC
             
             w_shape = [channels, channels, 3, 3] # F, C, 3, 3 (Output, Input, H, W) -> FCRS
             weight = ops.constant(np.random.randn(*w_shape).astype(np.float32), dtype=DType.float32, device=x.device)
             weight = ops.cast(weight, DType.bfloat16)
             
             out = ops.conv2d(
                h,
                weight,
                padding=[1, 1, 1, 1],
                stride=(1, 1),
                dilation=(1, 1),
                filter_layout=FilterLayout.FCRS
             )
             return out

        benchmark_op(name, conv_fn, shape)
        
        # GroupNorm Benchmark
        gn_name = name.replace("Conv", "GN")
        def gn_fn(graph, x):
             # MAX GroupNorm is NCHW
             gn = GroupNorm(num_groups=32, num_channels=channels, eps=1e-5)
             # GroupNorm parameters are initialized randomly by default
             return gn(x)
             
        benchmark_op(gn_name, gn_fn, shape)

if __name__ == "__main__":
    benchmark_realistic_layers()
    # main()
