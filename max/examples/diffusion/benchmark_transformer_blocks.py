
import argparse
import time
import numpy as np

from max.driver import load_devices, DeviceSpec, Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.engine import InferenceSession, Model
from max.tensor import Tensor
from max import functional as F
from max.nn.legacy.kernels import flash_attention_gpu
from max.nn.legacy.attention.mask_config import MHAMaskVariant

# Helper for manual Linear as input
def manual_linear_input(x, w, name_suffix=""):
    return ops.matmul(x, w)

# Helper for SwiGLU
def manual_swiglu(x):
    x1, x2 = F.chunk(x, chunks=2, axis=-1)
    return F.silu(x1) * x2

def benchmark_component(
    name: str,
    create_graph_fn,
    input_shapes: dict[str, list[int]],
    weight_shapes: dict[str, list[int]],
    device: Device,
    dtype: DType,
    num_warmup: int = 5,
    num_iter: int = 20,
):
    print(f"\nBenchmarking {name}...")
    
    # Create inputs
    inputs = {}
    # Data inputs
    for key, shape in input_shapes.items():
        arr = np.random.randn(*shape).astype(np.float32)
        t = Tensor.from_dlpack(arr).to(DeviceRef.from_device(device)).cast(dtype)
        inputs[key] = t.driver_tensor

    # Weight inputs
    for key, shape in weight_shapes.items():
        arr = np.random.randn(*shape).astype(np.float32) * 0.02
        t = Tensor.from_dlpack(arr).to(DeviceRef.from_device(device)).cast(dtype)
        inputs[key] = t.driver_tensor

    # Compile model
    session = InferenceSession([device])
    try:
        model = create_graph_fn(session, inputs, dtype)
    except Exception as e:
        print(f"Failed to compile {name}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Verify run
    print("  Running warmup...")
    # Prepare input list strictly matching graph inputs order
    # We need to inspect graph input names or ensure create_graph_fn uses inputs dict order
    # Max Graph inputs iteration relies on order of creation or input_types list
    # The 'inputs' dict is passed to create_graph_fn which constructs TensorType list
    # We should reconstruct the list of driver tensors in correct order
    
    # Let's inspect model.input_names if available, or assume create_graph_fn returns (model, OrderedInputKeys)
    # But current API session.load(graph) returns Model.
    # We can assume input_types order matches keys in input_shapes then weight_shapes.
    
    ordered_keys = list(input_shapes.keys()) + list(weight_shapes.keys())
    input_list = [inputs[k] for k in ordered_keys]
    
    for _ in range(num_warmup):
        outputs = model.execute(*input_list)
        for o in outputs:
            o.to_numpy() # Sync

    # Benchmark
    print(f"  Running {num_iter} iterations...")
    latencies = []
    for _ in range(num_iter):
        t0 = time.time()
        outputs = model.execute(*input_list)
        for o in outputs:
            o.to_numpy() # Sync
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

    avg_lat = np.mean(latencies)
    print(f"  Result: {avg_lat:.2f} ms")
    return avg_lat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len-img", type=int, default=4096)
    parser.add_argument("--seq-len-txt", type=int, default=512)
    parser.add_argument("--component", type=str, default="all", choices=["all", "mlp", "attn", "parallel"])
    args = parser.parse_args()
    
    print("Initializing device...")
    devices = load_devices([DeviceSpec.accelerator()])
    device = devices[0]
    dtype = DType.bfloat16
    
    # Config (Confirmed 6144)
    DIM = 6144
    HEADS = 48
    DIM_HEAD = 128
    
    print(f"Config: Dim={DIM}, Heads={HEADS}")
    
    # 1. Flux2FeedForward (MLP)
    def create_mlp(session, inputs, dtype):
        print("Creating MLP graph (WEIGHTS AS INPUTS)...")
        # Define shapes
        inner_dim = int(DIM * 3.0)
        
        # Keys matching 'inputs' dict
        data_keys = ["x"]
        weight_keys = ["w_in", "w_out"]
        
        input_types = []
        for k in data_keys:
            input_types.append(TensorType(dtype, shape=inputs[k].shape, device=inputs[k].device))
        for k in weight_keys:
            input_types.append(TensorType(dtype, shape=inputs[k].shape, device=inputs[k].device))
            
        with Graph("mlp", input_types=input_types) as graph:
            x = graph.inputs[0]
            w_in = graph.inputs[1]
            w_out = graph.inputs[2]
            
            # Linear In
            h = manual_linear_input(x, w_in)
            
            # SwiGLU
            h = manual_swiglu(h)
            
            # Linear Out
            out = manual_linear_input(h, w_out)
            
            graph.output(out.cast(DType.float32))
        
        print("Loading MLP session...")
        return session.load(graph)

    # 2. Flux2Attention (Dual Stream)
    def create_attn(session, inputs, dtype):
        print("Creating Attention graph (WEIGHTS AS INPUTS)...")
        data_keys = ["hidden_states", "encoder_hidden_states"]
        weight_keys = ["w_q", "w_k", "w_v", "w_add_q", "w_add_k", "w_add_v", "w_out", "w_enc_out"]
        
        input_types = []
        for k in data_keys:
             input_types.append(TensorType(dtype, shape=inputs[k].shape, device=inputs[k].device))
        for k in weight_keys:
             input_types.append(TensorType(dtype, shape=inputs[k].shape, device=inputs[k].device))

        with Graph("attn", input_types=input_types) as graph:
            x = graph.inputs[0]
            enc = graph.inputs[1]
            
            # Weights mapping
            idx = 2
            w_q = graph.inputs[idx]; idx+=1
            w_k = graph.inputs[idx]; idx+=1
            w_v = graph.inputs[idx]; idx+=1
            w_add_q = graph.inputs[idx]; idx+=1
            w_add_k = graph.inputs[idx]; idx+=1
            w_add_v = graph.inputs[idx]; idx+=1
            w_out = graph.inputs[idx]; idx+=1
            w_enc_out = graph.inputs[idx]; idx+=1
            
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            enc_seq = enc.shape[1]
            inner_dim = HEADS * DIM_HEAD
            
            # QKV
            q = manual_linear_input(x, w_q)
            k = manual_linear_input(x, w_k)
            v = manual_linear_input(x, w_v)
            
            added_q = manual_linear_input(enc, w_add_q)
            added_k = manual_linear_input(enc, w_add_k)
            added_v = manual_linear_input(enc, w_add_v)
            
            # Reshape
            q = F.reshape(q, [batch_size, seq_len, HEADS, DIM_HEAD])
            k = F.reshape(k, [batch_size, seq_len, HEADS, DIM_HEAD])
            v = F.reshape(v, [batch_size, seq_len, HEADS, DIM_HEAD])
            
            added_q = F.reshape(added_q, [batch_size, enc_seq, HEADS, DIM_HEAD])
            added_k = F.reshape(added_k, [batch_size, enc_seq, HEADS, DIM_HEAD])
            added_v = F.reshape(added_v, [batch_size, enc_seq, HEADS, DIM_HEAD])
            
            # Concat
            q = F.concat([added_q, q], axis=1)
            k = F.concat([added_k, k], axis=1)
            v = F.concat([added_v, v], axis=1)
            
            # Flash Attention
            scale = 1.0 / (DIM_HEAD**0.5)
            # MaskVariant.NULL_MASK (no mask)
            hidden_states = flash_attention_gpu(
                q, k, v, 
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=scale
            )
            
            total_seq = hidden_states.shape[1]
            hidden_states = F.reshape(hidden_states, [batch_size, total_seq, inner_dim])
            hidden_states = hidden_states.cast(dtype)
            
            encoder_out = hidden_states[:, :enc_seq, :]
            hidden_out = hidden_states[:, enc_seq:, :]
            
            # Out Proj
            hidden_out = manual_linear_input(hidden_out, w_out)
            encoder_out = manual_linear_input(encoder_out, w_enc_out)
            
            graph.output(hidden_out.cast(DType.float32), encoder_out.cast(DType.float32))
            
        return session.load(graph)

    # 3. Flux2ParallelSelfAttention (Single Stream)
    def create_parallel_attn(session, inputs, dtype):
        print("Creating Parallel Attention graph (WEIGHTS AS INPUTS)...")
        data_keys = ["hidden_states"]
        weight_keys = ["w_fused", "w_out"]
        
        input_types = []
        for k in data_keys:
             input_types.append(TensorType(dtype, shape=inputs[k].shape, device=inputs[k].device))
        for k in weight_keys:
             input_types.append(TensorType(dtype, shape=inputs[k].shape, device=inputs[k].device))
        
        with Graph("parallel_attn", input_types=input_types) as graph:
            x = graph.inputs[0]
            w_fused = graph.inputs[1]
            w_out = graph.inputs[2]
            
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            
            inner_dim = HEADS * DIM_HEAD
            mlp_hidden_dim = int(DIM * 4.0)
            
            fused = manual_linear_input(x, w_fused)
            
            # Split
            qkv_dim = inner_dim * 3
            qkv = fused[:, :, :qkv_dim]
            mlp_states = fused[:, :, qkv_dim:]
            
            q, k, v = F.chunk(qkv, 3, axis=-1)
            
            q = F.reshape(q, [batch_size, seq_len, HEADS, DIM_HEAD])
            k = F.reshape(k, [batch_size, seq_len, HEADS, DIM_HEAD])
            v = F.reshape(v, [batch_size, seq_len, HEADS, DIM_HEAD])
            
            scale = 1.0 / (DIM_HEAD**0.5)
            attn_out = flash_attention_gpu(
                q, k, v, 
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=scale
            )
            attn_out = F.reshape(attn_out, [batch_size, seq_len, inner_dim])
            attn_out = attn_out.cast(dtype)
            
            mlp_out = manual_swiglu(mlp_states)
            
            combined = F.concat([attn_out, mlp_out], axis=-1)
            
            out = manual_linear_input(combined, w_out)
            
            graph.output(out.cast(DType.float32))
        
        return session.load(graph)

    # Run benchmarks
    
    if args.component in ["all", "mlp"]:
        inner_dim = int(DIM * 3.0)
        benchmark_component(
            "Flux2FeedForward (Dual Stream MLP)",
            create_mlp,
            {"x": [1, args.seq_len_img, DIM]},
            {
                "w_in": [DIM, inner_dim * 2],
                "w_out": [inner_dim, DIM]
            },
            device,
            dtype
        )
    
    if args.component in ["all", "attn"]:
        inner_dim = HEADS * DIM_HEAD
        benchmark_component(
            "Flux2Attention (Dual Stream Attention)",
            create_attn,
            {
                "hidden_states": [1, args.seq_len_img, DIM],
                "encoder_hidden_states": [1, args.seq_len_txt, DIM]
            },
            {
                "w_q": [DIM, inner_dim], "w_k": [DIM, inner_dim], "w_v": [DIM, inner_dim],
                "w_add_q": [DIM, inner_dim], "w_add_k": [DIM, inner_dim], "w_add_v": [DIM, inner_dim],
                "w_out": [inner_dim, DIM], "w_enc_out": [inner_dim, DIM]
            },
            device,
            dtype
        )
    
    if args.component in ["all", "parallel"]:
        total_seq_len = args.seq_len_img + args.seq_len_txt
        inner_dim = HEADS * DIM_HEAD
        mlp_hidden_dim = int(DIM * 4.0)
        fused_dim = inner_dim * 3 + mlp_hidden_dim * 2
        benchmark_component(
            "Flux2ParallelSelfAttention (Single Stream Fused)",
            create_parallel_attn,
            {"hidden_states": [1, total_seq_len, DIM]},
            {
                "w_fused": [DIM, fused_dim],
                "w_out": [inner_dim + mlp_hidden_dim, DIM]
            },
            device,
            dtype
        )

if __name__ == "__main__":
    main()
