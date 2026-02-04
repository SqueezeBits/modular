
import time
import numpy as np
from max.driver import Accelerator, Buffer as DriverTensor
from max.dtype import DType
from max.tensor import Tensor
from max.graph import DeviceRef, DeviceKind, Graph, TensorType, ops
from max.engine import InferenceSession


def benchmark_scalar_upload(device_id=0, iterations=50, num_steps=50):
    """Benchmark scalar upload approaches for scheduler timestep/dt computation.
    
    Compares:
    1. Baseline: Python-side t/dt compute + Tensor.cast (slow)
    2. Optimized: Python-side t/dt compute + Buffer bit manipulation (faster)
    3. Graph: Pre-upload sigmas, compute t/dt on GPU via compiled graph (fastest)
    """
    device = Accelerator(device_id)
    device_ref = DeviceRef(DeviceKind.GPU, device_id)
    print(f"Benchmarking on device: {device}")
    print(f"Iterations: {iterations}, Simulated steps: {num_steps}")
    
    # Simulate scheduler sigmas (like FlowMatch Euler)
    sigmas_np = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)
    timesteps_np = (sigmas_np[:-1] * 1000.0).astype(np.float32)  # [num_steps]
    
    # ==========================================================================
    # 1. Baseline: Python-side compute + Tensor.cast (original slow path)
    # ==========================================================================
    start_time = time.perf_counter()
    for step_idx in range(min(iterations, num_steps)):
        t = timesteps_np[step_idx]
        dt = sigmas_np[step_idx + 1] - sigmas_np[step_idx]
        
        t_tensor = (
            Tensor.from_dlpack(np.array(t / 1000.0, dtype=np.float32))
            .to(device_ref)
            .cast(DType.bfloat16)
        )
        dt_tensor = (
            Tensor.from_dlpack(np.array(dt, dtype=np.float32))
            .to(device_ref)
            .cast(DType.bfloat16)
        )
    end_time = time.perf_counter()
    baseline_ms = (end_time - start_time) / min(iterations, num_steps) * 1000
    print(f"\n1. Baseline (Tensor.cast): {baseline_ms:.4f} ms/step")

    # ==========================================================================
    # 2. Optimized: Python-side compute + Buffer bit manipulation
    # ==========================================================================
    start_time = time.perf_counter()
    for step_idx in range(min(iterations, num_steps)):
        t = timesteps_np[step_idx]
        dt = sigmas_np[step_idx + 1] - sigmas_np[step_idx]
        
        # Manual float32 -> bfloat16 truncation
        t_u16 = (np.array(t / 1000.0, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)[None]
        dt_u16 = np.array((np.array(dt, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16))
        
        timestep_drv = DriverTensor.from_dlpack(t_u16).to(device).view(DType.bfloat16)
        dt_drv = DriverTensor.from_dlpack(dt_u16).to(device).view(DType.bfloat16, shape=[])
    end_time = time.perf_counter()
    optimized_ms = (end_time - start_time) / min(iterations, num_steps) * 1000
    print(f"2. Optimized (Buffer+shift): {optimized_ms:.4f} ms/step")
    print(f"   Speedup vs Baseline: {baseline_ms / optimized_ms:.2f}x")

    # ==========================================================================
    # 3. Graph: Pre-upload sigmas, compute t/dt on GPU
    # ==========================================================================
    
    # Build the graph: input is step_index (int64), outputs are t (bf16) and dt (bf16)
    # Graph inputs are defined via input_types parameter
    def forward(step_idx_in, sigmas_in):
        # step_idx_in: [1] int64
        # sigmas_in: [num_steps + 1] float32
        
        # Gather sigma at step_idx and step_idx + 1
        sigma_i = ops.gather(sigmas_in, step_idx_in, axis=0)  # [1]
        step_idx_plus_1 = step_idx_in + ops.constant(np.array([1], dtype=np.int64), dtype=DType.int64, device=device_ref)
        sigma_i_plus_1 = ops.gather(sigmas_in, step_idx_plus_1, axis=0)  # [1]
        
        # t = sigma_i (already normalized 0-1 in sigmas)
        t_out = ops.cast(sigma_i, DType.bfloat16)
        
        # dt = sigma_i+1 - sigma_i
        dt_f32 = sigma_i_plus_1 - sigma_i
        dt_out = ops.cast(dt_f32, DType.bfloat16)
        
        # Squeeze dt to scalar
        dt_out_scalar = ops.squeeze(dt_out, axis=0)
        
        return t_out, dt_out_scalar
    
    graph = Graph(
        name="scheduler_scalar_graph",
        forward=forward,
        input_types=[
            TensorType(DType.int64, [1], device_ref),       # step_idx
            TensorType(DType.float32, [num_steps + 1], device_ref),  # sigmas
        ],
    )
    
    # Compile graph
    session = InferenceSession(devices=[device])
    model = session.load(graph, custom_extensions=[])
    
    # Pre-upload sigmas to GPU once
    sigmas_gpu = DriverTensor.from_dlpack(sigmas_np).to(device)
    
    # Warmup
    step_idx_buf = DriverTensor.from_dlpack(np.array([0], dtype=np.int64)).to(device)
    _ = model.execute(step_idx_buf, sigmas_gpu)
    
    # Benchmark
    start_time = time.perf_counter()
    for step_idx in range(min(iterations, num_steps)):
        step_idx_buf = DriverTensor.from_dlpack(np.array([step_idx], dtype=np.int64)).to(device)
        outputs = model.execute(step_idx_buf, sigmas_gpu)
        t_drv = outputs[0]
        dt_drv = outputs[1]
    end_time = time.perf_counter()
    graph_ms = (end_time - start_time) / min(iterations, num_steps) * 1000
    print(f"3. Graph (GPU compute): {graph_ms:.4f} ms/step")
    print(f"   Speedup vs Baseline: {baseline_ms / graph_ms:.2f}x")
    print(f"   Speedup vs Optimized: {optimized_ms / graph_ms:.2f}x")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n--- Summary ---")
    print(f"Baseline:  {baseline_ms:.4f} ms/step")
    print(f"Optimized: {optimized_ms:.4f} ms/step ({baseline_ms/optimized_ms:.1f}x faster)")
    print(f"Graph:     {graph_ms:.4f} ms/step ({baseline_ms/graph_ms:.1f}x faster)")


if __name__ == "__main__":
    benchmark_scalar_upload()
