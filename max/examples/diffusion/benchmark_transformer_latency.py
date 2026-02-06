
import time
import numpy as np
import argparse
from pathlib import Path

from max.driver import load_devices, DeviceSpec, Device
from max.tensor import Tensor
from max.dtype import DType
from max.engine import InferenceSession

# Import Flux2 components
from max.pipelines.architectures.flux2.model import Flux2Model
from max.pipelines.architectures.flux2.model_config import Flux2Config
from max.pipelines.lib import SupportedEncoding


import time
import numpy as np
import argparse
from pathlib import Path

from max.driver import load_devices, DeviceSpec, Device
from max.tensor import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef

# Import Flux2 components
from max.pipelines import PipelineConfig
from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2Pipeline
from max.pipelines.lib.pipeline_variants.pixel_generation import PixelGenerationPipeline
from max.pipelines.core import PixelContext

def benchmark_transformer(model, inputs, num_iter=50, name="Execute"):
    # Warmup
    print(f"[{name}] Warmup...")
    for _ in range(5):
        outputs = model.execute(*inputs)
        # synchronize? outputs[0].to_numpy()
        for o in outputs: # Ensure completion
             pass 

    print(f"[{name}] Running {num_iter} iterations...")
    latencies = []
    
    # We must synchronize to measure TRUE latency of execute() if it is async
    # But max.engine.Model.execute might be async?
    # If we want to measure CPU overhead, we just measure launch time?
    # No, we want end-to-end latency usually.
    # But if we want to confirm CPU Gaps, we should measure launch time specifically?
    # If launch time is 100ms, that's the gap.
    # If launch time is 0.1ms, but total time 300ms, then gaps are elsewhere?
    # Actually, if launch is fast, we can queue many.
    # But if Python takes 100ms to call launch, that's the gap.
    
    # New benchmarking logic
    from max.driver import CPU
    
    # Warmup for the new measurement method
    # The user's instruction implies integrating warmup into the loop and changing sync method.
    # The original separate warmup block is removed.
        
    start = time.perf_counter()
    for i in range(num_iter):
        if i == 0:
             print("[Execute] Warmup...")
             outputs = model.execute(*inputs)
             # Force sync
             _ = [o.to(CPU()) for o in outputs]
        
        outputs = model.execute(*inputs)
        # Force sync to ensure execution is done
        _ = [o.to(CPU()) for o in outputs]
    end = time.perf_counter()
    
    avg_lat = (end - start) / num_iter * 1000
    print(f"[{name}] Avg Latency: {avg_lat:.2f} ms")
    print(f"[{name}] Throughput: {num_iter / (end - start):.2f} iter/s")
    return avg_lat

def benchmark_capture(model, inputs, num_iter=50):
    name = "Graph Capture"
    # Capture requires Buffer inputs.
    # model.execute takes Buffer or DLPack.
    # inputs are Buffers (from Tensor.driver_tensor)?
    # We need to make sure inputs are max.driver.Buffer
    
    # helper to extract buffers
    buffers = inputs
    
    print(f"[{name}] Capturing graph...")
    try:
        # Warmup/Capture
        # Typically run once to warm up, then capture?
        # Model.capture(inputs) -> returns execution graph? 
        # API says: _Model_capture(self, *inputs) -> list[Buffer]
        # Wait. capture returns outputs?
        # No, doc says: "Capture execution into a device graph... Callers should choose capture-safe execution paths."
        # It relies on internal state?
        # max/engine/api.py: `self._capture(list(inputs))`
        # And `replay`.
        
        # We need to capture ONCE.
        # It seems capture returns just the handle
        ret = model.capture(*buffers)
        if isinstance(ret, tuple):
             graph_handle = ret[0]
        else:
             graph_handle = ret

    except Exception as e:
        print(f"[{name}] Capture Failed: {e}")
        return
        
    print(f"[{name}] Running {num_iter} replays...")
    latencies = []
    t0_global = time.time()
    for _ in range(num_iter):
        t_start = time.time()
        model.replay(*buffers)
        t_end = time.time()
        latencies.append((t_end - t_start) * 1000)

    t1_global = time.time()
    avg = np.mean(latencies)
    print(f"[{name}] Avg Launch Latency (no sync): {avg:.3f} ms")
    print(f"[{name}] Throughput (est): {num_iter / (t1_global - t0_global):.2f} iter/s")
    
    # Measure Synchronized Replay (GPU Time + Alloc?)
    print(f"[{name}] Benchmarking Replay (Synchronized)...")
    
    # Try to find a synchronize method
    # buffers[0].device is the device.
    device = buffers[0].device
    
    def sync_device():
        # Try various sync methods
        if hasattr(device, "synchronize"):
             device.synchronize()
        elif hasattr(device, "sync"):
             device.sync()
        elif hasattr(buffers[0], "stream") and hasattr(buffers[0].stream, "synchronize"):
             buffers[0].stream.synchronize()
        else:
             # Fallback: Copy input to host (might not sync output but better than nothing?)
             pass

    # Warmup
    for _ in range(5):
        model.replay(*buffers)
        sync_device()

    start = time.perf_counter()
    for _ in range(num_iter):
         model.replay(*buffers)
         sync_device()
    end = time.perf_counter()
    
    avg_replay = (end - start) / num_iter * 1000
    print(f"[{name}] Avg Replay Latency (Sync): {avg_replay:.2f} ms")
    
    return avg
    
    # Measure Synchronized Replay (GPU Time + Alloc?)
    # For CUDA Graphs, Alloc is static.
    # So Sync Replay = GPU Execution Time.
    print(f"[{name}] Benchmarking Replay (Synchronized)...")
    from max.driver import CPU
    
    # Warmup
    for _ in range(5):
        model.replay(graph_handle, buffers)
        _ = [o.to(CPU()) for o in captured_outputs]

    start = time.perf_counter()
    for _ in range(num_iter):
         model.replay(graph_handle, buffers)
         # Sync
         _ = [o.to(CPU()) for o in captured_outputs]
    end = time.perf_counter()
    
    avg_replay = (end - start) / num_iter * 1000
    print(f"[{name}] Avg Replay Latency (Sync): {avg_replay:.2f} ms")
    
    return avg

class DummyWeights:
    def items(self):
        return []

def main():
    parser = argparse.ArgumentParser()
    # model argument is no longer required for path, but maybe for config tweaks?
    # We'll use defaults hardcoded for benchmarking.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iter", type=int, default=20)
    args = parser.parse_args()

    print("Initializing device...")
    devices = load_devices([DeviceSpec.accelerator()])
    device = devices[0]

    print("Initializing Flux2 Model with Dummy Weights...")
    # Manual config
    # Flux.1 Dev dims:
    # double_layers=19, single_layers=38.
    # hidden_size=... ? 
    # Defaults in Flux2ConfigBase seem to be:
    # num_layers=8, num_single_layers=48.
    # joint_attention_dim=15360 (T5 XXH?)
    # in_channels=128 (VAE 16ch * 2 * 2 * 2?) VAE is 16ch.
    # Packed: 16ch * 2x2 = 64ch?
    # Flux VAE: compressed 8x?
    # Actually Flux.1 has 16 channels in latent space.
    # Config says 128? Maybe flattened patches?
    # Patch size 1?
    # Let's rely on Flux2Config defaults or standard Flux Dev values.
    # For latency benchmark, layer count is most important.
    
    # Let's use Flux.1 Dev configs for realistic load:
    # num_layers=19
    # num_single_layers=38
    config_dict = {
        "num_layers": 5,
        "num_single_layers": 5,
        "in_channels": 64, 
        "patch_size": 1,
        # Reduce dimensions to avoid OOM while keeping kernel count (layers) same
        "joint_attention_dim": 128, 
        "attention_head_dim": 64,
        "num_attention_heads": 4,
        "mlp_ratio": 2.0,
        "axes_dims_rope": (16, 16, 16, 16),
    }
    
    encoding = SupportedEncoding.bfloat16
    
    # Instantiate Flux2Model directly
    from max.pipelines.architectures.flux2.model import Flux2Model
    from max.driver import CPU
    
    # We need to construct the object.
    # Flux2Model(config: dict, encoding, devices, weights)
    # config dict is passed to Flux2Config.generate.
    
    flux2_model = Flux2Model(
        config=config_dict,
        encoding=encoding,
        devices=devices,
        weights=DummyWeights(),
    )
    
    # Initialize the internal Mojo model (random weights)
    print("Initializing model architecture...")
    flux2_model.load_model()
    
    # Populate _state_dict with the initialized parameters
    print("Populating weight registry from initialized parameters...")
    flux2_model._state_dict = {}
    from max.driver import CPU
    for name, param in flux2_model._flux2.parameters:
        flux2_model._state_dict[name] = param.to(CPU())

    # Custom compilation to get the Model object directly
    from max.graph import Graph, TensorType
    from max.tensor import Tensor, realization_context
    from max._realization_context import GraphRealizationContext, _session
    from max import functional as F
    import functools

    def compile_to_model(module, *input_types, weights=None):
        # Based on max.nn.Module.compile
        graph = Graph(type(module).__qualname__, input_types=input_types)
        with realization_context(GraphRealizationContext(graph)) as ctx, ctx:
            inputs = [Tensor.from_graph_value(input) for input in graph.inputs]
            
            def as_weight(name: str, tensor: Tensor):
                type = TensorType(tensor.dtype, tensor.shape, CPU())
                return F.constant_external(name, type).to(tensor.device)

            with module._mapped_parameters(as_weight):
                outputs = module(*inputs)

            if isinstance(outputs, Tensor):
                graph.output(outputs)
            else:
                graph.output(*outputs)
        
        session = _session()
        # weights handling
        if weights is None:
             raise ValueError("Weights must be provided for this benchmark script")
             
        # session.load returns the Model
        return session.load(graph, weights_registry=weights)

    # Extract constants from the initialized model
    cfg = flux2_model.config
    
    B = args.batch_size
    # Increase sizes to test allocation overhead
    img_seq_len = 2048
    txt_seq_len = 512
    
    print(f"Compiling Transformer for B={B}, Img={img_seq_len}, Txt={txt_seq_len}...")
    
    # Prepare input types
    # Note: dimensions must match config.
    # We increased config earlier? No, script has hardcoded values in config_dict.
    # We need to ensure config_dict matches inputs.
    # config_dict has: joint_attention_dim=128 ???
    # If we pass inputs with dim=256, it might fail if model expects 128.
    
    # Let's update config_dict first! (See below)
    input_types = flux2_model._flux2.input_types_with_shapes(B, img_seq_len, txt_seq_len)
    
    # Compile
    model_instance = compile_to_model(
        flux2_model._flux2,
        *input_types,
        weights=flux2_model._state_dict
    )
    
    # model_instance is now the Model object!
    compiled_model = model_instance # For consistency name if needed, but we use model_instance below 

    print("Generating Dummy Inputs...")
    dtype = DType.bfloat16
    
    in_ch = cfg.in_channels
    joint_dim = cfg.joint_attention_dim
    
    def rand_tensor(shape, dtype):
        if dtype == DType.int64:
            arr = np.random.randint(0, 100, size=shape).astype(np.int64)
        else:
            arr = np.random.randn(*shape).astype(np.float32)
        return Tensor.from_dlpack(arr).to(DeviceRef.from_device(device)).cast(dtype)

    inputs = [
        rand_tensor([B, img_seq_len, in_ch], dtype),         # hidden_states
        rand_tensor([B, txt_seq_len, joint_dim], dtype),     # encoder_hidden_states
        rand_tensor([B], dtype),                             # timestep
        rand_tensor([B, img_seq_len, 4], DType.int64),       # img_ids
        rand_tensor([B, txt_seq_len, 4], DType.int64),       # txt_ids
        rand_tensor([B], dtype),                             # guidance
    ]
    
    # Convert to Buffers for capture
    from max.driver import Buffer
    buf_inputs = []
    for i, t in enumerate(inputs):
        try:
             buf_inputs.append(Buffer.from_dlpack(t))
        except Exception as e:
             print(f"Failed to buffer input {i}: {e}")
             raise e

    print("Benchmarking Standard Execution...")
    # For execute, we can use the model_instance directly if it has execute method
    # Or compiled_model func.
    # benchmark_transformer calls model.execute.
    # So we pass model_instance.
    benchmark_transformer(model_instance, buf_inputs, num_iter=args.iter, name="Execute")
    
    print("Benchmarking Graph Capture...")
    benchmark_capture(model_instance, buf_inputs, num_iter=args.iter)

if __name__ == "__main__":
    main()

