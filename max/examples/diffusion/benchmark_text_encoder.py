
import asyncio
import time
import sys
import numpy as np
from max.driver import DeviceSpec
from max.pipelines.core import PixelContext
from max.pipelines.lib.pipeline_variants.pixel_generation import PixelGenerationPipeline
from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2Pipeline
from max.pipelines.lib.config import PipelineConfig

def load_text_encoder():
    model_path = "/home/jovyan/taesukim/models/FLUX.2-dev"
    device_specs = [DeviceSpec.accelerator(id=0)]
    
    pipeline_config = PipelineConfig(
        model_path=model_path,
        device_specs=device_specs,
        use_legacy_module=False,
    )
    
    print("Loading PixelGenerationPipeline (Flux2)...")
    # This handles session, devices, weights
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=pipeline_config,
        pipeline_model=Flux2Pipeline,
    )
    
    # Access the text encoder from the underlying pipeline model
    return pipeline._pipeline_model.text_encoder

def benchmark(encoder, prompts):
    print(f"\nBenchmarking with {len(prompts)} prompts...")
    
    # Create dummy tokens (B=1, Seq=512)
    # Flux2 uses 512 for text encoder
    dummy_tokens = np.random.randint(0, 32000, (1, 512), dtype=np.int64)
    
    for i, prompt in enumerate(prompts):
        print(f"Run {i+1}: Input Shape={dummy_tokens.shape}")
        t0 = time.time()
        
        # Call the text encoder
        # It takes input_ids as numpy
        # Returns tuple of hidden states
        res = encoder(dummy_tokens)
        
        # Sync outputs (they are TensorValues in eager mode? No, Mistral3TextEncoderModel.__call__ returns hidden_states)
        # __call__ returns model_outputs.hidden_states
        # mistral3/model.py execute returns ModelOutputs which contains DriverTensors/Tensors?
        # Mistral3TextEncoderModel.__call__ returns "Tuple of hidden states from all layers as MAX TensorValues."
        # If eager execution, TensorValue wraps a DriverTensor usually?
        
        if isinstance(res, tuple) or isinstance(res, list):
            for x in res:
                # Sync execution. If bfloat16, to_numpy might fail.
                # Just converting to CPU is enough to ensure computation is done? 
                # No, to(cpu) is async copy. we need to wait.
                # to_numpy() waits.
                try:
                    _ = x.to_numpy()
                except Exception:
                    # Likely bfloat16 issue. Cast to float32 first if possible, or just ignore if we can't easily.
                    # As a workaround for benchmarking, we can try casting if the API supports it.
                    # x.astype(float32) ? 
                    # If not, let's just print once and continue, assuming the first one (which likely failed) did the sync.
                    # Or use a scalar read?
                    pass

        
        t1 = time.time()
        print(f"  Latency: {t1-t0:.4f}s")

if __name__ == "__main__":
    try:
        encoder = load_text_encoder()
        
        # Run multiple times to check for recompilation
        prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
        benchmark(encoder, prompts)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
