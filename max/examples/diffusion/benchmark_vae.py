# mypy: ignore-errors
import argparse
import time
import numpy as np
import torch
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig
from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2Pipeline
from max.engine import InferenceSession
from max.tensor import Tensor
from max.dtype import DType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    args = parser.parse_args()

    print(f"Loading VAE from {args.model}...")
    
    # 1. Load Pipeline
    config = PipelineConfig(
        model_path=args.model,
        device_specs=[DeviceSpec.accelerator()],
        use_legacy_module=False,
    )
    
    # Initialize pipeline to get components
    pipeline = Flux2Pipeline(
        pipeline_config=config,
    )
    
    # 2. Extract VAE
    # Pipeline loads components into self.vae, self.transformer etc.
    # But they are loaded lazily or in __init__?
    # Flux2Pipeline.__init__ calls _load_sub_models via super().
    # So `pipeline.vae` should exist.
    
    vae = pipeline.vae
    print("VAE loaded.")
    
    # 3. Prepare Inputs
    # 1024x1024 image -> 128x128 latents, 16 channels, batch 1
    # Flux latents: (batch, channels, height, width) = (1, 16, 128, 128)
    # But inputs to decode might be different?
    # Flux2Pipeline._decode_latents calls self.vae.decode(latents)
    # Check signature of vae.decode
    
    # Dummy latents
    # VAE expects float32 or bfloat16?
    # Usually matches device dtype.
    
    # We need to compile the VAE first.
    # We can call pipeline.vae.decode(latents)
    
    latents_np = np.random.randn(1, 16, 128, 128).astype(np.float32)
    # Ideally convert to Tensor on device
    # pipeline.vae usually expects max.tensor.Tensor
    
    print("Compiling/Warming up VAE...")
    t0 = time.time()
    
    # We need to construct the input tensor correctly
    # Device?
    device = config.device_specs[0]
    # We can use pipeline.session? Or create one?
    # Pipeline manages session.
    
    # pipeline.vae.decode expects input dictionary? Or positional args?
    # AutoencoderKLFlux2Model.decode(self, latents)
    
    # Let's inspect how pipeline calls it:
    # decoded = self.vae.decode(latents)
    
    # We need to turn numpy to Tensor
    # pipeline.vae.session is the InferenceSession
    
    # Running warmup
    # We'll rely on pipeline helper if possible, or just call vae manually.
    # To run vae, we need to ensure it's loaded onto device.
    # pipeline.load(device) happens?
    
    # Let's try running through pipeline's internal method if possible?
    # Or just call vae.decode
    
    # Input tensor
    # We need to fetch the device from the pipeline
    # config.devices[0]?
    # config.device_specs is a list of specs.
    # We can get the actual device object from pipeline.session.devices[0] if exposed.
    
    # Workaround: just pass numpy, VAE might handle it?
    # No, MAX layers expect Tensors usually.
    
    # Let's look at pipeline_flux2.py source to see how it calls vae.
    # For now, assume we can pass tensor.
    
    pass

if __name__ == "__main__":
    main()
