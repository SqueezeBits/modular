# mypy: ignore-errors
"""Benchmark to identify overhead in MAX Flux2 pipeline per-call."""

import argparse
import time
import numpy as np
import torch
import asyncio
import os

from max.driver import DeviceSpec
from max.pipelines import PipelineConfig
from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2Pipeline
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.pipeline_variants.pixel_generation import PixelGenerationPipeline
from max.interfaces import PixelGenerationRequest, RequestID, PixelGenerationInputs
from max.pipelines.core import PixelContext
from max.tensor import Tensor


class InstrumentedFlux2Pipeline(Flux2Pipeline):
    """Flux2Pipeline with per-component timing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing = {}
    
    def execute(self, model_inputs, callback_queue=None, output_type="np"):
        self.timing = {}
        
        # 1. Prompt encoding
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        prompt_embeds, text_ids = self._prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )
        torch.cuda.synchronize()
        self.timing["prompt_encoding"] = time.perf_counter() - t0

        # 2. Latent setup
        t0 = time.perf_counter()
        dtype = prompt_embeds.dtype
        latents = (
            Tensor.from_dlpack(model_inputs.latents)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )
        latents = self._patchify_latents(latents)
        latents = self._pack_latents(latents)

        latent_image_ids = (
            Tensor.from_dlpack(model_inputs.latent_image_ids)
            .to(self.transformer.devices[0])
        )

        guidance = Tensor.full(
            [latents.shape[0]],
            model_inputs.guidance_scale,
            device=self.transformer.devices[0],
            dtype=dtype,
        )
        torch.cuda.synchronize()
        self.timing["latent_setup"] = time.perf_counter() - t0

        # 3. Scheduler setup
        t0 = time.perf_counter()
        from max.pipelines.architectures.flux2.pipeline_flux2 import compute_empirical_mu
        image_seq_len = latents.shape[1].dim
        num_inference_steps = model_inputs.num_inference_steps
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)
        base_sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)
        self.scheduler.set_timesteps(sigmas=base_sigmas, mu=mu)
        self._sigmas_cpu = np.ascontiguousarray(self.scheduler.sigmas)
        
        sigmas = Tensor.from_dlpack(self._sigmas_cpu).to(self.transformer.devices[0])
        batch_size = prompt_embeds.shape[0].dim
        timesteps = self.scheduler.timesteps
        num_timesteps = timesteps.shape[0]
        torch.cuda.synchronize()
        self.timing["scheduler_setup"] = time.perf_counter() - t0

        # 4. Pre-create tensors
        t0 = time.perf_counter()
        timestep_tensors_drv = []
        for t in timesteps:
            timestep_np = np.full((batch_size,), t, dtype=np.float32) / 1000.0
            timestep_tensor = (
                Tensor.from_dlpack(timestep_np)
                .to(self.transformer.devices[0])
                .cast(dtype)
            )
            timestep_tensors_drv.append(timestep_tensor.driver_tensor)

        dt_tensors_drv = []
        for i in range(num_timesteps):
            dt_val = float(self._sigmas_cpu[i + 1] - self._sigmas_cpu[i])
            dt_np = np.array(dt_val, dtype=np.float32)
            dt_tensor = (
                Tensor.from_dlpack(dt_np)
                .to(self.transformer.devices[0])
                .cast(dtype)
            )
            dt_tensors_drv.append(dt_tensor.driver_tensor)
        torch.cuda.synchronize()
        self.timing["tensor_creation"] = time.perf_counter() - t0

        # 5. Transformer compilation
        t0 = time.perf_counter()
        text_seq_len = prompt_embeds.shape[1].dim
        compiled_model = self.transformer._ensure_compiled(
            batch_size=batch_size,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
        )
        torch.cuda.synchronize()
        self.timing["transformer_compile"] = time.perf_counter() - t0

        # 6. Scheduler step graph
        t0 = time.perf_counter()
        if self._scheduler_step_model is None:
            device = self.transformer.devices[0]
            self._build_scheduler_step_graph(dtype, device)
        torch.cuda.synchronize()
        self.timing["scheduler_graph_build"] = time.perf_counter() - t0

        # 7. Denoising loop
        encoder_hidden_states_drv = prompt_embeds.driver_tensor
        guidance_drv = guidance.driver_tensor
        txt_ids_drv = text_ids.driver_tensor
        img_ids_drv = latent_image_ids.driver_tensor
        latents_drv = latents.driver_tensor

        t0 = time.perf_counter()
        from tqdm import tqdm
        for i in tqdm(range(num_timesteps), desc="Denoising"):
            timestep_drv = timestep_tensors_drv[i]
            dt_drv = dt_tensors_drv[i]

            noise_pred_drv = compiled_model(
                latents_drv,
                encoder_hidden_states_drv,
                timestep_drv,
                img_ids_drv,
                txt_ids_drv,
                guidance_drv,
            )[0]

            latents_drv = self._scheduler_step(latents_drv, noise_pred_drv, dt_drv)
        
        torch.cuda.synchronize()
        self.timing["denoising_loop"] = time.perf_counter() - t0

        # 8. VAE decode
        t0 = time.perf_counter()
        latents = Tensor.from_dlpack(latents_drv)
        batch_size = latents.shape[0].dim
        image_list = []
        for b in range(batch_size):
            latents_b = latents[b : b + 1]
            latent_image_ids_b = latent_image_ids[b : b + 1]
            image_b = self._decode_latents(
                latents_b,
                latent_image_ids_b,
                model_inputs.height,
                model_inputs.width,
                output_type=output_type,
            )
            image_list.append(image_b)
        torch.cuda.synchronize()
        self.timing["vae_decode"] = time.perf_counter() - t0

        from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2PipelineOutput
        return Flux2PipelineOutput(images=image_list)
    
    def print_timing(self):
        print("\n--- Per-call Component Timing ---")
        total = 0
        for k, v in self.timing.items():
            print(f"  {k:25s}: {v*1000:8.2f} ms")
            total += v
        print(f"  {'TOTAL':25s}: {total*1000:8.2f} ms")


async def run_benchmark(model_path, num_steps=1, num_runs=5):
    print(f"Loading model from {model_path}...")
    
    config = PipelineConfig(
        model_path=os.path.normpath(model_path),
        device_specs=[DeviceSpec.accelerator()],
        use_legacy_module=False,
    )
    
    tokenizer = PixelGenerationTokenizer(
        model_path=model_path,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=512,
    )
    
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=InstrumentedFlux2Pipeline,
    )
    
    request = PixelGenerationRequest(
        request_id=RequestID(),
        model_name="benchmark",
        prompt="A cat",
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        guidance_scale=3.5,
    )
    
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[PixelContext](batch={context.request_id: context})
    
    # Warmup
    print("Warming up (3 runs)...")
    for _ in range(3):
        pipeline.execute(inputs)
    
    # Benchmark
    print(f"\nRunning {num_runs} instrumented runs...")
    all_timings = []
    for i in range(num_runs):
        pipeline.execute(inputs)
        pipeline._pipeline_model.print_timing()
        all_timings.append(pipeline._pipeline_model.timing.copy())
    
    # Average
    print("\n=== AVERAGE TIMING ===")
    avg_timing = {}
    for k in all_timings[0].keys():
        avg_timing[k] = np.mean([t[k] for t in all_timings])
    
    total = 0
    for k, v in avg_timing.items():
        print(f"  {k:25s}: {v*1000:8.2f} ms")
        total += v
    print(f"  {'TOTAL':25s}: {total*1000:8.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--steps", type=int, default=1, help="Number of steps")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.model, args.steps, args.runs))


if __name__ == "__main__":
    main()
