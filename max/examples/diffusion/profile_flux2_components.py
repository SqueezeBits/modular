# mypy: ignore-errors
import argparse
import time
import numpy as np
import warnings
import asyncio
from typing import Literal
from tqdm import tqdm

import torch  # For cuda.synchronize() in profiling
try:
    from diffusers import Flux2Pipeline as DiffusersFlux2Pipeline
except ImportError:
    DiffusersFlux2Pipeline = None

from max.driver import DeviceSpec, Accelerator, Buffer as DriverTensor
from max.dtype import DType
from max.pipelines import PipelineConfig
from max.pipelines.architectures.flux2.pipeline_flux2 import Flux2Pipeline, Flux2ModelInputs, Flux2PipelineOutput, FluxMistral3TextEncoder
from max.tensor import Tensor
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.pipeline_variants.pixel_generation import PixelGenerationPipeline
from max.interfaces import PixelGenerationRequest, RequestID, PixelGenerationInputs
from max.pipelines.core import PixelContext

# Suppress warnings
warnings.filterwarnings("ignore")

print(f"DEBUG: Flux2Pipeline components: {Flux2Pipeline.components}")
print(f"DEBUG: FluxMistral3TextEncoder imported: {FluxMistral3TextEncoder}")

class ProfiledFlux2Pipeline(Flux2Pipeline):
    """Subclass of Flux2Pipeline with internal profiling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiling_stats = {
            "prompt_encoding": [],
            "compilation": [],
            "denoising_step": [],
            "transformer_forward": [],
            "scheduler_step": [],
            "vae_decode": [],
            "total_execute": []
        }

    def execute(
        self,
        model_inputs: Flux2ModelInputs,
        callback_queue = None,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Flux2PipelineOutput:
        
        t0_execute = time.time()
        
        # 1. Encode prompts
        t0 = time.time()
        prompt_embeds, text_ids = self._prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )
        self.profiling_stats["prompt_encoding"].append(time.time() - t0)

        # 2. Denoise setup
        dtype = prompt_embeds.dtype
        latents: Tensor = (
            Tensor.from_dlpack(model_inputs.latents)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )
        # Patchify then pack latents (critical for correct output)
        latents = self._patchify_latents(latents)
        latents = self._pack_latents(latents)

        latent_image_ids: Tensor = (
            Tensor.from_dlpack(model_inputs.latent_image_ids)
            .to(self.transformer.devices[0])
        )

        guidance = Tensor.full(
            [latents.shape[0]],
            model_inputs.guidance_scale,
            device=self.transformer.devices[0],
            dtype=dtype,
        )

        # Store sigmas on CPU for optimized scheduler step (no GPU sync)
        self._sigmas_cpu = model_inputs.sigmas.copy()

        sigmas = (
            Tensor.from_dlpack(model_inputs.sigmas)
            .to(self.transformer.devices[0])
        )
        batch_size = prompt_embeds.shape[0].dim

        timesteps: np.ndarray = model_inputs.timesteps
        num_timesteps = timesteps.shape[0]

        # DriverTensor approach: No pre-creation loops needed here

        # Compile
        t0 = time.time()
        image_seq_len = latents.shape[1].dim
        text_seq_len = prompt_embeds.shape[1].dim
        compiled_model = self.transformer._ensure_compiled(
            batch_size=batch_size,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
        )
        self.profiling_stats["compilation"].append(time.time() - t0)

        # Build scheduler step graph if not already built
        if self._scheduler_step_model is None:
            device = self.transformer.devices[0]
            self._build_scheduler_step_graph(dtype, device)

        # Pre-convert unchanging tensors
        encoder_hidden_states_drv = prompt_embeds.driver_tensor
        guidance_drv = guidance.driver_tensor
        txt_ids_drv = text_ids.driver_tensor
        img_ids_drv = latent_image_ids.driver_tensor

        for i in tqdm(range(num_timesteps), desc="Denoising"):
            t0_step = time.time()
            self._current_timestep = i

            # Manual float32 -> bfloat16 conversion (truncation) to avoid Tensor.cast overhead
            t = timesteps[i]
            dt = self._sigmas_cpu[i + 1] - self._sigmas_cpu[i]
            t_u16 = (np.array(t / 1000.0, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)[None]
            dt_u16 = (np.array(dt, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)[None]

            timestep_drv = DriverTensor.from_dlpack(t_u16).to(self.transformer.devices[0]).view(DType.bfloat16)
            dt_drv = DriverTensor.from_dlpack(dt_u16).to(self.transformer.devices[0]).view(DType.bfloat16, shape=[])

            latents_drv = latents.driver_tensor

            # Transformer forward pass
            t0_transformer = time.time()
            
            # Sync before timing if possible? MAX syncs on output access.
            # But driver_tensor execution might be async on device.
            # Accessing output forces sync.
            output_drv_list = compiled_model(
                latents_drv,
                encoder_hidden_states_drv,
                timestep_drv,
                img_ids_drv,
                txt_ids_drv,
                guidance_drv,
            )

            noise_pred_drv = output_drv_list[0]
            torch.cuda.synchronize()
            
            self.profiling_stats["transformer_forward"].append(time.time() - t0_transformer)

            # Scheduler step using compiled graph
            t0_scheduler = time.time()
            latents_drv = self._scheduler_step(latents_drv, noise_pred_drv, dt_drv)
            latents = Tensor.from_dlpack(latents_drv)
            torch.cuda.synchronize()
            self.profiling_stats["scheduler_step"].append(time.time() - t0_scheduler)
            
            self.profiling_stats["denoising_step"].append(time.time() - t0_step)

            if callback_queue is not None:
                image = self._decode_latents(
                    latents,
                    latent_image_ids,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
                callback_queue.put_nowait(image)
        
        # Print intermediate stats in case of crash during VAE
        print("\n--- Intermediate Profiling Results (Pre-VAE) ---")
        print(f"Steps: {len(self.profiling_stats['denoising_step'])}")
        if self.profiling_stats['denoising_step']:
            print(f"Prompt Encoding:       {np.mean(self.profiling_stats['prompt_encoding'])*1000:.2f} ms")
            print(f"Compilation:           {np.mean(self.profiling_stats['compilation'])*1000:.2f} ms")
            print(f"Total Step Time:       {np.mean(self.profiling_stats['denoising_step'])*1000:.2f} ms")
            print(f"  Transformer Fwd:     {np.mean(self.profiling_stats['transformer_forward'])*1000:.2f} ms")
            print(f"  Scheduler Step:      {np.mean(self.profiling_stats['scheduler_step'])*1000:.2f} ms")
        
        # 3. Decode
        t0 = time.time()
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
        self.profiling_stats["vae_decode"].append(time.time() - t0)
        
        self.profiling_stats["total_execute"].append(time.time() - t0_execute)
        
        return Flux2PipelineOutput(images=image_list)

async def profile_flux2(model_path, num_steps=10):
    print(f"Profiling Flux2 pipeline components...")
    
    config = PipelineConfig(
        model_path=model_path,
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
        pipeline_model=ProfiledFlux2Pipeline,
    )
    
    request = PixelGenerationRequest(
        request_id=RequestID(),
        model_name="profile",
        prompt="A cat in a garden",
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        guidance_scale=3.5,
    )
    
    print("Preparing inputs...")
    t0_prepare = time.time()
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )
    print(f"Preparation time (Tokenizer + Init): {(time.time() - t0_prepare)*1000:.2f} ms")
    
    # Run a full e2e warmup pass to compile all graphs (incl. VAE)
    print("Running warmup pass (untimed)...")
    pipeline.execute(inputs)
    print("Warmup complete.")
    
    # Reset profiling stats for the actual timed run
    pipeline._pipeline_model.profiling_stats = {
        "prompt_encoding": [],
        "compilation": [],
        "transformer_forward": [],
        "scheduler_step": [],
        "denoising_step": [],
        "vae_decode": [],
        "total_execute": [],
    }
    
    print("Executing pipeline (timed)...")
    pipeline.execute(inputs)
    
    # Extract stats from the internal model
    stats = pipeline._pipeline_model.profiling_stats
    
    print("\n--- Profiling Results ---")
    print(f"Steps: {num_steps}")
    
    print(f"Prompt Encoding:       {np.mean(stats['prompt_encoding'])*1000:.2f} ms")
    print(f"Compilation:           {np.mean(stats['compilation'])*1000:.2f} ms")
    print(f"VAE Decode:            {np.mean(stats['vae_decode'])*1000:.2f} ms")
    print(f"Total Execution:       {np.mean(stats['total_execute'])*1000:.2f} ms")
    
    print("\n--- Per Step Averages ---")
    print(f"Total Step Time:       {np.mean(stats['denoising_step'])*1000:.2f} ms")
    print(f"  Transformer Fwd:     {np.mean(stats['transformer_forward'])*1000:.2f} ms")
    print(f"  Scheduler Step:      {np.mean(stats['scheduler_step'])*1000:.2f} ms")
    print(f"  Overhead/Other:      {(np.mean(stats['denoising_step']) - np.mean(stats['transformer_forward']) - np.mean(stats['scheduler_step']))*1000:.2f} ms")

class TimingWrapper:
    def __init__(self, obj, name, stats_dict):
        self.obj = obj
        self.name = name
        self.stats_dict = stats_dict

    def __call__(self, *args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.time()
        res = self.obj(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.time()
        if self.name not in self.stats_dict:
            self.stats_dict[self.name] = []
        self.stats_dict[self.name].append(t1 - t0)
        return res

    def __getattr__(self, name):
        return getattr(self.obj, name)

def profile_diffusers(model_path, num_steps=10):
    if DiffusersFlux2Pipeline is None:
        print("Diffusers not installed.")
        return

    print(f"Profiling Diffusers Flux2 pipeline...")
    pipe = DiffusersFlux2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        revision="refs/pr/1", 
    ).to("cuda")

    print("Compiling transformer...")
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

    stats = {}
    
    # Wrap components
    # pipe.encode_prompt is strictly for prompt encoding
    original_encode_prompt = pipe.encode_prompt
    pipe.encode_prompt = TimingWrapper(original_encode_prompt, "prompt_encoding", stats)

    # pipe.transformer is already compiled, wrap it
    pipe.transformer = TimingWrapper(pipe.transformer, "transformer_forward", stats)
    
    # pipe.scheduler.step
    pipe.scheduler.step = TimingWrapper(pipe.scheduler.step, "scheduler_step", stats)
    
    # pipe.vae.decode
    pipe.vae.decode = TimingWrapper(pipe.vae.decode, "vae_decode", stats)

    print("Warming up (and compiling)...")
    # First run triggers compilation
    t0_warmup = time.time()
    pipe(
        prompt="A cat",
        num_inference_steps=2, # Short warmup
        height=1024,
        width=1024,
        guidance_scale=3.5,
        max_sequence_length=512,
        output_type="pil"
    )
    print(f"Warmup done in {time.time() - t0_warmup:.2f}s")
    
    # Clear stats from warmup
    for k in stats:
        stats[k] = []

    print("Executing profiled run...")
    t0_total = time.time()
    pipe(
        prompt="A cat",
        num_inference_steps=num_steps,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        max_sequence_length=512,
        output_type="pil"
    )
    total_time = time.time() - t0_total

    print("\n--- Profiling Results (Diffusers) ---")
    print(f"Steps: {num_steps}")
    
    # Prompt encoding runs once
    if "prompt_encoding" in stats:
        print(f"Prompt Encoding:       {np.mean(stats['prompt_encoding'])*1000:.2f} ms")
    
    # Transformer runs num_steps times
    if "transformer_forward" in stats:
        print(f"Transformer Fwd:       {np.mean(stats['transformer_forward'])*1000:.2f} ms")
        
    # Scheduler runs num_steps times
    if "scheduler_step" in stats:
        print(f"Scheduler Step:        {np.mean(stats['scheduler_step'])*1000:.2f} ms")
        
    # VAE runs once (if output_type=pil/np)
    if "vae_decode" in stats:
        print(f"VAE Decode:            {np.mean(stats['vae_decode'])*1000:.2f} ms")
        
    print(f"Total Execution:       {total_time*1000:.2f} ms")
    
    # Estimate breakdown
    # Total step time approximation
    step_time_avg = 0
    if "transformer_forward" in stats and "scheduler_step" in stats:
        step_time_avg = np.mean(stats['transformer_forward']) + np.mean(stats['scheduler_step'])
        print("\n--- Per Step Averages ---")
        print(f"Est. Step Time:        {step_time_avg*1000:.2f} ms (Transformer + Scheduler)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to FLUX.2 model")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps for profiling")
    parser.add_argument("--framework", type=str, choices=["max", "diffusers"], default="max", help="Framework to profile")
    args = parser.parse_args()

    if args.framework == "max":
        asyncio.run(profile_flux2(args.model, args.steps))
    else:
        profile_diffusers(args.model, args.steps)

