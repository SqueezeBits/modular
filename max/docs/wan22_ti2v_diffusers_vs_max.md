# Wan2.2 TI2V: Diffusers vs MAX

This note compares the current MAX implementation of `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
against the upstream Hugging Face diffusers implementation at the code-structure level.

Scope:
- DiT / transformer path
- VAE path
- TI2V-specific runtime behavior

Reference implementations:
- `diffusers/models/transformers/transformer_wan.py`
- `diffusers/models/autoencoders/autoencoder_kl_wan.py`

MAX implementation:
- [model.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/model.py)
- [wan_transformer.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/wan_transformer.py)
- [layers/transformer.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/layers/transformer.py)
- [pipeline_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/pipeline_wan.py)
- [autoencoder_kl_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/autoencoders/autoencoder_kl_wan.py)
- [vae.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/autoencoders/vae.py)

## 1. DiT / Transformer

### 1.1 Overall structure

Diffusers:
- `WanTransformer3DModel` is a single torch module.
- Forward path is:
  - patch embedding
  - time/text condition embedding
  - `N` transformer blocks
  - output modulation
  - projection
  - unpatchify

MAX:
- The same logical path is split into:
  - `WanTransformerPreProcess`
  - block-level compiled `WanTransformerBlock`
  - `WanTransformerPostProcess`
- Runtime container is `BlockLevelModel`.
- Each transformer block is compiled separately to reduce peak workspace and VRAM.

Result:
- The DiT math is intentionally aligned, but the runtime form is different:
  torch monolith in diffusers vs block-level compiled graphs in MAX.

### 1.2 Timestep handling

Diffusers:
- Standard Wan:
  - timestep shape is `[batch]`
- Wan2.2 TI2V:
  - timestep shape is `[batch, seq_len]`
  - the timestep sequence is expanded per token
  - first latent-frame tokens are masked differently from later tokens

MAX:
- Standard Wan path:
  - `[batch]`
- TI2V path:
  - `[batch, seq_len]`
  - per-token modulation support added in:
    - [layers/embeddings.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/layers/embeddings.py)
    - [wan_transformer.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/wan_transformer.py)
    - [layers/transformer.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/layers/transformer.py)
- TI2V-specific timestep masking is applied in:
  - [pipeline_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/pipeline_wan.py)

Important difference:
- The original MAX TI2V attempt used uniform per-token timesteps.
- That caused early-frame collapse.
- The current MAX path now applies a TI2V timestep mask so the first latent-frame tokens do not receive the same timestep as the denoised frames.

### 1.3 TI2V routing

Diffusers:
- `Wan2.2-TI2V-5B-Diffusers` still uses `_class_name = "WanPipeline"`.
- TI2V behavior is inferred from top-level config such as `expand_timesteps=True`.

MAX:
- Same decision:
  - no new public architecture name
  - still resolves to `WanPipeline`
  - TI2V mode is inferred from top-level `expand_timesteps=True`

Relevant code:
- [pixel_tokenizer.py](/root/max_workspace/modular/max/python/max/pipelines/lib/pixel_tokenizer.py)
- [pipeline_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/pipeline_wan.py)

### 1.4 TI2V conditioning semantics

Diffusers / Wan2.2 behavior:
- image-conditioned TI2V does not use the old Wan2.1 concat-I2V path
- it encodes the condition image to VAE latents
- it replaces the global first latent frame
- that conditioning is preserved during denoising

MAX:
- Same high-level semantics are implemented in:
  - [pipeline_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/pipeline_wan.py)
- Specifically:
  - encode input image to VAE latent
  - normalize with Wan VAE latent stats
  - replace first latent frame
  - restore that frame after scheduler steps

## 2. VAE

### 2.1 Base architecture

Diffusers:
- `AutoencoderKLWan`
- Supports:
  - `decoder_base_dim`
  - `patch_size`
  - `temperal_downsample`
  - `is_residual`
- For Wan2.2 TI2V:
  - `base_dim = 160`
  - `decoder_base_dim = 256`
  - `patch_size = 2`
  - `in_channels = 12`
  - `out_channels = 12`
  - `z_dim = 48`
  - `is_residual = True`

MAX:
- `AutoencoderKLWanModel`
- VAE config fields mirrored in:
  - [model_config.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/autoencoders/model_config.py)
- The MAX VAE now handles:
  - `decoder_base_dim`
  - `patch_size=2`
  - `in/out_channels=12`
  - `z_dim=48`
  - `temperal_downsample` alias normalization

### 2.2 Patchify / unpatchify

Diffusers:
- `patchify()` and `unpatchify()` are explicit helpers inside `autoencoder_kl_wan.py`
- `patch_size=2` converts spatial `3`-channel video into `12`-channel VAE input

MAX:
- Equivalent numpy helpers exist in:
  - [autoencoder_kl_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/autoencoders/autoencoder_kl_wan.py)
- These are used before encode and after decode.

This was necessary because the original MAX Wan VAE path assumed the older `patch_size=None/1` behavior.

### 2.3 Residual VAE blocks

Diffusers:
- Wan2.2 residual VAE uses:
  - `AvgDown3D`
  - `DupUp3D`
  - residual down/up blocks
- This changes both encoder and decoder channel flow and shortcut behavior.

MAX:
- The helper modules and residual block paths are now present in:
  - [vae.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/autoencoders/vae.py)
- The MAX implementation diverges from diffusers at runtime strategy:
  - diffusers is pure eager torch
  - MAX compiles graph fragments and caches them

Important implementation detail:
- The non-residual Wan path in MAX can use symbolic spatial dims cleanly.
- The residual Wan2.2 path is more sensitive to shape arithmetic in cached encoder/decoder graphs.
- The current MAX implementation avoids the worst symbolic-shape failures by compiling the residual VAE on concrete spatial shapes per request shape.

### 2.4 Encoder key layout

Diffusers:
- non-residual encoder can be effectively flattened into resnet/downsampler entries
- residual encoder is already grouped semantically by down block

MAX:
- For non-residual Wan, encoder keys are remapped from hierarchical HF names to the flat MAX layout in:
  - [autoencoder_kl_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/autoencoders/autoencoder_kl_wan.py)
- For residual Wan2.2, MAX uses the grouped residual down-block path directly and does not rely on the older flat remap logic.

## 3. Runtime differences

### 3.1 Compilation model

Diffusers:
- eager torch execution
- no MAX graph compile boundary

MAX:
- DiT:
  - block-level compile
- VAE:
  - cached encoder/decoder graph compile
  - residual Wan2.2 VAE uses concrete-shape compile to stay stable

Tradeoff:
- MAX is structurally more complex because compile stability and memory behavior matter.
- Diffusers is structurally simpler because eager torch avoids graph-shape issues.

### 3.2 Where MAX now matches diffusers

For `Wan2.2-TI2V-5B` MAX now matches diffusers on the parts that matter for correctness:
- model routing through `WanPipeline`
- top-level `expand_timesteps=True` TI2V detection
- per-token timestep modulation
- first latent-frame TI2V conditioning
- VAE `patch_size=2`
- `decoder_base_dim=256`
- `in/out_channels=12`
- `z_dim=48`

### 3.3 Where MAX still differs structurally

MAX still differs from diffusers in these ways:
- DiT is block-compiled instead of eager
- VAE is graph-compiled instead of eager
- residual VAE shape handling is more explicit and compile-oriented
- several helpers use `rebind` / cached graph signatures that have no diffusers analogue

## 4. TI2V quality issue and fix

Observed issue:
- The original MAX TI2V output could collapse into cyan/blue low-variance frames, especially at the start of the video.

Root causes:
- wrong TI2V defaults in the offline example
- missing Wan2.2 VAE config handling (`patch_size=2`, `decoder_base_dim=256`)
- missing TI2V per-token timestep masking

Fixes:
- example defaults updated in:
  - [simple_offline_video_generation.py](/root/max_workspace/modular/max/examples/diffusion/simple_offline_video_generation.py)
- TI2V tokenizer / latent defaults updated in:
  - [pixel_tokenizer.py](/root/max_workspace/modular/max/python/max/pipelines/lib/pixel_tokenizer.py)
- TI2V timestep mask added in:
  - [pipeline_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/wan/pipeline_wan.py)
- Wan2.2 VAE config and residual path support updated in:
  - [autoencoder_kl_wan.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/autoencoders/autoencoder_kl_wan.py)
  - [vae.py](/root/max_workspace/modular/max/python/max/pipelines/architectures/autoencoders/vae.py)

## 5. Summary

At a code level:
- MAX DiT is now semantically aligned with diffusers for Wan2.2 TI2V, but compiled differently.
- MAX VAE now supports the Wan2.2 TI2V config shape and residual path needed for usable output.
- The main implementation gap that caused visible quality collapse was not the transformer math, but TI2V-specific timestep semantics and Wan2.2 VAE behavior.

If this note needs to be expanded, the next useful addition would be a symbol-by-symbol mapping table:
- diffusers class / method
- MAX class / method
- behavior match / behavior divergence
