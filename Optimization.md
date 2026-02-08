# Optimization Log

## Text Encoder Optimization

**Goal:** Reduce `_prepare_prompt_embeddings` latency from ~300ms to ~30ms.

**Changes:**
- Implemented `_ensure_prompt_tile_model` to replace eager `F.tile` and `reshape` with a compiled graph.
- Implemented `_ensure_text_ids_model` to replace numpy-based text ID generation with a compiled graph (on-device generation).
- Updated `_prepare_prompt_embeddings` to use these compiled graphs.

**Results:**
- Baseline: ~300ms (as reported)
- Optimized: 15.44ms

**Verification:**
Run `simple_offline_generation.py` with `FLUX2_DEBUG=1`.

## VAE Latency Optimization

**Goal:** Reduce VAE decode latency and remove graph construction overhead (~400ms gap).

**Changes:**
- **DriverTensor Slicing:** Modified `Flux2Pipeline.execute` to slice `DriverTensor` (Buffer) instead of `Tensor`. Slicing `Tensor` in Python triggers implicit graph construction (`ops.slice_tensor`), adding significant overhead.
- **Bypass Model.__call__:** Overrode `AutoencoderKLFlux2Model.decode` to directly call `model.execute()` when inputs are `DriverTensor`, bypassing `Model.__call__` overhead.

**Results:**
- Eliminated ~430ms latency gap between denoising and VAE.
- Removed trace artifacts (`slice_tensor`, `to_mlir`, `parameters`) during decode.

**Verification:**
Run `simple_offline_generation.py` with `FLUX2_DEBUG=1`.

