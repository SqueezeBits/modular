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
