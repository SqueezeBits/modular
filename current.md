# WAN Pipeline V2 Migration — Current Status

## What Was Done

**module_v3 → nn.layer API migration** for the WAN video generation pipeline.

Migrated from the old API (`max.nn.module_v3`, `max.experimental.tensor.Tensor`, `max.experimental.functional`) to the new v2 API (`max.nn.layer.Module`, `max.graph.TensorValue`, `max.graph.ops`, `max.nn.linear.Linear`, `max.driver.Buffer`).

### Files Modified (5 files, +920 -677 lines)

1. **`max/python/max/pipelines/architectures/flux2/layers/embeddings.py`**
   - WAN depends on `TimestepEmbedding`, `Timesteps`, `apply_rotary_emb` from here
   - `F.*` → `ops.*`, `Tensor` → `TensorValue`, `Module[[...], ...]` → `Module`, `forward` → `__call__`
   - `Linear(in, out, bias=True)` → `Linear(in_dim=, out_dim=, dtype=, device=, has_bias=)`

2. **`max/python/max/pipelines/architectures/wan/wan_transformer.py`**
   - 11 module classes migrated
   - New `WanConv3d(Module)` class using `Weight` + `ops.conv3d` (NDHWC/QRSCF layout)
   - `Tensor.ones/zeros` → `Weight("name", dtype, shape, device)`
   - `flash_attention_gpu` called directly (no `F.functional()` wrapper)
   - `ModuleList` → `LayerList`

3. **`max/python/max/pipelines/architectures/wan/model.py`**
   - `_BlockLevelModel`: `CompileWrapper` → `Model`, takes/returns `Buffer`
   - `compile_model()`: `F.lazy()` + `CompileWrapper` → `Graph` + `module.load_state_dict()` + `session.load()`
   - `_compute_wan_rope`: `Tensor.from_dlpack().to()` → `Buffer.from_numpy().to()`
   - `_rope_cache`: `tuple[Tensor, Tensor]` → `tuple[Buffer, Buffer]`

4. **`max/python/max/pipelines/architectures/wan/pipeline_wan.py`**
   - Runtime tensor types at transformer boundary: `Tensor` → `Buffer`
   - `prompt_embeds.driver_tensor` to extract `Buffer` from `Tensor` for transformer input
   - `Tensor.from_dlpack(buffer)` to wrap Buffer back to Tensor for guidance model
   - **Intentionally kept** `F` and `Tensor` imports for UMT5 text encoder and guidance model (still on module_v3)

5. **`max/python/max/pipelines/architectures/autoencoders/autoencoder_kl_wan.py`**
   - 23+ classes migrated (1724 → 1823 lines)
   - New conv wrappers: `WanCausalConv3d`, `WanCausalConv3dCached`, `WanConv2dPermuted`, `WanConv2d`
   - `AutoencoderKLWanModel`: compilation uses `Graph` + `load_state_dict` + `session.load`
   - **Intentionally kept** `F` and `Tensor` for runtime decode methods

## What Needs Testing

**NO TESTS HAVE BEEN RUN YET.** The migration is code-complete but unverified.

### Test Command (E2E Video Generation)

```bash
./bazelw run //max/examples/diffusion:simple_offline_video_generation -- \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative-prompt '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走' \
  --height 720 --width 1280 --num-frames 81 \
  --guidance-scale 4.0 --guidance-scale-2 3.0 \
  --num-inference-steps 40 \
  --output t2v_out.mp4 --fps 16
```

### Unit Tests

```bash
./bazelw test //max/tests/integration/architectures/wan/...
```

## Key Migration Patterns (Quick Reference)

| Old (module_v3)                        | New (nn.layer)                                              |
|----------------------------------------|-------------------------------------------------------------|
| `from max.nn.module_v3 import Module`  | `from max.nn.layer import Module`                           |
| `from max.nn.module_v3 import Linear`  | `from max.nn.linear import Linear`                          |
| `from max.experimental.tensor import Tensor` | `from max.graph import TensorValue` / `from max.driver import Buffer` |
| `from max.experimental import functional as F` | `from max.graph import ops`                          |
| `Module[[In], Out]`                    | `Module`                                                    |
| `def forward(self, ...)`              | `def __call__(self, ...)`                                   |
| `Tensor.ones/zeros([dim])`            | `Weight("name", dtype, [dim], device)`                      |
| `Linear(in, out, bias=True)`          | `Linear(in_dim=, out_dim=, dtype=, device=, has_bias=)`     |
| `F.reshape/permute/concat`            | `ops.reshape/permute/concat`                                |
| `CompileWrapper(module, types, weights)` | `Graph` + `load_state_dict` + `session.load()`            |
| `Tensor.from_dlpack(arr).to(dev)`     | `Buffer.from_numpy(arr).to(dev)`                            |

## Known Design Decisions

- **Buffer↔Tensor bridging**: UMT5 text encoder and guidance model still use module_v3 `Tensor`. At the transformer boundary, we convert using `tensor.driver_tensor` (Tensor→Buffer) and `Tensor.from_dlpack(buffer)` (Buffer→Tensor).
- **Conv3d layout**: Weights stored in FCQRS (PyTorch) layout with manual NCDHW↔NDHWC permute for input/output.
- **ops.pad format**: Uses `[d0_before, d0_after, d1_before, d1_after, ...]` (NOT PyTorch's reversed-pairs format).
- **ops.range**: `ops.range(start, stop, step, dtype=, device=)` replaces `F.arange(0, n, dtype=, device=)`.

## Detailed Plan

Full migration plan is at: `.claude/plans/vast-scribbling-balloon.md`

## Branch

`add/wan-pipeline/full-pipeline`
