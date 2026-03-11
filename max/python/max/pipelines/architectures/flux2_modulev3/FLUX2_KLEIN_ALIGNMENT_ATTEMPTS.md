# FLUX.2-Klein Alignment Attempts

Date: 2026-03-11

## Scope

This note summarizes the FLUX.2-Klein prompt-alignment experiments that were
tried while comparing MAX outputs against diffusers.

This is separate from `FLUX2_PADDING_HANDOFF.md`.

- `FLUX2_PADDING_HANDOFF.md` explains the general FLUX.2 prompt-padding work.
- this document focuses on the Klein-specific mismatch investigation
- the main question here was: why does `FLUX.2-klein` still diverge from
  diffusers even after the FLUX.2-dev-style padding fix was ported to Qwen3

## Baseline Observation

The first strong signal came from comparing saved intermediate tensors from
diffusers and MAX.

Observed behavior:

- `latent_model_input_0` matched
- the first major divergence appeared at `prompt_embeds`
- `noise_pred_0` then diverged immediately afterward

The helper script used for this was:

```bash
python max/tests/integration/tools/compare_flux2_klein_dumps.py
```

The important finding was:

- diffusers used a right-padded prompt layout
- MAX was restoring embeddings as if they belonged in a different position
- the best-shift analysis showed that the meaningful embedding rows were similar
  in content but misaligned in sequence position

This made prompt alignment the first place to focus.

## Why FLUX.2-dev Worked but Klein Did Not

`FLUX.2-dev` uses Mistral3 and left padding.

That matters because with left padding plus causal attention:

- padded prefix tokens do not have access to later real tokens
- their hidden states tend to stay nearly inert
- so "compact tokens + zero-padding restore" is a reasonable approximation

`FLUX.2-klein` uses Qwen3 and right padding.

That changes the behavior:

- padded suffix tokens can still attend to earlier real tokens
- the padded positions can therefore produce nonzero hidden states
- so "compact tokens + zero-padding restore" is not equivalent to full padded
  execution with an attention mask

This was the central reason Klein required separate investigation.

## Attempt 1: Reuse the FLUX.2-dev Style Fix

### Idea

Port the Mistral3 approach directly to Qwen3:

- compact tokens in the tokenizer
- run the text encoder on only valid tokens
- restore to target prompt length after encoding

### Why it was tried

This was the simplest extension of the working `FLUX.2-dev` solution.

### Result

This did not match diffusers.

Reason:

- the original Qwen3/Klein port initially restored the sequence as if the same
  positional assumption used by Mistral3 still applied
- but Klein does not share Mistral3's left-padding behavior

## Attempt 2: Change Qwen3 Restore Direction to Right Zero-Padding

### Idea

Since diffusers' Klein tokenizer uses right padding, switch Qwen3 restore logic
to right zero-padding instead of left zero-padding.

### Why it was tried

The diffusers debug prints showed:

- `input_ids` had real tokens first
- `attention_mask` had a valid prefix followed by zeros

So the prompt layout clearly matched right padding.

### Result

This fixed the obvious sequence-direction mismatch, but it still did not match
diffusers well enough.

Reason:

- right zero-padding restores sequence length
- but it still forces padded positions to zero
- diffusers does not do that
- in Qwen3 right-padding mode, padded query positions can still attend to
  earlier real tokens and produce nonzero hidden states

So this attempt corrected alignment direction, but not the underlying
attention-mask semantics.

## Attempt 3: Use `valid_length` Padded Flash Attention

### Idea

Use MAX's existing padded flash-attention kernel instead of zero-padding the
text-encoder output.

Implemented shape of the experiment:

1. stop compacting tokens for `FLUX.2-klein`
2. preserve the tokenizer attention mask
3. convert the mask to `valid_length = mask.sum()`
4. pass that valid length into Qwen3 attention via
   `flash_attention_gpu(..., valid_length=...)`

### Why it was tried

This was the most promising compiled-path approximation already supported by the
existing kernel stack.

The expected benefit was:

- keep execution compiled
- keep full padded token sequence
- avoid hand-restoring prompt embeddings with zeros

### Code shape of the experiment

The experiment required these categories of changes:

- `pixel_tokenizer.py`
  - keep full padded tokens for `FLUX.2-klein`
- `PixelContext`
  - preserve both positive and negative prompt masks
- `Flux2Pipeline` / `Flux2KleinPipeline`
  - convert masks to `uint32 valid_length`
- Qwen3 text encoder
  - accept `valid_length` as a compiled input
  - pass it through every attention layer

There were also follow-up compile fixes:

- Qwen3 text encoder was changed to batch-first execution
- the text-encoder input types were made symbolic in batch dimension
- this was needed because the padded attention kernel compile path was sensitive
  to shape provenance and internal rebind expectations

### Intermediate issue: compile failures

The first versions of this attempt failed during model compilation.

The compile errors came from the padded flash-attention kernel path and looked
like shape/rebind mismatches in generated KGEN.

The following adjustments were needed before runtime execution worked:

- make Qwen3 text encoder batch-first instead of creating batch size `1` inside
  the graph with `unsqueeze`
- make the text-encoder input types use symbolic batch shape rather than a
  literal `1`
- pass `valid_length` as a batch-shaped input that matches the attention input
  contract more directly

After those changes, the Klein pipeline compiled and ran end-to-end again.

### Final result of Attempt 3

The model compiled and executed, but output quality was still poor.

Verification command:

```bash
HF_TOKEN=... ./bazelw run //max/tests/integration/accuracy:verify_pipelines -- \
  --pipeline black-forest-labs/FLUX.2-klein-4B-bfloat16 \
  --devices gpu:0
```

Measured result after the padded-attention attempt:

- `SSIM = 0.354734`
- `LPIPS = 0.584610`
- `MAE = 0.265`

This means the attempt succeeded as an execution-path experiment, but failed as
an accuracy fix.

## Why the `valid_length` Attempt Still Fails Semantically

`valid_length` only tells the kernel that each batch element has a contiguous
valid prefix.

That is not the same as diffusers / HF right-padding execution with an explicit
`attention_mask`.

For Klein/Qwen3, the important difference is:

- padded suffix tokens still exist as query positions
- those padded queries can attend to earlier real key/value tokens
- therefore padded positions can produce nonzero hidden states

The `valid_length` approach is only a contiguous-length approximation.

It is good enough to compile and run, but it does not reproduce the full
behavior of a real `attention_mask` for this model.

## Current Conclusion

The investigation so far supports the following conclusion:

- `FLUX.2-dev` can get away with compact tokens plus zero-padding restore
  because left padding makes the approximation much safer
- `FLUX.2-klein` cannot rely on the same trick
- right zero-padding alone is insufficient
- `valid_length` padded flash attention is also insufficient for parity

So the likely correct next step is:

- implement an explicit mask-aware attention path for the Qwen3 text encoder
- match the HF behavior more directly with
  `qk^T + attention_mask -> softmax -> v`

This will probably be slower than the flash-attention approximation, but it is
the most defensible path if exact or near-exact parity with diffusers is the
goal.

## Useful Commands

Compare saved intermediates:

```bash
python max/tests/integration/tools/compare_flux2_klein_dumps.py
```

Run Klein verification:

```bash
HF_TOKEN=... ./bazelw run //max/tests/integration/accuracy:verify_pipelines -- \
  --pipeline black-forest-labs/FLUX.2-klein-4B-bfloat16 \
  --devices gpu:0
```

## Short Summary

What was tried:

1. reuse the FLUX.2-dev compact-token restore approach
2. switch Qwen3 restore direction to right zero-padding
3. replace zero-padding restore with `valid_length` padded flash attention

Why these were tried:

- each one was the next-cheapest extension of the current compiled path
- each preserved as much of the existing MAX implementation as possible

What was learned:

- Klein is different from FLUX.2-dev because right padding changes padded-token
  hidden-state behavior
- compiled approximations based only on sequence length are not enough
- real attention-mask semantics are likely required
