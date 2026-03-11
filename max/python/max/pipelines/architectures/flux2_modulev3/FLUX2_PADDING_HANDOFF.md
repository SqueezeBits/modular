# FLUX.2 Prompt Padding Handoff

Date: 2026-03-10

Update: 2026-03-11

## Purpose

This document explains why the recent FLUX.2 prompt-padding change was made,
how it was implemented, what assumptions it depends on, and what a future
session should know before extending or revisiting it.

The main goal of this change is to preserve correct prompt sequence length for
FLUX.2 prompt embeddings after the old prompt-embedding preparation path was
removed and its logic was integrated into the text encoder.

## Background

There was an earlier attempt to fix FLUX.2 prompt padding in
`https://github.com/modular/modular/pull/6013`.

That older approach worked against a previous structure where:

- the pipeline still had a separate prompt-embedding preparation step
- selected text-encoder hidden states were returned to the pipeline
- the pipeline itself could pad those hidden states back to the tokenizer's
  padded sequence length

That structure no longer exists on current `main`.

Today, the Mistral3 text encoder used by `Flux2Pipeline` already returns fused
prompt embeddings directly. The selected hidden states are stacked and flattened
inside the text encoder implementation, not in the pipeline.

As a result, the old PR could not be applied directly, because the previous
insertion point for padding no longer exists.

## Original Problem

In the current FLUX.2 path, the tokenizer still produces padded tokens, but the
FLUX.2-specific logic in `pixel_tokenizer.py` compacts the token IDs down to
only valid tokens:

- `input_ids_array = input_ids_array[attention_mask_array]`
- `attention_mask_array` is then replaced with all-ones

This means the downstream execution path loses the original padded prompt
length. Once the prompt is compacted, the text encoder only sees the valid token
count, not the tokenizer's padded target length.

That creates a mismatch:

- the diffusion transformer expects prompt embeddings aligned to the padded
  prompt length
- the text encoder only sees the compact prompt length
- the old pipeline-level logic that used to re-pad embeddings no longer exists

## Design Options Considered

### Option 1: Reintroduce padding with the old mask-based pipeline approach

This was the closest match to PR 6013, but it no longer fits the current code
structure well because prompt-embedding fusion now happens inside the text
encoder.

### Option 2: Use `ArchConfig.get_max_seq_len()` as the source of truth

This is now the intended source of truth.

Current reason:

- `PipelineRegistry` already asks the architecture config for pixel tokenizer
  max length
- `Flux2ArchConfig.initialize()` now resolves that length and writes it back to
  `pipeline_config.model.max_length`
- the pipeline and text encoder can therefore reuse the same resolved value
  without keeping a separate Flux2-only default path

### Option 3: Keep compact tokens, but let the text encoder restore padded
prompt length from a configured target sequence length

This is the approach that was implemented.

It matches the current architecture better because:

- the tokenizer can stay compact for execution
- the text encoder remains the owner of fused prompt embeddings
- the pipeline receives already-restored prompt embeddings and only needs to use
  their final sequence length

## Final Design

The adopted design is:

1. Keep the FLUX.2 tokenizer behavior that compacts valid tokens.
2. Define the prompt padding target as the tokenizer's configured max length for
   FLUX.2.
3. Store that target length on the Mistral3 text encoder wrapper.
4. After the text encoder produces fused prompt embeddings for compact tokens,
   left-pad them back to the target sequence length.
5. Make the FLUX.2 pipeline derive `text_ids` from the restored embedding
   length, not from the compact input token length.

## Important Assumption

This implementation assumes that `black-forest-labs/FLUX.2-dev` uses left
padding.

That assumption is intentionally documented in code comments, but not inferred
dynamically at runtime.

This is acceptable for the current requested implementation, but it is also the
main limitation of the design:

- if a future FLUX.2 tokenizer changes its padding side
- or if a new related model does not share the same padding behavior

then the current hardcoded zero-padding logic will be wrong.

In that case, the safer future solution is to preserve a real padding mask and
restore embeddings by mask-aware scatter instead of hardcoded per-model
padding.

## Files Changed

### 1. `max/python/max/pipelines/architectures/flux2_modulev3/arch.py`

Changed the Flux2 architecture config so that `get_max_seq_len()` resolves the
effective prompt length and synchronizes it back into
`pipeline_config.model.max_length`.

Behavior now:

- `Flux2Pipeline` and `Flux2KleinPipeline` default to 512
- `pipeline_config.model.max_length` still overrides the 512 default if
  explicitly provided
- the tokenizer path and pipeline path both see the same resolved max length

Why this matters:

- this makes the architecture config the actual source of truth for FLUX.2
  prompt length
- the text encoder can use the same resolved target length that the tokenizer
  used

### 2. `max/python/max/pipelines/architectures/mistral3/text_encoder/model.py`

Added:

- `target_prompt_seq_len` property backed by text-encoder config

Behavior now:

- the text encoder wrapper reads `target_prompt_seq_len` directly from its
  initial config during `__init__`
- the wrapper compiles once with that target length already baked into the
  transformer config
- it still preserves an eager validation check for compact token length
  exceeding the configured target

Why this matters:

- prompt padding no longer runs eagerly after text encoding
- the compiled text encoder owns both fused embedding creation and padding

### 3. `max/python/max/pipelines/architectures/mistral3/text_encoder/mistral3.py`

Added:

- `target_prompt_seq_len` as a compile-time transformer config field
- `_pad_prompt_embeddings()` inside the transformer module

Behavior now:

- after the selected hidden states are stacked and flattened, the transformer
  optionally restores right padding inside the compiled graph for Qwen3 / Flux2
  Klein
- the pad width is expressed from the symbolic compact sequence length and the
  configured target prompt length

Why this matters:

- the prompt-length restoration step now lives inside the compiled text-encoder
  forward path itself
- this removes the previous eager post-processing boundary
### 4. `max/python/max/pipelines/architectures/flux2_modulev3/pipeline_flux2.py`

Added:

- `default_text_encoder_prompt_seq_len = 512`
- `_resolve_text_encoder_prompt_seq_len()`
- `_get_component_config_dict()` override for text-encoder config injection

Updated behavior:

- before submodels are constructed, the pipeline injects the resolved target
  prompt length into the text-encoder `config_dict`
- the value is resolved from `pipeline_config.model.max_length` if provided,
  otherwise from the default 512
- this applies to both the Mistral3 and Qwen3 Flux2 text-encoder wrappers
  because both are loaded through the same `text_encoder` component path
- `prepare_prompt_embeddings()` no longer assumes that input token length is the
  final prompt embedding length
- instead, it computes `seq_len` from `prompt_embeds.shape[1]` after text
  encoding

Why this matters:

- before this change, `prepare_prompt_embeddings()` used the compact token count
  as the sequence length
- after the change, it uses the restored padded embedding length
- this keeps `text_ids` aligned with the actual prompt embeddings seen by the
  diffusion transformer

### 5. `max/python/max/pipelines/architectures/qwen3/text_encoder/model.py`

Added:

- `target_prompt_seq_len` property backed by text-encoder config

Behavior now:

- the Qwen3 text encoder wrapper reads `target_prompt_seq_len` directly from
  its initial config during `__init__`
- it compiles once with padding behavior already expressed in the transformer
  config

Why this matters:

- Flux2 Klein can now follow the same compact-token then restore-padding path
  as Flux2
- the text-encoder contract is now aligned across the two Flux2 variants

### 6. `max/python/max/pipelines/architectures/qwen3/text_encoder/qwen3.py`

Added:

- `target_prompt_seq_len` as a compile-time transformer config field
- `_pad_prompt_embeddings()` inside the transformer module

Behavior now:

- after the selected hidden states are stacked and flattened, the transformer
  optionally restores right padding inside the compiled graph
- the pad width is expressed from the symbolic compact sequence length and the
  configured target prompt length

Why this matters:

- Flux2 Klein no longer depends on padded tokens surviving until the text
  encoder
- prompt-length restoration now lives inside the compiled Qwen3 path itself

### 7. `max/python/max/pipelines/lib/pixel_tokenizer.py`

Changed:

- FLUX.2-Klein now follows the same compact-token behavior as FLUX.2

Behavior now:

- tokenizer output is padded to max length
- FLUX.2 and FLUX.2-Klein both compact tokens down to valid positions
- attention mask is replaced by all-ones after compaction

## End-to-End Flow After This Change

1. `PipelineRegistry` constructs the pixel tokenizer.
2. For FLUX.2 / FLUX2-Klein, `Flux2ArchConfig.get_max_seq_len()` resolves the
   primary tokenizer max length to 512 by default, unless overridden by
   `pipeline_config.model.max_length`.
3. `PixelGenerationTokenizer.encode()` pads to that max length.
4. FLUX.2 and FLUX.2-Klein compact-token logic remove padding tokens before
   execution.
5. `Flux2Pipeline` configures the Flux2 text encoder with the same resolved
   prompt length stored by the architecture config.
6. The compiled text encoder recompiles with that target prompt length baked
   into its graph.
7. The compiled text encoder produces fused prompt embeddings for the compact
   tokens and restores them back to the target padded prompt length.
   Mistral3 / FLUX.2-dev uses left padding, while Qwen3 / Flux2 Klein uses
   right padding.
8. `Flux2Pipeline.prepare_prompt_embeddings()` uses the restored embedding
   sequence length to create `text_ids`.
9. The diffusion transformer receives prompt embeddings and `text_ids` that are
   aligned to the padded prompt length again.

## Why This Change Is Reasonable

This change is not a perfect upstream-faithful reproduction of the original
FLUX.2 text-encoder behavior.

The official implementation keeps padded `input_ids` and `attention_mask`
together through the text-encoder forward pass.

This MAX-side implementation instead does:

- compact-token execution
- post-encoding restoration to padded length

That is a pragmatic compromise because:

- it works with the current MAX code structure
- it avoids reopening the old hidden-state preparation path
- it requires fewer changes than threading real attention masks through the
  current Mistral3 text-encoder execution path

## Known Limitations

1. The approach depends on model-specific padding side.

Currently:

- Mistral3 / FLUX.2-dev assumes left padding
- Qwen3 / Flux2 Klein assumes right padding

If padding side changes, the current implementation is no longer correct.

2. The tokenizer still discards the original padding mask for FLUX.2.

This means the system cannot currently recover arbitrary padding layouts.

3. The solution is specific to the current Flux2 text-encoder wrappers.

It should not be blindly copied to other text encoders unless the same compact
token and padding-side assumptions hold.

## Recommended Future Improvements

If a future session wants a more robust solution, the next step should be:

- preserve the original padding mask through FLUX.2 tokenization
- keep compact-token execution if desired
- restore fused embeddings using the mask instead of hardcoded per-model
  padding

That would remove the dependency on hardcoded `padding_side` assumptions and
make the system more portable across related models.

If strict upstream parity becomes important, the stronger follow-up would be:

- pass padded `input_ids` and a real `attention_mask` through the text encoder
- let the encoder attend with the same mask that the tokenizer produced

That is a larger change than the current patch.

## Validation Performed

The following validation was performed after the code changes:

- `python -m py_compile` on the modified files

No end-to-end FLUX.2 runtime inference test was executed as part of this
handoff.

## Quick Summary For The Next Session

If you need to continue this work, the main facts are:

- the old PR 6013 pipeline padding approach no longer fit current `main`
- the fix was moved to the Mistral3 text encoder wrapper
- FLUX.2 prompt length now comes from tokenizer max length, defaulting to 512
- prompt embeddings are restored by model-specific zero-padding after text
  encoding
- the pipeline now trusts `prompt_embeds.shape[1]`, not compact token length
- the design is intentionally pragmatic, not fully mask-faithful
- the main risk is the hard dependency on architecture-specific padding side
