# Exact tiled attention v2 (opt-in foundation)

Status: host arithmetic plus opt-in forward/backward device kernels qualified
on exact Apple, NVIDIA, and AMD fixtures. This does not alter the v1 default;
large-shape evidence shows v2's memory/speed tradeoff is not promotable.

V2 fixes the logical KV tile at 32 elements. Rows visit tiles and cells in
ascending order. Each tile computes its maximum with `identical_fmax`; the
running `(max, denominator, weighted_value)` triple is rescaled exactly once
per logical tile using `portable_exp32(old_max - new_max)`. Every multiply,
add, division, and rescale is rounded to float32 at the spelling represented
by `tools/attention_v2_oracle.py`. Device block size, warp width, and vendor
may not alter this order. Partial tiles behave as if absent cells do not exist.
The weighted-value update is one pinned `fma(weight, value, accumulator)`;
rescaling remains a separately rounded multiply.
Backward dot products and gradient folds likewise use one pinned FMA per
term; exponentiation uses the same `portable_expf` leaf as forward.

This is intentionally not v1 arithmetic. A separating fixture must differ in
bits from materialized v1 while remaining numerically close. V1 remains the
default until v2 has independent Apple, NVIDIA, and AMD columns.

The GPU forward may retain only per-query max, denominator, and output; it may
not allocate score or probability tensors proportional to `L*S`. Backward
must recompute tiles in the same ascending order, first obtaining the fixed
row normalizer, then visiting tiles again to form dQ/dK/dV. dK and dV folds
are logically ordered by ascending query row; atomics and vendor-dependent
partition reductions are forbidden. A first implementation may serialize
that fold before introducing a fixed query-tile tree with its own oracle.

Required promotion gates:

1. CPU oracle repeatability and a required v1/v2 separating fixture.
2. GPU output, max, and denominator bit equality to the oracle across tail
   tiles, signed zero, repeated maxima, extreme exponents, masks, and windows.
3. Backward bit equality for dQ/dK/dV and projection gradients, including a
   recomputation sabotage that must move bits.
4. Prefill/split/decode equivalence inside v2 and repeatability across launch
   geometries and all three GPU columns.
5. End-to-end quality parity reported separately from bit identity.
6. Measured peak memory and time at GPT-3-small-like batch scaling. The memory
   gate compares the linear workspace formula against the materialized
   score/probability pair; no performance default follows from the formula.

Run the current executable stage with:

```sh
python -m unittest tools.tests.test_attention_v2_oracle -v
```

## NI20 selectable model profile, source delivery 2026-10-06

`MOJOLEARN_IDN_ATTN_SOFTMAX=2` selects the new
`attention-online-tile32.fp32.v2` model profile in IDENTICAL builds only;
`MOJOLEARN_IDN_ALL_OFF` suppresses it. The default remains V1. This source has
not been compiled, executed, tested, timed or verified. Prior standalone
qualification prose does not qualify these model adapters.

The native arithmetic source is
`transformer/impl/llama/attention_v2_model_contract.mojo`. It uses the existing
standalone V2 graph with direct model-layout addressing: token-major Q,
context, dContext and dQ; head-major K/V and dK/dV. Online tiles contain 32
consecutive visible logical keys, starting at the row's first visible key.
This is a numerical-profile constant across all vendors, unrelated to GPU
block geometry, dimensions in a benchmark or host worker partitioning. Every
score is an ascending depth FMA chain followed by a rounded multiply by the
existing scale. Each tile updates maximum, rescales denominator/context once,
and consumes its keys ascending. Final context uses the explicit division.

Backward uses the same fixed online normalizer and final-max probabilities,
with the existing standalone V2 closed-form gradient contract. dQ folds keys
ascending; dK/dV have one owning key, folding query heads within the KV group
ascending and then query tokens ascending. They never use atomic adds or
vendor subgroup reductions. The remaining KV slice, RoPE, projection-weight,
normalization and residual derivatives retain their existing paths.

Real device entrypoints are `eager_attention_forward` and
`llama_decoder_layer_backward_device`. The owning forward stages record an
`attn_v2` tag, so backward and later materialization cannot silently use V1.
Native host `transformer_block_oracle` and `transformer_block_backward_oracle`
call the shared native arithmetic. Both optimized Byte host variants,
`block_fast` and `block_par`, call it too; the parallel variant retains its
worker allocation and passes each chunk's absolute query origin. Thus normal
Transformer, ByteTrainer/ByteLayerPool and Samba block callers reach the
profile, including retained native tapes.

Ordinary identity traces remain complete. V2 diagnostic scores are scaled
serial dots; masked absent keys are negative infinity; their exponent and
weight are positive zero. Visible diagnostic probabilities use the final
online maximum/denominator. Context retains its online rescaling rounding and
is not reconstructed from the diagnostic probabilities. Backward diagnostics
likewise describe the V2 derivative, with positive-zero absent contributions.
Host and device use the same diagnostic functions. No trace request changes
the model output's arithmetic.

Existing fixed15 and nonzero attention-softcap profiles keep their original
routes. Nonempty score plants use their existing V1 instrumentation contract;
ordinary trace with no plants uses V2. V1 backward sabotage defines are refused
before V2 writes rather than advertised as valid V2 negative controls. These
instrumentation exclusions must be represented in later qualification scope.
Legal model rows always include their own causal key; standalone arbitrary
all-masked-row semantics are not broadened by this integration.

Byte checkpoint names include `-attention-online-tile32-v2`. Samba's native
`training_experiment_profile` includes it, with mismatch refusal on load.
Native Transformer extensions export `transformer_attention_profile` for
artifact provenance. Baseline/candidate A/B must hold other numerical flags
fixed before testing combinations. Complete host/NVIDIA/AMD/Apple identity,
training quality, finite/refusal behavior, trace stages, causal/window/decode
behavior, odd/tail layouts, GQA gradients, and complete NVIDIA+AMD workload
measurements remain required before any promotion. No default was promoted.
