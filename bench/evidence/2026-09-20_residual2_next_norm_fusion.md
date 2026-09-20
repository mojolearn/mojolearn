# Exact residual2 to next-block RMSNorm fusion

The ByteLM training and logits orchestrators now let block `i`'s final
residual kernel also produce block `i+1`'s RMSNorm1 buffers. The kernel still
materializes `residual2`, so the trace and backward retain the same state;
the next block skips only the norm launch that has already run. The fused
kernel preserves the residual add spelling and the serial ascending RMS
sum-of-squares fold.

There is no separate final normalization in the current ByteLM architecture:
the last block's `residual2` feeds the LM head directly, so the final block
keeps the ordinary residual kernel.

## Route and timing

The automatic route is Apple-only, plain RMSNorm without bias, and
`M <= 2048`. `MOJOLEARN_DISABLE_RESIDUAL2_NEXT_NORM` provides an explicit
baseline build. Other vendors, LayerNorm/bias variants, and larger shapes use
the unchanged two-kernel path.

`bench/residual_norm_price_main.mojo` measures the same residual-plus-norm
boundary twelve times at `d_model=768`. Stable Apple M4 repetitions after
warmup gave about 6.24 ms baseline versus 5.33 ms fused at `M=2048`, a 14.5%
boundary win. A twelve-layer model has eleven cross-block boundaries, or
about 0.83 ms saved by this measurement. At batch 16 (`M=32768`) the medians
were 93.14 versus 102.58 ms, a 10.1% regression; this shape is routed to the
baseline.

All three dedicated hashes matched:

- `M=2048`: residual `6316337721413083941`, sumsq
  `4193963273563581572`, output `15384826769618229439`.
- `M=32768`: residual `5723610026210370341`, sumsq
  `7762877517947080980`, output `5881589094275598053`.

## Exact production gate

`training/checks/train_step_check.mojo` uses two layers, so it exercises one
cross-block fusion through the complete forward, backward, optimizer, trace,
checkpoint, split-composition, and determinism paths. The fused and explicit
disabled builds produced the same clean digest `463245ce6c97e68d` and the
same negative-control and optimizer digests. A clean sequential fused run
passed every clause with the required 34 trace records. (Two concurrent gate
runs share a fixed trace path and therefore must not be used as a hygiene
gate.)
