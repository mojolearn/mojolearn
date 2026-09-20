# Exact residual1 + RMSNorm2 fusion

The GPT/transformer forward used two kernels for S22 followed by S1--S4:
materialize `residual1 = x + o_proj`, then read `residual1` twice in RMSNorm.
`residual_rms_norm_kernel` retains the materialized residual required by the
trace and backward, but feeds the identically spelled add directly into the
serial ascending sum-of-squares fold. It removes one launch and one complete
read of `[M, d_model]`; every per-element operation and reduction order is
unchanged.

Apple routing is deliberately limited to the measured GPT-3-small batch-one
regime (`M <= 2048`, plain RMSNorm without bias). Other columns and larger
Apple shapes retain the two-kernel path. The explicit
`MOJOLEARN_FUSE_RESIDUAL1_NORM2` define remains a qualification override.

## Apple M4 timing

`bench/residual_norm_price_main.mojo` runs the boundary twelve times at
`d_model=768`, with 11 measured repetitions after warmup. Residual, sumsq and
normalized-output digests matched exactly at every shape.

| M / GPT batch at L2048 | baseline median ms | fused median ms | result |
|---:|---:|---:|---:|
| 2048 / 1 | 6.425 | 5.311 | 17.3% faster |
| 4096 / 2 | 11.613 | 11.528 | neutral, +0.7% |
| 8192 / 4 | 22.435 | 22.674 | neutral, -1.1% |
| 32768 / 16 | 92.693 | 102.567 | 10.7% slower; routed to baseline |

The large-shape result is why this is a measured Apple shape route rather
than a global structural default. The repeated-stage batch-one saving is
about 1.114 ms across twelve layers, or 0.093 ms per layer.

## Exactness gates

- `transformer_check.mojo`: 17 cases, 30/30 stages and 349,206 cells exact.
- `transformer_backward_check.mojo`: 17 cases, 37/37 stages and 412,172 cells
  exact, proving all retained forward activations and resulting gradients.
- Dedicated GPT-small boundary hashes:
  residual `6316337721413083941`, sumsq `4193963273563581572`, output
  `15384826769618229439`, equal baseline/fused.

Bias-GELU and bias-SiLU were already fused in production. Dropout remains
outside the transformer profile, so neither was duplicated in this change.
