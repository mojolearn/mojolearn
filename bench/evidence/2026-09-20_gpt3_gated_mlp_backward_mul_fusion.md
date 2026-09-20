# GPT-3-small gated-MLP backward dual-product fusion

## Scope

- Base: `0164644ff`.
- Mode: explicit `MOJOLEARN_NUMERIC_IDENTICAL`, FP32.
- Hardware: Apple M4 Max local GPU; no cloud resource used.
- Production stage: S21 backward in
  `transformer/checks/transformer_backward.mojo`.
- GPT-3-small intermediate width: 3,072; token rows 2,048, 8,192, and
  32,768 (6,291,456 through 100,663,296 cells).

The baseline launches `bwd_mul_kernel` twice:

```
d_silu = d_gated * up
d_up   = d_gated * silu
```

The candidate performs those same two independent `ftz(pinned_mul(...))`
expressions in one thread and launch. It shares only the bit-copy load of
`d_gated`; neither product consumes or changes the other. Arithmetic order,
rounding, output storage, and all later consumers are unchanged.

## Interleaved A/B

Each row below is nanoseconds. Baseline and fused order alternated over seven
post-warm-up rounds. Complete baseline and candidate outputs were compared by
`uint32` bits before timing; all three shapes had zero mismatches.

| rows | baseline samples | fused samples | median fused / baseline |
|---:|---|---|---:|
| 2,048 | 2318000, 3056000, 3220000, 3642000, 3247000, 2435000, 2414000 | 1886000, 2020000, 3204000, 3363000, 1953000, 1826000, 1914000 | 0.639 |
| 8,192 | 8624000, 7963000, 7423000, 7542000, 9925000, 8029000, 7323000 | 10484000, 6590000, 6473000, 6471000, 6206000, 6716000, 6006000 | 0.813 |
| 32,768 | 73445000, 32532000, 35124000, 31783000, 29384000, 34164000, 34071000 | 24488000, 25802000, 26569000, 28361000, 23745000, 24918000, 26474000 | 0.757 |

The stage removes one launch. Per cell, its explicit global traffic falls
from two shared-input reads plus two other reads and two writes (24 bytes) to
one shared-input read plus two other reads and two writes (20 bytes), a 16.7%
reduction. Retained and scratch storage are unchanged.

## Exactness and integration gates

Commands used `MOJOLEARN_BUILD_JOBS=2` and explicit identical defines.

```
pixi run mojo run -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  transformer/checks/transformer_backward_check.mojo
```

Passed 17 fixture cases, all 37 backward stages, and all 412,172 compared
cells bitwise against the host oracle. Launch- and batch-order contracts in
the unchanged downstream GEMMs remain covered by that gate.

```
pixi run mojo run -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  training/checks/byte_lm_head_v2_integration_check.mojo
```

Passed `BYTE_LM_HEAD_V2_INTEGRATION_OK`, including the repeated ByteTrainer
update check. No dropout path was altered: production transformer training
continues to refuse dropout, so its seed and mask contract is unchanged.

## Adjacent audit disposition

- Adam/AdamW is already one production launch; denominator and quotient
  scratch collapse to one cell unless intermediate recording is explicitly
  requested. No fusion was warranted.
- Biased GELU and biased SiLU forward epilogues are already fused.
- Residual-plus-normalization fusion was not reopened: the existing audit
  found it cannot preserve the pinned row reduction behavior. This change
  instead fuses only two independent pointwise products.
