# Exact tiled attention v2 (opt-in foundation)

Status: host arithmetic and executable gates only. This does not alter the v1
default and makes no GPU-performance or cross-device claim.

V2 fixes the logical KV tile at 32 elements. Rows visit tiles and cells in
ascending order. Each tile computes its maximum with `identical_fmax`; the
running `(max, denominator, weighted_value)` triple is rescaled exactly once
per logical tile using `portable_exp32(old_max - new_max)`. Every multiply,
add, division, and rescale is rounded to float32 at the spelling represented
by `tools/attention_v2_oracle.py`. Device block size, warp width, and vendor
may not alter this order. Partial tiles behave as if absent cells do not exist.

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
