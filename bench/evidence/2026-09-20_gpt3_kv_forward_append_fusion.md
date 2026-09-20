# GPT-3-small K/V forward-cache append fusion

## Scope and invariant

- Base: `429862da5`; explicit `MOJOLEARN_NUMERIC_IDENTICAL`, FP32.
- Hardware: local Apple M4 Max; no cloud resource used.
- Path: non-windowed transformer forward cache append.
- Shapes: GPT-3-small K/V width 768 at 2,048, 8,192, and 32,768 rows.

The old path launched `kv_append_kernel` separately for K and V. Both calls
classified the same output cache cell as old-cache or fresh-token data and
computed the same source index; only the source and destination pointers
differed. `kv_append2_kernel` computes that integer routing once, then performs
the original two independent bit copies. It performs no floating-point
operation and changes no reduction, cache layout, or lifetime.

Launches fall from two to one. Explicit global traffic remains two reads and
two writes (16 bytes per cell pair), while the division/modulo-heavy cache
index calculation is halved. Storage is unchanged. Windowed gather/ring paths
remain untouched pending their own production-shape qualification.

## Interleaved production-shape A/B

Nine post-warm-up alternating samples per shape, nanoseconds. Both complete
K and V outputs were compared as `uint32` bits before timing; all mismatch
counts were zero.

| rows | baseline samples | fused samples | median fused / baseline |
|---:|---|---|---:|
| 2,048 | 4205000, 2736000, 2773000, 2693000, 2809000, 2785000, 2745000, 2667000, 2719000 | 1591000, 1599000, 1575000, 1602000, 1555000, 1837000, 1571000, 1551000, 1593000 | 0.580 |
| 8,192 | 10955000, 11009000, 10666000, 10301000, 9835000, 9690000, 9687000, 11933000, 9675000 | 5815000, 5689000, 5712000, 5464000, 5606000, 5200000, 5173000, 5281000, 7514000 | 0.544 |
| 32,768 | 34438000, 27584000, 28329000, 31229000, 34052000, 36461000, 38926000, 37883000, 35878000 | 14018000, 14130000, 14694000, 17676000, 22448000, 19928000, 29156000, 35847000, 23904000 | 0.579 |

Every measured round improved. Median stage time falls 42.0--45.6%.

## Gates

All commands used `MOJOLEARN_BUILD_JOBS=2` and an explicit identical define.

- `transformer/checks/transformer_check.mojo`: passed 17 fixtures, 30/30
  forward stages, 349,206 compared cells, and 30/30 trace records. The matrix
  includes linear cache-hot-tail and four windowed cases; window routing is
  therefore verified unchanged.
- `transformer/checks/transformer_backward_check.mojo`: passed 17 fixtures,
  37/37 stages, 412,172 cells, and 37/37 trace records.
- `training/checks/byte_lm_head_v2_integration_check.mojo`: passed
  `BYTE_LM_HEAD_V2_INTEGRATION_OK`, including repeated ByteTrainer updates.

## Rejected adjacent expansion

The window gather and ring write also occur in K/V pairs, but GPT-3-small
training uses the linear cache path measured here. They were left unchanged:
combining three cache operations in one patch would broaden regression risk
without a representative windowed production benchmark. Q/K RoPE likewise
remains canonical and unfused because GQA gives Q and K different widths and
duplicating its audited arithmetic/sign controls would create a second
numerical spelling.
