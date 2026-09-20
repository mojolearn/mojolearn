# Exact norm2 backward residual fan-in fusion

Apple M4, IDENTICAL. The final RMSNorm backward cell kernel already computes
and stores the recorded `norm2_dx`. It now optionally performs the immediately
following residual fan-in from that same rounded local value while also writing
the separately recorded `d_residual1`. The operation remains exactly
`ftz(ftz(norm2_dx) + ftz(in_d_residual2))`; no reduction or arithmetic order
moves. This removes one launch and one complete reread of `norm2_dx` per
transformer layer.

## Exactness

The production and `MOJOLEARN_BWD_NORM2_RESIDUAL_SPLIT_TRIAL=1` control builds
both passed the transformer backward gate: 17 cases, 37/37 stages and 412,172
cells bit-identical to the host reference. Their complete 37-stage trace SHA
was identical:

```
a4166b19a0af95ecb9f4978ac8476f61035aa98474c2a449e4ffe97679ec68c8
```

`pixi run check-train-step` also passed every clause, including all thirteen
device/oracle stages, eight-step composition, determinism and optimizer state.

## Repeated-stage price

`bench/norm2_residual_backward_price_main.mojo` times the old two-launch seam
against the fused spelling using resident buffers. Nine synchronized calls per
arm produced these medians:

| shape | split ms | fused ms | speedup | isolated saving across 12 layers |
|---|---:|---:|---:|---:|
| 2048x768 | 2.312 | 1.192 | 1.940x | 13.440 ms |
| 8192x768 | 4.250 | 2.318 | 1.833x | 23.184 ms |
| 32768x768 | 13.976 | 9.102 | 1.535x | 58.488 ms |

These are isolated Apple stage timings, not whole-step timings. The portable
kernel compiles for every backend, but production routing is Apple IDENTICAL
only until NVIDIA and AMD run the same alternating benchmark. Other vendors,
FAST, and the explicit split trial retain the old two-launch path.
