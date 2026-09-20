# Gated-MLP backward fusion (Apple IDENTICAL)

The S21 gate-product backward and clean S20 SiLU backward now share one
portable kernel on the measured Apple IDENTICAL route.  The kernel still
materializes `d_silu`, `d_up`, and `d_gate`.  Its local `d_silu` value is
rounded with the same `ftz(pinned_mul(...))` spelling before both the store
and the downstream multiply, preserving the split store/load seam's bits.
Sabotage, FAST, NVIDIA, and AMD builds retain the split route.

## Exactness gates

- Clean fused transformer backward: 17 fixtures, 37/37 stages and 412,172
  cells bit-identical to the host reference.
- Fused and `MOJOLEARN_BWD_GATED_SILU_SPLIT_TRIAL=1` trace SHA-256:
  `a4166b19a0af95ecb9f4978ac8476f61035aa98474c2a449e4ffe97679ec68c8`.
- `MOJOLEARN_BUILD_JOBS=2 pixi run check-train-step`: all clauses green;
  all thirteen oracle stages identical and same-device determinism green.

## Timing

`pixi run mojo run -I . bench/gated_silu_backward_price_main.mojo`, Apple
GPU, seven alternating whole-call A/B samples per shape. Values are median
milliseconds; each shape is token rows by GPT-3-small intermediate width.

| shape | split | fused | speedup | saving/layer |
|---|---:|---:|---:|---:|
| 2048 x 3072 | 3.133 | 2.871 | 1.091x | 0.262 ms |
| 8192 x 3072 | 10.090 | 8.874 | 1.137x | 1.216 ms |
| 32768 x 3072 | 39.797 | 34.064 | 1.168x | 5.733 ms |

At 32,768 token rows the isolated weighted saving is 68.796 ms across 12
layers.  Backend routes remain conservative until separately measured.
