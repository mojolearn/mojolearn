# Direction-pass diagnostics, NVIDIA L40S, 2026-09-30

lane/neural-net-experiment f7d0f2179, linalg and mamba bindings and the int15 price harness built for sm_89.

## gemm-int8 (board cell 4096^3, 2.46 s vs torch._int_mm 48 ms): the time is Python operand conversion
Board form `(codes, 0)`: `_int8_operand(a codes)` 1253 ms, `_int8_operand(b codes)` 1212 ms, output array 36 ms,
binding 6.1 ms (uploads+alloc 1.5, kernel 1.4, download 3.0). The float32 form (quantize on the device) takes
48.1 ms whole. The kernel is ~98 TOPS; the fix is the codes form's host conversion.

## Mamba-3 backward at the board's Samba shape (B=2 L=512 d_model=384), 37.6 ms per layer
s16_s15 23.7 ms (63%), s17_operands 9.0 ms (24%), angle 2.0 ms (5%), everything else under 2%.
Session counters: 5 backward reuses, 0 recomputes.

## fixed15 plan sweep at mlp_up (MOJOLEARN_INT15_PLAN=0..10), TFLOPS of the tuned product
| row | best plan | best | default dispatch (ceiling run) |
|---|---|---:|---:|
| mlp_up.t512 (512x14336x4096) | 3 | 34.3 | 18.2 |
| mlp_up.t512.bwd_dw | 0 | 52.0 | 47.2 |
| mlp_up.t512.bwd_dx | 0 | 45.2 | 40.6 |
Every plan gives the same digests (6 per row across all 11 plans). Defaults unchanged by this pass.

## Row-block planes quantizer (default) vs MOJOLEARN_INT15_ROW_BLOCK=0, best training operation
| row | row block | parallel | digests |
|---|---:|---:|---|
| qkv.t512 | 43.0 TFLOPS (0.399 ms) | 24.3 (0.706 ms) | same |
| mlp_up.t512 | 14.9 (4.04 ms) | 11.8 (5.08 ms) | same |
| backward rows | unchanged within 1-4% | | same |
