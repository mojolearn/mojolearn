# Kernel map of one T3 optimizer step (written before any rental)

Shape: batch 4, length 2048 (M = 8,192 token rows a shard), d_model 768, 12
heads of 64, FF 2,048, 12 layers, vocabulary 50,257, K = 64 shards a step.
Source read: `python/mojolearn/parallel_training.py`,
`bindings/_mojolearn_byte_lm.mojo`, `training/byte_lm.mojo`,
`training/byte_lm_parallel.mojo`, `transformer/impl/llama/modeling_llama.mojo`,
`transformer/checks/transformer_backward.mojo`,
`transformer/impl/llama/fused_attention.mojo`,
`gemm/checks/gemm_identical.mojo`, `gemm/checks/gemm_backward.mojo`,
`checks/kernel_matrix.mojo`.

## The step

One optimizer step = 64 x `shard_gradient_fold` (one `byte_gradient_device`
at the shard's ids, then an ordered add of the shard's 162,147,840-float
gradient into the device total) + one AdamW update + the host hashes of the
state and the summed gradient. Everything below is PER SHARD unless marked.

| phase | kernels (launches per shard) | GEMM FLOP per shard | vendor-specific path on AMD? |
|---|---|---|---|
| ids upload, weight unpack | 2 H2D, 12 block copies + 2 copies | 0 | shared |
| embedding forward | 1 gather kernel | 0 | shared |
| per layer x12, forward | RMSNorm (fused with the previous residual), q/k/v/o projections (4 GEMM), rope + kv append, fused attention forward (`fwd_r2`), corner flag, regime scan, residual add, RMSNorm, gate/up (2 GEMM), silu, gate product, down (1 GEMM), residual add; one host sync per layer | 12 x 116.0 GF = 1.39 TF | GEMM: `_shipped_body_kpack_hg` sends every k=768 call with m >= 4096 to the TUNED 128x128 kernel on AMD ONLY (NVIDIA runs the kpack body); attention: AMD's default arm word lacks `_bswz` (NVIDIA has it) |
| head forward | 1 GEMM 8192 x 50257 x 768 | 0.632 TF | GEMM as above (TUNED 128x128 on AMD) |
| cross entropy forward/backward | refuse scan, 13 forward kernels, 3 backward kernels over 8192 x 50257 cells | 0 | shared |
| head backward | dA (8192 x 768, k = 50257), dB (50257 x 768, k = 8192) | 1.265 TF | kpack body, group rule on long k (S = 110 on AMD, 132 on NVIDIA) |
| per layer x12, backward | down dA/dB, silu/gate backward, gate/up dA and dB (4 GEMM), fan-in, RMSNorm backward (+ its dW GEMM), o dA/dB, fused attention backward (`dq_tiled_pf`, `kvgrid_dkdv_pf`, `zdot_estash_dres_pf`, corner flag, regime scan), rope backward, q/k/v dA and dB (6 GEMM), fan-in, RMSNorm backward, residual add; one host sync per layer | 12 x 231.9 GF = 2.78 TF | GEMM dA at k = 768 -> TUNED on AMD, the rest kpack; attention as forward |
| embedding backward | PLAN_SCAN at M = 8192 (< 16384): 50,257 x 8,192 integer probes | 0 | shared |
| pack grads, fold add | 12 block copies + 2 copies, 1 ordered add of 162 M floats | 0 | shared |
| once per step | AdamW (optskip_noshadow_rows16 glue arm on AMD and NVIDIA), device validation scans, export + host sha256 | 0 | shared |

GEMM total 6.07 TFLOP a shard, 388 TFLOP a step. At the measured 139 s a step
(2.17 s a shard) that is at most 2.8 TFLOP/s if GEMM were the whole shard.

## The GEMM kernels and what differs by column

Both GEMM kernels (`identical_gemm_tuned_kernel` and
`identical_gemm_kpack_kernel`) are ONE source for every column; the column
enters only through kernel-matrix rows:

- the per-step seam `_tuned_step`: NVIDIA `fma.rn` + `mul.rn.ftz` (2 issue
  slots); AMD `v_fmac` + post-round class flush (5 slots with a hazard nop,
  measured 2026-09-18); Apple a software flush plus a block admission;
- the fold flush: hardware `mul.rn.ftz` on NVIDIA, software `ftz` elsewhere;
- ksplit block parallelism S: 132 NVIDIA, 110 AMD (the MI250X CU count; the
  MI300X/MI325X have 304);
- shared-memory pages: two 20,480-byte pages per 128x128 block on both, which
  on AMD's 64 KB LDS allows ONE resident block (4 waves, 1 wave per SIMD) per
  CU;
- the AMD-only dispatch override for k = 768 (TUNED kernel instead of kpack).

Every change this lane makes to them is behind an AMD row; NVIDIA compiles the
same lines it compiled before, and its bits must still be re-proven before a
release (owed, not rented here).

## The last itemization on AMD (history, B1/L2048, MI300X, 2026-09-18)

`2e7d346a5^:bench/results/e1g/2026-09-18_013257-amd-mi300x-hotaisle-step-breakdown/remote/step-breakdown/breakdown.tsv`:
real step 688 ms; GEMM 565 ms (81 percent) at 2.72 TFLOP/s and FLAT across
call shapes (2.51 to 2.85); attention kernels 92 ms (13 percent); everything
else 37 ms. The H100 on the same instrument: 207 ms, GEMM 119.5 ms (58
percent) at 12.84 TFLOP/s. Since then AMD took the class flush (-8.4 percent
of the step).

## Measured on this lane

(filled in from the box: `profile/`)
