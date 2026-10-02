# PR #78 byte-LM host inference kernels (lane/neural-pass73)

CPU-only change in `training/byte_lm_host.mojo` / `byte_lm_host_kernels.mojo`. Main vs lane, two round sets each; digests equal
(lm-infer ee872b5ebd23a4e0).

| Box | lm-infer main ms | lm-infer lane ms | samba-infer (untouched) | transformer-infer (untouched) |
|---|---|---|---|---|
| EPYC 9374F (RunPod nvc3, 13-thread quota; main 0ddf81e8b, lane 28bd7bf23; R2 `measurements/2026-10-01/pr78-nvc3.tar.gz`) | 311.0 / 230.3 | 235.7 / 205.4 | 192.9 / 221.5 vs 195.3 / 199.8 | 30.0 / 32.1 vs 20.7 / 25.2 (noise) |
| M2 Pro (lane 9d3b5532a, R2 `pr78-cpu-m2pro`) | 356 / 362 | 225 / 231 | | |

Torch CPU on the capped EPYC (13 threads), eager-fp32 / compile-fp32 / compile-bf16 ms: lm-infer 80.8 / 54.5 / 34.5,
samba-infer 96.8 / 47.4 / 40.9, transformer-infer 10.3 / 6.4 / 5.8. These replace the throttled (128-thread) board rows.

Same bits; faster on the EPYC (the deciding box for a CPU change) and the M2 Pro: merged.
