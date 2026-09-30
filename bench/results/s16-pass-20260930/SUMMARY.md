# S16 pass (PR #6, lane/neural-net-experiment ba7cfd4d6) on NVIDIA L4 (2x L4 pod nvc1), 2026-09-30

Job `tools/s16_job.sh` (nvc1-0001). Overlay: mojolearn 0.8.31, Python sources and eight IDENTICAL bindings from the head.

## Bits
- `tools/probe_ftz_hw.mojo`: PASS over all 2^32 float32 words (non-NaN words that differ: 0; NaN payload words: 16,777,213, refused by the contract).
- Mamba-3 y and all 10 gradients, two calls, B2 L512 d384 and B8 L512 d768: identical for S16 arms regs2, regs, shared, smem48, naive; the angle-naive build; and the `MOJOLEARN_FTZ_HW_OFF=1` build. Equal to the L40S strides-pass digests.
- Neural board baseline digests (`--set nvidia`) equal with the flush on and off.
- Classical (LU 1024, SGD-reg/clf 20000, LARS 200000, IVF 40000) digests equal to the classical-pass L40S evidence on main.
- Kernel PCA 256/1000/10000 output sha256 equal to the NVIDIA L40S evidence on main (which equals Apple M4 and CPU).
- gemm-int8 digest 9b16c7064e10cecd, as every earlier run. Session checks pass.

## Time (L4)
| | before | after |
|---|---|---|
| S16 qk+s15, board shape (naive -> regs2 / regs) | 65.1 ms | 10.8 / 9.2 ms |
| S16 qk+s15, default shape (naive -> regs2 / regs) | 567 ms | 86.0 / 83.5 ms |
| angle stage, board shape (naive -> staged) | 6.3 ms | 0.8 ms |
| Mamba-3 backward sum of stages, board shape (flush off -> on) | 62.3 ms | 36.4 ms |
| samba-train-step (flush off -> on) | 190.8 ms | 148.7 ms |
| transformer-forward (flush off -> on) | 6.54 ms | 5.43 ms |
| samba-forward / lm-forward / lm-train-step (flush off -> on) | 16.3 / 91.0 / 116.0 ms | 14.9 / 85.9 / 109.1 ms |

`regs` is 3-15% faster than the `regs2` default on the L4 (same bits). AMD and Apple: not run.
