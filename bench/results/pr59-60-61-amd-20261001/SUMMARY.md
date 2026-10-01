# PR #59, #60, #61 on the MI325X: main against each branch (Oct 1)

Job: `tools/ab_job.sh amd p596061` (lane/ab-measure). Each tree's neural bindings were built from source over the released 0.8.33 wheel and timed twice, interleaved, plus one phase-timer build per tree.
Raw files are in R2 at `measurements/2026-10-01/ab-p596061-amd.tar.gz`.

**Base caveat.** The three branches start from main 1f776f534, which is before #55 (AMD GEMM tile minimum). The `main` arm is today's main, which includes #55. So the end-to-end lm-train-step of each branch should be compared with the pre-#55 baseline (100.6 ms in pr57-pr58-gpu-20261001), not with the 75 ms main arm. The attention phase timers are not affected by #55. From now on, ab_job merges origin/main into each branch first.

Every arm gave lm-forward digest 4a8e781b0739a038 and the same loss trace: **same bits**.

## lm-train-step (ms, two runs)

| arm | run 1 | run 2 | vs pre-#55 base 100.6 |
|---|---|---|---|
| main (with #55) | 74.9 | 75.1 | n/a |
| #59 4-wide fwd/dq reads | 95.3 | 95.5 | -5% |
| #60 4-wide dkdv/zdot reads | 99.2 | 99.1 | -1.5% |
| #61 AMD MODE flush | 91.3 | 91.6 | -9% |

## Attention backward phases (ms per layer, phase-timer build)

| arm | bwd_dq_tiled_pf | bwd_kvgrid_dkdv_pf | bwd_zdot_estash_dres_pf |
|---|---|---|---|
| main | 1.69 | 0.89 | 0.61 |
| #59 | 1.00 | 0.89 | 0.60 |
| #60 | 1.69 | 0.83 | 0.42 |
| #61 | 1.00 | 0.64 | 0.55 |

All three cut AMD backward kernel time with the same bits. Confirmation on today's main (with #55 and #58 TQ16) follows before any merge.
