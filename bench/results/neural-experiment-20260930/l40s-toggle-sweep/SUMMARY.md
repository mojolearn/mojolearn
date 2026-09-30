# L40S toggle-round sweep, 2026-09-30

Branch lane/neural-toggle-sweep-20260930 (= lane/neural-net-experiment e2a624067 build code, plus runner).
One build of transformer, mamba, byte_lm (sm_89, identical) over the mojolearn 0.8.31 wheel. Session checks
(reuse, refusals, lifetime, budget) and the fresh-prefill check (188928 cells) PASSED.
`neural_experiments.py --set nvidia`, three passes (10, 20, 20 calls). Ratio = toggle median / baseline median,
median of the three passes [min-max]. Every toggle's output digest and loss series equal baseline's in every pass.

| toggle | transformer-fwd | mamba3-fwd | samba-fwd | samba-train | lm-fwd | lm-train | verdict |
|---|---|---|---|---|---|---|---|
| no_retain_weights | 1.16 [1.02-1.17] | 1.17 [1.15-1.18] | 1.14 [0.99-1.18] | 0.96 | 0.95 | 0.99 | keep retention ON |
| legacy_fresh_entry | 1.07 [1.02-1.08] | 1.01 | 1.05 | 0.98 | 1.01 | 1.00 | keep session route ON |
| no_stage_reset | 0.89 [0.80-0.91] | 1.01 | 0.99 | 0.98 | 1.00 | 1.00 | KEEP: skip the reset |
| speculative_attn | 0.99 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | no gain on NVIDIA |
| swiglu_fused | 0.98 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | no gain on NVIDIA |
| no_layer_sync | 0.98 | 1.00 | 0.99 | 1.00 | 1.08 | 1.00 | no gain |
| all_on | 0.88 | 1.03 | 0.97 | 1.00 | 1.11 [1.02-1.13] | 0.99 | slower lm-fwd; do not combine |

GEMM plan arms on transformer-forward (one pass): tuned128 0.99, half 0.98, quarter 1.01, kpack 1.03,
kfoldv 0.99, all same digest: no plan beats shipped at this shape.

Baseline medians (ms, pass 1): transformer-fwd 2.905, mamba3-fwd 2.676, samba-fwd 9.920, samba-train 148.0,
lm-fwd 52.5, lm-train 45.2. lm-train-step does not move under any toggle: its cost is in the backward kernels.
Pass 1 attempt without torch in the venv failed before timing (kept as sweep-nvidia.attempt1-no-torch.*).
AMD and Apple not run.
