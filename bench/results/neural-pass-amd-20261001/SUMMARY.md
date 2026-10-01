# MI325X neural pass: PR #55 confirmation, attention arms, MFMA forward trial, PR #56 (Oct 1)

Box: DigitalOcean MI325X. The released 0.8.33 overlay venv was used, with the measure branch bindings rebuilt.
Raw files are in R2 at `measurements/2026-10-01/neural-pass-amd.tar.gz` and `measurements/2026-10-01/pr56-amd.tar.gz`; summary-raw.txt holds the condensed lines.
Every run gave lm-forward digest 4a8e781b0739a038 and the same loss trace.

## PR #55 (GEMM tile minimum 1024 blocks on AMD) — MERGED
Times in ms.

| run | lm-forward | lm-train-step |
|---|---|---|
| default (1024) run 1 | 41.2 | 75.0 |
| default (1024) run 2 | 41.3 | 75.0 |
| TILE=0 control (old plan) | 55.7 | 101.0 |

The gemm 4096³ race was 85.7 ms both ways (digest 535b4c27bd9313d1). check_device_default_dispatch and the 8 GEMM gates are green. NVIDIA is unchanged (see neural-pass-nvidia-20261001).

## Attention arms (AMD)
The default (41.3 / 74.3 ms) is the best valid arm. kvgrid_r64 regresses lm-train-step to 273.8 ms; no_dres and the bswz toggle are slower. fgrid_r64, kvsplit, kvrecompute, zdefer and zlag are invalid compositions and were refused.

## MFMA forward trial (-D MOJOLEARN_ATTN_FWD_MFMA=1)
lm-forward 39.6 ms, lm-train-step 73.2 ms, with the same digest and losses. That is 1.7 ms faster on the forward pass than the default arm.

## PR #56 (scratch pool) on AMD (built on main before #55)
Times in ms (lm-forward / lm-train-step).

| run | pool | no pool |
|---|---|---|
| run 1 | 54.1 / 99.6 | 54.6 / 101.3 |
| run 2 | 54.5 / 99.6 | 55.0 / 100.8 |

lm-train-step is about 1.5% faster with the pool. The phase timers show fwd/bwd regime scans of 0.037 ms (pool) against 0.066/0.069 ms (no pool).
