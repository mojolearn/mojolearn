# PR #57 (pipelined logits download) and PR #58 (dq query rows per block) on the L40S and MI325X (Oct 1)

Both PRs were measured with the released 0.8.33 overlay venv and the measure-branch bindings rebuilt. The base was main 74510f024 for #57 and 1f776f534 for #58, both before #55.
Raw files are in R2 at `measurements/2026-10-01/pr5{7,8}-{nvidia,amd}.tar.gz`.
Every run gave lm-forward digest 4a8e781b0739a038 and the same loss trace.

## PR #57: neutral on both GPUs
Times in ms (lm-forward / lm-train-step).

| run | L40S | MI325X |
|---|---|---|
| default (chunk 2M) run 1 | 50.7 / 41.1 | 54.1 / 100.6 |
| default (chunk 2M) run 2 | 50.8 / 42.2 | 54.5 / 100.9 |
| STAGE=0 run 1 | 49.4 / 41.1 | 54.4 / 100.8 |
| STAGE=0 run 2 | 53.2 / 41.0 | 54.4 / 101.0 |
| chunk 1M | 52.6 / 41.3 | 54.0 / 100.1 |
| chunk 4M | 53.5 / 41.3 | 55.6 / 100.9 |

The M3 decides whether #57 merges.

## PR #58: TQ16 wins on both GPUs
lm-train-step in ms, two runs each; bwd_dq_tiled_pf per layer from the phase-timer build.

| arm | L40S train-step | L40S dq | MI325X train-step | MI325X dq |
|---|---|---|---|---|
| TQ64 (old default) | 41.2 / 42.5 | 0.61 | 100.7 / 100.5 | 1.69 |
| TQ32 | 39.3 / 39.3 | 0.38 | 93.1 / 93.1 | 0.70 |
| TQ16 | 38.9 / 38.6 | 0.32 | 91.1 / 90.6 | 0.45 |

lm-forward is unchanged. The writing lane makes TQ16 the NVIDIA and AMD default on lane/neural-pass50; a confirm run of the plain build follows before the merge.
