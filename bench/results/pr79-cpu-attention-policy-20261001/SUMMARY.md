# PR #79 CPU transformer block policy (lane/neural-pass75)

CPU-only change: lane policy vs the staged path (restore switch), same tree. Digests equal on both boxes:
htc sha 93b0da9f81be0dc5, transformer-infer d2e67d7563b0edb2, samba-infer ddce61948b8456e0.

| Box | host_threads_ab_check policy ms | attention phase ms | transformer-infer ms | samba-infer ms |
|---|---|---|---|---|
| EPYC 9374F (RunPod nvc3, 13-thread quota, head fa3b56787, R2 `measurements/2026-10-01/pr79-nvc3.tar.gz`) | 52.8 / 56.0 vs 67.6 / 70.5 | 30.4 / 20.6 vs 38.0 / 26.8 | 22.1 / 21.4 vs 29.8 / 26.2 | 178.2 / 156.7 vs 190.4 / 222.7 |
| M2 Pro (head e4ac74164, R2 `pr79-cpu-m2pro`) | 163.7 vs 190.8 / 193.1 | 91 vs 119.4 | 28.5-31.3 vs 30.5-30.7 | 187.3-187.6 vs 189-224 |

Pairs are lane / staged per round set. Same bits, faster on the EPYC (the deciding box for a CPU change) and the M2 Pro: merged.
