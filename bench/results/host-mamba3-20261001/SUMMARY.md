# Mamba-3 host oracle over host tasks, host GEMM packs uninitialized, loss keep-alive fix (PR #15, lane/neural-pass9), 2026-10-01

CPU column on two hosts: AMD EPYC 9575F (Zen 5, the DO box) and AMD EPYC 9374F (Zen 4, the NVIDIA pod).

| check | Zen 5 | Zen 4 |
|---|---|---|
| `check-gemm-host-rows` | PASS, 5427 cases, 0 differ | PASS |
| `host_threads_ab_check --model mamba3` (one thread vs policy) | PASS, sha256 e0de9d88212ad12c both; 200.3 -> 22.6 ms (8.85x) | PASS, e0de9d88212ad12c; 366.5 -> 70.1 ms |
| `host_threads_ab_check --model transformer` | PASS, ac99eab06d1db1d2 both; 506.2 -> 103.2 ms | PASS, ac99eab06d1db1d2 |
| CPU train gate | PASS 640/640 | PASS 640/640 |
| host GEMM bench, one thread / policy (lm_head 512x8192x384) | 98.8 / 800.0 GFLOP/s | — |

Same output sha256 on both hosts and the author's laptop (e0de9d88, the serial build's).

Board cells, Zen 5, CPU column, main (through PR #14) vs PR #15, same quality:

| cell | main | PR #15 |
|---|---|---|
| mamba3-infer | 40.72 ms | 10.17 ms (4.0x) |
| samba-infer | 227.52 ms (mean_nll 5.635909657868805) | 97.81 ms (same) |
| transformer-infer | 20.26 ms | 20.23 ms |
| lm-host-train-step | 524.16 ms | 523.01 ms (same losses) |

`check-mamba3-corpus` not run: the task needs a dump directory (`$MOJOLEARN_MAMBA_CORPUS_DUMP`) from a prior run.
