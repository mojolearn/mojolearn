# Masked keys skipped in the host attention (PR #16) and in lm-infer's kernels (PR #17), 2026-10-01

Hosts: AMD EPYC 9575F (Zen 5) + MI325X; AMD EPYC 9374F (Zen 4) + L40S.

**PR #16 (lane/neural-pass10):**
| check | Zen 5 / MI325X | Zen 4 / L40S |
|---|---|---|
| host_threads_ab transformer / mamba3 | PASS, ac99eab06d1db1d2 / e0de9d88212ad12c (unchanged from PR #15) | PASS, same |
| CPU train gate | 640/640 | 640/640 |
| GPU check-transformer, clause (a) | 30/30 stages bit-identical to the host oracle, 349,206 cells | same |
| GPU check-transformer-backward, clause (a) | PASS | 37/37 stages, 412,172 cells |
| GPU forward / backward readback checks (`-D MOJOLEARN_STEP_PHASE_TIMERS=1`, `MOJOLEARN_IDENTITY_TRACE` set) | PASS 30/30 and 37/37, two executions | not rerun with the define |
| host transformer one thread / policy (ab tool) | 412.8 / 90.9 ms (PR #15: 506.2 / 103.2) | — |
| cells transformer-infer / lm-host-train-step / samba-infer | 19.4 / 531 / 107 ms (main: 20.2 / 523 / 97.8; within run-to-run spread at 64 threads) | 41.2 / 1299 / 258 ms |

**PR #17 (lane/neural-pass11):** host gate PASS 144/144 and sweep PASS 864/864 (threads 1, 8, 64) on both hosts.
lm-infer cell (CPU column): Zen 5 151.1 ms vs main 147.9 ms (even); Zen 4 323.0 ms. The author's 10-core laptop: 3751 -> 2880 ms.
