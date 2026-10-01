# Host GEMM output/scratch lists uninitialized, transformer mask as lane spans (PR #18, lane/neural-pass12), 2026-10-01

AMD EPYC 9575F (Zen 5, 64 cores), CPU column, the PR #15 run order.

| check | result |
|---|---|
| check-gemm-host-rows | PASS (5427 cases, 0 differ) |
| host_threads_ab mamba3 / transformer | PASS, e0de9d88212ad12c / ac99eab06d1db1d2 (unchanged) |
| CPU train gate | 640/640 |
| host GEMM bench at the policy, lm_head 512x8192x384 | 1261 GFLOP/s (PR #15 run: 800), same digest 8283376667491874104 |
| host GEMM bench at the policy, mamba3_in_proj 2048x1728x384 | 1548 GFLOP/s (PR #15 run: 893), same digest 11721081545930648343 |
| one thread (lm_head / in_proj) | 99.6 / 106.6 GFLOP/s (unchanged: the gain is per call, not in the kernel) |
| cells samba-infer / transformer-infer / lm-host-train-step | 101.5 / 22.0 / 542.4 ms (flat within the 64-thread spread) |
