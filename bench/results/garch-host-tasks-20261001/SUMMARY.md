# garch series as host tasks inside the GPU binding (PR #21, lane/neural-pass15), 2026-10-01

AMD MI325X box (host EPYC 9575F). Board garch cell, ours only, 3 rounds:

| run | synthetic | taxi-hourly |
|---|---|---|
| released 0.8.32 (one GPU thread per series) | 5,208.7 ms | 8,888.4 ms |
| this branch (series over host tasks) | 56.1 ms (93x) | 69.3 ms (128x) |
| this branch, `MOJOLEARN_SEQ_GARCH_HOST_MAX=0` (the device run) | 5,200.9 ms | 8,865.5 ms |
| digest, all three | 10215ebce6ad0f6a | 78356d7d495daa98 |

The board's arch-cpu opponent on this box read 52.7 / 43.3 ms (0.8.25 board). NVIDIA: not run.
