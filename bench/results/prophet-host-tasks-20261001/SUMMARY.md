# prophet series as host tasks inside the GPU binding (PR #30, lane/neural-pass23), 2026-10-01

Board prophet cell, ours only, 3 rounds:
| run | AMD MI325X synthetic / taxi-hourly | NVIDIA L40S synthetic / taxi-hourly |
|---|---|---|
| released 0.8.32 (device) | 31,537.4 / 37,240.1 ms | 5,331.6 / 6,296.5 ms |
| branch (host tasks) | 195.9 / 200.6 ms (161x / 186x) | 178.7 / 185.6 ms (30x / 34x) |
| branch, MOJOLEARN_SEQ_PROPHET_HOST_MAX=0 | 32,097.8 / 36,515.4 ms | 5,384.9 / 6,362.0 ms |
| digest (all runs, both vendors) | 8d3c6f8fb148f95f / 66f5dd106a7a3f88 | same |
The prophet-cpu opponent read ~248-282 ms on AMD (0.8.25 board).
