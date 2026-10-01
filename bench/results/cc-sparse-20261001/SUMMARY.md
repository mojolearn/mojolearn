# connected-components as a sparse walk (PR #27, lane/neural-pass22), 2026-10-01

cc_sparse_check PASS and x_neighbors graph_check PASS on the MI325X box and the L40S pod.
Board connected-components cell, released 0.8.32 (before) vs the branch (after), 3 rounds:

| | taxi | istella |
|---|---|---|
| AMD before / after | 962.1 / 55.5 ms (17x) | 1,107.2 / 56.6 ms (20x) |
| NVIDIA before / after | 942.3 / 118.4 ms (8x) | 1,096.9 / 105.5 ms (10x) |
| digest, all runs on both vendors | f9dd8da0c6fcd7d8 | 7134dcb78da4dc5e |
