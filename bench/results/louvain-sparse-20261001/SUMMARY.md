# Louvain as a sparse walk of the dense item (PR #20, lane/neural-pass14), 2026-10-01

| check | AMD MI325X box | NVIDIA L40S pod |
|---|---|---|
| louvain_sparse_check (labels, modularity bits, levels vs o_louvain and the dense item) | PASS, 0 differences (n 1200: sparse 16.4 vs dense 96.8 ms; n 6000 sparse 141 ms) | PASS |
| x_neighbors graph_check (GPU column) | PASS | PASS |

Board louvain cell, released 0.8.32 (before) vs 0.8.32 + this branch (after), digests:

| | taxi | istella |
|---|---|---|
| AMD before / after | 37,238 / 560 ms (66x) | 38,915 / 621 ms (63x) |
| NVIDIA before / after | 54,791 / 897 ms (61x) | 62,733 / 989 ms (63x) |
| digest, every run on both vendors | 3599bb95ca20a3e0 | 3adeefd363c2dea1 |
