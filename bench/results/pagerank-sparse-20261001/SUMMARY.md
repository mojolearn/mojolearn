# PageRank over the nonzero cells (PR #35, lane/neural-pass30), 2026-10-01

Board pagerank cell, ours, 3 rounds. AMD/NVIDIA: released 0.8.32 (before) vs branch (after); restore = branch with
MOJOLEARN_PR_SPARSE=0. M3 Ultra (Metal): main vs branch source trees.
| | istella before / after / restore | taxi before / after / restore |
|---|---|---|
| AMD MI325X | 572.8 / 113.2 / 579.2 ms (5.1x) | 581.7 / 113.2 / 581.1 ms (5.1x) |
| NVIDIA L40S | 1,208.6 / 34.6 / 1,223.4 ms (35x) | 1,357.7 / 25.7 / 1,191.0 ms (53x) |
| Apple M3 Ultra | 673.5 / 39.8 / 663.0 ms (17x) | 672.9 / 36.2 / 684.7 ms (19x) |
| digest, every run on all three vendors | 7eb6c05c6a688609 | 8084077034791453 |
