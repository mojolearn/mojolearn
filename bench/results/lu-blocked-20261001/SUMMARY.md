# Blocked LU: panel steps, tiled trailing update (PR #37, lane/neural-pass32), 2026-10-01

Board lu-factor / lu-solve cells (synthetic, n 8192), ours, 3 rounds; restore = branch with MOJOLEARN_XD_LU_PANEL=0.
| | lu-factor before / after / restore | lu-solve before / after / restore |
|---|---|---|
| AMD MI325X (released 0.8.32 vs branch) | 10,679.9 / 10,246.2 / 11,006.4 ms | 10,426.5 / 10,222.0 / 10,581.9 ms |
| NVIDIA L40S (released 0.8.32 vs branch) | 6,281.1 / 4,516.9 / 6,295.9 ms (1.39x) | 6,393.0 / 4,515.5 / 6,303.8 ms (1.42x) |
| Apple M3 Ultra (main vs branch) | 22,275.5 / 14,509.0 / 22,296.8 ms (1.54x) | 22,278.8 / 14,487.8 / 22,245.5 ms (1.54x) |
| digest, every run on all three vendors | 43e06346de01739b | 43e06346de01739b |
AMD barely moves (4%): its cost is elsewhere (follow-up asked of the author).
