# Embedding binding transfers (PR #33, lane/neural-pass28), 2026-10-01

`check-embedding`: GREEN (clauses a, b, c, f) on the AMD MI325X, the NVIDIA L40S and the Apple M3 Ultra.
Board embedding cell (synthetic), ours, 3 rounds; digests identical before/after on every machine:
| | before | after | digest |
|---|---|---|---|
| AMD (released 0.8.32 vs branch) | 185.3 ms | 84.3 ms (2.2x) | 2753e61a6dde36e3 |
| NVIDIA (released 0.8.32 vs branch) | 331.4 ms | 183.3 ms (1.8x) | 2753e61a6dde36e3 |
| M3 Ultra (main vs branch, Metal) | 798.0 ms | 101.0 ms (7.9x) | 5c23514a916c0930 |
The cell's inputs come from torch, and a seeded torch.randn differs between the Mac's arm64 torch and x86 torch
(randn(4096) at seed 7: 14193c5e9017ae6b vs df1f2643101fd189), so the Mac digest is not comparable across vendors;
the cross-vendor proof is check-embedding. Files: amd/, nvidia/ (race2-* are the runs with torch installed), m3-ultra/.
