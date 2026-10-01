# TreeSHAP rows over host tasks (PR #36, lane/neural-pass31), 2026-10-01

Board tree-shap cell, ours, 3 rounds; restore = branch with MOJOLEARN_XTREES_SHAP_TASKS=1 (the serial walk).
| | istella before / after / restore | taxi before / after / restore |
|---|---|---|
| AMD MI325X | 8,081.3 / 418.4 / 8,133.3 ms (19x) | 3,815.6 / 197.1 / 3,846.1 ms (19x) |
| NVIDIA L40S | 14,167.5 / 450.0 / 12,615.4 ms (31x) | 5,907.5 / 197.8 / 6,327.7 ms (30x) |
| Apple M3 Ultra (main vs branch) | 10,268.3 / 848.7 / 10,292.2 ms (12x) | 4,165.7 / 342.6 / 4,164.5 ms (12x) |
| digest, every run on all three vendors | 9c42e3fbec797f77 | 8fd935518dd93613 |
