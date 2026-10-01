# PR #75 isotonic sort in the binding (lane/neural-pass70 at 448c3f086, stacked on #73), 2026-10-01

Branch board driver, board-prepared data, ours vs sklearn-cpu, 3 rounds.

## M3 Ultra (~/pr75-metal)

| dataset | ours ms (0.8.33 board) | ours ms (#75) | sklearn ms | held-out r2 ours / sklearn |
|---|---|---|---|---|
| istella | 954.8 | 62.5 | 38.0 | 0.187985457 / 0.187985459 |
| taxi | 1462.0 | 100.6 | 92.0 | 0.897068843 / 0.897068843 |

L40S (nvc2) and MI325X (pass16) pending.
