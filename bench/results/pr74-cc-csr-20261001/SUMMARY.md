# PR #74 connected_components CSR (lane/neural-pass69), 2026-10-01

Branch board driver (CSR handoff), board-prepared data, ours vs networkx-cpu, 3 rounds.

## M3 Ultra (~/pr74-metal)

| dataset | ours ms (0.8.33 board) | ours ms (#74) | networkx ms | labels |
|---|---|---|---|---|
| istella | 77.0 | 5.95 | 6.67 | digest 7134dcb78da4dc5e = networkx, ARI 1.0, 81 components |
| taxi | 82.3 | 6.13 | 6.36 | digest f9dd8da0c6fcd7d8 = networkx, ARI 1.0, 588 components |

L40S (nvc2) and MI325X (pass15) pending.
