# PR #74 connected_components CSR (lane/neural-pass69), 2026-10-01

Branch board driver (CSR handoff), board-prepared data, ours vs networkx-cpu, 3 rounds.

## M3 Ultra (~/pr74-metal)

| dataset | ours ms (0.8.33 board) | ours ms (#74) | networkx ms | labels |
|---|---|---|---|---|
| istella | 77.0 | 5.95 | 6.67 | digest 7134dcb78da4dc5e = networkx, ARI 1.0, 81 components |
| taxi | 82.3 | 6.13 | 6.36 | digest f9dd8da0c6fcd7d8 = networkx, ARI 1.0, 588 components |

MI325X (DigitalOcean, pass15, 0.8.34 overlay, R2 `measurements/2026-10-01/cc74-amd.tar.gz`): taxi 56.79 -> 8.88 ms, istella 51.64 -> 7.13 ms (networkx 5.3 / 5.6), ARI 1.0 both.

L40S / EPYC (RunPod nvc3, head 04c65bffb, opponents capped at the 13-CPU quota, R2 `cc74-nvc3`): istella 162.2 -> 14.4 ms, taxi -> 22.3 ms (networkx 14.2 / 19.5), ARI 1.0 both.

Same labels on all three vendors, faster on NVIDIA and AMD: merged.
