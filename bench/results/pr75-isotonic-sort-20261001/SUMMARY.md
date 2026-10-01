# PR #75 isotonic sort in the binding (lane/neural-pass70 at 448c3f086, stacked on #73), 2026-10-01

Branch board driver, board-prepared data, ours vs sklearn-cpu, 3 rounds.

## M3 Ultra (~/pr75-metal)

| dataset | ours ms (0.8.33 board) | ours ms (#75) | sklearn ms | held-out r2 ours / sklearn |
|---|---|---|---|---|
| istella | 954.8 | 62.5 | 38.0 | 0.187985457 / 0.187985459 |
| taxi | 1462.0 | 100.6 | 92.0 | 0.897068843 / 0.897068843 |

MI325X (DigitalOcean, pass16, R2 `measurements/2026-10-01/iso75-amd.tar.gz`): taxi 2413 -> 100.8 ms, istella 1416 -> 79.7 ms (sklearn 82 / 35), r2 equal to 0.8.33 exactly.

L40S / EPYC (RunPod nvc3, head 8d8343dc5, R2 `measurements/2026-10-01/iso75-nvc3.tar.gz`): taxi 2933 -> 197.9 ms, istella 1403 -> 125.5 ms (sklearn 155 / 65), r2 0.8970688429121374 / 0.18798545682734658 = 0.8.33.

Same answers on all three vendors, faster on NVIDIA and AMD: merged.
