# PR #76 KNNImputer host route (lane/neural-pass71, stacked on #74)

knn-imputer race, ours (default route) vs the 0.8.33 board row; masked_rmse must equal 0.8.33.

| Box | taxi ms (before -> after) | istella | sklearn | masked_rmse |
|---|---|---|---|---|
| M3 Ultra (R2 `measurements/2026-10-01/knn76-metal.tar.gz`) | 75.65 -> 1.60 | | | equal |
| MI325X (pass17, R2 `measurements/2026-10-01/knn76-amd.tar.gz`) | 101 -> 1.9 | 1550 -> 51 | 2.2 / 16.8 | equal |
| L40S / EPYC (nvc2, head f2fd807bf, R2 `measurements/2026-10-01/knn76-nvidia.tar.gz`) | 175.13 -> 2.38 | fit 103-123 ms (board 2357), infer 85 s vs sklearn 830-850 s | 4.34 | 6.151695683 = 0.8.33 |

The NVIDIA istella race hit the 2400 s ceiling during sklearn's inference rounds (rc=124); ours completed every round.

M2 Pro: threaded host imputer 16.8 s vs serial 94.4 s on the #77 probe, same digest.

Same answers on all three vendors, faster on NVIDIA and AMD: merged after #74.
