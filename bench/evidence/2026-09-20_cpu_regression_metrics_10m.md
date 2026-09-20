# CPU regression metrics, 10M rows

- Hardware: Apple arm64, macOS 26.5.2; local CPU host binding.
- Commit baseline: `263fca21e`; deterministic arithmetic fixture, seedless.
- Payload: two Float32 arrays, 10,000,000 rows, 80,000,000 bytes.
- Workload SHA-256: `8f3a80345c2a0b573dfe0e161b80c5ddf5438fe4cf138a423711acd08899934f`.
- Protocol: two warmups, seven timed calls per binary. Times below are raw milliseconds.

| metric | baseline samples | candidate samples | baseline median | candidate median | speedup | exact result |
|---|---|---|---:|---:|---:|---:|
| MSE | 46.804, 47.009, 47.009, 46.699, 47.319, 46.815, 46.648 | 13.009, 12.666, 13.418, 12.286, 12.436, 12.566, 12.799 | 46.815 | 12.666 | 3.70x | `7.999029207894637e-07` |
| MAE | 50.374, 48.713, 48.433, 48.010, 48.530, 48.315, 48.403 | 12.917, 15.258, 13.205, 12.898, 12.950, 12.942, 16.209 | 48.433 | 12.950 | 3.74x | `0.0007741686422377825` |
| RMSE | 48.434, 48.659, 48.403, 48.629, 48.370, 49.260, 52.434 | 13.818, 14.199, 14.648, 14.524, 18.521, 15.454, 15.622 | 48.629 | 14.648 | 3.32x | `0.0008943729335442185` |

The candidate changes only independent 256-row partial generation. The ascending partial fold, division, and root are unchanged. Separate `/usr/bin/time -l` processes reported peak RSS 274,972,672 bytes baseline and 195,444,736 bytes candidate, a 79,527,936-byte reduction. `build_metrics_host.sh` passed.
