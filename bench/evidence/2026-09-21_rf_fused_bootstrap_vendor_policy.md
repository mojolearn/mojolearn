# RF fused bootstrap and label gather vendor policy

The fused route writes each sampled row ID and its sampled label in the same
Philox kernel. It keeps the original thread/subsequence mapping, rejection
loop, output index, seed, and stream order. Weighted bootstrap, no-bootstrap,
ExtraTrees, and the sorted-row experiment keep their prior paths.

The H100 trial used source witness `56ab28755fe5ed4ca6a577864dd16c5597c185c3`,
an NVIDIA H100 80GB HBM3, and the canonical Taxi and Istella-S R2 objects
(archive SHA-256 `10d5d35f...aab6cc15` and `31f04237...2f6ffef`). Every
public IDENTICAL RandomForest cell trained 100 depth-16 trees on 1,000,000
rows, predicted 100,000 rows, excluded one warmup, and retained five complete
fits in each of three alternating fresh processes.

All 12 records completed. Every repeat matched the complete five-array forest,
fitted metadata and classes, full predictions, full probabilities, accuracy,
and log loss across arms. All within-process timing spreads were at most
1.072. Under the current policy, a candidate qualifies when exactness and
quality match and the median of process medians improves on both datasets;
slowest/fastest extremes and stability remain recorded diagnostics.

| vendor | dataset | baseline median (ms) | fused median (ms) | fused / baseline | decision |
| --- | --- | ---: | ---: | ---: | --- |
| NVIDIA H100 | Taxi | 938.706 | 931.836 | 0.99268 | default on |
| NVIDIA H100 | Istella-S | 1307.490 | 1303.676 | 0.99708 | default on |
| AMD MI325X | Taxi | — | — | 1.0013 | default off |
| AMD MI325X | Istella-S | — | — | 0.9994 | default off |

NVIDIA therefore selects fusion by default. `MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER_OFF=1`
restores the two-launch route. HIP remains off because Taxi regressed; the
positive define remains available for experiments. Apple remains off because
its only measurements were short directional smoke runs rather than the full
repeated matrix.

The H100 receipt is external at
`mojolearn-evidence/2026-09-21_rf_scaler_training_h100/rf/`; its original
`verdict.json` SHA-256 is `be9b4ea1...d13065316`. The MI325X receipt is at
`mojolearn-evidence/training-amd-2026-09-21_114800-mi325x-retry/remote/rf-fused-bootstrap/`.
Both provider leases were destroyed and verified absent after collection.
