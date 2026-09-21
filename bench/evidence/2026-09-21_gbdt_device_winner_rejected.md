# GBDT device winner fold: rejected

The default-off candidate moved the strict sequential winner fold from the
host to the device for Depthwise and Lossguide training. It left the other
FAST split-cost changes disabled, including partition-stat propagation and
histogram scheduling. A sabotage build changed the selected threshold after a
defined device winner; all model, prediction, probability, and loss-curve
hashes moved, proving that the candidate route executed.

Both provider trials used exactly the pinned Taxi and Istella-S R2 objects,
1,000,000 training rows, 100 depth-6 trees, one excluded full-shape warmup,
five retained fits, and three alternating fresh processes per arm. Promotion
required identical full model text, predictions, probabilities, loss curves,
accuracy, and log loss, plus a candidate median of process medians below the
baseline. Stability and slowest/fastest ratios were recorded as diagnostics.

| provider | dataset | policy | candidate / baseline | exact + quality | decision |
| --- | --- | --- | ---: | --- | --- |
| NVIDIA H100 | Taxi | Depthwise | 1.007451 | pass | reject |
| NVIDIA H100 | Taxi | Lossguide | 0.999666 | pass | reject |
| NVIDIA H100 | Istella-S | Depthwise | 1.053930 | pass | reject |
| NVIDIA H100 | Istella-S | Lossguide | 1.052296 | pass | reject |
| AMD MI300X | Taxi | Depthwise | 1.008455 | pass | reject |
| AMD MI300X | Taxi | Lossguide | 1.000428 | pass | reject |
| AMD MI300X | Istella-S | Depthwise | 1.013117 | pass | reject |
| AMD MI300X | Istella-S | Lossguide | 1.054181 | pass | reject |

Every MI300X cell also passed binding/source provenance, loaded-array hashes,
the sabotage reach check, and the recorded 1.10 stability diagnostic. The
candidate is slower in every MI300X cell and three of four H100 cells; Taxi
Lossguide's 0.03% H100 improvement does not qualify a shared default.

The candidate source and trial-only harness were removed after this result.
Full MI300X records are outside the repository at
`mojolearn-evidence/rf-gbdt-mi300x-2026-09-21/run2/remote/gbdt_device_winner/`.
H100 records are at
`mojolearn-evidence/2026-09-21_rf_gdw_h100/pulled/gbdt-device-winner/`.

Two setup failures are preserved rather than hidden. The first MI300X attempt
found that the archive image lacked `rsync` and that the RF body required a Git
checkout for its commit witness. The second attempt initially stopped before
timing because a retired wide-only RF control still expected Taxi to fall back
to the original route, and the GBDT archive helper ignored the exported commit
witness. The corrected runs reused the same guarded second lease after saving
the failed-attempt directories. The final VM was deleted with HTTP 204 and
verified absent by GET 404; its local dead-man was cancelled only after that
verification.
