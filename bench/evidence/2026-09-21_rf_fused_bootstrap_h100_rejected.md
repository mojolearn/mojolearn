# RF fused-bootstrap H100 trial — rejected

The default-off `MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER` candidate combined the
default Philox bootstrap-row write with the sampled-label gather. It was run
from source witness `56ab28755fe5ed4ca6a577864dd16c5597c185c3` on an NVIDIA
H100 80GB HBM3. Taxi and Istella-S came from the canonical R2 objects (archive
SHA-256 `10d5d35f...aab6cc15` and `31f04237...2f6ffef`). Each public
IDENTICAL RandomForest cell trained 100 depth-16 trees on 1,000,000 rows,
predicted 100,000 rows, excluded one warmup, and retained five complete fits.

All 12 dataset x arm x outer-process records completed. Every process reported
the intended binding selection, and every repeat across both arms matched the
complete five-array forest plus fitted metadata, full predictions, full
probabilities, accuracy, and logloss. All process spreads passed the 1.10
limit. The timing gate still rejected the candidate:

| dataset | baseline median (ms) | fused median (ms) | median fused/base | slowest fused / fastest baseline |
| --- | ---: | ---: | ---: | ---: |
| Taxi | 938.706 | 931.836 | 0.99268 | 1.04303 |
| Istella-S | 1307.490 | 1303.676 | 0.99708 | 1.00451 |

The promotion rule required exact bytes and quality on both datasets and a
conservative ratio at most 0.98. The fusion is effectively flat and does not
justify another path, so it remains default-off.

The complete local receipt is
`/Users/andrewhendel/mojolearn-evidence/2026-09-21_rf_scaler_training_h100/rf/`;
`verdict.json` has SHA-256 `be9b4ea1...d13065316`. RunPod pod
`vl7kbqczche8o5` was deleted after the receipts were pulled: DELETE returned
HTTP 204 and the verification GET returned HTTP 404.
