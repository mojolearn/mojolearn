# FAST resident GBDT binary class codes (2026-09-20)

This change is limited to FAST `Logloss`/`CrossEntropy` class prediction.
The existing resident model still performs identical quantization and tree
traversal and produces the same Float32 raw cursor. Instead of materializing
sigmoid probability columns and applying a second argmax pass, the resident
call writes Int64 codes directly: `raw > 0` is class one; zero remains class
zero, exactly matching first-max over `[1 - sigmoid(raw), sigmoid(raw)]`.
Rows are independent and split over the existing host worker pool.

IDENTICAL, DETERMINISTIC, multiclass, probability, raw-score, and regression
paths retain their old dispatch. A missing new binding also falls back.
Depthwise and Lossguide support binary classification but explicitly refuse
CatBoost `MultiClass`, so no unsupported 3/8-class claim is made.

## Large-data generalization

Apple M4 Metal; 12 trees; alternating old/new order; five-run medians after
warmup. `old` is resident `predict_proba` plus first-max/decode. `new` is the
direct resident-code public classifier prediction. All arrays matched exactly.

| policy | seed | features | depth | rows | labels | old s | new s | speedup | digest prefix |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| Depthwise | 11 | 8 | 4 | 500k | balanced | .018351 | .017424 | 1.053x | `6f1c104ed8905ff7` |
| Depthwise | 29 | 16 | 6 | 1m | 90% majority | .036425 | .034837 | 1.046x | `ea2f7c58fb4fdf79` |
| Depthwise | 47 | 64 | 8 | 500k | threshold/tie-heavy | .066448 | .065500 | 1.014x | `5d912adf135e2df8` |
| Lossguide | 11 | 16 | 8 | 1m | balanced | .037947 | .035865 | 1.058x | `a3f45a3e30b6bbb2` |
| Lossguide | 29 | 64 | 4 | 500k | 90% majority | .064313 | .063047 | 1.020x | `0dd6608482f52889` |
| Lossguide | 47 | 8 | 6 | 1m | threshold/tie-heavy | .031484 | .029251 | 1.076x | `54c19fd4b749ad65` |

No robust regression remained after parallelizing the code conversion. The
gain is deliberately described as modest: traversal dominates these shallow
12-tree fixtures. The Python output boundary also shrinks: raw/probability
and intermediate code arrays are replaced by the final 8-byte-per-row codes.
No process-RSS claim is made because retained device/pinned workspaces dominate
the allocator high-water mark.

Regression profiling at 500k/1m rows found no duplicate boundary to remove:
public regression prediction already returns the resident raw Float32 output
directly. Repeated output hashes matched for two Depthwise and one Lossguide
shape; no regressor code was changed.
