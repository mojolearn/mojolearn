# Pointwise, ordered and categorical boosting: two H100s

RunPod `rbtojh7e0esekh`, two H100 80GB HBM3, IDENTICAL, sm_90a.
All builds and model execution occurred on the pod. R2 enwik8 staging and
corpus SHA are recorded. No local compilation or tests.

Passed:

- Nine pointwise full fits: 513/8193 rows, 73 features, binary/half-byte and
  5/6/7/8-bit one-byte widths, weighted RMSE and Logloss. Every trace record
  and raw histogram dump matches; full models, loss curves and predictions
  match. Failed-fit publication is atomic.
- Six native comparisons over the existing 400,000-row fractional-stat fixture:
  every histogram bit after a full pass and sibling-subtracted partial pass,
  all three packing policies, with/without one-hot flags. The fixture checks
  22,116 bits-as-UInt32 cells in total, without hashing away any mismatch.
- Sixteen OrderedRMSE fits: 65/513 rows, four border counts, forward/reversed
  permutations, signed gradients, zero/fractional weights. Original traces,
  full models and predictions match; invalid permutations do not publish.
- Four ExperimentalTwoLevelFeatureFreq fits: 65/513 rows, both source orders,
  weighted input, full serialized model/CTR state and predictions, atomic
  invalid-target refusal. Both levels reach the existing distributed greedy
  histogram seam; categorical generation and level transitions are unchanged.
- Four adapter fits: classifier/regressor, greedy/pointwise, weighted training,
  evaluation sets, labels/margins/probabilities/regression outputs and atomic
  failed-fit publication.
- All 16 previous greedy fixtures pass, with JSON receipts and output hashes
  exactly unchanged (`greedy-golden-comparison.log`).

The first gather treated interleaved weight/target pairs as separate planes.
Raw histograms and trace checks caught it even where winners matched. Failure
logs, traces and dumps are retained in `initial-layout-failure`; commit
`a88216112` corrected the copy span. The unchanged gates pass afterward.
An initial build started before the toolchain download finished; its refusal
is retained too. No failing arithmetic assertions were removed.

Source: the pod received full commit `eaec62839`. `source-overlay.tgz` contains
the changes from `be853dfe2` through `80b5efe79`, including the corrected native
implementation. `final-python-overlay.tgz` adds adapter support at `25a44efb5`.
The base binary was built from the initial commit, the final GBDT binary from
the corrected native sources. Both hashes and all build/job logs are retained.
The main merge at `2fb5c6da9` adds independent CPU-host surfaces; these were not
shipped to the pod and do not alter this GPU implementation.

Scope: two H100s only. Root data, categorical state, histograms and models
remain; pointwise workers currently clone complete index/histogram buffers and
gather only owned bins. No resident pooled-state capacity, speedup, eight-GPU
or new cross-vendor claim. The pod remains leased for the next Gram batch.
