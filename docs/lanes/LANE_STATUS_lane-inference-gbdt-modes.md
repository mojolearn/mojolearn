# Lane status: lane/inference-gbdt-modes (2026-09-15)

Public, bitwise-identical CPU `predict` and `predict_proba` from saved models
of the newest gradient boosting modes (the gbdt-ordered-rmse,
gbdt-feature-freq, gbdt-pointwise-l2-bayesian-eval and gbdt-categorical-ctr
lanes), through `mojolearn.host_model` and HostGBDT on the shipped forest
host binding.

## Design (stage 0, read only)

What the saved models hold. All 36 Apple model texts of the four lanes
(nine fixtures each, dumped by lane/cpu-training-gbdt-ordered) use only the
records HostGBDT already parses: `format`, `features`, `trees`, `losses`,
`bias`, `feature` of type `float` or `cat`, `tree`, `split` (with the
trailing `split_type take_bin` on a one-hot split), `leaf` and `loss`. None
carries `ctr_columns`, `ctr_table`, `ctr_entry`, `tensor_ctr_registry` or
`feature_freq_tensor`:

- OrderedRMSE: symmetric trees over float features, RMSE, no bias.
- ExperimentalTwoLevelFeatureFreq: one depth-two symmetric tree; the source
  columns are `type cat` (dense codes, borders `code + 0.5`). The CPU
  verifier refuses a tree whose level winner is the tensor column, so no
  lane fixture has a tensor CTR in its model.
- Pointwise L2 with Bayesian bootstrap and eval set: symmetric Logloss trees
  with a `bias` (boost from average); `predict_proba` is the sigmoid pair.
- The gbdt-categorical-ctr lane: its categorical column has two categories,
  at or below the GPU `one_hot_max_size`, so it is one-hot (`type cat`,
  TakeBin splits). No CTR table is built on any fixture.

What blocked host inference. `HostGBDT.from_file` refused every `estimator`
member other than `GradientBoosting`, and OrderedRMSE and
ExperimentalTwoLevelFeatureFreq inherit `GradientBoosting.save`, which
writes `type(self).__name__`. The one-hot walk (`_binarize` counting
`value > border` over the `code + 0.5` borders, TakeBin equality in the
packed layout) existed but had never been measured against a GPU column.

What a real CTR model needs at predict time, and how the reference does
it. CatBoost hashes each categorical value (`CalcCatFeatureHash`), combines
the hashes of a projection, looks the bucket up in the learn CTR table and
forms the value with `TModelCtr::Calc`
(`libs/model/static_ctr_provider.cpp:14-122`, `online_ctr.h:289`):
`(count + prior_num) / (denominator + prior_denom)`, then shift and scale;
an unseen bucket takes `Calc(0, denominator)` for Counter and FeatureFreq
and `Calc(0, 0)` for Borders. The result column is then quantized like a
float feature. Our format carries the counts, priors, shift, scale,
denominator and class axis in `ctr_table` and `ctr_entry`, keyed by the
dense category code instead of a hash (deviation 56), and
`gbdt/models/ctr_value_table.mojo::expand_raw_columns` applies them. That
module imports only `gbdt/ctrs/ctr.mojo` and `std.math`, so the forest host
binding can reuse it unchanged ahead of `_binarize`: Python parses the
tables into flat arrays, Mojo rebuilds `TCtrValueTable`s and calls
`expand_raw_columns`. Tensor CTRs (`feature_freq_tensor`) go through
`TTensorCtrRegistry.expand_for_model_apply`, whose module imports
`max.gpu.host` and would have to be split before a no-accelerator build can
import it.

## Done (stage 1)

- `python/mojolearn/_gbdt_host.py`: `GBDT_ESTIMATORS` admits
  `GradientBoosting`, `OrderedRMSE` and `ExperimentalTwoLevelFeatureFreq`
  archives (the two subclasses with loss RMSE only); `HostGBDT.estimator`
  is the saved class name. CTR and tensor CTR records still refuse by name.
- `python/mojolearn/host_surface.py`: forest kinds `gbdt_ordered_rmse`,
  `gbdt_feature_freq`, `gbdt_pointwise_bayesian_eval`,
  `gbdt_categorical_onehot`; the two classes. README and SUPPORT_MATRIX
  regenerated.
- `tools/forest_host_gate.py`: the four kinds with `make` support (reverse
  row permutation for OrderedRMSE, the `coded-2-4-v1` fixture transform for
  the categorical kinds, weights for the pointwise kind).
- `tools/identity_break.py`: `MOJOLEARN_IDENTITY_HOST_INFER` (`1` or a lane
  list) predicts the held-out rows through `host_model(<saved file>)`; the
  JSON records `host_infer`.
- `python/mojolearn/tests/test_gbdt_host_modes.py`.

## Evidence

- Apple M4, one core, bindings built from 479a9575e: the four lanes with
  `MOJOLEARN_IDENTITY_HOST_INFER`, diffed against the 166-lane GPU columns
  with `--require-columns 4 --owed-json`: train IDENTICAL=36, infer/model
  IDENTICAL=72, batch IDENTICAL=36, 0 OWED. The infer cells are HostGBDT's
  predictions on the saved file. Record:
  `bench/results/identity_break/2026-09-15_inference-gbdt-modes/`.
- `test_gbdt_host_modes` (7) and `test_host_surface` (110) pass on the M4;
  the runtime test ran (not skipped) against those bindings.
- Pending on one RunPod CPU pod: the x86 column, the forest sabotage column
  (`-D MOJOLEARN_FOREST_HOST_SABOTAGE=1`, must read DIVERGENT) and the
  installed-wheel check (`tools/inf_gbdt_wheel_models.py`).

## Not public, and why

- Models with `ctr_table` records (a categorical column above
  `one_hot_max_size`): no committed GPU column predicts one, and the CPU
  verifier refuses to train one, so any cell would be OWED with no hash on
  either side. The host apply is designed above and not built.
- Models with `feature_freq_tensor` records: same, plus the module split.
- Forest gate fixtures for the four new kinds: OWED to the next release
  record (make and record need a GPU).
