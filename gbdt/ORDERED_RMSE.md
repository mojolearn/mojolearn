# Ordered RMSE

`gbdt.train.train_ordered_rmse` is the native raw-float entry for the supported
ordered boosting path. It returns the existing `TrainedModel`, usable with
`predict_floats` and the normal model serialization machinery.

```mojo
from gbdt.train import train_ordered_rmse, predict_floats

# x is column-major; permutation contains each original row id exactly once.
var model = train_ordered_rmse(
    ctx, x, y, n_rows, n_features, permutation,
    n_estimators=100, max_depth=6,
    learning_rate=Float32(0.03), l2_leaf_reg=Float32(3.0),
)
var prediction = predict_floats(ctx, model, x, n_rows)
```

Compile the calling program with `-D MOJOLEARN_NUMERIC_IDENTICAL=1` to retain
the bitwise arithmetic profile. The ordered path uses the existing GPU tree
searcher and Newton leaf estimator; it does not train on the host.

Each iteration chooses one shared tree structure. Each fold owns an independent
approximation cursor in permutation order. Its leaf estimates use only its
prefix rows, then update both the prefix and its quality-evaluation tail.
Quality-tail labels may influence structure scoring, as in ordered boosting,
but never that fold's leaf estimation. A separate full-data cursor estimates
the exported tree. Learning-rate rescaling is rounded before cursor addition,
matching exported-model prediction.

The public entry supports finite numeric columns, RMSE, one explicit
permutation, zero initial bias, one Newton leaf-estimation step, depth 1–8,
nonnegative sample weights with positive total mass, and one GPU. The lower
`fit_ordered_rmse` entry additionally accepts explicit one-hot flags and the
supported fold scorers. Missing values, categorical CTR permutations, other
objectives, bootstrap, score noise and early stopping are not options on this
entry. This does not claim full external CatBoost parity or installed Python
`boosting_type="Ordered"` support.

`checks/ordered_rmse_check.mojo` covers prefix-label isolation with a leakage
negative control, persistent fold cursors against an independent float64
replay, weighted multi-tree fit/predict, invalid permutations, zero-mass
prefixes at L2=0, and standalone/mixed constant-tree prediction with bias.
It refuses non-IDENTICAL builds and emits `ORDERED_BITS` records. Cross-vendor
qualification requires those records to match, in addition to each native
check passing. The older `ordered_boosting_check.mojo` remains a separate
fold-axis wiring check.

AMD MI325X and NVIDIA RTX 4090 pass this native gate at `6dd44ac5`, with
all 130 ordered records matching by bits. Weighted CTR checks also pass.
See the [comparison](../bench/results/resume/2026-09-06-ordered-mamba-knn/cross-device-continued.json).

## Python source entry

`mojolearn.ensemble.OrderedRMSE` exposes the same bounded native trainer:

```python
from mojolearn.ensemble import OrderedRMSE

model = OrderedRMSE(
    n_estimators=100, max_depth=6, numeric_mode="identical",
)
model.fit(X, y, permutation=row_order, sample_weight=weights)
prediction = model.predict(X_new)
model.save("ordered.npz")
restored = OrderedRMSE.load("ordered.npz")
```

`row_order` must be an integer bijection of original row ids; X, y and weights
remain in original row order. The explicit ordering avoids a hidden host RNG.
The narrow constructor refuses categorical/objective/bootstrap/early-stopping
options rather than forwarding them to plain boosting. Prediction and model
serialization use the existing GBDT format. No loss curve or best iteration is
computed: those attributes are `None`.

The binding requires a rebuilt GBDT extension exporting
`gbdt_fit_ordered_rmse`; older binaries fail with an explicit upgrade/build
message. Frozen source `eb835021` now passes the real installed Python gate
on AMD/NVIDIA in all three modes, with the complete IDENTICAL model and
72 prediction cells matching; see the [installed lane record](../bench/results/resume/2026-09-06-installed-gap-closure/installed-lane-comparison.json).
The full NVIDIA candidate still fails unrelated sequence jobs and is not
release-qualified. `test_ordered_rmse_surface.py`
checks buffer order, parameter forwarding and refusals without native work;
the real installed GPU record covers fit/predict/save/load separately. As with the other estimators, `numeric_mode` selects a compiled
arithmetic tier; it does not enable general CatBoost feature parity.

`tools/ordered_rmse_surface_check.py` is the real installed native gate (also
executable with `runpy.run_path(..., run_name="__main__")`). It fits a weighted
32-row three-tree model, checks held-out-row prediction and save/load bits,
and requires repeat model/prediction identity in pinned modes. It reads mode
and vendor from the actual binary and emits complete model text and prediction
bits as `ORDERED_PYTHON_JSON`; AMD/NVIDIA IDENTICAL records must be compared
separately. The controller sets `MOJOLEARN_NUMERIC_MODE` and CPU limits before
launching it. Follow-up `29a8c848` persists the effective numeric mode in saved
models and restores it before binding, even with a changed process default.
Old files without the optional mode field keep the historical default behavior;
select their intended tier explicitly. The updated gate removes the manual
mode reset and passes all three modes in a separate NVIDIA wrapper overlay.
That wrapper follow-up still needs refreshed AMD/Apple and final-wheel qualification.
