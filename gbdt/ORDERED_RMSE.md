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
