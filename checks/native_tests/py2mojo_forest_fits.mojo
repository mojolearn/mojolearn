# SPDX-License-Identifier: Apache-2.0
"""Host random-forest and decision-tree fits and predictions for the Mojo checks that replaced numeric Python
tests (lane py2mojo-C, 2026-11-22; docs/plans/PY2MOJO_CONVERSION.md batches 3 and 4). Importable; no `main()`, no
`test_*`.

A fit is what `RandomForestClassifier.fit` / `RandomForestRegressor.fit` / `DecisionTree*.fit` (splitter 'best')
run on the host column: the door's configuration (bindings/trees_public_forest_door.mojo `_rf_base`,
`_rf_arguments` for a DecisionTree*, `_rf_config`: the 16 RF slots in `_mojolearn_rf.mojo`'s order),
`forest_request` (bindings/forest_config_run.mojo), then the host image's fit body `_rf_fit_forest`
(core/mojolearn_rf_host_run.mojo, what `RfHostForestImage.fit` runs). Its `RfHostForest` arrays are the export
the door copies (`rf_host_export_into` is a straight copy). Predictions are the host image's two routes:
STRICT (`sequential`, the identical-mode default) through core/forest_host_strict_run.mojo and RESIDENT
(`parallel_groves`) through core/forest_host_groves_run.mojo. The depth and leaf count are `tree_shape`
(xtrees/ops.mojo, `get_depth` / `get_n_leaves`). X is row-major for unweighted fits and column-major for
weighted ones, as the door stages it. Each fit keeps its own parallel builder; the callers run independent fits
as case-table rows across the host cores.
"""
from std.math import min, max
from core.entry_table import entry_ref_unresolved
from core.forest_fit_request import FOREST_ARG_NO_INT
from bindings.forest_config_run import (
    ForestConfig, forest_request, FOREST_FEATURES_ALL, FOREST_FEATURES_SQRT, FOREST_FEATURES_LOG2,
    FOREST_FEATURES_FRACTION,
)
from xtrees.trees_forest_run import ForestFit
from core.mojolearn_rf_host_run import _rf_fit_forest
from core.forest_host_strict_run import (
    forest_strict_prepare_host_native, forest_strict_predict_host_check, forest_strict_predict_host_walk,
    forest_strict_release_host_native,
)
from core.forest_host_groves_run import (
    forest_prepare_host_native, forest_predict_resident_host_native, forest_release_host_native,
)
from xtrees.ops import tree_shape

comptime INT32_MAX = 2147483647
#: the RF criterion codes (python/mojolearn/randomforest.py, ensemble/host/rf_oracle.mojo RF_*)
comptime RF_GINI = 0
comptime RF_ENTROPY = 1
comptime RF_SQUARED = 2
comptime RF_POISSON = 4
comptime RF_GAMMA = 5
comptime RF_INVERSE_GAUSSIAN = 6


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct RfSpec(Copyable, Movable):
    """The constructor arguments a fit reads (python/mojolearn/randomforest.py defaults)."""
    var classifier: Bool
    var n_estimators: Int
    var max_depth: Int
    var features: Int
    var fraction: Float64
    var n_bins: Int
    var min_samples_leaf: Int
    var min_samples_split: Int
    var bootstrap: Bool
    var max_samples: Float64
    var seed: Int
    var n_streams: Int
    var criterion: Int

    def __init__(out self, classifier: Bool, n_estimators: Int = 100, seed: Int = 0):
        """RandomForestClassifier / RandomForestRegressor defaults: max_features 'sqrt' / 1.0."""
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.max_depth = INT32_MAX
        self.features = FOREST_FEATURES_SQRT if classifier else FOREST_FEATURES_FRACTION
        self.fraction = 1.0
        self.n_bins = 128
        self.min_samples_leaf = 1
        self.min_samples_split = 2
        self.bootstrap = True
        self.max_samples = 1.0
        self.seed = seed
        self.n_streams = 4
        self.criterion = RF_GINI if classifier else RF_SQUARED

    @staticmethod
    def decision_tree(classifier: Bool, max_depth: Int) -> Self:
        """DecisionTreeClassifier / DecisionTreeRegressor (splitter 'best'): `_rf_arguments`' fixed single-tree
        arguments (one tree, no bootstrap, one stream, max_samples None) over the RF defaults, max_features None
        (every feature), random_state None (seed 0); a regressor left at n_bins None takes the host image's
        `dt_default_bins`, 128."""
        var s = Self(classifier, 1, 0)
        s.max_depth = max_depth
        s.features = FOREST_FEATURES_ALL
        s.bootstrap = False
        s.n_streams = 1
        return s^


def _int(mut c: ForestConfig, value: Int):
    c.ints.append(value)
    c.floats.append(Float64(value))
    c.bad.append(UInt8(0))


def _float(mut c: ForestConfig, value: Float64):
    c.ints.append(0)
    c.floats.append(value)
    c.bad.append(UInt8(FOREST_ARG_NO_INT))


def rf_config(s: RfSpec) -> ForestConfig:
    """`_rf_config`: rows, cols, classes (filled by `forest_request`), n_estimators, max_depth, max_leaves (-1),
    the feature fraction (filled), n_bins, min_samples_leaf, min_samples_split, min_impurity_decrease (0.0),
    bootstrap, max_samples, the seed, n_streams, max_batch_size (4096)."""
    var c = ForestConfig(False, List[Int](), List[Float64](), List[UInt8]())
    _int(c, 0)
    _int(c, 0)
    _int(c, 0)
    _int(c, s.n_estimators)
    _int(c, s.max_depth)
    _int(c, -1)
    _int(c, 0)
    _int(c, s.n_bins)
    _int(c, s.min_samples_leaf)
    _int(c, s.min_samples_split)
    _float(c, 0.0)
    _int(c, 1 if s.bootstrap else 0)
    _float(c, s.max_samples)
    _int(c, s.seed)
    _int(c, s.n_streams)
    _int(c, 4096)
    c.criterion = s.criterion
    c.features.form = s.features
    c.features.fraction = s.fraction
    return c^


struct Forest(Movable):
    """One fitted forest's export arrays and counts."""
    var offsets: List[Int32]
    var colid: List[Int32]
    var quesval: List[Float32]
    var left: List[Int32]
    var leaves: List[Float32]
    var trees: Int
    var outputs: Int
    var features: Int

    def __init__(out self, var offsets: List[Int32], var colid: List[Int32], var quesval: List[Float32],
                 var left: List[Int32], var leaves: List[Float32], trees: Int, outputs: Int, features: Int):
        self.offsets = offsets^
        self.colid = colid^
        self.quesval = quesval^
        self.left = left^
        self.leaves = leaves^
        self.trees = trees
        self.outputs = outputs
        self.features = features

    def model(self) -> List[Int]:
        return [Int(self.offsets.unsafe_ptr()), Int(self.colid.unsafe_ptr()), Int(self.quesval.unsafe_ptr()),
                Int(self.left.unsafe_ptr()), Int(self.leaves.unsafe_ptr())]

    def nodes(self) -> Int:
        return len(self.colid)


def column_major(x: List[Float32], n: Int, d: Int) -> List[Float32]:
    var out = List[Float32](length=n * d, fill=Float32(0))
    for i in range(n):  # small-loop(fixture staging: the weighted fit's column-major X)
        for j in range(d):  # small-loop(the features of one row)
            out[j * n + i] = x[i * d + j]
    return out^


def rf_fit(s: RfSpec, x: List[Float32], n: Int, d: Int, codes: List[Int32], y: List[Float32], classes: Int,
           weights: List[Float32]) raises -> Forest:
    """One host fit. `x` row-major n x d; `codes` the int32 class codes (classifier) or `y` the float32 targets
    (regressor); `weights` empty for an unweighted fit."""
    var weighted = len(weights) > 0
    var staged = column_major(x, n, d) if weighted else x.copy()
    var w_max = 0.0
    if weighted and not s.classifier:
        for w in weights:  # small-loop(the largest weight, as the door's `_largest`)
            w_max = max(w_max, Float64(w))
    var fit = ForestFit(s.classifier, rowmajor_ok=not weighted, weights=Int(weights.unsafe_ptr()) if weighted else 0,
                        weight_max=w_max)
    var y_addr = Int(codes.unsafe_ptr()) if s.classifier else Int(y.unsafe_ptr())
    var request = forest_request(rf_config(s), entry_ref_unresolved(), n, d, classes if s.classifier else 0,
                                 Int(staged.unsafe_ptr()), y_addr, fit, False)
    request.gil_free = True
    var f: Forest
    if s.classifier:
        if weighted:
            var r = _rf_fit_forest[True, False](request)
            f = Forest(r.offsets.copy(), r.colid.copy(), r.quesval.copy(), r.left_child.copy(), r.leaves.copy(),
                       r.n_trees, r.num_outputs, d)
        else:
            var r = _rf_fit_forest[True, True](request)
            f = Forest(r.offsets.copy(), r.colid.copy(), r.quesval.copy(), r.left_child.copy(), r.leaves.copy(),
                       r.n_trees, r.num_outputs, d)
    else:
        if weighted:
            var r = _rf_fit_forest[False, False](request)
            f = Forest(r.offsets.copy(), r.colid.copy(), r.quesval.copy(), r.left_child.copy(), r.leaves.copy(),
                       r.n_trees, r.num_outputs, d)
        else:
            var r = _rf_fit_forest[False, True](request)
            f = Forest(r.offsets.copy(), r.colid.copy(), r.quesval.copy(), r.left_child.copy(), r.leaves.copy(),
                       r.n_trees, r.num_outputs, d)
    _ = staged^
    _ = request^
    _ = codes
    _ = y
    _ = weights
    return f^


def predict(f: Forest, x: List[Float32], rows: Int, groves: Bool = False) raises -> List[Float32]:
    """`predict_proba` (classifier: rows x classes votes) or `predict` (regressor: rows values) on the STRICT
    route (`sequential`) or the RESIDENT route (`parallel_groves`); `x` row-major."""
    var out = List[Float32](length=rows * f.outputs, fill=Float32(-1))
    var model = f.model()
    if groves:
        var handle = forest_prepare_host_native[True](model, f.features, f.trees, f.outputs, -1)
        try:
            var wrote = forest_predict_resident_host_native[True](handle, Int(x.unsafe_ptr()), Int(out.unsafe_ptr()),
                                                                  rows, f.features, f.outputs)
            require(wrote == rows, "resident predict wrote " + String(wrote) + " rows")
        finally:
            forest_release_host_native[True](handle)
    else:
        var handle = forest_strict_prepare_host_native[True](model, f.features, f.trees, f.outputs)
        try:
            var trees = forest_strict_predict_host_check[True](handle, rows, f.features, f.outputs)
            forest_strict_predict_host_walk[True](handle, Int(x.unsafe_ptr()), Int(out.unsafe_ptr()), rows,
                                                  f.features, trees, f.outputs)
        finally:
            forest_strict_release_host_native[True](handle)
    _ = model^
    _ = x
    return out^


def argmax_rows(p: List[Float32], rows: Int, k: Int) -> List[Int]:
    """The classifier's `predict`: the first maximum of each row's votes."""
    var out = List[Int](length=rows, fill=0)
    for r in range(rows):  # small-loop(oracle side: one prediction block)
        var best = 0
        for c in range(1, k):  # small-loop(the classes of one row)
            if p[r * k + c] > p[r * k + best]:
                best = c
        out[r] = best
    return out^


def tree_depth_leaves(f: Forest, tree: Int) raises -> Tuple[Int, Int]:
    var res = List[Int32](length=2, fill=Int32(0))
    var start = Int(f.offsets[tree])
    var cnt = Int(f.offsets[tree + 1]) - start
    tree_shape(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(f.left.unsafe_ptr()) + 4 * start), cnt,
               MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(res.unsafe_ptr())))
    return (Int(res[0]), Int(res[1]))


def accuracy(pred: List[Int], want: List[Int32]) -> Float64:
    var hit = 0
    for i in range(len(want)):  # small-loop(oracle: one held-out block)
        if pred[i] == Int(want[i]):
            hit += 1
    return Float64(hit) / Float64(len(want))


def r2(pred: List[Float32], want: List[Float32]) -> Float64:
    """scikit-learn's r2_score in float64."""
    var n = len(want)
    var mean = 0.0
    for v in want:  # small-loop(oracle: one held-out block)
        mean += Float64(v)
    mean /= Float64(n)
    var ss_res = 0.0
    var ss_tot = 0.0
    for i in range(n):  # small-loop(oracle: one held-out block)
        var e = Float64(want[i]) - Float64(pred[i])
        var t = Float64(want[i]) - mean
        ss_res += e * e
        ss_tot += t * t
    return 1.0 - ss_res / ss_tot
