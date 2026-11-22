# SPDX-License-Identifier: Apache-2.0
"""Host ExtraTrees fits (and DecisionTree*(splitter='random'), which fits one inner extra-trees tree) for the Mojo
checks that replaced numeric Python tests (lane bm-trees, 2026-11-22). Importable; no `main()`, no `test_*`.

A fit is what `ExtraTrees*.fit` runs on the host column: the door's configuration (bindings/trees_public_forest_door.mojo
`_et_config`: the 22 slots of `_mojolearn_trees.mojo` from python/mojolearn/extratrees.py's defaults, and for a
random-splitter tree `dt_random_fit`'s inner settings: one tree, no bootstrap, max_features None as 1.0, the seed 0
when None), `forest_request` (bindings/forest_config_run.mojo) with the deviance criteria's target check
(`forest_target_check`), then the host image's typed fit `et_host_fit_result` (core/mojolearn_et_host_run.mojo, what
`EtHostForest.fit` runs), kept and exported through the image's own registry operations, and predicted through its
sequential walk (`et_host_predict_walk`, XF_PREDICT). X is row-major (the door's `rowmajor_ok` fits); a classifier's
y is its float32 class codes. Each fit keeps its own parallel builder; callers run independent fits as case-table
rows across the host cores.
"""
from std.math import isfinite, min, max
from core.entry_table import entry_ref_unresolved
from core.forest_fit_request import FOREST_ARG_NO_INT
from bindings.forest_config_run import ForestConfig, forest_request, forest_target_check
from xtrees.trees_forest_run import ForestFit
from core.mojolearn_et_host_run import (
    ET_HOST_EXPORTS, et_host_export_into, et_host_fit_result, et_host_predict_walk, et_host_retain,
)
from checks.native_tests.py2mojo_forest_fits import Forest

#: python/mojolearn/extratrees.py `_MF_*` and `_CRITERION_*`
comptime MF_SQRT = -1
comptime MF_ALL = -3
comptime ET_GINI = 0
comptime ET_ENTROPY = 1
comptime ET_MSE = 2
comptime ET_POISSON = 4
comptime ET_GAMMA = 5
comptime ET_INVERSE_GAUSSIAN = 6


struct EtSpec(Copyable, Movable):
    """The constructor arguments a fit reads (python/mojolearn/extratrees.py defaults; -1 for None)."""
    var classifier: Bool
    var n_estimators: Int
    var max_depth: Int
    var max_leaf_nodes: Int
    #: the two max_features slots (`_max_features_slots`): a sentinel or count, and a float fraction
    var mf_spec: Int
    var mf_fraction: Float64
    var criterion: Int
    var bootstrap: Bool
    var seed: Int

    def __init__(out self, classifier: Bool, n_estimators: Int = 100, seed: Int = 0):
        """ExtraTreesClassifier / ExtraTreesRegressor defaults: max_features 'sqrt' / 1.0, gini / squared_error."""
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.max_depth = -1
        self.max_leaf_nodes = -1
        self.mf_spec = MF_SQRT if classifier else 0
        self.mf_fraction = 0.0 if classifier else 1.0
        self.criterion = ET_GINI if classifier else ET_MSE
        self.bootstrap = False
        self.seed = seed

    @staticmethod
    def random_tree(classifier: Bool, max_depth: Int, max_leaf_nodes: Int, seed: Int) -> Self:
        """`dt_random_fit`'s inner forest: one tree, no bootstrap, max_features 1.0 (the tree's None), the tree's
        depth, leaf cap, criterion (DecisionTree* defaults: gini / squared_error) and seed."""
        var s = Self(classifier, 1, seed)
        s.max_depth = max_depth
        s.max_leaf_nodes = max_leaf_nodes
        s.mf_spec = 0
        s.mf_fraction = 1.0
        return s^


def _int(mut c: ForestConfig, value: Int):
    c.ints.append(value)
    c.floats.append(Float64(value))
    c.bad.append(UInt8(0))


def _float(mut c: ForestConfig, value: Float64):
    c.ints.append(0)
    c.floats.append(value)
    c.bad.append(UInt8(FOREST_ARG_NO_INT))


def et_config(s: EtSpec) -> ForestConfig:
    """`_et_config`'s 22 slots: rows, cols, classes (filled by `forest_request`), n_estimators, max_depth,
    min_samples_split 2, min_samples_leaf 1, min_weight_fraction_leaf 0.0, the two max_features slots,
    min_impurity_decrease 0.0, bootstrap, oob_score 0, the seed, warm_start 0, ccp_alpha 0.0, class_weight 0,
    monotonic_cst 0, max_samples 0 (None), max_leaf_nodes, device 1, the criterion code."""
    var c = ForestConfig(True, List[Int](), List[Float64](), List[UInt8]())
    _int(c, 0)
    _int(c, 0)
    _int(c, 0)
    _int(c, s.n_estimators)
    _int(c, s.max_depth)
    _int(c, 2)
    _int(c, 1)
    _float(c, 0.0)
    if s.mf_spec != 0:
        _int(c, s.mf_spec)
        _float(c, 0.0)
    else:
        _int(c, 0)
        _float(c, s.mf_fraction)
    _float(c, 0.0)
    _int(c, 1 if s.bootstrap else 0)
    _int(c, 0)
    _int(c, s.seed)
    _int(c, 0)
    _float(c, 0.0)
    _int(c, 0)
    _int(c, 0)
    _int(c, 0)
    _int(c, s.max_leaf_nodes)
    _int(c, 1)
    _int(c, s.criterion)
    if not s.classifier:
        # `_target_codes`: the deviance criteria's codes the target check reads
        c.target_code = s.criterion
        c.poisson = ET_POISSON
        c.gamma = ET_GAMMA
        c.inverse_gaussian = ET_INVERSE_GAUSSIAN
    return c^


def et_fit(s: EtSpec, x: List[Float32], n: Int, d: Int, y: List[Float32], classes: Int) raises -> Forest:
    """One host fit. `x` row-major n x d; `y` the float32 class codes (classifier) or targets (regressor).
    Refuses (in the door's words) a target the deviance criterion cannot fit."""
    var config = et_config(s)
    if not s.classifier:
        var finite = True
        var lo = Float64(y[0])
        var hi = Float64(y[0])
        for v in y:  # small-loop(the target statistics the door reads, `_target_check`)
            finite = finite and isfinite(v)
            lo = min(lo, Float64(v))
            hi = max(hi, Float64(v))
        forest_target_check(config, finite, lo, hi)
    var request = forest_request(config, entry_ref_unresolved(), n, d, classes if s.classifier else 0,
                                 Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), ForestFit(s.classifier, rowmajor_ok=True),
                                 False)
    request.gil_free = True
    var handle: Int
    if s.classifier:
        handle = et_host_retain(et_host_fit_result[True, True](request))
    else:
        handle = et_host_retain(et_host_fit_result[False, True](request))
    _ = request^
    _ = x
    _ = y
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    var sizes = List[Int](length=16, fill=0)
    registry[].sizes_into(handle, Int(sizes.unsafe_ptr()))
    var trees = sizes[0]
    var nodes = sizes[1]
    var outputs = sizes[2]
    var offsets = List[Int32](length=trees + 1, fill=Int32(0))
    var colid = List[Int32](length=nodes, fill=Int32(0))
    var quesval = List[Float32](length=nodes, fill=Float32(0))
    var left = List[Int32](length=nodes, fill=Int32(0))
    var leaves = List[Float32](length=nodes * outputs, fill=Float32(0))
    var dst: List[Int] = [Int(offsets.unsafe_ptr()), Int(colid.unsafe_ptr()), Int(quesval.unsafe_ptr()),
                          Int(left.unsafe_ptr()), Int(leaves.unsafe_ptr())]
    try:
        et_host_export_into(handle, dst, trees, nodes, outputs)
    finally:
        registry[].release(handle)
    _ = dst^
    return Forest(offsets^, colid^, quesval^, left^, leaves^, trees, outputs, d)


def et_predict(f: Forest, x: List[Float32], rows: Int) raises -> List[Float32]:
    """`predict_proba` (classifier: rows x classes votes) or `predict` (regressor) on the image's sequential walk."""
    var out = List[Float32](length=rows * f.outputs, fill=Float32(-1))
    var model = f.model()
    var wrote = et_host_predict_walk(model, Int(x.unsafe_ptr()), Int(out.unsafe_ptr()), rows, f.features, f.trees,
                                     f.outputs)
    if wrote != rows:
        raise Error("et predict wrote " + String(wrote) + " rows")
    _ = model^
    _ = x
    return out^
