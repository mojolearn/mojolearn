# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the ExtraTrees estimators (`extratrees/`).

Kept in its OWN extension for the same reason `_mojolearn_estimators` and
`_mojolearn_gbdt` are: an independently changing binding stops being a merge
point, and the extratrees lane changes independently of both. Arrays cross as
borrowed NumPy addresses; all device buffers and contexts live for one call
and no pointer is retained.

THE MODEL CROSSES AS FLAT ARRAYS, NOT AS A HANDLE. The gbdt boundary is
handle-free through its model TEXT; this lane has no text format, so the
forest crosses as the arrays `TreeMetaDataNode` already is (deviation 146's
layout argument): per-node `colid` / `quesval` / `left_child_id`, the flat
`vector_leaf`, and a `tree_offsets` prefix so tree `t` is the node range
`[offsets[t], offsets[t+1])`. `et_predict` rebuilds the forest from those
arrays and calls the PORTED `forest_vote` -- the traversal is
`decisiontree.cuh:394-413` through `flatnode.mojo`, not a reimplementation at
this boundary. `instance_count` and `best_metric_val` are not carried: the
traversal never reads either (`flatnode.mojo` says so of `best_metric_val`
explicitly), and a field the boundary carries but nothing reads is the
present-but-dead state rule 3 forbids.

FIT RETURNS PYTHON LISTS, one element at a time under the GIL. On a
100-tree forest of covtype-sized trees that is a few million appends and it
is the dominant cost of the CALL (not of the fit). Named rather than hidden;
the fix, if it is ever needed, is a two-call sizes-then-fill protocol or a
bytes serialization, both of which change this surface.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from mojo_only.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext

from extratrees.estimator import (
    ExtraTreesConfig,
    FitResult,
    fit_extra_trees_classifier,
    fit_extra_trees_classifier_device,
    fit_extra_trees_regressor,
    fit_extra_trees_regressor_device,
)
from extratrees.ported.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_MSE,
)
from extratrees.ported.decisiontree.flatnode import (
    SparseTreeNode,
    TreeMetaDataNode,
)
from extratrees.ported.randomforest.randomforest import Forest, forest_vote


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def _copy_f32(addr: PythonObject, n: Int) raises -> List[Float32]:
    """Borrowed NumPy memory into an owned List, read while the GIL-holding
    caller keeps the array alive (the `_arrays.py` contract)."""
    var p = _f32_ptr(Int(py=addr))
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(p[i])
    return out^


comptime N_FIT_PARAMS = 22
"""`params` for both fit entry points, in this exact order -- the wrapper
names the same order in the same words, because a silent reordering here is a
wrong answer rather than a failure:

     0  n_rows
     1  n_features
     2  n_classes        (classifier only; MUST be 0 for the regressor)
     3  n_estimators
     4  max_depth        (-1 = sklearn's None)
     5  min_samples_split
     6  min_samples_leaf
     7  min_weight_fraction_leaf  (float)
     8  max_features_spec         (positive count, 0 = fraction at slot 9,
                                   -1 sqrt, -2 log2, -3 all -- the
                                   estimator.mojo sentinels)
     9  max_features_fraction     (float)
    10  min_impurity_decrease     (float)
    11  bootstrap        (0/1)
    12  oob_score        (0/1)
    13  random_state
    14  warm_start       (0/1)
    15  ccp_alpha        (float)
    16  has_class_weight (0/1)
    17  has_monotonic_cst(0/1)
    18  max_samples      (sklearn's max_samples RESOLVED TO A COUNT by the
                          wrapper: 0 = None; honoured with bootstrap=1,
                          refused by name otherwise -- DEVIATION 460. This
                          slot was a 0/1 `max_samples_set` flag while
                          bootstrap was refused)
    19  max_leaf_nodes   (-1 = sklearn's None)
    20  device           (0 host, 1 GPU)
    21  criterion        (the `decisiontree.mojo` CRITERION_* code:
                          0 GINI, 1 ENTROPY for the classifier; 2 MSE for
                          the regressor -- DEVIATION 459)

Every refused sklearn parameter RIDES THROUGH so the refusal fires in
`refuse_unported` by name, in one place, rather than being re-decided at this
boundary. The criterion RIDES AS SLOT 21 since DEVIATION 459 (it did not
while each entry point had exactly one criterion): `et_classifier_fit`
admits GINI and ENTROPY, `et_regressor_fit` admits MSE, and the OTHER
criteria are refused by name in the WRAPPER, which cites the same
UNPORTED.tsv rows validity_check does.
"""


def _config_from(
    params: PythonObject, base: ExtraTreesConfig
) raises -> ExtraTreesConfig:
    """Slots 3-19 and 21 written OVER `base`. Read under the GIL.

    `base` carries the defaults the slots then overwrite; since DEVIATION
    459 the criterion is slot 21 and `base`'s criterion is only a default
    the slot replaces. `et_classifier_fit` passes `ExtraTreesConfig()` and
    `et_regressor_fit` passes `ExtraTreesConfig().for_regression()`. Every
    slot overwrites its field AFTER the base is taken,
    so no default can shadow what the caller sent -- DEVIATION 458 is what
    happened when the regressor applied `for_regression()` the other way round
    and its `max_features_spec = ALL` default overwrote slots 8-9 of every
    regressor fit.
    """
    var config = base.copy()
    config.n_estimators = Int32(Int(py=params[3]))
    config.max_depth = Int32(Int(py=params[4]))
    config.min_samples_split = Int32(Int(py=params[5]))
    config.min_samples_leaf = Int32(Int(py=params[6]))
    config.min_weight_fraction_leaf = Float64(py=params[7])
    config.max_features_spec = Int(py=params[8])
    config.max_features_fraction = Float64(py=params[9])
    config.min_impurity_decrease = Float32(Float64(py=params[10]))
    config.bootstrap = Int(py=params[11]) != 0
    config.oob_score = Int(py=params[12]) != 0
    config.random_state = UInt64(Int(py=params[13]))
    config.warm_start = Int(py=params[14]) != 0
    config.ccp_alpha = Float64(py=params[15])
    config.has_class_weight = Int(py=params[16]) != 0
    config.has_monotonic_cst = Int(py=params[17]) != 0
    config.max_samples = Int(py=params[18])
    config.max_leaf_nodes = Int32(Int(py=params[19]))
    # slot 21, DEVIATION 459: the criterion code, written over the base's
    # default like every other slot.
    config.criterion = Int32(Int(py=params[21]))
    return config^


def _forest_out(result: FitResult) raises -> PythonObject:
    """The fitted forest as `[offsets, colid, quesval, left_child, leaves,
    meta]`, all Python lists.

    `meta` is `[n_trees, num_outputs, depth_cap_bound, resolved_max_depth,
    max_features_count, n_sampled_rows]` -- the last four are what
    `FitResult`/`FitPlan` exist to report rather than hide (`n_sampled_rows`
    is the bootstrap sample size the fit used, 0 without bootstrap;
    DEVIATION 460)."""
    var offsets = Python.list()
    var colid = Python.list()
    var quesval = Python.list()
    var left_child = Python.list()
    var leaves = Python.list()
    var total = 0
    offsets.append(PythonObject(0))
    for t in range(len(result.forest.trees)):
        var n = result.forest.trees[t].num_nodes()
        total += n
        offsets.append(PythonObject(total))
        for i in range(n):
            var node = result.forest.trees[t].sparsetree[i]
            colid.append(PythonObject(Int(node.colid)))
            quesval.append(PythonObject(Float64(node.quesval)))
            left_child.append(PythonObject(Int(node.left_child_id)))
        for i in range(len(result.forest.trees[t].vector_leaf)):
            leaves.append(
                PythonObject(Float64(result.forest.trees[t].vector_leaf[i]))
            )
    var meta = Python.list()
    meta.append(PythonObject(Int(result.forest.n_trees)))
    meta.append(PythonObject(Int(result.forest.num_outputs)))
    meta.append(PythonObject(1 if result.depth_cap_bound else 0))
    meta.append(PythonObject(Int(result.plan.params.max_depth)))
    meta.append(PythonObject(result.plan.max_features_count))
    meta.append(PythonObject(Int(result.plan.n_sampled_rows)))
    var out = Python.list()
    out.append(offsets)
    out.append(colid)
    out.append(quesval)
    out.append(left_child)
    out.append(leaves)
    out.append(meta)
    return out


def et_classifier_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit `ExtraTreesClassifier`. `x` is COLUMN-major float32
    (n_rows * n_features), `y` is float32 class CODES in [0, n_classes).
    See `N_FIT_PARAMS` for `params`; returns `_forest_out`'s lists."""
    if len(params) != N_FIT_PARAMS:
        raise Error(
            "et_classifier_fit: params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_classes = Int(py=params[2])
    var device = Int(py=params[20]) != 0
    var config = _config_from(params, ExtraTreesConfig())
    if config.criterion != CRITERION_GINI and config.criterion != CRITERION_ENTROPY:
        raise Error(
            "et_classifier_fit: criterion (slot 21) must be GINI (0) or"
            " ENTROPY (1); got " + String(config.criterion)
        )
    var x = _copy_f32(x_addr, n_rows * n_features)
    var y = _copy_f32(y_addr, n_rows)

    var result: FitResult
    with GILReleased(Python()):
        if device:
            var ctx = DeviceContext()
            result = fit_extra_trees_classifier_device(
                ctx,
                x,
                y,
                Int32(n_rows),
                Int32(n_features),
                Int32(n_classes),
                config,
            )
        else:
            result = fit_extra_trees_classifier(
                x, y, Int32(n_rows), Int32(n_features), Int32(n_classes),
                config,
            )
    return _forest_out(result)


def et_regressor_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit `ExtraTreesRegressor`. Same contract; slot 2 MUST be 0."""
    if len(params) != N_FIT_PARAMS:
        raise Error(
            "et_regressor_fit: params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if Int(py=params[2]) != 0:
        raise Error("et_regressor_fit: n_classes (slot 2) must be 0")
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var device = Int(py=params[20]) != 0
    var config = _config_from(params, ExtraTreesConfig().for_regression())
    if config.criterion != CRITERION_MSE:
        raise Error(
            "et_regressor_fit: criterion (slot 21) must be MSE (2); got "
            + String(config.criterion)
        )
    var x = _copy_f32(x_addr, n_rows * n_features)
    var y = _copy_f32(y_addr, n_rows)

    var result: FitResult
    with GILReleased(Python()):
        if device:
            var ctx = DeviceContext()
            result = fit_extra_trees_regressor_device(
                ctx, x, y, Int32(n_rows), Int32(n_features), config
            )
        else:
            result = fit_extra_trees_regressor(
                x, y, Int32(n_rows), Int32(n_features), config
            )
    return _forest_out(result)


def et_predict_binding(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The forest's averaged vote per row, through the PORTED traversal.

    Model arrays are int32/float32 as `_forest_out` laid them out (the
    wrapper converts the lists once and keeps NumPy arrays). `x` here is
    ROW-major (the traversal reads `row[offset + colid]`). `out` is
    n_rows * num_outputs float32 and receives `forest_vote`'s average --
    per-class probabilities for the classifier (argmax is the wrapper's,
    exactly as `RandomForest::predict` argmaxes over `predict_proba`), the
    mean prediction for the regressor. `params` is `[n_rows, n_features,
    n_trees, num_outputs]`. Returns rows written.
    """
    if len(params) != 4:
        raise Error(
            "et_predict: params must hold [n_rows, n_features, n_trees,"
            " num_outputs], got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_trees = Int(py=params[2])
    var num_outputs = Int(py=params[3])
    if n_trees < 1 or num_outputs < 1:
        raise Error("et_predict: n_trees and num_outputs must be >= 1")

    var offsets_p = _i32_ptr(Int(py=offsets_addr))
    var colid_p = _i32_ptr(Int(py=colid_addr))
    var quesval_p = _f32_ptr(Int(py=quesval_addr))
    var left_p = _i32_ptr(Int(py=left_child_addr))
    var leaves_p = _f32_ptr(Int(py=leaves_addr))
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))

    var wrote = 0
    with GILReleased(Python()):
        # Rebuild the forest in `TreeMetaDataNode`'s own layout so the
        # traversal that runs is flatnode.mojo's, not a copy of it here.
        # `instance_count` and `best_metric_val` are zero: the traversal
        # reads neither, and the docstring above says so where a caller can
        # see it.
        var forest = Forest(Int32(num_outputs))
        for t in range(n_trees):
            var lo = Int(offsets_p[t])
            var hi = Int(offsets_p[t + 1])
            if lo < 0 or hi < lo:
                raise Error("et_predict: tree_offsets are not a prefix scan")
            var nodes = List[SparseTreeNode[DType.float32]](
                capacity=hi - lo
            )
            var vleaf = List[Float32](capacity=(hi - lo) * num_outputs)
            for i in range(lo, hi):
                nodes.append(
                    SparseTreeNode[DType.float32](
                        colid_p[i], quesval_p[i], 0.0, left_p[i], 0
                    )
                )
                for k in range(num_outputs):
                    vleaf.append(leaves_p[i * num_outputs + k])
            forest.trees.append(
                TreeMetaDataNode[DType.float32](
                    Int32(t), 0, 0, Int32(num_outputs), vleaf^, nodes^
                )
            )
        forest.n_trees = Int32(n_trees)

        var row = List[Float32](capacity=n_rows * n_features)
        for i in range(n_rows * n_features):
            row.append(xp[i])
        for r in range(n_rows):
            var vote = forest_vote(forest, row, r * n_features)
            for k in range(num_outputs):
                op[r * num_outputs + k] = vote[k]
            wrote += 1
    return PythonObject(wrote)


def trees_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `mojo_only/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_trees() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_trees")
        m.def_function[trees_vendor_binding]("trees_vendor")
        m.def_function[et_classifier_fit_binding]("et_classifier_fit")
        m.def_function[et_regressor_fit_binding]("et_regressor_fit")
        m.def_function[et_predict_binding]("et_predict")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_trees: ", e))
