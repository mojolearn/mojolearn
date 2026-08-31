# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the RandomForest estimators (`ensemble/`).

Kept in its OWN extension for the same reason `_mojolearn_trees` is: an
independently changing binding stops being a merge point, and the ensemble
lane changes independently of the extratrees lane. Arrays cross as borrowed
NumPy addresses; all device buffers and contexts live for one call and no
pointer is retained.

THE MODEL CROSSES AS FLAT ARRAYS, exactly the `_mojolearn_trees` protocol
(deviation 146's layout argument): per-node `colid` / `quesval` /
`left_child_id`, the flat `vector_leaf`, and a `tree_offsets` prefix so tree
`t` is the node range `[offsets[t], offsets[t+1])`. The predict bindings
rebuild `RandomForestMetaData` from those arrays and call the PORTED
`RandomForest.predict` / `predict_proba` (`randomforest.cuh:382-436`), not a
reimplementation at this boundary. `instance_count` and `best_metric_val`
are not carried: the traversal reads neither.

WHY THIS BINDS `ensemble/` AND NOT `extratrees/ported/randomforest`: the
extratrees surface refuses `bootstrap=True` by name because ITS copy of the
row sampler has no caller; `ensemble/` is the dedicated cuML RandomForest
port and its `fit_forest` IS `RandomForest::fit` (`randomforest.cuh:286-370`)
with the with-replacement `RowSampler` wired. This extension is that
sampler's first Python caller.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from mojo_only.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceBuffer, DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import (
    BinScales,
    ClassificationBin,
    RegressionBin,
)
from mojo_only.fixed_point import choose_scale
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)
from ensemble.decisiontree.decisiontree import (
    ENTROPY,
    GAMMA,
    GINI,
    INVERSE_GAUSSIAN,
    MSE,
    POISSON,
    DecisionTreeParams,
    TreeMetaDataNode,
    criterion_name,
)
from ensemble.flatnode import SparseTreeNode
from ensemble.randomforest import (
    CLASSIFICATION,
    REGRESSION,
    RF_params,
    RandomForest,
    RandomForestMetaData,
    fit_forest,
)

comptime DT = DType.float32
comptime CLT = DType.int32
comptime RLT = DType.float32
comptime ClsObj = ClassificationObjectiveFunction[DT, CLT, ClassificationBin]
comptime RegObj = RegressionObjectiveFunction[DT, RLT, RegressionBin]


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


comptime N_RF_FIT_PARAMS = 16
"""`params` for both fit entry points, in this exact order -- the wrapper
names the same order in the same words:

     0  n_rows
     1  n_cols
     2  n_classes             (classifier only; MUST be 0 for the regressor)
     3  n_trees
     4  max_depth
     5  max_leaves            (-1 = unlimited, cuML's own sentinel)
     6  max_features_fraction (float in (0, 1]; the WRAPPER resolves
                               'sqrt'/'log2'/counts to a fraction, exactly
                               as cuML's python layer does)
     7  max_n_bins
     8  min_samples_leaf
     9  min_samples_split
    10  min_impurity_decrease (float)
    11  bootstrap             (0/1)
    12  max_samples           (float; ignored and reset to 1.0 by the port
                               when bootstrap is 0, `randomforest.cuh:304`)
    13  seed
    14  n_streams             (cuML's python default is 4; DEVIATION 117)
    15  max_batch_size        (cuML default 4096)

The split criterion is NOT a slot: it crosses as its own integer argument
(`criterion`, the `CRITERION` enumerator value from `decisiontree.mojo`)
so the two fit entry points can gate it by LABEL TYPE before the engine
sees it -- see DEVIATION 407 at `_check_criterion` below.
"""

# DEVIATION 407 (2026-08-23, RF lane). THE CRITERION SELECTOR CROSSES THE
# PYTHON BOUNDARY.
# THEIRS: `randomforest_common.pyx:104-151` maps the strings
# 'gini'/'entropy'/'mse'/'poisson'/'gamma'/'inverse_gaussian' (and the
# digits '0'..'7') onto `CRITERION`, refuses MAE by NotImplementedError, and
# passes the enumerator down as `split_criterion`; nothing in their python
# layer checks that the enumerator suits the LABEL TYPE. The C++ `default:`
# arm of `GainPerSplit` (`objectives.cuh:132-136`, `:331-338`) returns
# `-max()` for every candidate, so a regression enumerator handed to the
# classifier (or vice versa) fits a forest of STUMPS without a word.
# OURS: the wrapper maps sklearn-shaped names onto the same enumerators
# (`python/mojolearn/randomforest.py::_CLS_CRITERIA` / `_REG_CRITERIA`)
# and each fit binding REFUSES BY NAME an enumerator its objective has no
# arm for -- a silent stump is the "reached but inert" failure class
# `ensemble/mojo_only/criteria_check.mojo` arm A/B exists to catch, and
# the Python surface must not reopen it. Until this deviation the binding
# hardcoded GINI / MSE and the wrapper refused every other name; that text
# is gone.


def _cls_criteria() -> List[Int]:
    """The enumerators `ClassificationObjectiveFunction.GainPerSplit` has
    an arm for (`objectives.cuh:132-136`)."""
    return [GINI, ENTROPY]


def _reg_criteria() -> List[Int]:
    """The enumerators `RegressionObjectiveFunction.GainPerSplit` has an
    arm for (`objectives.cuh:331-338`); MAE is enumerated but armless,
    theirs and ours."""
    return [MSE, POISSON, GAMMA, INVERSE_GAUSSIAN]


def _check_criterion(
    who: String, criterion: Int, allowed: List[Int]
) raises:
    """DEVIATION 407: refuse an enumerator the objective has no arm for."""
    for i in range(len(allowed)):
        if allowed[i] == criterion:
            return
    var names = String("")
    for i in range(len(allowed)):
        if i > 0:
            names += "/"
        names += criterion_name(allowed[i])
    raise Error(
        who
        + ": split criterion "
        + criterion_name(criterion)
        + " ("
        + String(criterion)
        + ") has no arm in this objective's GainPerSplit"
        + " (objectives.mojo, DEVIATION 407); accepted: "
        + names
    )


def _rf_params_from(params: PythonObject, criterion: Int) raises -> RF_params:
    """Slots 3-15 into `RF_params`, read under the GIL."""
    return RF_params(
        n_trees=Int32(Int(py=params[3])),
        bootstrap=Int(py=params[11]) != 0,
        max_samples=Float32(Float64(py=params[12])),
        seed=UInt64(Int(py=params[13])),
        n_streams=Int32(Int(py=params[14])),
        tree_params=DecisionTreeParams(
            max_depth=Int32(Int(py=params[4])),
            max_leaves=Int32(Int(py=params[5])),
            max_features=Float32(Float64(py=params[6])),
            max_n_bins=Int32(Int(py=params[7])),
            min_samples_leaf=Int32(Int(py=params[8])),
            min_samples_split=Int32(Int(py=params[9])),
            split_criterion=criterion,
            min_impurity_decrease=Float32(Float64(py=params[10])),
            max_batch_size=Int32(Int(py=params[15])),
        ),
    )


def _forest_out(
    forest: RandomForestMetaData[DT, DT]
) raises -> PythonObject:
    """The fitted forest as `[offsets, colid, quesval, left_child, leaves,
    meta]`, all Python lists; `meta` is `[n_trees, num_outputs]`.
    `vector_leaf` is indexed BY NODE (`decisiontree.cuh:387`), sized
    nodes * num_outputs with internal-node slots dead, and crosses as-is so
    the predict side can index it the way the traversal does."""
    var offsets = Python.list()
    var colid = Python.list()
    var quesval = Python.list()
    var left_child = Python.list()
    var leaves = Python.list()
    var num_outputs = 1
    if len(forest.trees) > 0:
        num_outputs = Int(forest.trees[0].num_outputs)
    var total = 0
    offsets.append(PythonObject(0))
    for t in range(len(forest.trees)):
        ref tree = forest.trees[t]
        var n = len(tree.sparsetree)
        total += n
        offsets.append(PythonObject(total))
        for i in range(n):
            ref node = tree.sparsetree[i]
            colid.append(PythonObject(Int(node.ColumnId())))
            quesval.append(PythonObject(Float64(node.QueryValue())))
            left_child.append(PythonObject(Int(node.LeftChildId())))
        for i in range(len(tree.vector_leaf)):
            leaves.append(PythonObject(Float64(tree.vector_leaf[i])))
    var meta = Python.list()
    meta.append(PythonObject(len(forest.trees)))
    meta.append(PythonObject(num_outputs))
    var out = Python.list()
    out.append(offsets)
    out.append(colid)
    out.append(quesval)
    out.append(left_child)
    out.append(leaves)
    out.append(meta)
    return out


def _forest_out_i32(
    forest: RandomForestMetaData[DT, CLT]
) raises -> PythonObject:
    """`_forest_out` for the classifier's label type. Two copies because the
    two metadata types do not unify; the bodies must stay identical."""
    var offsets = Python.list()
    var colid = Python.list()
    var quesval = Python.list()
    var left_child = Python.list()
    var leaves = Python.list()
    var num_outputs = 1
    if len(forest.trees) > 0:
        num_outputs = Int(forest.trees[0].num_outputs)
    var total = 0
    offsets.append(PythonObject(0))
    for t in range(len(forest.trees)):
        ref tree = forest.trees[t]
        var n = len(tree.sparsetree)
        total += n
        offsets.append(PythonObject(total))
        for i in range(n):
            ref node = tree.sparsetree[i]
            colid.append(PythonObject(Int(node.ColumnId())))
            quesval.append(PythonObject(Float64(node.QueryValue())))
            left_child.append(PythonObject(Int(node.LeftChildId())))
        for i in range(len(tree.vector_leaf)):
            leaves.append(PythonObject(Float64(tree.vector_leaf[i])))
    var meta = Python.list()
    meta.append(PythonObject(len(forest.trees)))
    meta.append(PythonObject(num_outputs))
    var out = Python.list()
    out.append(offsets)
    out.append(colid)
    out.append(quesval)
    out.append(left_child)
    out.append(leaves)
    out.append(meta)
    return out


def rf_classifier_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
    criterion: PythonObject,
) raises -> PythonObject:
    """Fit the cuML-port RandomForest classifier. `x` is COLUMN-major
    float32 (n_rows * n_cols, the layout `fit_forest`'s default expects);
    `y` is int32 class CODES in [0, n_classes). See `N_RF_FIT_PARAMS`.
    `criterion` is GINI (0) or ENTROPY (1); anything else is refused by
    name (DEVIATION 407)."""
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            "rf_classifier_fit: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_classes = Int(py=params[2])
    if n_classes < 2:
        raise Error("rf_classifier_fit: n_classes must be >= 2")
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _i32_ptr(Int(py=y_addr))
    var crit = Int(py=criterion)
    _check_criterion("rf_classifier_fit", crit, _cls_criteria())
    var rf_params = _rf_params_from(params, crit)

    var forest: RandomForestMetaData[DT, CLT]
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
        var hy = ctx.enqueue_create_host_buffer[CLT](n_rows)
        ctx.synchronize()
        for i in range(n_rows * n_cols):
            hx.unsafe_ptr().unsafe_store(i, xp[i])
        for i in range(n_rows):
            hy.unsafe_ptr().unsafe_store(i, yp[i])
        var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
        ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
        var dy = ctx.enqueue_create_buffer[CLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        # unweighted: `fit_forest` reads sample weights only through
        # `sample_weight_host`, which stays empty; the device buffer is the
        # 1-element placeholder `rf_bench.mojo` also passes.
        var dsw = ctx.enqueue_create_buffer[DT](1)
        ctx.synchronize()
        forest = fit_forest[ClsObj](
            ctx, dx, dy, dsw, n_rows, n_cols, n_classes, rf_params
        )
        ctx.synchronize()
        _ = dx^
        _ = dy^
        _ = dsw^
        _ = hx^
        _ = hy^
        # DEVIATION 1946: THE CONTEXT DIES LAST. Mojo destroys a value at its
        # LAST USE, so without this line `ctx`'s last use is the
        # `synchronize()` above and the five buffers -- two of them PINNED
        # HOST allocations -- are freed against a context that is already
        # gone. `ensemble/mojo_only/rf_ctx_order_probe.mojo` is that ordering
        # in 60 lines with no Python. It is the same class as DEVIATION 1944
        # (a device buffer freed against a context that is not the live one),
        # and it is why the pure-Mojo probe passed where this binding hung:
        # `rf_ctx_probe.mojo::one_fit` takes `ctx` as a BORROWED argument, so
        # the caller's frame keeps it alive past every release.
        _ = ctx^
    return _forest_out_i32(forest)


def rf_regressor_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
    criterion: PythonObject,
) raises -> PythonObject:
    """Fit the cuML-port RandomForest regressor. Same contract; slot 2
    MUST be 0 and `y` is float32. `criterion` is MSE (2), POISSON (4),
    GAMMA (5) or INVERSE_GAUSSIAN (6); anything else is refused by name
    (DEVIATION 407). The log criteria are DEVIATION 406's: they RUN in
    both numeric modes, comparable to cuML in neither."""
    if len(params) != N_RF_FIT_PARAMS:
        raise Error(
            "rf_regressor_fit: params must hold "
            + String(N_RF_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    if Int(py=params[2]) != 0:
        raise Error("rf_regressor_fit: n_classes (slot 2) must be 0")
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var crit = Int(py=criterion)
    _check_criterion("rf_regressor_fit", crit, _reg_criteria())
    var rf_params = _rf_params_from(params, crit)

    var forest: RandomForestMetaData[DT, RLT]
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
        var hy = ctx.enqueue_create_host_buffer[RLT](n_rows)
        ctx.synchronize()
        for i in range(n_rows * n_cols):
            hx.unsafe_ptr().unsafe_store(i, xp[i])
        for i in range(n_rows):
            hy.unsafe_ptr().unsafe_store(i, yp[i])
        var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
        ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
        var dy = ctx.enqueue_create_buffer[RLT](n_rows)
        ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
        var dsw = ctx.enqueue_create_buffer[DT](1)
        ctx.synchronize()
        # THE LABEL SCALE IS NOT OPTIONAL. `RegressionBin` accumulates
        # `label_sum` in fixed point through `BinScales.label_scale`
        # (DEVIATION 101b), and the host chooses the scale once per fit
        # from the sum of label magnitudes (`fixed_point.choose_scale`).
        # The unit scale TRUNCATES every |label| < 1 to zero, which fits a
        # forest of zero-leaved stumps -- the build gate caught exactly
        # that. The weight plane stays 1.0: unweighted `Weight()` is a
        # count, which is exact.
        var mag = Float64(0.0)
        for i in range(n_rows):
            var v = Float64(yp[i])
            mag += v if v >= 0.0 else -v
        var scales = BinScales(
            Float32(choose_scale(mag, n_rows)), Float32(1.0)
        )
        # `n_unique_labels` is 1 for regression, exactly what
        # `rf_regressor_fit`'s cuML counterpart passes.
        forest = fit_forest[RegObj](
            ctx, dx, dy, dsw, n_rows, n_cols, 1, rf_params, scales
        )
        ctx.synchronize()
        _ = dx^
        _ = dy^
        _ = dsw^
        _ = hx^
        _ = hy^
        # DEVIATION 1946, as in `rf_classifier_fit_binding`: the context
        # outlives every buffer created on it.
        _ = ctx^
    return _forest_out(forest)


def _rebuild_trees(
    offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    colid_p: MutPointer[Int32, MutUntrackedOrigin],
    quesval_p: MutPointer[Float32, MutUntrackedOrigin],
    left_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    n_trees: Int,
    num_outputs: Int,
) raises -> List[TreeMetaDataNode[DT]]:
    """The flat arrays back into `TreeMetaDataNode`'s own layout, so the
    traversal that runs is the port's. `instance_count` and
    `best_metric_val` are zero: the traversal reads neither."""
    var trees = List[TreeMetaDataNode[DT]](capacity=n_trees)
    for t in range(n_trees):
        var lo = Int(offsets_p[t])
        var hi = Int(offsets_p[t + 1])
        if lo < 0 or hi < lo:
            raise Error("rf_predict: tree_offsets are not a prefix scan")
        var nodes = List[SparseTreeNode[DT]](capacity=hi - lo)
        var vleaf = List[Float32](capacity=(hi - lo) * num_outputs)
        for i in range(lo, hi):
            nodes.append(
                SparseTreeNode[DT](
                    colid_p[i], quesval_p[i], 0.0, Int64(left_p[i]), 0
                )
            )
            for k in range(num_outputs):
                vleaf.append(leaves_p[i * num_outputs + k])
        trees.append(
            TreeMetaDataNode[DT](
                Int32(t), 0, 0, 0.0, vleaf^, nodes^, Int32(num_outputs)
            )
        )
    return trees^


def _default_rf_params(n_trees: Int) raises -> RF_params:
    """A benign `RF_params` for the predict-side metadata: predict reads the
    forest's trees, not these knobs, but the structs require one."""
    return RF_params(
        n_trees=Int32(n_trees),
        bootstrap=True,
        max_samples=Float32(1.0),
        seed=UInt64(0),
        n_streams=Int32(4),
        tree_params=DecisionTreeParams(
            max_depth=Int32(16),
            max_leaves=Int32(-1),
            max_features=Float32(1.0),
            max_n_bins=Int32(128),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            split_criterion=GINI,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(4096),
        ),
    )


def rf_predict_proba_binding(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Classifier probabilities through the PORTED `predict_proba`
    (`predict` stopped one line early, per its docstring). Model arrays are
    int32/float32 as `_forest_out` laid them out; `x` is ROW-major; `out`
    receives n_rows * num_outputs float32. `params` is `[n_rows, n_cols,
    n_trees, num_outputs]`. The wrapper's argmax over these IS cuML's
    `predict`. Returns rows written."""
    if len(params) != 4:
        raise Error(
            "rf_predict_proba: params must hold [n_rows, n_cols, n_trees,"
            " num_outputs], got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_trees = Int(py=params[2])
    var num_outputs = Int(py=params[3])
    if n_trees < 1 or num_outputs < 2:
        raise Error(
            "rf_predict_proba: n_trees must be >= 1 and num_outputs >= 2"
        )
    var offsets_p = _i32_ptr(Int(py=offsets_addr))
    var colid_p = _i32_ptr(Int(py=colid_addr))
    var quesval_p = _f32_ptr(Int(py=quesval_addr))
    var left_p = _i32_ptr(Int(py=left_child_addr))
    var leaves_p = _f32_ptr(Int(py=leaves_addr))
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))

    var wrote = 0
    with GILReleased(Python()):
        var rf_params = _default_rf_params(n_trees)
        var trees = _rebuild_trees(
            offsets_p, colid_p, quesval_p, left_p, leaves_p,
            n_trees, num_outputs,
        )
        var forest = RandomForestMetaData[DT, CLT](
            trees^, rf_params, Int32(n_cols)
        )
        var rf = RandomForest[DT, CLT](
            rf_params=rf_params, rf_type=CLASSIFICATION
        )
        var rows = List[Float32](capacity=n_rows * n_cols)
        for i in range(n_rows * n_cols):
            rows.append(xp[i])
        var probs = List[Float32](capacity=n_rows * num_outputs)
        for _ in range(n_rows * num_outputs):
            probs.append(0.0)
        rf.predict_proba(rows, n_rows, n_cols, probs, forest)
        for i in range(n_rows * num_outputs):
            op[i] = probs[i]
        wrote = n_rows
    return PythonObject(wrote)


def rf_predict_reg_binding(
    offsets_addr: PythonObject,
    colid_addr: PythonObject,
    quesval_addr: PythonObject,
    left_child_addr: PythonObject,
    leaves_addr: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Regressor predictions through the PORTED `RandomForest.predict`.
    Same array contract; `out` is n_rows float32; `params` is `[n_rows,
    n_cols, n_trees, 1]`. Returns rows written."""
    if len(params) != 4:
        raise Error(
            "rf_predict_reg: params must hold [n_rows, n_cols, n_trees, 1],"
            " got "
            + String(len(params))
        )
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_trees = Int(py=params[2])
    var num_outputs = Int(py=params[3])
    if n_trees < 1 or num_outputs != 1:
        raise Error(
            "rf_predict_reg: n_trees must be >= 1 and num_outputs must be 1"
        )
    var offsets_p = _i32_ptr(Int(py=offsets_addr))
    var colid_p = _i32_ptr(Int(py=colid_addr))
    var quesval_p = _f32_ptr(Int(py=quesval_addr))
    var left_p = _i32_ptr(Int(py=left_child_addr))
    var leaves_p = _f32_ptr(Int(py=leaves_addr))
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))

    var wrote = 0
    with GILReleased(Python()):
        var rf_params = _default_rf_params(n_trees)
        var trees = _rebuild_trees(
            offsets_p, colid_p, quesval_p, left_p, leaves_p, n_trees, 1
        )
        var forest = RandomForestMetaData[DT, RLT](
            trees^, rf_params, Int32(n_cols)
        )
        var rf = RandomForest[DT, RLT](
            rf_params=rf_params, rf_type=REGRESSION
        )
        var rows = List[Float32](capacity=n_rows * n_cols)
        for i in range(n_rows * n_cols):
            rows.append(xp[i])
        var preds = List[Float32](capacity=n_rows)
        for _ in range(n_rows):
            preds.append(0.0)
        rf.predict(rows, n_rows, n_cols, preds, forest)
        for i in range(n_rows):
            op[i] = preds[i]
        wrote = n_rows
    return PythonObject(wrote)


def rf_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `mojo_only/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_rf() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_rf")
        m.def_function[rf_vendor_binding]("rf_vendor")
        m.def_function[rf_classifier_fit_binding]("rf_classifier_fit")
        m.def_function[rf_regressor_fit_binding]("rf_regressor_fit")
        m.def_function[rf_predict_proba_binding]("rf_predict_proba")
        m.def_function[rf_predict_reg_binding]("rf_predict_reg")
        return m.finalize()
    except e:
        abort(String("failed to initialize _mojolearn_rf: ") + String(e))
        return PythonObject(None)
