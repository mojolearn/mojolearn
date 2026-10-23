# SPDX-License-Identifier: Apache-2.0
# The boosted and bagged tree ensembles' public operations in one native
# call each: BaggingClassifier/BaggingRegressor, AdaBoostClassifier/
# AdaBoostRegressor, DARTRegressor/DARTClassifier and RandomTreesEmbedding
# (python/mojolearn/_expansion_trees.py). This is the Python orchestration
# those methods ran, step for step: the parameter checks with their wording,
# the member draws, the member trees assembled with their constructors'
# stored state and fitted by the forest owner (bindings/trees_public_forest.mojo:
# the same fit entries, sessions and exports the members' own Python
# methods launched), the x_trees glue kernels in the same order with the
# same arguments, and the fitted state. Every kernel and its arguments are
# the ones the Python path launched, so the fitted ensembles and their
# predictions keep their bits.
#
# Members are mojolearn tree estimators (DecisionTree*, RandomForest*,
# ExtraTrees*): a member of any other class is a Python object whose
# methods native code may not call, and is refused by name.
#
# Round trips removed against the Python path: the per-member Python
# `fit` / `predict` calls (each a Python argument conversion, parameter
# refresh and label pass) run in this one native call; X is converted to
# float32 once per operation and handed to every member (AdaBoost's members
# reuse its one column-major copy, a session's members its one device
# copy); DART no longer opens (and uploads X into) a forest data session its
# round never reads (the round's own `x_trees_dart_open` holds X; the
# session's finite scan is kept); RandomTreesEmbedding's `fit_transform`
# converts X once for the fit and the transform.
from std.math import ceil
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.py_attr import py_attr
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_wrap_array, _store_code
from bindings.trees_public_inputs import (
    TREES_NONFINITE, TreesMatrix, TreesLabels, trees_builtins, trees_table, trees_entry, trees_need,
    trees_helper, trees_env, trees_empty, trees_vector, trees_reshape, trees_address, trees_f32,
    trees_all_finite, trees_check_sample_weight, trees_encode_labels, trees_decode_labels, trees_cast,
    trees_weighted_rows, trees_call,
)
from bindings.trees_public_forest import (
    ForestFit, forest_fit, forest_predict, forest_predict_proba, forest_get_params, forest_refresh,
    forest_capture_mode, forest_effective_mode, forest_export, forest_fit_arrays, forest_kind,
    _check_mode, _rf_params, _store_model, _int, _float, _repr, _str, _truthy, _is_bool, _is_integral,
    _refuse, _classifier, _launch, _y_rows_check, _dt_tier_bins,
)


# ------------------------------------------------------------- utilities


def _mode(fields: PythonObject, context: PythonObject) raises -> String:
    """The tier `NumericModeMixin._bind` resolves: the estimator's
    `numeric_mode`, else the process default."""
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    return String(_check_mode(mode))


def _xt(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The estimator's x_trees binding (`_bind()`)."""
    return trees_table(context, _mode(fields, context), "_mojolearn_x_trees")


def _x(table: PythonObject, name: String) raises -> PythonObject:
    return trees_need(table, name, "mojolearn: the x_trees binding has no " + name, "ImportError")


def _seed(random_state: PythonObject, context: PythonObject) raises -> Int:
    """`_trees_seed`."""
    if random_state is None:
        return 0
    if _is_bool(random_state) or not _is_integral(random_state, context) or Bool(py=random_state < PythonObject(0)):
        raise Error("random_state must be None or a nonnegative int (a RandomState object is refused:"
                    + " draws come from the lane's own counter RNG, xtrees/ops.mojo)")
    return Int(py=_int(random_state))


def _sub_seed(seed: Int, k: Int) -> Int:
    """`_trees_sub_seed`: the k-th member's random_state."""
    return (seed * 1000003 + 7919 * (k + 1)) % 2147483647


def _P(v: Int) raises -> PythonObject:
    return PythonObject(v)


def _F(v: Float64) raises -> PythonObject:
    return PythonObject(v)


def _A(array: PythonObject) raises -> PythonObject:
    return PythonObject(trees_address(array))


def _f64_word(array: PythonObject, index: Int) raises -> Float64:
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(array))[index]


def _i32_word(array: PythonObject, index: Int) raises -> Int:
    return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(array))[index])


def _x_f32(source: PythonObject, context: PythonObject) raises -> TreesMatrix:
    """`as_f32_c(X, ndim=2, name="X")`."""
    return trees_f32(source, 2, "C", "X", context)


def _as_array(m: TreesMatrix, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A C-order float32 matrix as an Array a member's fit or predict reads:
    the matrix's own Array, or (a borrowed foreign buffer) one copy."""
    if py_type(m.keep) is context["array_class"] and Int(py=m.keep._addr) == m.addr:
        return m.keep
    var out = trees_empty(shape, "<f4", context)
    trees_cast(m.addr, "<f4", trees_address(out), "<f4", m.rows * m.cols, context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _f32_vector(value: PythonObject, context: PythonObject) raises -> TreesMatrix:
    """`as_f32_c(value, ndim=1, ...)` of a member's output."""
    return trees_f32(value, 1, "C", "prediction", context)


def _i32_codes(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`as_i32_c(value, ndim=1, name="predicted codes")` of a member's
    decoded int64 labels (its classes are the ensemble's codes)."""
    var n = Int(py=value.size)
    var out = trees_vector(n, "<i4", context)
    trees_cast(trees_address(value), String(value.dtype), trees_address(out), "<i4", n, context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _fsum(values: PythonObject, context: PythonObject) raises -> Float64:
    """`_portable_math.fsum` of a list of floats: the core helper `fsum_f64`."""
    var b = trees_builtins()
    var src = py_call(context["array_type"], PythonObject("d"), values)
    var out = py_call(context["array_type"], PythonObject("d"), Python.list(PythonObject(0.0)))
    var status = Int(py=py_call_typed(trees_helper(context, "fsum_f64"), py_call(src.buffer_info)[0], PythonObject(len(src)),
                                      py_call(out.buffer_info)[0]))
    # The helper reads src through a raw integer address; last-use destruction
    # after buffer_info/len must not free it before fsum_f64 consumes the data.
    _ = src
    var total = Float64(py=out[0])
    if status != 0:
        if total != 0:
            raise Error("OverflowError: math range error")
        raise Error("-inf + inf in fsum")
    return total


def _copy_i32(array: PythonObject, start: Int, stop: Int, context: PythonObject) raises -> PythonObject:
    """`array[start:stop]` of a flat int32 Array: a block copy in C."""
    var b = trees_builtins()
    var view = py_call(b.memoryview, array._store)
    var raw = py_call(py_call(py_attr(view, "__getitem__"), py_call(b.slice, PythonObject(start), PythonObject(stop))).tobytes)
    var store = py_call(context["array_type"], PythonObject("i"))
    _ = py_call(store.frombytes, raw)
    return classical_wrap_array(store, Python.tuple(stop - start), "<i4", "C", context)


# ------------------------------------------------------------- x_trees glue


def _arange(xt: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    var out = trees_vector(n, "<i4", context)
    if n > 0:
        _ = py_call_typed(_x(xt, "x_trees_iota_i32"), _A(out), Python.list(_P(n)))
        _ = out  # keep the Python object alive through the native call above
    return out


def _class_major(xt: PythonObject, inits: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    var k = len(inits)
    var vals = py_call(context["array_type"], PythonObject("d"), inits)
    var vals_array = classical_wrap_array(vals, Python.tuple(k), "<f8", "C", context)
    var out = trees_vector(k * n, "<f8", context)
    if k * n > 0:
        _ = py_call_typed(_x(xt, "x_trees_fill_class_major_f64"), _A(vals_array), _A(out), Python.list(_P(n), _P(k)))
        _ = out  # keep the Python object alive through the native call above
        _ = vals_array  # keep the Python object alive through the native call above
    return out


def _indices(xt: PythonObject, pool: Int, draw: Int, replace: Bool, seed: Int, stream: Int,
             context: PythonObject) raises -> PythonObject:
    var out = trees_vector(draw, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_sample_indices"), _A(out),
                      Python.list(_P(pool), _P(draw), _P(1 if replace else 0), _P(seed), _P(stream)))
    _ = out  # keep the Python object alive through the native call above
    return out


def _gather(xt: PythonObject, x_addr: Int, n_src: Int, d_src: Int, rows: PythonObject, cols: PythonObject,
            context: PythonObject) raises -> PythonObject:
    var r = len(rows)
    var c = len(cols)
    var out = trees_empty(Python.tuple(r, c), "<f4", context)
    _ = py_call_typed(_x(xt, "x_trees_gather_f32"), _P(x_addr), _A(rows), _A(cols), _A(out),
                      Python.list(_P(n_src), _P(d_src), _P(r), _P(c)))
    _ = out  # keep the Python object alive through the native call above
    return out


def _gather_codes(xt: PythonObject, codes: PythonObject, rows: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = trees_vector(len(rows), "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_gather_i32"), _A(codes), _A(rows), _A(out), Python.list(_P(len(codes)), _P(len(rows))))
    _ = out  # keep the Python object alive through the native call above
    return out


def _gather_vec(xt: PythonObject, v_addr: Int, n: Int, rows: PythonObject, context: PythonObject) raises -> PythonObject:
    """A float32 vector gathered at rows (as an n x 1 matrix)."""
    var col = trees_vector(1, "<i4", context)
    var out = _gather(xt, v_addr, n, 1, rows, col, context)
    return trees_reshape(out, Python.tuple(len(rows)), context)


def _acc_cols(xt: PythonObject, acc: PythonObject, x: PythonObject, cols: PythonObject, n: Int, k: Int,
              context: PythonObject) raises:
    var xa = trees_f32(x, 2, "C", "sub-estimator output", context)
    _ = py_call_typed(_x(xt, "x_trees_accumulate_cols"), _A(acc), _P(xa.addr), _A(cols),
                      Python.list(_P(n), _P(k), _P(len(cols)), _F(1.0)))
    _ = xa  # keep the Python object alive through the native call above
    _ = xa^


def _acc(xt: PythonObject, acc: PythonObject, x: PythonObject, n: Int, context: PythonObject) raises:
    var xa = trees_f32(x, 1, "C", "sub-estimator output", context)
    _ = py_call_typed(_x(xt, "x_trees_accumulate"), _A(acc), _P(xa.addr), Python.list(_P(n), _F(1.0)))
    _ = xa  # keep the Python object alive through the native call above
    _ = xa^


def _acc_votes(xt: PythonObject, acc: PythonObject, codes: PythonObject, n: Int, k: Int, on: Float64,
               off: Float64) raises:
    _ = py_call_typed(_x(xt, "x_trees_accumulate_onehot"), _A(acc), _A(codes), Python.list(_P(n), _P(k), _F(on), _F(off)))


def _scale(xt: PythonObject, acc: PythonObject, divisor: Float64) raises:
    _ = py_call_typed(_x(xt, "x_trees_scale"), _A(acc), Python.list(_P(Int(py=acc.size)), _F(divisor)))


def _argmax(xt: PythonObject, acc: PythonObject, n: Int, k: Int, context: PythonObject) raises -> PythonObject:
    var out = trees_vector(n, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_argmax_rows"), _A(acc), _A(out), Python.list(_P(n), _P(k)))
    _ = out  # keep the Python object alive through the native call above
    return out


def _sub_cols(member: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_sub_cols`: a member's classes (the ensemble's codes) as int32."""
    var b = trees_builtins()
    var classes = py_fields(member)["classes_"]
    var store = py_call(context["array_type"], PythonObject("i"), py_call(b.map, b.int, classes))
    return classical_wrap_array(store, Python.tuple(len(classes)), "<i4", "C", context)


# ---------------------------------------------------------------- members


def _connected(cls: PythonObject, context: PythonObject) raises -> Bool:
    var b = trees_builtins()
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        if py_contains(context["trees_classes"], mro[i]):
            return True
    return False


def _require_tree(base: PythonObject, owner: String, context: PythonObject) raises:
    """A member must be a mojolearn tree estimator (its fit and predict run
    in this owner); any other estimator is a Python object."""
    if not _connected(py_type(base), context):
        raise Error("NotImplementedError: " + owner + ": estimator=" + _repr(base) + " is not a mojolearn "
                    + "DecisionTree*, RandomForest* or ExtraTrees* estimator; the native ensemble fits only "
                    + "those members. Refused by name rather than accepted and ignored.")


def _new_member(cls: PythonObject, settings: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cls(**settings)` of a connected tree class: the state its
    constructor stores (every parameter, its default unless set, and the
    constructor flag), assembled without running Python."""
    var b = trees_builtins()
    var member = py_call(b.object.__new__, cls)
    var fields = py_fields(member)
    _ = py_call(fields.update, context["trees_defaults"][cls])
    _ = py_call(fields.update, settings)
    py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
    return member


def _clone(base: PythonObject, seed: Int, context: PythonObject) raises -> PythonObject:
    """`_trees_clone(base, random_state=seed)`."""
    var params = forest_get_params(base, context)
    if py_contains(params, "random_state"):
        py_setitem(params, "random_state", _P(seed))
    return _new_member(py_type(base), params, context)


def _fitted(member: PythonObject) raises:
    """The flag the fit wrapper sets after a member's `fit`."""
    py_setitem(py_fields(member), "_mojolearn_fitted", PythonObject(True))


def _member_fit(member: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                context: PythonObject) raises:
    """`member.fit(X, y[, sample_weight=...])`."""
    _ = forest_fit(member, Python.tuple(X, y, sample_weight, PythonObject(False), PythonObject(None)), context)
    _fitted(member)


def _member_predict(member: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    return forest_predict(member, Python.tuple(X), context)


def _member_proba(member: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    return forest_predict_proba(member, Python.tuple(X), context)


def _member_mode(member: PythonObject, context: PythonObject) raises -> String:
    """The tier a clone of `member` fits under (`_capture_fit_mode`)."""
    var mode = py_get(py_fields(member), "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    return String(_check_mode(mode))


def _is_class(member: PythonObject, key: String, context: PythonObject) raises -> Bool:
    """`type(member) is <the connected class>` (exactly, no subclass)."""
    return py_type(member) is context[key]


struct Session(Copyable, Movable):
    """An open forest data session (`ForestDataSession`): the binding and
    the handle, or none."""
    var native: PythonObject
    var handle: Int

    def __init__(out self, native: PythonObject, handle: Int):
        self.native = native
        self.handle = handle

    def close(mut self) raises:
        if self.handle > 0:
            var handle = self.handle
            self.handle = 0
            _ = py_call_typed(trees_entry(self.native, "rf_data_session_close"), _P(handle))


def _session_choice(default: String) raises -> String:
    """`forest_data_session_choice(None, default)`: '' (none), 'exact' or
    'share' (the forest owner's tier rule leaves 'share' to the caller)."""
    var choice = trees_env("MOJOLEARN_FOREST_SESSION", default)
    var clean = String(py_call(py_call(PythonObject(choice).strip).lower))
    if clean == "" or clean == "0" or clean == "off" or clean == "no":
        return ""
    if clean == "1" or clean == "on" or clean == "exact":
        return "exact"
    if clean == "share":
        return "share"
    raise Error("MOJOLEARN_FOREST_SESSION must be 0, 1 (exact) or share")


def _member_session(member: PythonObject, x: TreesMatrix, row_major: Bool, x_finite: Bool, default: String,
                    context: PythonObject) raises -> Session:
    """`_trees_member_session`: the data session for the member fits of one
    ensemble on the forest binding a clone of `member` resolves (handle 0:
    the choice is off, or the binary has no session entry). It stands in for
    every member's finite scan of X, so X is scanned here unless the caller
    already did."""
    var choice = _session_choice(default)
    var mode = _member_mode(member, context)
    var native = trees_table(context, mode, "_mojolearn_rf")
    if choice == "" or trees_entry(native, "rf_data_session_open") is None:
        return Session(native, 0)
    if not x_finite and not trees_all_finite(x.addr, x.rows * x.cols, False, context):
        raise Error(TREES_NONFINITE)
    var share = choice == "share" and mode == "fast"
    var handle = py_call_typed(trees_entry(native, "rf_data_session_open"), _P(x.addr),
                               Python.list(_P(x.rows), _P(x.cols), _P(1 if row_major else 0), _P(1 if share else 0)))
    return Session(native, Int(py=handle))


def _switches(xt: PythonObject) raises -> Int:
    var query = trees_entry(xt, "x_trees_fast_switches")
    return 0 if query is None else Int(py=py_call_typed(query))


def _ada_session_default(fields: PythonObject, xt: PythonObject, context: PythonObject) raises -> String:
    """`_trees_ada_session_default`."""
    var fast = _mode(fields, context) == "fast"
    var bits = _switches(xt)
    if fast and (bits & 4) != 0:
        return "share"
    if fast and (bits & 2) != 0:
        return "1"
    if not fast and (bits & 128) != 0:
        return "1"
    return "0"


def _member_store(member: PythonObject, result: PythonObject, cols: Int) raises:
    """A member fit's model arrays and dimensions (the member's own fit
    stored the same five arrays and three fields)."""
    _store_model(py_fields(member), result, cols)


def _member_fit_weighted(member: PythonObject, xcm: TreesMatrix, codes: PythonObject, w32: PythonObject,
                         encoded: Optional[TreesLabels], session: Session, context: PythonObject) raises:
    """`DecisionTreeClassifier._fit_weighted(Xcm, codes, w32, x_finite=True,
    encoded=..., session=...)` on a column-major float32 X already scanned."""
    var fields = forest_refresh(member, "dt_classifier", context)
    var mode = forest_capture_mode(fields, context)
    var labels = encoded.value().copy() if encoded else trees_encode_labels(codes, context)
    py_setitem(fields, "classes_", labels.classes)
    var k = len(labels.classes)
    py_setitem(fields, "n_classes_", _P(k))
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var xtrees = trees_table(context, mode, "_mojolearn_x_trees")
    var weights = trees_weighted_rows(w32, py_get(fields, "class_weight"), labels, xtrees, context)
    var rf = trees_table(context, mode, "_mojolearn_rf")
    _y_rows_check(labels.n, xcm.rows)
    var params = _rf_params(fields, "dt_classifier", xcm.rows, xcm.cols, k, rf, context)
    var criterion = fields["_cfg"]["criterion"]
    var result = PythonObject(None)
    if session.handle > 0:
        var entry = trees_need(session.native, "rf_classifier_fit_weighted_session_export",
                               "rebuild the forest binding for forest data sessions")
        result = forest_export(session.native, py_call_typed(entry, _P(session.handle), _A(labels.codes), params, criterion,
                                                             _P(weights.addr)), "into", context)
        _ = weights  # keep the Python object alive through the native call above
    else:
        var fit = ForestFit("rf_classifier_fit_weighted", True, Python.list(_P(weights.addr)), True)
        _ = weights  # keep the Python object alive through the native call above
        result = _launch(rf, fit, xcm, trees_address(labels.codes), params, criterion, False, context)
        _ = weights  # keep the Python object alive through the native call above
    # Both fit routes receive only the codes' address. The model owns the
    # classes table, not this freshly encoded label buffer.
    _ = labels
    _member_store(member, forest_fit_arrays(result, context), xcm.cols)
    _fitted(member)
    _ = weights^


def _member_fit_session_rows(member: PythonObject, session: Session, rows: PythonObject, y32: PythonObject,
                             x_cols: Int, context: PythonObject) raises:
    """`RandomForestRegressor._fit_in_session(session, y, rows=rows)`: the
    squared-error member fit on the session's X at `rows`."""
    var kind = forest_kind(member, context)
    var fields = forest_refresh(member, kind, context)
    var mode = forest_capture_mode(fields, context)
    var y = trees_f32(y32, 1, "C", "y", context)
    var code = fields["_cfg"]["criterion"]
    if Int(py=code) != Int(py=context["trees_rf_tables"]["reg_criteria"]["squared_error"]):
        raise Error("a forest data session fits squared_error members")
    var n_rows = len(rows)
    _y_rows_check(y.rows, n_rows)
    var rf = trees_table(context, mode, "_mojolearn_rf")
    var params = _rf_params(fields, kind, n_rows, x_cols, 0, rf, context)
    var entry = trees_need(session.native, "rf_regressor_fit_session_rows_export",
                           "rebuild the forest binding for forest data sessions")
    var result = forest_export(session.native, py_call_typed(entry, _P(session.handle), _A(rows), _P(y.addr), params, code),
                               "into", context)
    _ = y  # keep the Python object alive through the native call above
    _member_store(member, forest_fit_arrays(result, context), x_cols)
    _ = y  # keep the Python object alive through the native call above
    _fitted(member)
    _ = y^


# ---------------------------------------------------------------- Bagging


def _bagging_validate(fields: PythonObject, context: PythonObject) raises:
    """`_BaggingBase._validate_params`."""
    if _truthy(fields["oob_score"]) and not _truthy(fields["bootstrap"]):
        raise Error("Out of bag estimation only available if bootstrap=True")
    if _truthy(fields["warm_start"]):
        _refuse("warm_start=True", "there is no incremental fit here.")
    if fields["n_jobs"] is not None:
        _refuse("n_jobs", "the sub-fits run one after another on the device.")
    if _truthy(fields["verbose"]):
        _refuse("verbose", "nothing logs per estimator.")
    if Int(py=_int(fields["n_estimators"])) < 1:
        raise Error("n_estimators must be >= 1")
    _ = _seed(fields["random_state"], context)


def _count(v: PythonObject, total: Int, name: String, context: PythonObject) raises -> Int:
    """`_BaggingBase._count`."""
    var b = trees_builtins()
    var k = 0
    if _is_integral(v, context) and not _is_bool(v):
        k = Int(py=_int(v))
    elif Bool(py=py_call(b.isinstance, v, context["trees_real_type"])):
        var f = Float64(py=_float(v))
        if not (f > 0.0 and f <= 1.0):
            raise Error(name + "=" + _str(v) + " is outside (0, 1]")
        k = Int(py=_int(_F(f * Float64(total))))
    else:
        raise Error(name + " must be an int or a float")
    if not (1 <= k and k <= total):
        raise Error(name + " resolves to " + String(k) + ", outside [1, " + String(total) + "]")
    return k


def _bagging_batched_ok(fields: PythonObject, base: PythonObject, n_rows: Int, n: Int, n_feat: Int, d: Int,
                        sample_weight: PythonObject, context: PythonObject) raises -> Bool:
    """`_BaggingBase._batched_ok`: a bag of best-splitter decision trees over
    every feature, without weights or out-of-bag scoring, is one forest fit."""
    if trees_env("MOJOLEARN_HR2_BAGGING_MEMBERS", "") == "1":
        return False
    if not _is_class(base, "trees_dt_classifier", context) and not _is_class(base, "trees_dt_regressor", context):
        return False
    var bf = py_fields(base)
    if not Bool(py=bf["splitter"] == PythonObject("best")) or py_get(bf, "max_leaf_nodes") is not None:
        return False
    if sample_weight is not None or _truthy(fields["oob_score"]) or _truthy(fields["bootstrap_features"]) or n_feat != d:
        return False
    if not _truthy(fields["bootstrap"]) and n_rows != n:
        return False
    return True


def _bagging_fit_batched(model: PythonObject, fields: PythonObject, xa: PythonObject, d: Int, y: PythonObject,
                         base: PythonObject, n_rows: Int, n: Int, seed: Int, context: PythonObject) raises:
    """`_BaggingBase._fit_batched`: the members as one forest fit."""
    var bf = py_fields(base)
    var classifier = _is_class(base, "trees_dt_classifier", context)
    var bootstrap = _truthy(fields["bootstrap"])
    var common = Python.dict()
    py_setitem(common, "n_estimators", _int(fields["n_estimators"]))
    py_setitem(common, "criterion", bf["criterion"])
    py_setitem(common, "max_depth", bf["max_depth"])
    py_setitem(common, "min_samples_split", bf["min_samples_split"])
    py_setitem(common, "min_samples_leaf", bf["min_samples_leaf"])
    py_setitem(common, "max_features", PythonObject(1.0) if bf["max_features"] is None else bf["max_features"])
    py_setitem(common, "min_impurity_decrease", bf["min_impurity_decrease"])
    py_setitem(common, "bootstrap", PythonObject(bootstrap))
    py_setitem(common, "max_samples", _F(Float64(n_rows) / Float64(n)) if bootstrap else PythonObject(None))
    py_setitem(common, "random_state", _P(seed))
    if classifier:
        py_setitem(common, "n_bins", bf["n_bins"])
    else:
        py_setitem(common, "n_bins", _dt_tier_bins(bf, trees_table(context, _member_mode(base, context), "_mojolearn_rf")))
    py_setitem(common, "numeric_mode", py_get(bf, "numeric_mode"))
    var forest = PythonObject(None)
    if classifier:
        py_setitem(common, "class_weight", bf["class_weight"])
        forest = _new_member(context["trees_rf_classifier"], common, context)
    else:
        forest = _new_member(context["trees_rf_regressor"], common, context)
    _member_fit(forest, xa, y, PythonObject(None), context)
    py_setitem(fields, "_batched", Python.tuple(forest, base))
    py_setitem(fields, "_estimators", PythonObject(None))
    var xt = _xt(fields, context)
    var cols = _arange(xt, d, context)
    var trees = Int(py=_int(py_fields(forest)["_n_trees"]))
    var features = Python.list()
    for i in range(trees):  # driver-loop: over the member trees (one shared all-columns array)
        _ = py_call(features.append, cols)
    py_setitem(fields, "estimators_features_", features)
    py_setitem(fields, "_oob_rows", Python.list())
    py_setitem(fields, "n_features_in_", _P(d))


def _unseen_rows(xt: PythonObject, rows: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_BaggingBase._oob_of`: the rows a member never drew, ascending."""
    var out = trees_vector(max(1, n), "<i4", context)
    var k = Int(py=py_call_typed(_x(xt, "x_trees_unseen_rows"), _A(rows), _A(out), Python.list(_P(len(rows)), _P(n))))
    _ = out  # keep the Python object alive through the native call above
    return _copy_i32(out, 0, k, context)


def _bagging_fit_bags(model: PythonObject, fields: PythonObject, X: PythonObject, y_all: PythonObject,
                      classifier: Bool, sample_weight: PythonObject, context: PythonObject) raises:
    """`_BaggingBase._fit_bags`."""
    var x = _x_f32(X, context)
    var n = x.rows
    var d = x.cols
    var xa = _as_array(x, Python.tuple(n, d), context)
    var seed = _seed(fields["random_state"], context)
    var n_rows = _count(fields["max_samples"], n, "max_samples", context)
    var n_feat = _count(fields["max_features"], d, "max_features", context)
    var base = fields["estimator"]
    if base is None:
        base = _new_member(context["trees_dt_classifier" if classifier else "trees_dt_regressor"], Python.dict(), context)
    var owner = "BaggingClassifier" if classifier else "BaggingRegressor"
    _require_tree(base, owner, context)
    if _bagging_batched_ok(fields, base, n_rows, n, n_feat, d, sample_weight, context):
        _bagging_fit_batched(model, fields, xa, d, y_all, base, n_rows, n, seed, context)
        return
    var sw: Optional[TreesMatrix] = None
    if sample_weight is not None:
        sw = trees_check_sample_weight(sample_weight, n, context)
    var xt = _xt(fields, context)
    var members = Python.list()
    var features = Python.list()
    var oob_rows = Python.list()
    py_setitem(fields, "_estimators", members)
    py_setitem(fields, "_batched", PythonObject(None))
    py_setitem(fields, "estimators_features_", features)
    py_setitem(fields, "_oob_rows", oob_rows)
    var bootstrap = _truthy(fields["bootstrap"])
    var bootstrap_features = _truthy(fields["bootstrap_features"])
    var oob = _truthy(fields["oob_score"])
    var all_rows = n_rows == n and not bootstrap
    var all_cols = n_feat == d and not bootstrap_features
    var count = Int(py=_int(fields["n_estimators"]))
    for k in range(count):  # driver-loop: over the member estimators
        var rows = _arange(xt, n, context) if all_rows else _indices(xt, n, n_rows, bootstrap, seed, 2 * k, context)
        var cols = _arange(xt, d, context) if all_cols else _indices(xt, d, n_feat, bootstrap_features, seed, 2 * k + 1, context)
        var xs = xa if (all_rows and all_cols) else _gather(xt, x.addr, n, d, rows, cols, context)
        _ = x  # keep the Python object alive through the native call above
        var ys = y_all
        if not all_rows:
            ys = _gather_codes(xt, y_all, rows, context) if classifier else _gather_vec(xt, trees_address(y_all), n, rows, context)
        var member = _clone(base, _sub_seed(seed, k), context)
        var weights = PythonObject(None)
        if sw:
            # every row: the weights as given (the member's own check reads
            # the same float32 values); else the drawn rows' weights
            weights = sample_weight if all_rows else _gather_vec(xt, sw.value().addr, n, rows, context)
        _member_fit(member, xs, ys, weights, context)
        _ = x  # keep the Python object alive through the native call above
        _ = py_call(members.append, member)
        _ = py_call(features.append, cols)
        if oob:
            _ = py_call(oob_rows.append, _unseen_rows(xt, rows, n, context))
    py_setitem(fields, "n_features_in_", _P(d))
    _ = x^


def _bagging_fitted(fields: PythonObject) raises -> Bool:
    return py_get(fields, "_batched") is not None or py_get(fields, "_estimators") is not None


def _bagging_x(fields: PythonObject, X: PythonObject, context: PythonObject) raises -> TreesMatrix:
    """`_BaggingBase._check_X`."""
    if not _bagging_fitted(fields):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var x = _x_f32(X, context)
    var fitted = Int(py=_int(fields["n_features_in_"]))
    if x.cols != fitted:
        raise Error("X has " + String(x.cols) + " features, fit saw " + String(fitted))
    return x^


def _all_columns(cols: PythonObject, d: Int) raises -> Bool:
    """`cols.tolist() == list(range(d))`."""
    if len(cols) != d:
        return False
    for j in range(d):  # driver-loop: over the member's feature subset (columns-sized)
        if _i32_word(cols, j) != j:
            return False
    return True


def _sub_x(xt: PythonObject, x: TreesMatrix, xa: PythonObject, cols: PythonObject, rows_all: PythonObject,
           context: PythonObject) raises -> PythonObject:
    """`_BaggingBase._sub_X`."""
    if _all_columns(cols, x.cols):
        return xa
    return _gather(xt, x.addr, x.rows, x.cols, rows_all, cols, context)


def _members(fields: PythonObject) raises -> PythonObject:
    """The fitted member list (`estimators_`) of the per-member path."""
    return fields["_estimators"]


def _bagging_member_scores(xt: PythonObject, member: PythonObject, xs: PythonObject, m: Int, k: Int,
                           context: PythonObject) raises -> PythonObject:
    """`BaggingClassifier._member_proba`: a member's class scores, dense
    m x k float64 in the ensemble's columns."""
    var out = trees_vector(m * k, "<f8", context)
    _acc_cols(xt, out, _member_proba(member, xs, context), _sub_cols(member, context), m, k, context)
    return out


def _bagging_oob_outputs(xt: PythonObject, fields: PythonObject, x: TreesMatrix, k: Int, classifier: Bool,
                         context: PythonObject) raises -> PythonObject:
    """`_BaggingBase._oob_outputs`: each member's output on its out-of-bag
    rows, added in member order; (sums, counts)."""
    var n = x.rows
    var acc = _class_major(xt, Python.list(_F(0.0)), n * k, context)
    var counts = trees_vector(n, "<i4", context)
    var members = _members(fields)
    var features = fields["estimators_features_"]
    var oob_rows = fields["_oob_rows"]
    for i in range(len(members)):  # driver-loop: over the member estimators
        var oob = oob_rows[i]
        var m = len(oob)
        if m == 0:
            continue
        var xs = _gather(xt, x.addr, n, x.cols, oob, features[i], context)
        var out = PythonObject(None)
        if classifier:
            out = _bagging_member_scores(xt, members[i], xs, m, k, context)
        else:
            var p = _f32_vector(_member_predict(members[i], xs, context), context)
            out = trees_vector(m, "<f8", context)
            trees_cast(p.addr, "<f4", trees_address(out), "<f8", m, context)
            _ = p  # keep the Python object alive through the native call above
            _ = out  # keep the Python object alive through the native call above
        _ = py_call_typed(_x(xt, "x_trees_accumulate_rows"), _A(acc), _A(out), _A(oob), Python.list(_P(n), _P(k), _P(m)))
        _ = out  # keep the Python object alive through the native call above
        _ = oob  # keep the Python object alive through the native call above
        _ = acc  # keep the Python object alive through the native call above
        _ = py_call_typed(_x(xt, "x_trees_count_rows"), _A(counts), _A(oob), Python.list(_P(n), _P(m)))
        _ = oob  # keep the Python object alive through the native call above
        _ = counts  # keep the Python object alive through the native call above
    return Python.tuple(acc, counts)


def bagging_fit(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    """BaggingClassifier/BaggingRegressor `fit`."""
    var fields = py_fields(model)
    _bagging_validate(fields, context)
    var X = args[0]
    var sample_weight = args[2]
    if classifier:
        var labels = trees_encode_labels(args[1], context)
        py_setitem(fields, "classes_", labels.classes)
        var k = len(labels.classes)
        py_setitem(fields, "n_classes_", _P(k))
        if k < 2:
            raise Error("y has fewer than 2 classes")
        _bagging_fit_bags(model, fields, X, labels.codes, True, sample_weight, context)
        if _truthy(fields["oob_score"]):
            # `BaggingClassifier._set_oob_score`
            var xt = _xt(fields, context)
            var x = _x_f32(X, context)
            var n = x.rows
            var pair = _bagging_oob_outputs(xt, fields, x, k, True, context)
            var acc = pair[0]
            _ = py_call_typed(_x(xt, "x_trees_normalize_rows"), _A(acc), Python.list(_P(n), _P(k)))
            _ = acc  # keep the Python object alive through the native call above
            py_setitem(fields, "oob_decision_function_", trees_reshape(acc, Python.tuple(n, k), context))
            var pred = _argmax(xt, acc, n, k, context)
            var hits = Int(py=py_call_typed(_x(xt, "x_trees_count_equal"), _A(pred), _A(labels.codes), Python.list(_P(n))))
            _ = pred  # keep the Python object alive through the native call above
            py_setitem(fields, "oob_score_", _F(Float64(hits) / Float64(n)))
            _ = pred  # keep the Python object alive through the native call above
        return model
    var y32 = trees_f32(args[1], 1, "C", "y", context)
    var y_array = _as_array(y32, Python.tuple(y32.rows), context)
    _bagging_fit_bags(model, fields, X, y_array, False, sample_weight, context)
    if _truthy(fields["oob_score"]):
        # `BaggingRegressor._set_oob_score`
        var xt = _xt(fields, context)
        var x = _x_f32(X, context)
        var n = x.rows
        var pair = _bagging_oob_outputs(xt, fields, x, 1, False, context)
        var pred = trees_vector(n, "<f8", context)
        var words = trees_vector(4, "<f8", context)
        var flags = trees_vector(4, "<i4", context)
        _ = py_call_typed(_x(xt, "x_trees_oob_r2"),
                          Python.list(_A(pair[0]), _A(pair[1]), _A(y_array), _A(pred), _A(words), _A(flags)),
                          Python.list(_P(n)))
        _ = flags  # keep the Python object alive through the native call above
        _ = words  # keep the Python object alive through the native call above
        _ = pred  # keep the Python object alive through the native call above
        _ = y_array  # keep the Python object alive through the native call above
        if _i32_word(flags, 0) != 0 or _i32_word(flags, 1) != 0:
            raise Error("BaggingRegressor oob_score: a non-finite target or out-of-bag prediction, "
                        + "or a sum that overflows float64")
        py_setitem(fields, "oob_prediction_", pred)
        var ss_tot = _f64_word(words, 1)
        var ss_res = _f64_word(words, 2)
        var score = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
        py_setitem(fields, "oob_score_", _F(score))
    return model


def _bagging_proba(model: PythonObject, fields: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`BaggingClassifier.predict_proba`."""
    var x = _bagging_x(fields, X, context)
    var n = x.rows
    var k = Int(py=_int(fields["n_classes_"]))
    var xa = _as_array(x, Python.tuple(n, x.cols), context)
    var batched = py_get(fields, "_batched")
    if batched is not None:
        var proba = _member_proba(batched[0], xa, context)
        var out = trees_empty(Python.tuple(n, k), "<f8", context)
        trees_cast(trees_address(proba), "<f4", trees_address(out), "<f8", n * k, context)
        _ = out  # keep the Python object alive through the native call above
        _ = proba  # keep the Python object alive through the native call above
        return out
    var xt = _xt(fields, context)
    var acc = _class_major(xt, Python.list(_F(0.0)), n * k, context)
    var rows_all = _arange(xt, n, context)
    var members = _members(fields)
    var features = fields["estimators_features_"]
    for i in range(len(members)):  # driver-loop: over the member estimators
        var xs = _sub_x(xt, x, xa, features[i], rows_all, context)
        _acc_cols(xt, acc, _member_proba(members[i], xs, context), _sub_cols(members[i], context), n, k, context)
    _scale(xt, acc, Float64(len(members)))
    return trees_reshape(acc, Python.tuple(n, k), context)


def bagging_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _bagging_proba(model, py_fields(model), args[0], context)


def bagging_predict(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    """BaggingClassifier/BaggingRegressor `predict`."""
    var fields = py_fields(model)
    if classifier:
        var p = _bagging_proba(model, fields, args[0], context)
        var n = Int(py=p.shape[0])
        var k = Int(py=_int(fields["n_classes_"]))
        return trees_decode_labels(fields["classes_"], _argmax(_xt(fields, context), p, n, k, context), context)
    var x = _bagging_x(fields, args[0], context)
    var n = x.rows
    var xa = _as_array(x, Python.tuple(n, x.cols), context)
    var batched = py_get(fields, "_batched")
    if batched is not None:
        var p = _f32_vector(_member_predict(batched[0], xa, context), context)
        var out = trees_vector(n, "<f8", context)
        trees_cast(p.addr, "<f4", trees_address(out), "<f8", n, context)
        _ = out  # keep the Python object alive through the native call above
        _ = p  # keep the Python object alive through the native call above
        return out
    var xt = _xt(fields, context)
    var acc = trees_vector(n, "<f8", context)
    var rows_all = _arange(xt, n, context)
    var members = _members(fields)
    var features = fields["estimators_features_"]
    for i in range(len(members)):  # driver-loop: over the member estimators
        _acc(xt, acc, _member_predict(members[i], _sub_x(xt, x, xa, features[i], rows_all, context), context), n, context)
    _scale(xt, acc, Float64(len(members)))
    return acc


# --------------------------------------------------------------- AdaBoost


def _ada_validate(fields: PythonObject, classifier: Bool, context: PythonObject) raises:
    """`AdaBoost*._validate_params`."""
    if classifier:
        var algorithm = fields["algorithm"]
        if not Bool(py=algorithm == PythonObject("SAMME")) and not Bool(py=algorithm == PythonObject("deprecated")):
            _refuse("algorithm=" + _repr(algorithm), "SAMME.R was removed from the reference (sklearn 1.6);"
                    + " SAMME is the algorithm.")
    else:
        var losses = context["trees_ada_losses"]
        if not py_contains(losses, fields["loss"]):
            raise Error("loss must be one of " + _repr(py_call(trees_builtins().tuple, losses)) + ", got "
                        + _repr(fields["loss"]))
    if Int(py=_int(fields["n_estimators"])) < 1:
        raise Error("n_estimators must be >= 1")
    if not Float64(py=_float(fields["learning_rate"])) > 0:
        raise Error("learning_rate must be > 0")
    _ = _seed(fields["random_state"], context)


def _normalized_weights(xt: PythonObject, sample_weight: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_trees_normalized_weights_native`: uniform 1 / n, or the checked
    weights over their exactly rounded total (`x_trees_exact_sum`)."""
    if sample_weight is None:
        return _class_major(xt, Python.list(_F(1.0 / Float64(n))), n, context)
    var sw = trees_check_sample_weight(sample_weight, n, context)
    var out = trees_vector(n + 1, "<f8", context)
    var flags = trees_vector(8, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_exact_sum"), _P(sw.addr), _A(out), _A(flags), Python.list(_P(n), _P(1)))
    _ = flags  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = sw  # keep the Python object alive through the native call above
    if _i32_word(flags, 0) != 0 or _i32_word(flags, 1) != 0 or _i32_word(flags, 2) != 0 or _i32_word(flags, 3) != 0:
        raise Error("sample_weight must be finite and nonnegative")
    if _i32_word(flags, 4) == 0:
        raise Error("Sample weights must contain at least one non-zero number.")
    var w = trees_vector(n, "<f8", context)
    trees_cast(trees_address(out) + 8, "<f8", trees_address(w), "<f8", n, context)
    _ = w  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = sw^
    return w


def _w32(xt: PythonObject, w: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_AdaBoostBase._w32`: the boosting weights times n, float32."""
    var out = trees_vector(n, "<f4", context)
    _ = py_call_typed(_x(xt, "x_trees_scale_to_f32"), _A(w), _A(out), Python.list(_P(n), _F(Float64(n))))
    _ = out  # keep the Python object alive through the native call above
    return out


def _colmajor(xt: PythonObject, x: TreesMatrix, context: PythonObject) raises -> TreesMatrix:
    """`_trees_colmajor` of a C-order float32 X: the lane's transpose."""
    var out = trees_empty(Python.tuple(x.rows, x.cols), "<f4", context, "F")
    _ = py_call_typed(_x(xt, "x_trees_transpose_f32"), _P(x.addr), _A(out), Python.list(_P(x.rows), _P(x.cols)))
    _ = out  # keep the Python object alive through the native call above
    var _keep_ret_out = TreesMatrix(out, trees_address(out), x.rows, x.cols, False)
    _ = out  # keep the Python object alive through the native call above
    return _keep_ret_out^


def ada_fit(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    """AdaBoostClassifier (SAMME) / AdaBoostRegressor (AdaBoost.R2) `fit`."""
    var fields = py_fields(model)
    _ada_validate(fields, classifier, context)
    var x = _x_f32(args[0], context)
    var n = x.rows
    var d = x.cols
    var xa = _as_array(x, Python.tuple(n, d), context)
    var xt = _xt(fields, context)
    var seed = _seed(fields["random_state"], context)
    var m = Int(py=_int(fields["n_estimators"]))
    var lr = Float64(py=_float(fields["learning_rate"]))
    var members = Python.list()
    var alphas = Python.list()
    var errors = Python.list()
    if classifier:
        var labels = trees_encode_labels(args[1], context)
        py_setitem(fields, "classes_", labels.classes)
        var k = len(labels.classes)
        py_setitem(fields, "n_classes_", _P(k))
        if k < 2:
            raise Error("y has fewer than 2 classes")
        var base = fields["estimator"]
        if base is None:
            var settings = Python.dict()
            py_setitem(settings, "max_depth", _P(1))
            base = _new_member(context["trees_dt_classifier"], settings, context)
        _require_tree(base, "AdaBoostClassifier", context)
        var w = _normalized_weights(xt, args[2], n, context)
        var stats = trees_vector(4, "<f8", context)
        py_setitem(fields, "estimators_", members)
        py_setitem(fields, "estimator_weights_", alphas)
        py_setitem(fields, "estimator_errors_", errors)
        var codes = labels.codes
        # A best-splitter DecisionTreeClassifier member fits column-major X
        # with weights: X is transposed and scanned ONCE for every member.
        var dt_best = _is_class(base, "trees_dt_classifier", context) and Bool(py=py_fields(base)["splitter"] == PythonObject("best"))
        var xcm: Optional[TreesMatrix] = None
        var encoded: Optional[TreesLabels] = None
        var session = Session(PythonObject(None), 0)
        if dt_best:
            var cm = _colmajor(xt, x, context)
            if not trees_all_finite(cm.addr, n * d, False, context):
                raise Error(TREES_NONFINITE)
            _ = cm  # keep the Python object alive through the native calls above
            if trees_env("MOJOLEARN_ADABOOST_REENCODE", "") != "1":
                encoded = trees_encode_labels(codes, context)
            session = _member_session(base, cm, False, True, _ada_session_default(fields, xt, context), context)
            xcm = cm^
        var is_dt = Bool(py=py_call(trees_builtins().isinstance, base, context["trees_dt_classifier"]))
        try:
            for it in range(m):  # driver-loop: over the boosting rounds (one member each)
                var member = _clone(base, _sub_seed(seed, it), context)
                if xcm:
                    _member_fit_weighted(member, xcm.value(), codes, _w32(xt, w, n, context), encoded, session, context)
                else:
                    _member_fit(member, xa, codes, _w32(xt, w, n, context), context)
                var pred = _i32_codes(_member_predict(member, xa, context), context)
                _ = py_call_typed(_x(xt, "x_trees_samme_step"), _A(w), _A(pred), _A(codes), _A(stats),
                                  Python.list(_P(n), _P(k), _F(lr), _P(1 if it == m - 1 else 0), _P(1 if is_dt else 0)))
                _ = pred  # keep the Python object alive through the native call above
                _ = codes  # keep the Python object alive through the native call above
                _ = stats  # keep the Python object alive through the native call above
                _ = w  # keep the Python object alive through the native call above
                var status = _f64_word(stats, 0)
                var total = _f64_word(stats, 3)
                if status == 2.0:
                    if len(members) == 0:
                        raise Error("BaseClassifier in AdaBoostClassifier ensemble is worse than random,"
                                    + " ensemble can not be fit.")
                    break
                _ = py_call(members.append, member)
                _ = py_call(alphas.append, _F(_f64_word(stats, 1)))
                _ = py_call(errors.append, _F(_f64_word(stats, 2)))
                if status == 1.0 or not total > 0:
                    break
                if it < m - 1:
                    _scale(xt, w, total)
        finally:
            session.close()
        py_setitem(fields, "n_features_in_", _P(d))
        return model
    var y32 = trees_f32(args[1], 1, "C", "y", context)
    _y_rows_check(y32.rows, n)
    var base = fields["estimator"]
    if base is None:
        var settings = Python.dict()
        py_setitem(settings, "max_depth", _P(3))
        base = _new_member(context["trees_dt_regressor"], settings, context)
    _require_tree(base, "AdaBoostRegressor", context)
    var w = _normalized_weights(xt, args[2], n, context)
    var stats = trees_vector(4, "<f8", context)
    var cols = _arange(xt, d, context)
    py_setitem(fields, "estimators_", members)
    py_setitem(fields, "estimator_weights_", alphas)
    py_setitem(fields, "estimator_errors_", errors)
    # A best-splitter squared-error DecisionTreeRegressor member fits rows
    # of the SAME X staged on the device once (the member's rows gathered
    # there).
    var session = Session(PythonObject(None), 0)
    var default = _ada_session_default(fields, xt, context)
    if _session_choice(default) != "" and _is_class(base, "trees_dt_regressor", context):
        var bf = py_fields(base)
        var crit = bf["criterion"]
        if Bool(py=bf["splitter"] == PythonObject("best")) and (Bool(py=crit == PythonObject("squared_error"))
                                                                  or Bool(py=crit == PythonObject("mse"))):
            var native = trees_table(context, _member_mode(base, context), "_mojolearn_rf")
            if trees_entry(native, "rf_regressor_fit_session_rows_export") is not None:
                session = _member_session(base, x, True, False, default, context)
    var loss = Int(py=context["trees_ada_losses"][fields["loss"]])
    var is_dt = Bool(py=py_call(trees_builtins().isinstance, base, context["trees_dt_regressor"]))
    var y_array = _as_array(y32, Python.tuple(n), context)
    try:
        for it in range(m):  # driver-loop: over the boosting rounds (one member each)
            var rows = trees_vector(n, "<i4", context)
            _ = py_call_typed(_x(xt, "x_trees_weighted_sample"), _A(w), _A(rows), Python.list(_P(n), _P(n), _P(seed), _P(it)))
            _ = rows  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
            var member = _clone(base, _sub_seed(seed, it), context)
            var yr = _gather_vec(xt, y32.addr, n, rows, context)
            _ = y32  # keep the Python object alive through the native call above
            if session.handle > 0:
                _member_fit_session_rows(member, session, rows, yr, d, context)
                _ = y32  # keep the Python object alive through the native call above
            else:
                _member_fit(member, _gather(xt, x.addr, n, d, rows, cols, context), yr, PythonObject(None), context)
                _ = y32  # keep the Python object alive through the native call above
                _ = x  # keep the Python object alive through the native call above
            var pred = _f32_vector(_member_predict(member, xa, context), context)
            _ = py_call_typed(_x(xt, "x_trees_r2_step"), _A(w), _P(pred.addr), _A(y_array), _A(stats),
                              Python.list(_P(n), _P(loss), _F(lr), _P(1 if it == m - 1 else 0), _P(1 if is_dt else 0)))
            _ = pred  # keep the Python object alive through the native call above
            _ = y_array  # keep the Python object alive through the native call above
            _ = stats  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
            var status = _f64_word(stats, 0)
            var total = _f64_word(stats, 3)
            if status == 2.0:
                if len(members) == 0:
                    _ = py_call(members.append, member)
                    _ = py_call(alphas.append, _F(0.0))
                    _ = py_call(errors.append, _F(_f64_word(stats, 2)))
                break
            _ = py_call(members.append, member)
            _ = py_call(alphas.append, _F(_f64_word(stats, 1)))
            _ = py_call(errors.append, _F(_f64_word(stats, 2)))
            if status == 1.0 or not total > 0:
                break
            if it < m - 1:
                _scale(xt, w, total)
    finally:
        session.close()
    py_setitem(fields, "n_features_in_", _P(d))
    return model


def _ada_x(fields: PythonObject, X: PythonObject, context: PythonObject) raises -> TreesMatrix:
    """`_AdaBoostBase._check_X`."""
    if not py_contains(fields, "estimators_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var x = _x_f32(X, context)
    var fitted = Int(py=_int(fields["n_features_in_"]))
    if x.cols != fitted:
        raise Error("X has " + String(x.cols) + " features, fit saw " + String(fitted))
    return x^


def _ada_decision(fields: PythonObject, x: TreesMatrix, xt: PythonObject, context: PythonObject) raises -> PythonObject:
    """`AdaBoostClassifier._decision`: the members' weighted one-hot votes
    over the native fsum of their weights."""
    var n = x.rows
    var k = Int(py=_int(fields["n_classes_"]))
    var xa = _as_array(x, Python.tuple(n, x.cols), context)
    var acc = _class_major(xt, Python.list(_F(0.0)), n * k, context)
    var members = fields["estimators_"]
    var alphas = fields["estimator_weights_"]
    for i in range(len(members)):  # driver-loop: over the member estimators
        var pred = _i32_codes(_member_predict(members[i], xa, context), context)
        var a = Float64(py=_float(alphas[i]))
        _acc_votes(xt, acc, pred, n, k, a, -a / Float64(k - 1))
    _scale(xt, acc, _fsum(alphas, context))
    return acc


def ada_decision_function(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var x = _ada_x(fields, args[0], context)
    var xt = _xt(fields, context)
    var n = x.rows
    var k = Int(py=_int(fields["n_classes_"]))
    var acc = _ada_decision(fields, x, xt, context)
    if k == 2:
        var d = trees_vector(n, "<f8", context)
        _ = py_call_typed(_x(xt, "x_trees_margin2"), _A(acc), _A(d), Python.list(_P(n), _P(0)))
        _ = d  # keep the Python object alive through the native call above
        _ = acc  # keep the Python object alive through the native call above
        return d
    return trees_reshape(acc, Python.tuple(n, k), context)


def ada_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var x = _ada_x(fields, args[0], context)
    var xt = _xt(fields, context)
    var n = x.rows
    var k = Int(py=_int(fields["n_classes_"]))
    var acc = _ada_decision(fields, x, xt, context)
    if k == 2:
        var pairs = trees_vector(n * 2, "<f8", context)
        _ = py_call_typed(_x(xt, "x_trees_margin2"), _A(acc), _A(pairs), Python.list(_P(n), _P(2)))
        _ = pairs  # keep the Python object alive through the native call above
        _ = acc  # keep the Python object alive through the native call above
        acc = pairs
    else:
        _scale(xt, acc, Float64(k - 1))
    _ = py_call_typed(_x(xt, "x_trees_softmax_rows"), _A(acc), Python.list(_P(n), _P(k)))
    _ = acc  # keep the Python object alive through the native call above
    return trees_reshape(acc, Python.tuple(n, k), context)


def ada_predict(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var x = _ada_x(fields, args[0], context)
    var xt = _xt(fields, context)
    var n = x.rows
    if classifier:
        var k = Int(py=_int(fields["n_classes_"]))
        var acc = _ada_decision(fields, x, xt, context)
        var codes = PythonObject(None)
        if k == 2:
            codes = trees_vector(n, "<i4", context)
            _ = py_call_typed(_x(xt, "x_trees_margin2"), _A(acc), _A(codes), Python.list(_P(n), _P(1)))
            _ = codes  # keep the Python object alive through the native call above
            _ = acc  # keep the Python object alive through the native call above
        else:
            codes = _argmax(xt, acc, n, k, context)
        return trees_decode_labels(fields["classes_"], codes, context)
    # `AdaBoostRegressor.predict`: the weighted median of the members.
    var members = fields["estimators_"]
    var m = len(members)
    var xa = _as_array(x, Python.tuple(n, x.cols), context)
    var preds = trees_vector(m * n, "<f4", context)
    for j in range(m):  # driver-loop: over the member estimators
        var p = _f32_vector(_member_predict(members[j], xa, context), context)
        _ = py_call_typed(_x(xt, "x_trees_put_f32"), _A(preds), _P(p.addr), Python.list(_P(j * n), _P(n)))
        _ = p  # keep the Python object alive through the native call above
        _ = preds  # keep the Python object alive through the native call above
    var b = trees_builtins()
    var weights_store = py_call(context["array_type"], PythonObject("d"), py_call(b.map, b.float, fields["estimator_weights_"]))
    var weights = classical_wrap_array(weights_store, Python.tuple(m), "<f8", "C", context)
    var out = trees_vector(n, "<f4", context)
    _ = py_call_typed(_x(xt, "x_trees_weighted_median"), _A(preds), _A(weights), _A(out), Python.list(_P(n), _P(m)))
    _ = out  # keep the Python object alive through the native call above
    _ = weights  # keep the Python object alive through the native call above
    _ = preds  # keep the Python object alive through the native call above
    return out


# -------------------------------------------------------------------- DART


comptime _DART_VALUES_CAP = 1 << 26


def _dart_thr(v: Float64) -> Int:
    """`_DARTBase._dart_thr`: skip_drop for the binding's 53-bit draw."""
    if not v > 0.0:
        return 0
    if v >= 1.0:
        return 1 << 53
    return Int(ceil(v * 9007199254740992.0))


def _dart_bag(fields: PythonObject, xt: PythonObject, n: Int, it: Int, mut cache_start: Int, mut cache: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """`_DARTBase._bag`: the bagged rows of round `it` (None: every row)."""
    var freq = Int(py=_int(fields["subsample_freq"]))
    var frac = Float64(py=_float(fields["subsample"]))
    if freq <= 0 or not frac < 1.0:
        return PythonObject(None)
    var start = it - it % freq
    if cache is not None and cache_start == start:
        return cache
    var out = trees_vector(max(1, n), "<i4", context)
    var k = Int(py=py_call_typed(_x(xt, "x_trees_bag_rows"), _A(out),
                                 Python.list(_P(n), _P(_seed(fields["bagging_seed"], context)), _P(start), _F(frac))))
    _ = out  # keep the Python object alive through the native call above
    cache_start = start
    cache = _copy_i32(out, 0, k, context)
    return cache


def _dart_cols(fields: PythonObject, xt: PythonObject, d: Int, t: Int, context: PythonObject) raises -> PythonObject:
    """`_DARTBase._cols`: the sorted feature subset of tree `t` (None: all)."""
    var frac = Float64(py=_float(fields["colsample_bytree"]))
    if not frac < 1.0:
        return PythonObject(None)
    var cnt = max(Int(py=_int(_F(Float64(d) * frac + 0.5))), min(1, d))
    if cnt >= d:
        return PythonObject(None)
    return _indices(xt, d, cnt, False, _seed(fields["feature_fraction_seed"], context), t, context)


def _dart_member(fields: PythonObject, handle: PythonObject, c: Int, n: Int, d: Int, rows: PythonObject,
                 cols: PythonObject, max_depth: PythonObject, seed: Int, j: Int, context: PythonObject) raises -> PythonObject:
    """One DART member: a one-tree RandomForestRegressor fitted by
    `rf_regressor_fit_dart_export` on the round's device X and class-c
    target (`RandomForestRegressor._fit_in_dart`)."""
    var settings = Python.dict()
    py_setitem(settings, "n_estimators", _P(1))
    py_setitem(settings, "bootstrap", PythonObject(False))
    py_setitem(settings, "max_features", _F(1.0))
    py_setitem(settings, "max_depth", max_depth)
    py_setitem(settings, "max_leaves", _int(fields["num_leaves"]))
    py_setitem(settings, "min_samples_leaf", _int(fields["min_child_samples"]))
    py_setitem(settings, "n_bins", _int(fields["max_bin"]))
    py_setitem(settings, "random_state", _P(_sub_seed(seed, j)))
    py_setitem(settings, "n_streams", _P(1))
    py_setitem(settings, "numeric_mode", py_get(fields, "numeric_mode"))
    var tree = _new_member(context["trees_rf_regressor"], settings, context)
    var tf = forest_refresh(tree, "rf_regressor", context)
    var mode = forest_capture_mode(tf, context)
    var code = tf["_cfg"]["criterion"]
    if Int(py=code) != Int(py=context["trees_rf_tables"]["reg_criteria"]["squared_error"]):
        raise Error("a DART member fits squared_error")
    var rf = trees_table(context, mode, "_mojolearn_rf")
    var entry = trees_need(rf, "rf_regressor_fit_dart_export", "rebuild the forest binding for DART member fits on the device")
    var m = n if rows is None else len(rows)
    var dc = d if cols is None else len(cols)
    var params = _rf_params(tf, "rf_regressor", m, dc, 0, rf, context)
    var result = forest_export(rf, py_call_typed(entry, _int(handle), _P(c), _P(0) if rows is None else _A(rows),
                                                 _P(0) if cols is None else _A(cols), params, code), "into", context)
    _member_store(tree, forest_fit_arrays(result, context), dc)
    return tree


def _dart_boost(model: PythonObject, fields: PythonObject, x: TreesMatrix, y32: TreesMatrix, K: Int, kind: Int,
                context: PythonObject) raises -> PythonObject:
    """`_DARTBase._boost` and its device round (`_boost_loop_device`)."""
    var b = trees_builtins()
    var n = x.rows
    var d = x.cols
    var xt = _xt(fields, context)
    var seed = _seed(fields["random_state"], context)
    var drop_seed = _seed(fields["drop_seed"], context)
    var inits = Python.list()
    if kind == 0 or kind == 1:
        # `_trees_fsum_f32(b, y32) / n`: the exact sum, refused when not finite
        var out = trees_vector(1, "<f8", context)
        var flags = trees_vector(8, "<i4", context)
        _ = py_call_typed(_x(xt, "x_trees_exact_sum"), _P(y32.addr), _A(out), _A(flags), Python.list(_P(y32.rows), _P(0)))
        _ = flags  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        if _i32_word(flags, 0) != 0 or _i32_word(flags, 1) != 0 or _i32_word(flags, 2) != 0:
            raise Error("Input y contains NaN or infinity.")
        var mean = _f64_word(out, 0) / Float64(n)
        if kind == 0:
            _ = py_call(inits.append, _F(mean))
        else:
            if not (0.0 < mean and mean < 1.0):
                raise Error("y must hold both classes")
            _ = py_call(inits.append, _float(py_call_typed(_x(xt, "x_trees_log64"), _F(mean / (1.0 - mean)))))
    else:
        var cnt = trees_vector(K, "<i4", context)
        _ = py_call_typed(_x(xt, "x_trees_class_counts"), _P(y32.addr), _A(cnt), Python.list(_P(n), _P(K)))
        _ = cnt  # keep the Python object alive through the native call above
        var priors = trees_vector(K, "<f8", context)
        _ = py_call_typed(_x(xt, "x_trees_class_priors"), _A(cnt), _A(priors), Python.list(_P(n), _P(K)))
        _ = priors  # keep the Python object alive through the native call above
        _ = cnt  # keep the Python object alive through the native call above
        for c in range(K):  # driver-loop: over the K class starts
            _ = py_call(inits.append, _F(_f64_word(priors, c)))
    py_setitem(fields, "init_score_", inits[0] if K == 1 else inits)
    var lr = Float64(py=_float(fields["learning_rate"]))
    var l1 = Float64(py=_float(fields["reg_alpha"]))
    var mds = Float64(py=_float(fields["max_delta_step"]))
    var lam = Float64(py=_float(fields["reg_lambda"]))
    py_setitem(fields, "n_classes_", _P(K))
    var trees = Python.list()
    var tree_values = Python.list()
    py_setitem(fields, "trees_", trees)
    py_setitem(fields, "tree_values_", tree_values)
    py_setitem(fields, "tree_coefs_", Python.list())
    py_setitem(fields, "tree_weights_", Python.list())
    py_setitem(fields, "_bag_at", PythonObject(None))
    var depth_param = fields["max_depth"]
    var max_depth = PythonObject(None) if depth_param is None or Int(py=_int(depth_param)) <= 0 else _int(depth_param)
    var bagged = Float64(py=_float(fields["subsample"])) < 1.0 and Int(py=_int(fields["subsample_freq"])) > 0
    var sampled = Float64(py=_float(fields["colsample_bytree"])) < 1.0
    if not bagged and not sampled:
        # The Python fit opened a forest data session here whose X the
        # round never read (`x_trees_dart_open` holds X): only its finite
        # scan is kept, on the tier a one-tree regressor would use.
        var probe_mode = _mode(fields, context)
        var native = trees_table(context, probe_mode, "_mojolearn_rf")
        if _session_choice("1") != "" and trees_entry(native, "rf_data_session_open") is not None:
            if not trees_all_finite(x.addr, n * d, False, context):
                raise Error(TREES_NONFINITE)
    # `_dart_device`: the round's entries and the leaf-value cap.
    if trees_entry(xt, "x_trees_dart_open") is None:
        raise Error("NotImplementedError: mojolearn DART: this binary has no x_trees_dart_* round (a FAST CPU-only install or a "
                    + "MOJOLEARN_*DART_DEVICE_OFF build); the host loop is removed, so the fit is refused by name")
    var n_iters = Int(py=_int(fields["n_estimators"]))
    var node_cap = 2 * Int(py=_int(fields["num_leaves"])) - 1
    if not (1 <= node_cap and node_cap <= 65535 and n_iters * K * node_cap <= _DART_VALUES_CAP):
        raise Error("NotImplementedError: mojolearn DART: n_estimators * classes * (2 * num_leaves - 1) must be at most "
                    + String(_DART_VALUES_CAP) + " and num_leaves at most 32768 (the round's leaf-value store); refused by name")
    var all_cols = _arange(xt, d, context)
    var inits32_store = py_call(context["array_type"], PythonObject("f"), inits)
    var inits32 = classical_wrap_array(inits32_store, Python.tuple(len(inits)), "<f4", "C", context)
    var skip_thr = _dart_thr(Float64(py=_float(fields["skip_drop"])))
    var bad = trees_vector(1, "<i4", context)
    var weights64 = trees_vector(n_iters, "<f8", context)
    var coefs64 = trees_vector(n_iters * K, "<f8", context)
    var coef32 = trees_vector(n_iters * K, "<f4", context)
    var thr64 = trees_vector(n_iters, "<i8", context)
    var flags = trees_vector(n_iters, "<i4", context)
    var state = trees_vector(1, "<f8", context)
    var uniform = 1 if _truthy(fields["uniform_drop"]) else 0
    var xgb = 1 if _truthy(fields["xgboost_dart_mode"]) else 0
    var max_drop = Int(py=_int(fields["max_drop"]))
    var drop_rate = _float(fields["drop_rate"])
    var handle = py_call_typed(_x(xt, "x_trees_dart_open"), _P(x.addr), _P(y32.addr), _A(inits32),
                               Python.list(_P(n), _P(d), _P(K), _P(kind), _P(n_iters), _P(node_cap)))
    _ = inits32  # keep the Python object alive through the native call above
    var cache_start = -1
    var cache = PythonObject(None)
    try:
        for it in range(n_iters):  # driver-loop: over the boosting rounds
            _ = py_call_typed(_x(xt, "x_trees_dart_thresholds"), _A(weights64), _A(coefs64), _A(coef32), _A(thr64),
                              _A(state), Python.list(_P(it), _P(K), _P(uniform), _P(max_drop)), drop_rate)
            _ = thr64  # keep the Python object alive through the native call above
            _ = coef32  # keep the Python object alive through the native call above
            _ = coefs64  # keep the Python object alive through the native call above
            _ = weights64  # keep the Python object alive through the native call above
            _ = state  # keep the Python object alive through the native call above
            _ = py_call_typed(_x(xt, "x_trees_dart_step"), handle, _A(coef32), _A(thr64), _A(flags), _A(bad),
                              Python.list(), Python.list(_P(it), _P(drop_seed), _P(it), _P(skip_thr)))
            _ = flags  # keep the Python object alive through the native call above
            _ = thr64  # keep the Python object alive through the native call above
            _ = coef32  # keep the Python object alive through the native call above
            _ = bad  # keep the Python object alive through the native call above
            _ = inits32  # keep the Python object alive through the native call above
            if _i32_word(bad, 0) != 0:
                raise Error("RuntimeError: x_trees dart: a tree walk left its tree (child or column out of range)")
            var rescaled = py_call_typed(_x(xt, "x_trees_dart_rescale"), _A(weights64), _A(coefs64), _A(flags), _A(state),
                                         Python.list(_P(it), _P(K), _P(uniform), _P(xgb)), _F(lr))
            _ = state  # keep the Python object alive through the native call above
            _ = flags  # keep the Python object alive through the native call above
            _ = coefs64  # keep the Python object alive through the native call above
            _ = weights64  # keep the Python object alive through the native call above
            var shrink = _float(rescaled[1])
            _ = state  # keep the Python object alive through the native call above
            _ = flags  # keep the Python object alive through the native call above
            _ = coefs64  # keep the Python object alive through the native call above
            _ = weights64  # keep the Python object alive through the native call above
            var factor = _float(rescaled[2])
            _ = state  # keep the Python object alive through the native call above
            _ = flags  # keep the Python object alive through the native call above
            _ = coefs64  # keep the Python object alive through the native call above
            _ = weights64  # keep the Python object alive through the native call above
            var rows = _dart_bag(fields, xt, n, it, cache_start, cache, context)
            var m = 0 if rows is None else len(rows)
            for c in range(K):  # driver-loop: over the K member trees of one round
                var j = it * K + c
                var cols = _dart_cols(fields, xt, d, j, context)
                var tree = _dart_member(fields, handle, c, n, d, rows, cols, max_depth, seed, j, context)
                _ = inits32  # keep the Python object alive through the native call above
                var tf = py_fields(tree)
                if cols is not None:
                    var colid = tf["_colid"]
                    var cid = _copy_i32(colid, 0, Int(py=colid.size), context)
                    _ = py_call_typed(_x(xt, "x_trees_remap_cols"), _A(cid), _A(cols), Python.list(_P(Int(py=cid.size)), _P(len(cols))))
                    _ = cid  # keep the Python object alive through the native call above
                    _ = cols  # keep the Python object alive through the native call above
                    py_setitem(tf, "_colid", cid)
                var offsets = tf["_offsets"]
                var lo = _i32_word(offsets, 0)
                var n_nodes = _i32_word(offsets, 1) - lo
                var values = trees_vector(n_nodes, "<f4", context)
                _ = py_call_typed(_x(xt, "x_trees_dart_add"), handle, _A(tf["_colid"]), _A(tf["_quesval"]),
                                  _A(tf["_left_child"]), _A(values), _A(rows if rows is not None else all_cols),
                                  Python.list(_P(j), _P(c), _P(lo), _P(n_nodes), shrink, factor, _F(lam), _F(l1), _F(mds), _P(m)))
                _ = inits32  # keep the Python object alive through the native call above
                _ = values  # keep the Python object alive through the native call above
                _ = py_call(trees.append, tree)
                _ = py_call(tree_values.append, values)
    finally:
        _ = py_call_typed(_x(xt, "x_trees_dart_close"), handle, _A(bad))
        _ = bad  # keep the Python object alive through the native call above
        _ = inits32  # keep the Python object alive through the native call above
    if _i32_word(bad, 0) != 0:
        raise Error("RuntimeError: x_trees dart: a tree walk left its tree (child or column out of range)")
    var coefs = Python.list()
    for i in range(n_iters * K):  # driver-loop: over the tree coefficients (trees-sized)
        _ = py_call(coefs.append, _F(_f64_word(coefs64, i)))
    var weights = Python.list()
    for i in range(n_iters):  # driver-loop: over the tree weights (rounds-sized)
        _ = py_call(weights.append, _F(_f64_word(weights64, i)))
    py_setitem(fields, "tree_coefs_", coefs)
    py_setitem(fields, "tree_weights_", weights)
    py_setitem(fields, "_bag_at", PythonObject(None))
    py_setitem(fields, "n_features_in_", _P(d))
    return model


def dart_fit(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    """DARTRegressor / DARTClassifier `fit`."""
    var fields = py_fields(model)
    var x = _x_f32(args[0], context)
    if not classifier:
        var y32 = trees_f32(args[1], 1, "C", "y", context)
        _y_rows_check(y32.rows, x.rows)
        return _dart_boost(model, fields, x, y32, 1, 0, context)
    var labels = trees_encode_labels(args[1], context)
    py_setitem(fields, "classes_", labels.classes)
    var K = len(labels.classes)
    if K < 2:
        raise Error("y must hold at least two classes")
    _y_rows_check(labels.n, x.rows)
    var codes = trees_vector(labels.n, "<f4", context)
    trees_cast(trees_address(labels.codes), "<i4", trees_address(codes), "<f4", labels.n, context)
    _ = codes  # keep the Python object alive through the native call above
    var y32 = TreesMatrix(codes, trees_address(codes), labels.n, 1, True)
    _ = codes  # keep the Python object alive through the native call above
    var kind = 1 if K == 2 else 2
    py_setitem(fields, "_KIND", _P(kind))
    var _keep_ret_codes = _dart_boost(model, fields, x, y32, 1 if K == 2 else K, kind, context)
    _ = codes  # keep the Python object alive through the native call above
    return _keep_ret_codes^


def _dart_raw(fields: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_DARTBase._raw`: class-major raw scores (K * n) float64 from ONE
    native walk of every row through every tree."""
    if not py_contains(fields, "trees_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var x = _x_f32(X, context)
    var fitted = Int(py=_int(fields["n_features_in_"]))
    if x.cols != fitted:
        raise Error("X has " + String(x.cols) + " features, fit saw " + String(fitted))
    var n = x.rows
    var d = x.cols
    var K = Int(py=_int(py_get(fields, "n_classes_", _P(1))))
    var score = trees_vector(K * n, "<f8", context)
    if n == 0:
        return score
    var b = trees_builtins()
    var cols = Python.list()
    var ques = Python.list()
    var lefts = Python.list()
    var vals = Python.list()
    var sizes = Python.list()
    var coefs = Python.list()
    var trees = fields["trees_"]
    var values = fields["tree_values_"]
    var tree_coefs = fields["tree_coefs_"]
    var count = min(len(trees), min(len(values), len(tree_coefs)))
    for i in range(count):  # driver-loop: over the T fitted trees (one address per tree and array)
        var tf = py_fields(trees[i])
        var lo = _i32_word(tf["_offsets"], 0)
        _ = py_call(cols.append, _P(trees_address(tf["_colid"]) + 4 * lo))
        _ = py_call(ques.append, _P(trees_address(tf["_quesval"]) + 4 * lo))
        _ = py_call(lefts.append, _P(trees_address(tf["_left_child"]) + 4 * lo))
        _ = py_call(vals.append, _A(values[i]))
        _ = py_call(sizes.append, _P(_i32_word(tf["_offsets"], 1) - lo))
        _ = py_call(coefs.append, _float(tree_coefs[i]))
    var init = fields["init_score_"]
    var starts = Python.list(_float(init)) if K == 1 else py_call(b.list, py_call(b.map, b.float, init))
    var xt = _xt(fields, context)
    _ = py_call_typed(_x(xt, "x_trees_dart_predict"), _P(x.addr), Python.list(cols, ques, lefts, vals), sizes, coefs,
                      starts, _A(score), Python.list(_P(n), _P(d), _P(K)))
    _ = x  # keep the Python object alive through the native call above
    _ = score  # keep the Python object alive through the native call above
    _ = x^
    return score


def _dart_raw_rows(fields: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_DARTBase._raw_rows`: (n, K) row-major raw scores."""
    var raw = _dart_raw(fields, X, context)
    var K = Int(py=_int(fields["n_classes_"]))
    var n = Int(py=raw.size) // K
    var out = trees_vector(n * K, "<f8", context)
    _ = py_call_typed(_x(_xt(fields, context), "x_trees_transpose_f64"), _A(raw), _A(out), Python.list(_P(K), _P(n)))
    _ = out  # keep the Python object alive through the native call above
    _ = raw  # keep the Python object alive through the native call above
    return trees_reshape(out, Python.tuple(n, K), context)


def _multiclass(fields: PythonObject) raises -> Bool:
    return Int(py=_int(py_get(fields, "n_classes_", _P(1)))) > 1


def dart_predict(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not classifier:
        return _dart_raw(fields, args[0], context)
    var xt = _xt(fields, context)
    if _multiclass(fields):
        var acc = _dart_raw_rows(fields, args[0], context)
        var n = Int(py=acc.shape[0])
        var K = Int(py=acc.shape[1])
        return trees_decode_labels(fields["classes_"], _argmax(xt, acc, n, K, context), context)
    var raw = _dart_raw(fields, args[0], context)
    var n = Int(py=raw.size)
    var codes = trees_vector(n, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_positive_codes"), _A(raw), _A(codes), Python.list(_P(n)))
    _ = codes  # keep the Python object alive through the native call above
    _ = raw  # keep the Python object alive through the native call above
    return trees_decode_labels(fields["classes_"], codes, context)


def dart_decision_function(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if _multiclass(fields):
        return _dart_raw_rows(fields, args[0], context)
    return _dart_raw(fields, args[0], context)


def dart_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var xt = _xt(fields, context)
    if _multiclass(fields):
        var acc = _dart_raw_rows(fields, args[0], context)
        var n = Int(py=acc.shape[0])
        var K = Int(py=acc.shape[1])
        var flat = trees_reshape(acc, Python.tuple(n * K), context)
        _ = py_call_typed(_x(xt, "x_trees_softmax_rows"), _A(flat), Python.list(_P(n), _P(K)))
        _ = flat  # keep the Python object alive through the native call above
        return trees_reshape(flat, Python.tuple(n, K), context)
    var raw = _dart_raw(fields, args[0], context)
    var n = Int(py=raw.size)
    var acc = trees_vector(2 * n, "<f8", context)
    var zero = trees_vector(n, "<f8", context)
    _ = py_call_typed(_x(xt, "x_trees_stack_w64"), Python.list(_A(zero), _A(raw)), _A(acc), Python.list(_P(n)))
    _ = zero  # keep the Python object alive through the native call above
    _ = acc  # keep the Python object alive through the native call above
    _ = raw  # keep the Python object alive through the native call above
    _ = py_call_typed(_x(xt, "x_trees_softmax_rows"), _A(acc), Python.list(_P(n), _P(2)))
    _ = acc  # keep the Python object alive through the native call above
    _ = zero^
    return trees_reshape(acc, Python.tuple(n, 2), context)


# ---------------------------------------------------- RandomTreesEmbedding


def _rte_validate(fields: PythonObject, context: PythonObject) raises:
    """`RandomTreesEmbedding._validate_params`."""
    if _truthy(fields["sparse_output"]):
        _refuse("sparse_output=True", "the NumPy-free layer has no sparse container; the dense"
                + " one-hot is returned.")
    if fields["n_jobs"] is not None or _truthy(fields["verbose"]) or _truthy(fields["warm_start"]):
        _refuse("n_jobs/verbose/warm_start", "not carried in pass 1.")
    _ = _seed(fields["random_state"], context)


def _rte_fit(model: PythonObject, fields: PythonObject, x: TreesMatrix, context: PythonObject) raises:
    """`RandomTreesEmbedding.fit` on a C-order float32 X: uniform targets,
    one ExtraTreesRegressor fit, the leaf numbering."""
    var n = x.rows
    var d = x.cols
    var seed = _seed(fields["random_state"], context)
    var xt = _xt(fields, context)
    var u = trees_vector(n, "<f8", context)
    _ = py_call_typed(_x(xt, "x_trees_uniform"), _A(u), Python.list(_P(n), _P(seed), _P(0)))
    _ = u  # keep the Python object alive through the native call above
    var yr = trees_f32(u, 1, "C", "y", context)
    var settings = Python.dict()
    py_setitem(settings, "n_estimators", _int(fields["n_estimators"]))
    py_setitem(settings, "max_depth", fields["max_depth"])
    py_setitem(settings, "max_features", _P(1))
    py_setitem(settings, "min_samples_split", fields["min_samples_split"])
    py_setitem(settings, "min_samples_leaf", fields["min_samples_leaf"])
    py_setitem(settings, "max_leaf_nodes", fields["max_leaf_nodes"])
    py_setitem(settings, "min_impurity_decrease", fields["min_impurity_decrease"])
    py_setitem(settings, "random_state", _P(seed))
    py_setitem(settings, "numeric_mode", py_get(fields, "numeric_mode"))
    var forest = _new_member(context["trees_et_regressor"], settings, context)
    _member_fit(forest, _as_array(x, Python.tuple(n, d), context), yr.keep, PythonObject(None), context)
    py_setitem(fields, "forest_", forest)
    var ff = py_fields(forest)
    var offsets = ff["_offsets"]
    var n_trees = Int(py=offsets.size) - 1
    var nn = _i32_word(offsets, n_trees)
    var node_col = trees_vector(nn, "<i4", context)
    var col = Int(py=py_call_typed(_x(xt, "x_trees_leaf_numbering"), _A(ff["_left_child"]), _A(node_col), Python.list(_P(nn))))
    _ = node_col  # keep the Python object alive through the native call above
    py_setitem(fields, "_tree_base", _copy_i32(offsets, 0, n_trees, context))
    py_setitem(fields, "_node_col", node_col)
    py_setitem(fields, "n_trees_", _P(n_trees))
    py_setitem(fields, "n_output_features_", _P(col))
    _ = node_col  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", _P(d))


def _rte_transform(fields: PythonObject, x: TreesMatrix, context: PythonObject) raises -> PythonObject:
    """`RandomTreesEmbedding.transform` (`apply`, then the one-hot)."""
    if not py_contains(fields, "forest_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var n = x.rows
    var d = x.cols
    var fitted = Int(py=_int(fields["n_features_in_"]))
    if d != fitted:
        raise Error("X has " + String(d) + " features, fit saw " + String(fitted))
    var xt = _xt(fields, context)
    var ff = py_fields(fields["forest_"])
    var trees = Int(py=_int(fields["n_trees_"]))
    var width = Int(py=_int(fields["n_output_features_"]))
    var nodes = trees_vector(n * trees, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_apply"), _A(ff["_offsets"]), _A(ff["_colid"]), _A(ff["_quesval"]),
                      _A(ff["_left_child"]), _P(x.addr), _A(nodes), Python.list(_P(n), _P(d), _P(0), _P(trees)))
    _ = nodes  # keep the Python object alive through the native call above
    var out = trees_vector(n * width, "<f8", context)
    _ = py_call_typed(_x(xt, "x_trees_onehot_leaves"), _A(nodes), _A(fields["_tree_base"]), _A(fields["_node_col"]),
                      _A(out), Python.list(_P(n), _P(trees), _P(width)))
    _ = nodes  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return trees_reshape(out, Python.tuple(n, width), context)


def rte_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    _rte_validate(fields, context)
    var x = _x_f32(args[0], context)
    _rte_fit(model, fields, x, context)
    return model


def rte_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "forest_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    return _rte_transform(fields, _x_f32(args[0], context), context)


def rte_fit_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`fit(X).transform(X)` with X converted once."""
    var fields = py_fields(model)
    _rte_validate(fields, context)
    var x = _x_f32(args[0], context)
    _rte_fit(model, fields, x, context)
    return _rte_transform(fields, x, context)


# --------------------------------------------------------------- dispatch


def boost_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The connected schema of a boosted/bagged ensemble's class."""
    var b = trees_builtins()
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    var table = context["trees_boost_classes"]
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        var kind = py_get(table, mro[i])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: " + String(py_type(model).__name__) + " is not a connected tree ensemble")


def boost_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`trees_boost_<op>` of a connected Bagging*, AdaBoost*, DART* or
    RandomTreesEmbedding."""
    var kind = boost_kind(model, context)
    var classifier = kind.endswith("_classifier")
    if kind.startswith("bagging_"):
        if operation == "trees_boost_fit":
            return bagging_fit(model, args, classifier, context)
        if operation == "trees_boost_predict":
            return bagging_predict(model, args, classifier, context)
        if operation == "trees_boost_predict_proba":
            return bagging_predict_proba(model, args, context)
    elif kind.startswith("ada_"):
        if operation == "trees_boost_fit":
            return ada_fit(model, args, classifier, context)
        if operation == "trees_boost_predict":
            return ada_predict(model, args, classifier, context)
        if operation == "trees_boost_predict_proba":
            return ada_predict_proba(model, args, context)
        if operation == "trees_boost_decision_function":
            return ada_decision_function(model, args, context)
    elif kind.startswith("dart_"):
        if operation == "trees_boost_fit":
            return dart_fit(model, args, classifier, context)
        if operation == "trees_boost_predict":
            return dart_predict(model, args, classifier, context)
        if operation == "trees_boost_predict_proba":
            return dart_predict_proba(model, args, context)
        if operation == "trees_boost_decision_function":
            return dart_decision_function(model, args, context)
    elif kind == "rte":
        if operation == "trees_boost_fit":
            return rte_fit(model, args, context)
        if operation == "trees_boost_transform":
            return rte_transform(model, args, context)
        if operation == "trees_boost_fit_transform":
            return rte_fit_transform(model, args, context)
    raise Error("unknown tree ensemble operation " + operation + " for " + kind)
