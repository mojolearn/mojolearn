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
from bindings.trees_staged_input import trees_stage_members, trees_stage_rounds, trees_unstage
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.trees_side_records import trees_side_get, trees_side_put, trees_side_drop, trees_key_same
from bindings.classical_arrays import classical_wrap_array, _store_code
from bindings.trees_public_inputs import (
    TREES_NONFINITE, TreesLabels, trees_builtins, trees_table, trees_entry, trees_need,
    trees_helper, trees_env, trees_reshape, trees_address, trees_finite_f32,
    trees_encode_labels, trees_decode_labels, trees_cast,
    trees_weighted_rows, trees_call,
)
from bindings.admitted_input import (
    AdmittedX, admit_f32, admit_weight_words, admitted_words, admitted_array, admitted_vector,
)
# TOMBSTONE: TreesView / _view_of_array / trees_f32 / trees_all_finite / trees_check_sample_weight / trees_vector / trees_empty uses (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.trees_public_forest import (
    ForestFit, forest_fit, forest_predict, forest_predict_proba, forest_predict_codes, forest_get_params, forest_refresh,
    forest_capture_mode, forest_effective_mode, forest_export, forest_fit_arrays, forest_kind,
    _check_mode, _rf_params, _store_model, _int, _float, _repr, _str, _truthy, _is_bool, _is_integral,
    _refuse, _classifier, _launch, _y_rows_check, _dt_tier_bins, ForestHandle, forest_handle_done,
)
from bindings.trees_resident_table import (
    TREES_RESIDENT_DART, trees_resident_find, trees_resident_put, trees_resident_pin, trees_resident_watch,
    trees_resident_forget,
)
from bindings.native_model_records import model_identity_key, model_generation
from bindings.p3nc_fitted_freeze import fitted_freeze


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
    # The refusal text is built only on a miss (it was concatenated before
    # every successful lookup); same wording as trees_need's.
    var entry = py_get(table, PythonObject(name))
    if entry is None:
        raise Error("ImportError: mojolearn: the x_trees binding has no " + name)
    return entry


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


def _x_f32(source: PythonObject, context: PythonObject) raises -> AdmittedX:
    """`as_f32_c(X, ndim=2, name="X")`."""
    return admit_f32(source, 2, "X", context, allow_nonfinite=True)


def _as_array(m: AdmittedX, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A C-order float32 matrix as an Array a member's fit or predict reads:
    the matrix's own Array, or (a borrowed foreign buffer) one copy."""
    if py_type(m.array) is context["array_class"] and Int(py=m.array._addr) == m.address:
        return m.array
    var out = admitted_array(shape, "<f4", context)
    trees_cast(m.address, "<f4", trees_address(out), "<f4", m.rows * m.cols, context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _f32_vector(value: PythonObject, context: PythonObject) raises -> AdmittedX:
    """`as_f32_c(value, ndim=1, ...)` of a member's output."""
    return admit_f32(value, 1, "prediction", context, allow_nonfinite=True)


def _i32_codes(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`as_i32_c(value, ndim=1, name="predicted codes")` of a member's
    decoded int64 labels (its classes are the ensemble's codes)."""
    var n = Int(py=value.size)
    var out = admitted_vector(n, "<i4", context)
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
    var out = admitted_vector(n, "<i4", context)
    if n > 0:
        _ = py_call_typed(_x(xt, "x_trees_iota_i32"), _A(out), Python.list(_P(n)))
        _ = out  # keep the Python object alive through the native call above
    return out


def _class_major(xt: PythonObject, inits: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    var k = len(inits)
    var vals = py_call(context["array_type"], PythonObject("d"), inits)
    var vals_array = classical_wrap_array(vals, Python.tuple(k), "<f8", "C", context)
    var out = admitted_vector(k * n, "<f8", context)
    if k * n > 0:
        _ = py_call_typed(_x(xt, "x_trees_fill_class_major_f64"), _A(vals_array), _A(out), Python.list(_P(n), _P(k)))
        _ = out  # keep the Python object alive through the native call above
        _ = vals_array  # keep the Python object alive through the native call above
    return out


def _indices(xt: PythonObject, pool: Int, draw: Int, replace: Bool, seed: Int, stream: Int,
             context: PythonObject) raises -> PythonObject:
    var out = admitted_vector(draw, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_sample_indices"), _A(out),
                      Python.list(_P(pool), _P(draw), _P(1 if replace else 0), _P(seed), _P(stream)))
    _ = out  # keep the Python object alive through the native call above
    return out


# The gathers read trusted internal mojolearn Arrays (their `_addr` slot and
# shape), never arbitrary user objects; each Array must survive the
# raw-address call.
# TOMBSTONE: _gather_view / _gather_codes_view / _gather_vec_view (TreesView forms of the gathers) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
def _gather(xt: PythonObject, x_addr: Int, n_src: Int, d_src: Int, rows: PythonObject, cols: PythonObject,
            context: PythonObject) raises -> PythonObject:
    var r = Int(py=rows.shape[0])
    var c = Int(py=cols.shape[0])
    var out = admitted_array(Python.tuple(r, c), "<f4", context)
    _ = py_call_typed(_x(xt, "x_trees_gather_f32"), _P(x_addr), _P(trees_address(rows)), _P(trees_address(cols)),
                      _A(out), Python.list(_P(n_src), _P(d_src), _P(r), _P(c)))
    _ = rows
    _ = cols
    _ = out  # keep the owners alive through the native call above
    return out


def _gather_codes(xt: PythonObject, codes: PythonObject, rows: PythonObject, context: PythonObject) raises -> PythonObject:
    var r = Int(py=rows.shape[0])
    var out = admitted_vector(r, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_gather_i32"), _P(trees_address(codes)), _P(trees_address(rows)), _A(out),
                      Python.list(_P(Int(py=codes.shape[0])), _P(r)))
    _ = codes
    _ = rows
    _ = out  # keep the owners alive through the native call above
    return out


def _gather_vec(xt: PythonObject, v_addr: Int, n: Int, rows: PythonObject, context: PythonObject) raises -> PythonObject:
    """A float32 vector gathered at rows (as an n x 1 matrix)."""
    var col = admitted_vector(1, "<i4", context)
    var out = _gather(xt, v_addr, n, 1, rows, col, context)
    return trees_reshape(out, Python.tuple(Int(py=rows.shape[0])), context)


def _acc_cols(xt: PythonObject, acc: PythonObject, x: PythonObject, cols: PythonObject, n: Int, k: Int,
              context: PythonObject) raises:
    var xa = admit_f32(x, 2, "sub-estimator output", context, allow_nonfinite=True)
    _ = py_call_typed(_x(xt, "x_trees_accumulate_cols"), _A(acc), _P(xa.address), _P(trees_address(cols)),
                      Python.list(_P(n), _P(k), _P(Int(py=cols.shape[0])), _F(1.0)))
    _ = cols
    _ = xa  # keep the Python object alive through the native call above
    _ = xa^


def _acc(xt: PythonObject, acc: PythonObject, x: PythonObject, n: Int, context: PythonObject) raises:
    var xa = admit_f32(x, 1, "sub-estimator output", context, allow_nonfinite=True)
    _ = py_call_typed(_x(xt, "x_trees_accumulate"), _A(acc), _P(xa.address), Python.list(_P(n), _F(1.0)))
    _ = xa  # keep the Python object alive through the native call above
    _ = xa^


def _acc_votes(xt: PythonObject, acc: PythonObject, codes: PythonObject, n: Int, k: Int, on: Float64,
               off: Float64) raises:
    _ = py_call_typed(_x(xt, "x_trees_accumulate_onehot"), _A(acc), _A(codes), Python.list(_P(n), _P(k), _F(on), _F(off)))


def _scale(xt: PythonObject, acc: PythonObject, divisor: Float64) raises:
    _ = py_call_typed(_x(xt, "x_trees_scale"), _A(acc), Python.list(_P(Int(py=acc.size)), _F(divisor)))


def _argmax(xt: PythonObject, acc: PythonObject, n: Int, k: Int, context: PythonObject) raises -> PythonObject:
    var out = admitted_vector(n, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_argmax_rows"), _A(acc), _A(out), Python.list(_P(n), _P(k)))
    _ = out  # keep the Python object alive through the native call above
    return out


comptime _SUB_COLS_SLOT = "trees_boost_sub_cols_v1"


def _sub_cols(member: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_sub_cols`: a member's classes (the ensemble's codes) as int32.

    Built once per member classes (p3-trees): every predict rebuilt it per
    member (k `int` calls and a new Array). The record holds the classes
    object, its length and items by identity; any assignment or in-place
    edit of an exact list/tuple `classes_` changes the key and rebuilds.
    Other `classes_` types (an ndarray) are rebuilt on every call as before."""
    var b = trees_builtins()
    var classes = py_fields(member)["classes_"]
    var exact = py_type(classes) is b.list or py_type(classes) is b.tuple
    var key = Python.list(classes)
    if exact:
        _ = py_call(key.append, PythonObject(len(classes)))
        for i in range(len(classes)):  # driver-loop: over the member's k classes (identity record)
            _ = py_call(key.append, classes[i])
        var held = trees_side_get(_SUB_COLS_SLOT, member)
        if held is not None and trees_key_same(held[0], key):
            return held[1]
    var store = py_call(context["array_type"], PythonObject("i"), py_call(b.map, b.int, classes))
    var cols = classical_wrap_array(store, Python.tuple(len(classes)), "<i4", "C", context)
    if exact:
        _ = trees_side_put(_SUB_COLS_SLOT, member, Python.tuple(key, cols))
    return cols


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
    return _clone_from(base, forest_get_params(base, context), seed, context)


def _clone_from(base: PythonObject, template: PythonObject, seed: Int, context: PythonObject) raises -> PythonObject:
    """`_clone` from `template`, the base's `forest_get_params` read once
    before a member loop (lane perf-sweep-trees, 2026-11-05: every round
    re-read the unchanged base's parameters). get_params maps names to the
    stored values by identity, so a shallow copy is the same dictionary."""
    var params = py_call(py_attr(template, "copy"))
    if py_contains(params, "random_state"):
        py_setitem(params, "random_state", _P(seed))
    return _new_member(py_type(base), params, context)


def _member_like(cls: PythonObject, first: PythonObject, member_seed: Int) raises -> PythonObject:
    """A `cls` member holding `first`'s validated state (a clone's fields
    after `forest_refresh` and `forest_capture_mode`) with its own
    `random_state`: the state `_clone_from` / `_new_member` +
    `forest_refresh` + `forest_capture_mode` give it, key for key in the same
    order. Members of one template differ only in `random_state`, and the
    only derived field that reads it is `_cfg`'s `random_state` entry (the RF
    and ET derivations store the int seed; a member seed is always an exact
    nonnegative int), which gets its own mapping (lane ne2-b-members,
    N049 / N061 / N064)."""
    var b = trees_builtins()
    var member = py_call(b.object.__new__, cls)
    var mf = py_fields(member)
    _ = py_call(mf.update, first)
    if py_contains(mf, "random_state"):
        var own_seed = _P(member_seed)
        py_setitem(mf, "random_state", own_seed)
        var cfg = py_get(mf, "_cfg")
        if cfg is not None and py_type(cfg) is b.dict and py_contains(cfg, "random_state"):
            var own = py_call(b.dict, cfg)
            py_setitem(own, "random_state", own_seed)
            py_setitem(mf, "_cfg", own)
    return member


def _validated_clone(base: PythonObject, template: PythonObject, member_seed: Int, kind: String, memo: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`_clone_from(base, template, member_seed)` already through
    `forest_refresh(kind)` and `forest_capture_mode`: the constructor checks
    run for the first member of a fit loop only (`memo`, an empty list the
    loop owns, keeps that member's validated fields); every later member
    copies them with its own seed (`_member_like`). A member fit that
    receives such a member skips its own refresh. Same fields, same values:
    no fit or predict changes."""
    if len(memo) == 0:
        var member = _clone_from(base, template, member_seed, context)
        var fields = forest_refresh(member, kind, context)
        _ = forest_capture_mode(fields, context)
        _ = py_call(memo.append, py_call(trees_builtins().dict, fields))
        return member
    return _member_like(py_type(base), memo[0], member_seed)


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


def _member_codes(member: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`as_i32_c(member.predict(X))` of an AdaBoost member, whose classes are
    the ensemble's codes, without decoding and re-encoding (lane
    perf-trees-owed, 2026-11-06): the member's own class codes (int32, or
    int64 from the vote argmax) mapped through its classes. Classes 0..k-1
    in order (the usual case: every round sees every class) map to
    themselves: the codes are the answer (int64 narrowed once). Other exact
    int classes in int32 range go through one int32 gather. Anything else
    decodes as before. The int32 values are the decoded labels' in every case."""
    var pair = forest_predict_codes(member, Python.tuple(X), context)
    var codes = pair[1]
    var labels = classifier_label_list(pair[0], context)
    var k = len(labels)
    var int_t = trees_builtins().int
    var identity = k > 0
    var table = admitted_vector(max(k, 1), "<i4", context)
    var t = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(table))
    for c in range(k):  # driver-loop: over the member's k classes
        var v = labels[c]
        if py_type(v) is not int_t or Bool(py=v < PythonObject(0)) or Bool(py=v > PythonObject(2147483647)):
            _ = table
            return _i32_codes(trees_decode_labels(pair[0], codes, context), context)
        var code = Int(py=v)
        t[c] = Int32(code)
        identity = identity and code == c
    _ = table  # keep the table alive through the writes above
    var n = Int(py=codes.size)
    var narrow = codes
    if String(codes.dtype) != "<i4":
        narrow = admitted_vector(n, "<i4", context)
        trees_cast(trees_address(codes), String(codes.dtype), trees_address(narrow), "<i4", n, context)
        # `codes` is the only owner of the argmax result here (`pair` ended
        # above): without this use it is released before the cast reads it.
        _ = codes  # keep the Python object alive through the native call above
        _ = narrow  # keep the Python object alive through the native call above
    if identity:
        return narrow
    var out = admitted_vector(n, "<i4", context)
    _ = py_call_typed(trees_helper(context, "gather_i32"), PythonObject(trees_address(table)), PythonObject(k),
                      PythonObject(trees_address(narrow)), PythonObject(n), PythonObject(trees_address(out)))
    _ = table  # keep the Python object alive through the native call above
    _ = narrow  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


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


def _member_session(member: PythonObject, x: AdmittedX, row_major: Bool, x_finite: Bool, default: String,
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
    if not x_finite and not trees_finite_f32(x.address, x.rows * x.cols):
        raise Error(TREES_NONFINITE)
    var share = choice == "share" and mode == "fast"
    var handle = py_call_typed(trees_entry(native, "rf_data_session_open"), _P(x.address),
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


def _member_fit_weighted(member: PythonObject, xcm: AdmittedX, codes: PythonObject, w32: PythonObject,
                         encoded: Optional[TreesLabels], session: Session, context: PythonObject) raises:
    """`DecisionTreeClassifier._fit_weighted(Xcm, codes, w32, x_finite=True,
    encoded=..., session=...)` on a column-major float32 X already scanned.
    `member` comes from `_validated_clone(..., "dt_classifier", ...)`: its
    constructor state is refreshed and its mode captured already."""
    var fields = py_fields(member)
    var mode = String(fields["_fit_numeric_mode"])
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
                                                             _P(weights.address)), "into", context)
        _ = weights  # keep the Python object alive through the native call above
    else:
        var fit = ForestFit("rf_classifier_fit_weighted", True, Python.list(_P(weights.address)), True)
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
    squared-error member fit on the session's X at `rows`. `member` comes
    from `_validated_clone`: its constructor state is refreshed and its mode
    captured already."""
    var kind = forest_kind(member, context)
    var fields = py_fields(member)
    var mode = String(fields["_fit_numeric_mode"])
    var y = admit_f32(y32, 1, "y", context, allow_nonfinite=True)
    var code = fields["_cfg"]["criterion"]
    if Int(py=code) != Int(py=context["trees_rf_tables"]["reg_criteria"]["squared_error"]):
        raise Error("a forest data session fits squared_error members")
    var n_rows = len(rows)
    _y_rows_check(y.rows, n_rows)
    var rf = trees_table(context, mode, "_mojolearn_rf")
    var params = _rf_params(fields, kind, n_rows, x_cols, 0, rf, context)
    var entry = trees_need(session.native, "rf_regressor_fit_session_rows_export",
                           "rebuild the forest binding for forest data sessions")
    var result = forest_export(session.native, py_call_typed(entry, _P(session.handle), _A(rows), _P(y.address), params, code),
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
    var out = admitted_vector(max(1, n), "<i4", context)
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
    var sw: Optional[AdmittedX] = None
    if sample_weight is not None:
        sw = admit_weight_words(sample_weight, n, context)
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
    var template = forest_get_params(base, context) if count > 0 else Python.dict()
    for k in range(count):  # driver-loop: over the member estimators
        var rows = _arange(xt, n, context) if all_rows else _indices(xt, n, n_rows, bootstrap, seed, 2 * k, context)
        var cols = _arange(xt, d, context) if all_cols else _indices(xt, d, n_feat, bootstrap_features, seed, 2 * k + 1, context)
        var xs = xa if (all_rows and all_cols) else _gather(xt, x.address, n, d, rows, cols, context)
        _ = x  # keep the Python object alive through the native call above
        var ys = y_all
        if not all_rows:
            ys = _gather_codes(xt, y_all, rows, context) if classifier else _gather_vec(xt, trees_address(y_all), n, rows, context)
        var member = _clone_from(base, template, _sub_seed(seed, k), context)
        var weights = PythonObject(None)
        if sw:
            # every row: the weights as given (the member's own check reads
            # the same float32 values); else the drawn rows' weights
            weights = sample_weight if all_rows else _gather_vec(xt, sw.value().address, n, rows, context)
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


def _bagging_x(fields: PythonObject, X: PythonObject, context: PythonObject) raises -> AdmittedX:
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
    # The address is read once (it was re-read through the Python attribute
    # for every column).
    var words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(cols))
    for j in range(d):  # driver-loop: over the member's feature subset (columns-sized)
        if Int(words[j]) != j:
            _ = cols  # keep the Python object alive through the reads above
            return False
    _ = cols  # keep the Python object alive through the reads above
    return True


def _sub_x(xt: PythonObject, x: AdmittedX, xa: PythonObject, cols: PythonObject, rows_all: PythonObject,
           context: PythonObject) raises -> PythonObject:
    """`_BaggingBase._sub_X`."""
    if _all_columns(cols, x.cols):
        return xa
    return _gather(xt, x.address, x.rows, x.cols, rows_all, cols, context)


def _members(fields: PythonObject) raises -> PythonObject:
    """The fitted member list (`estimators_`) of the per-member path."""
    return fields["_estimators"]


def _bagging_member_scores(xt: PythonObject, member: PythonObject, xs: PythonObject, m: Int, k: Int,
                           context: PythonObject) raises -> PythonObject:
    """`BaggingClassifier._member_proba`: a member's class scores, dense
    m x k float64 in the ensemble's columns."""
    var out = admitted_vector(m * k, "<f8", context)
    _acc_cols(xt, out, _member_proba(member, xs, context), _sub_cols(member, context), m, k, context)
    return out


def _bagging_oob_outputs(xt: PythonObject, fields: PythonObject, x: AdmittedX, k: Int, classifier: Bool,
                         context: PythonObject) raises -> PythonObject:
    """`_BaggingBase._oob_outputs`: each member's output on its out-of-bag
    rows, added in member order; (sums, counts)."""
    var n = x.rows
    var acc = admitted_vector(n * k, "<f8", context)  # zero-filled: `_class_major([0.0])` without its fill program
    var counts = admitted_vector(n, "<i4", context)
    var members = _members(fields)
    var features = fields["estimators_features_"]
    var oob_rows = fields["_oob_rows"]
    for i in range(len(members)):  # driver-loop: over the member estimators
        var oob = oob_rows[i]
        var m = len(oob)
        if m == 0:
            continue
        var xs = _gather(xt, x.address, n, x.cols, oob, features[i], context)
        var out = PythonObject(None)
        if classifier:
            out = _bagging_member_scores(xt, members[i], xs, m, k, context)
        else:
            var p = _f32_vector(_member_predict(members[i], xs, context), context)
            out = admitted_vector(m, "<f8", context)
            trees_cast(p.address, "<f4", trees_address(out), "<f8", m, context)
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
            _ = labels  # the encoded-label owner outlives its raw-address read
            py_setitem(fields, "oob_score_", _F(Float64(hits) / Float64(n)))
            _ = pred  # keep the Python object alive through the native call above
        return model
    var y32 = admit_f32(args[1], 1, "y", context, allow_nonfinite=True)
    var y_array = _as_array(y32, Python.tuple(y32.rows), context)
    _bagging_fit_bags(model, fields, X, y_array, False, sample_weight, context)
    if _truthy(fields["oob_score"]):
        # `BaggingRegressor._set_oob_score`
        var xt = _xt(fields, context)
        var x = _x_f32(X, context)
        var n = x.rows
        var pair = _bagging_oob_outputs(xt, fields, x, 1, False, context)
        var pred = admitted_vector(n, "<f8", context)
        var words = admitted_vector(4, "<f8", context)
        var flags = admitted_vector(4, "<i4", context)
        _ = py_call_typed(_x(xt, "x_trees_oob_r2"),
                          Python.list(_A(pair[0]), _A(pair[1]), _A(y_array), _A(pred), _A(words), _A(flags)),
                          Python.list(_P(n)))
        # The tuple owns the accumulated predictions and counts. Extracting
        # their integer addresses does not retain either array through the
        # native call; releasing the tuple early corrupts its first rows.
        _ = pair
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
        var out = admitted_array(Python.tuple(n, k), "<f8", context)
        trees_cast(trees_address(proba), "<f4", trees_address(out), "<f8", n * k, context)
        _ = out  # keep the Python object alive through the native call above
        _ = proba  # keep the Python object alive through the native call above
        return out
    var xt = _xt(fields, context)
    var acc = admitted_vector(n * k, "<f8", context)  # zero-filled: `_class_major([0.0])` without its fill program
    var rows_all = _arange(xt, n, context)
    var members = _members(fields)
    var features = fields["estimators_features_"]
    # One device copy of X for the full-width members (trees_staged_input).
    var staged = trees_stage_members(members, trees_address(xa), n, x.cols, context)
    try:
        for i in range(len(members)):  # driver-loop: over the member estimators
            var xs = _sub_x(xt, x, xa, features[i], rows_all, context)
            _acc_cols(xt, acc, _member_proba(members[i], xs, context), _sub_cols(members[i], context), n, k, context)
    finally:
        trees_unstage(staged)
    _scale(xt, acc, Float64(len(members)))
    return trees_reshape(acc, Python.tuple(n, k), context)


def bagging_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _bagging_proba(model, py_fields(model), args[0], context)


def bagging_codes(model: PythonObject, fields: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """BaggingClassifier's predicted class codes (int32), undecoded."""
    var p = _bagging_proba(model, fields, X, context)
    var n = Int(py=p.shape[0])
    var k = Int(py=_int(fields["n_classes_"]))
    return _argmax(_xt(fields, context), p, n, k, context)


def bagging_predict(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    """BaggingClassifier/BaggingRegressor `predict`."""
    var fields = py_fields(model)
    if classifier:
        return trees_decode_labels(fields["classes_"], bagging_codes(model, fields, args[0], context), context)
    var x = _bagging_x(fields, args[0], context)
    var n = x.rows
    var xa = _as_array(x, Python.tuple(n, x.cols), context)
    var batched = py_get(fields, "_batched")
    if batched is not None:
        var p = _f32_vector(_member_predict(batched[0], xa, context), context)
        var out = admitted_vector(n, "<f8", context)
        trees_cast(p.address, "<f4", trees_address(out), "<f8", n, context)
        _ = out  # keep the Python object alive through the native call above
        _ = p  # keep the Python object alive through the native call above
        return out
    var xt = _xt(fields, context)
    var acc = admitted_vector(n, "<f8", context)
    var rows_all = _arange(xt, n, context)
    var members = _members(fields)
    var features = fields["estimators_features_"]
    var staged = trees_stage_members(members, trees_address(xa), n, x.cols, context)
    try:
        for i in range(len(members)):  # driver-loop: over the member estimators
            _acc(xt, acc, _member_predict(members[i], _sub_x(xt, x, xa, features[i], rows_all, context), context), n, context)
    finally:
        trees_unstage(staged)
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
    var sw = admit_weight_words(sample_weight, n, context)
    var out = admitted_vector(n + 1, "<f8", context)
    var flags = admitted_vector(8, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_exact_sum"), _P(sw.address), _A(out), _A(flags), Python.list(_P(n), _P(1)))
    _ = flags  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = sw  # keep the Python object alive through the native call above
    if _i32_word(flags, 0) != 0 or _i32_word(flags, 1) != 0 or _i32_word(flags, 2) != 0 or _i32_word(flags, 3) != 0:
        raise Error("sample_weight must be finite and nonnegative")
    if _i32_word(flags, 4) == 0:
        raise Error("Sample weights must contain at least one non-zero number.")
    var w = admitted_vector(n, "<f8", context)
    trees_cast(trees_address(out) + 8, "<f8", trees_address(w), "<f8", n, context)
    _ = w  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = sw^
    return w


def _w32(xt: PythonObject, w: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_AdaBoostBase._w32`: the boosting weights times n, float32."""
    var out = admitted_vector(n, "<f4", context)
    _ = py_call_typed(_x(xt, "x_trees_scale_to_f32"), _A(w), _A(out), Python.list(_P(n), _F(Float64(n))))
    _ = out  # keep the Python object alive through the native call above
    return out


def _colmajor(xt: PythonObject, x: AdmittedX, context: PythonObject) raises -> AdmittedX:
    """`_trees_colmajor` of a C-order float32 X: the lane's transpose."""
    var out = admitted_array(Python.tuple(x.rows, x.cols), "<f4", context, "F")
    _ = py_call_typed(_x(xt, "x_trees_transpose_f32"), _P(x.address), _A(out), Python.list(_P(x.rows), _P(x.cols)))
    _ = out  # keep the Python object alive through the native call above
    var _keep_ret_out = AdmittedX(out, trees_address(out), x.rows, x.cols, False)
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
        var stats = admitted_vector(4, "<f8", context)
        py_setitem(fields, "estimators_", members)
        py_setitem(fields, "estimator_weights_", alphas)
        py_setitem(fields, "estimator_errors_", errors)
        var codes = labels.codes
        # A best-splitter DecisionTreeClassifier member fits column-major X
        # with weights: X is transposed and scanned ONCE for every member.
        var dt_best = _is_class(base, "trees_dt_classifier", context) and Bool(py=py_fields(base)["splitter"] == PythonObject("best"))
        var xcm: Optional[AdmittedX] = None
        var encoded: Optional[TreesLabels] = None
        var session = Session(PythonObject(None), 0)
        if dt_best:
            var cm = _colmajor(xt, x, context)
            if not trees_finite_f32(cm.address, n * d):
                raise Error(TREES_NONFINITE)
            _ = cm  # keep the Python object alive through the native calls above
            if trees_env("MOJOLEARN_ADABOOST_REENCODE", "") != "1":
                encoded = trees_encode_labels(codes, context)
            session = _member_session(base, cm, False, True, _ada_session_default(fields, xt, context), context)
            xcm = cm^
        var is_dt = Bool(py=py_call(trees_builtins().isinstance, base, context["trees_dt_classifier"]))
        var template = forest_get_params(base, context) if m > 0 else Python.dict()
        # Every round's member predicts the same training X: staged on the
        # device once, after the first member is fitted (trees_stage_rounds).
        var rounds_staged = Python.list()
        # The best-splitter tree members are validated once per fit (lane
        # ne2-b-members, N064): every round re-ran forest_refresh, the full
        # constructor check and field rebuild, on a clone of the same base.
        var memo = Python.list()
        try:
            for it in range(m):  # driver-loop: over the boosting rounds (one member each)
                var member: PythonObject
                if xcm:
                    member = _validated_clone(base, template, _sub_seed(seed, it), "dt_classifier", memo, context)
                    _member_fit_weighted(member, xcm.value(), codes, _w32(xt, w, n, context), encoded, session, context)
                else:
                    member = _clone_from(base, template, _sub_seed(seed, it), context)
                    _member_fit(member, xa, codes, _w32(xt, w, n, context), context)
                if it == 0 and m > 1:
                    rounds_staged = trees_stage_rounds(member, trees_address(xa), n, d, context)
                var pred = _member_codes(member, xa, context)
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
            trees_unstage(rounds_staged)
            session.close()
        _ = xa  # the staged rows stay alive until they are unstaged above
        py_setitem(fields, "n_features_in_", _P(d))
        return model
    var y32 = admit_f32(args[1], 1, "y", context, allow_nonfinite=True)
    _y_rows_check(y32.rows, n)
    var base = fields["estimator"]
    if base is None:
        var settings = Python.dict()
        py_setitem(settings, "max_depth", _P(3))
        base = _new_member(context["trees_dt_regressor"], settings, context)
    _require_tree(base, "AdaBoostRegressor", context)
    var w = _normalized_weights(xt, args[2], n, context)
    var stats = admitted_vector(4, "<f8", context)
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
    var rounds_staged = Python.list()
    # The base's parameters read once and the session members validated once
    # per fit (lane ne2-b-members, N064): every round re-read the unchanged
    # base's parameters (`_clone`) and the session fit re-ran forest_refresh.
    var template = forest_get_params(base, context) if m > 0 else Python.dict()
    var memo = Python.list()
    var base_kind = forest_kind(base, context) if session.handle > 0 else String("")
    try:
        for it in range(m):  # driver-loop: over the boosting rounds (one member each)
            var rows = admitted_vector(n, "<i4", context)
            _ = py_call_typed(_x(xt, "x_trees_weighted_sample"), _A(w), _A(rows), Python.list(_P(n), _P(n), _P(seed), _P(it)))
            _ = rows  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
            var member: PythonObject
            if session.handle > 0:
                member = _validated_clone(base, template, _sub_seed(seed, it), base_kind, memo, context)
            else:
                member = _clone_from(base, template, _sub_seed(seed, it), context)
            var yr = _gather_vec(xt, y32.address, n, rows, context)
            _ = y32  # keep the Python object alive through the native call above
            if session.handle > 0:
                _member_fit_session_rows(member, session, rows, yr, d, context)
                _ = y32  # keep the Python object alive through the native call above
            else:
                _member_fit(member, _gather(xt, x.address, n, d, rows, cols, context), yr, PythonObject(None), context)
                _ = y32  # keep the Python object alive through the native call above
                _ = x  # keep the Python object alive through the native call above
            if it == 0 and m > 1:
                rounds_staged = trees_stage_rounds(member, trees_address(xa), n, d, context)
            var pred = _f32_vector(_member_predict(member, xa, context), context)
            _ = py_call_typed(_x(xt, "x_trees_r2_step"), _A(w), _P(pred.address), _A(y_array), _A(stats),
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
        trees_unstage(rounds_staged)
        session.close()
    _ = xa  # the staged rows stay alive until they are unstaged above
    py_setitem(fields, "n_features_in_", _P(d))
    return model


def _ada_x(fields: PythonObject, X: PythonObject, context: PythonObject) raises -> AdmittedX:
    """`_AdaBoostBase._check_X`."""
    if not py_contains(fields, "estimators_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var x = _x_f32(X, context)
    var fitted = Int(py=_int(fields["n_features_in_"]))
    if x.cols != fitted:
        raise Error("X has " + String(x.cols) + " features, fit saw " + String(fitted))
    return x^


def _ada_decision(fields: PythonObject, x: AdmittedX, xt: PythonObject, context: PythonObject) raises -> PythonObject:
    """`AdaBoostClassifier._decision`: the members' weighted one-hot votes
    over the native fsum of their weights."""
    var n = x.rows
    var k = Int(py=_int(fields["n_classes_"]))
    var xa = _as_array(x, Python.tuple(n, x.cols), context)
    var acc = admitted_vector(n * k, "<f8", context)  # zero-filled: `_class_major([0.0])` without its fill program
    var members = fields["estimators_"]
    var alphas = fields["estimator_weights_"]
    # One device copy of X for every one-tree member (trees_staged_input).
    var staged = trees_stage_members(members, trees_address(xa), n, x.cols, context)
    # The weights as one float64 store, read once (`array('d', alphas)` is
    # the same float conversion `_float` made per member, and the store fsum
    # reads): one conversion per predict instead of one per member plus the
    # fsum's own (lane perf-sweep-trees, 2026-11-05).
    var weights = py_call(context["array_type"], PythonObject("d"), alphas)
    if len(weights) < len(members):
        raise Error("IndexError: list index out of range")
    var wp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=py_call(weights.buffer_info)[0]))
    try:
        for i in range(len(members)):  # driver-loop: over the member estimators
            var pred = _member_codes(members[i], xa, context)
            var a = wp[unsafe_offset=i]
            _acc_votes(xt, acc, pred, n, k, a, -a / Float64(k - 1))
    finally:
        trees_unstage(staged)
    _ = weights  # keep the store alive through the reads above
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
        var d = admitted_vector(n, "<f8", context)
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
        var pairs = admitted_vector(n * 2, "<f8", context)
        _ = py_call_typed(_x(xt, "x_trees_margin2"), _A(acc), _A(pairs), Python.list(_P(n), _P(2)))
        _ = pairs  # keep the Python object alive through the native call above
        _ = acc  # keep the Python object alive through the native call above
        acc = pairs
    else:
        _scale(xt, acc, Float64(k - 1))
    _ = py_call_typed(_x(xt, "x_trees_softmax_rows"), _A(acc), Python.list(_P(n), _P(k)))
    _ = acc  # keep the Python object alive through the native call above
    return trees_reshape(acc, Python.tuple(n, k), context)


def _ada_codes_of(fields: PythonObject, x: AdmittedX, xt: PythonObject, context: PythonObject) raises -> PythonObject:
    """AdaBoostClassifier's predicted class codes (int32), undecoded."""
    var n = x.rows
    var k = Int(py=_int(fields["n_classes_"]))
    var acc = _ada_decision(fields, x, xt, context)
    var codes = PythonObject(None)
    if k == 2:
        codes = admitted_vector(n, "<i4", context)
        _ = py_call_typed(_x(xt, "x_trees_margin2"), _A(acc), _A(codes), Python.list(_P(n), _P(1)))
        _ = codes  # keep the Python object alive through the native call above
        _ = acc  # keep the Python object alive through the native call above
    else:
        codes = _argmax(xt, acc, n, k, context)
    return codes


def ada_codes(model: PythonObject, fields: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """AdaBoostClassifier's predicted class codes for X."""
    var x = _ada_x(fields, X, context)
    return _ada_codes_of(fields, x, _xt(fields, context), context)


def ada_predict(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var x = _ada_x(fields, args[0], context)
    var xt = _xt(fields, context)
    var n = x.rows
    if classifier:
        return trees_decode_labels(fields["classes_"], _ada_codes_of(fields, x, xt, context), context)
    # `AdaBoostRegressor.predict`: the weighted median of the members.
    var members = fields["estimators_"]
    var m = len(members)
    var xa = _as_array(x, Python.tuple(n, x.cols), context)
    var preds = admitted_vector(m * n, "<f4", context)
    var staged = trees_stage_members(members, trees_address(xa), n, x.cols, context)
    try:
        for j in range(m):  # driver-loop: over the member estimators
            var p = _f32_vector(_member_predict(members[j], xa, context), context)
            _ = py_call_typed(_x(xt, "x_trees_put_f32"), _A(preds), _P(p.address), Python.list(_P(j * n), _P(n)))
            _ = p  # keep the Python object alive through the native call above
            _ = preds  # keep the Python object alive through the native call above
    finally:
        trees_unstage(staged)
    var b = trees_builtins()
    # `array('d', ...)` converts each weight with float() itself: no map layer.
    var weights_store = py_call(context["array_type"], PythonObject("d"), fields["estimator_weights_"])
    var weights = classical_wrap_array(weights_store, Python.tuple(m), "<f8", "C", context)
    var out = admitted_vector(n, "<f4", context)
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
    var out = admitted_vector(max(1, n), "<i4", context)
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
                 cols: PythonObject, max_depth: PythonObject, seed: Int, j: Int, memo: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """One DART member: a one-tree RandomForestRegressor fitted by
    `rf_regressor_fit_dart_export` on the round's device X and class-c
    target (`RandomForestRegressor._fit_in_dart`).

    The member's settings are the same for every tree of one fit except its
    random_state, so its constructor checks run once (lane ne2-b-members,
    N061): the first member is built and validated (`forest_refresh`,
    `forest_capture_mode`, the criterion check), `memo` (an empty list the
    fit loop owns) keeps its validated fields, and every later member copies
    them with its own seed (`_member_like`). Same fields, same values."""
    if len(memo) > 0:
        var tree_like = _member_like(context["trees_rf_regressor"], memo[0], _sub_seed(seed, j))
        return _dart_member_fit(tree_like, py_fields(tree_like), String(py_fields(tree_like)["_fit_numeric_mode"]),
                                handle, c, n, d, rows, cols, context)
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
    if Int(py=tf["_cfg"]["criterion"]) != Int(py=context["trees_rf_tables"]["reg_criteria"]["squared_error"]):
        raise Error("a DART member fits squared_error")
    _ = py_call(memo.append, py_call(trees_builtins().dict, tf))
    return _dart_member_fit(tree, tf, mode, handle, c, n, d, rows, cols, context)


def _dart_member_fit(tree: PythonObject, tf: PythonObject, mode: String, handle: PythonObject, c: Int, n: Int, d: Int,
                     rows: PythonObject, cols: PythonObject, context: PythonObject) raises -> PythonObject:
    """The validated DART member's one-tree fit on the round's device X and
    class-c target, its arrays stored on the member."""
    var code = tf["_cfg"]["criterion"]
    var rf = trees_table(context, mode, "_mojolearn_rf")
    var entry = trees_need(rf, "rf_regressor_fit_dart_export", "rebuild the forest binding for DART member fits on the device")
    var m = n if rows is None else len(rows)
    var dc = d if cols is None else len(cols)
    var params = _rf_params(tf, "rf_regressor", m, dc, 0, rf, context)
    var result = forest_export(rf, py_call_typed(entry, _int(handle), _P(c), _P(0) if rows is None else _A(rows),
                                                 _P(0) if cols is None else _A(cols), params, code), "into", context)
    _member_store(tree, forest_fit_arrays(result, context), dc)
    return tree


def _dart_boost(model: PythonObject, fields: PythonObject, x: AdmittedX, y32: AdmittedX, K: Int, kind: Int,
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
        var out = admitted_vector(1, "<f8", context)
        var flags = admitted_vector(8, "<i4", context)
        _ = py_call_typed(_x(xt, "x_trees_exact_sum"), _P(y32.address), _A(out), _A(flags), Python.list(_P(y32.rows), _P(0)))
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
        var cnt = admitted_vector(K, "<i4", context)
        _ = py_call_typed(_x(xt, "x_trees_class_counts"), _P(y32.address), _A(cnt), Python.list(_P(n), _P(K)))
        _ = cnt  # keep the Python object alive through the native call above
        var priors = admitted_vector(K, "<f8", context)
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
            if not trees_finite_f32(x.address, n * d):
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
    var bad = admitted_vector(1, "<i4", context)
    var weights64 = admitted_vector(n_iters, "<f8", context)
    var coefs64 = admitted_vector(n_iters * K, "<f8", context)
    var coef32 = admitted_vector(n_iters * K, "<f4", context)
    var thr64 = admitted_vector(n_iters, "<i8", context)
    var flags = admitted_vector(n_iters, "<i4", context)
    var state = admitted_vector(1, "<f8", context)
    var uniform = 1 if _truthy(fields["uniform_drop"]) else 0
    var xgb = 1 if _truthy(fields["xgboost_dart_mode"]) else 0
    var max_drop = Int(py=_int(fields["max_drop"]))
    var drop_rate = _float(fields["drop_rate"])
    var handle = py_call_typed(_x(xt, "x_trees_dart_open"), _P(x.address), _P(y32.address), _A(inits32),
                               Python.list(_P(n), _P(d), _P(K), _P(kind), _P(n_iters), _P(node_cap)))
    _ = inits32  # keep the Python object alive through the native call above
    var cache_start = -1
    var cache = PythonObject(None)
    var memo = Python.list()
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
                var tree = _dart_member(fields, handle, c, n, d, rows, cols, max_depth, seed, j, memo, context)
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
                var values = admitted_vector(n_nodes, "<f4", context)
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
    # The previous fit's cached address lists hold its trees: drop them now,
    # with its kept device forest.
    trees_side_drop(_DART_SLOT, model)
    trees_resident_forget(model_identity_key(model))
    var x = _x_f32(args[0], context)
    if not classifier:
        var y32 = admit_f32(args[1], 1, "y", context, allow_nonfinite=True)
        _y_rows_check(y32.rows, x.rows)
        return _dart_boost(model, fields, x, y32, 1, 0, context)
    var labels = trees_encode_labels(args[1], context)
    py_setitem(fields, "classes_", labels.classes)
    var K = len(labels.classes)
    if K < 2:
        raise Error("y must hold at least two classes")
    _y_rows_check(labels.n, x.rows)
    var codes = admitted_vector(labels.n, "<f4", context)
    trees_cast(trees_address(labels.codes), "<i4", trees_address(codes), "<f4", labels.n, context)
    _ = codes  # keep the Python object alive through the native call above
    var y32 = admitted_words(codes, trees_address(codes), labels.n, 1)
    _ = codes  # keep the Python object alive through the native call above
    var kind = 1 if K == 2 else 2
    py_setitem(fields, "_KIND", _P(kind))
    var _keep_ret_codes = _dart_boost(model, fields, x, y32, 1 if K == 2 else K, kind, context)
    _ = codes  # keep the Python object alive through the native call above
    return _keep_ret_codes^


comptime _DART_SLOT = "trees_dart_forest_v1"


def _dart_key(model: PythonObject, fields: PythonObject, K: Int) raises -> PythonObject:
    """The validity key of the kept DART forest (N003, ne1-a-trees,
    2026-11-06): K, the model's generation word (bindings/native_model_records:
    a refit, set_params or __setstate__ bumps it), the identities of the
    three per-tree sequences and `init_score_`, and the sequences' lengths.
    Nine words, no per-tree walk: the O(T) identity walk of every tree's
    arrays it replaces cost about 9T Python reads per predict. The per-tree
    arrays the kept forest mirrors are frozen when it is built
    (`_dart_freeze`), so they cannot change in place; a sequence replaced by
    assignment changes its identity. Not noticed (as for any list a family
    does not own): an item assigned INTO an existing fitted list."""
    var trees = fields["trees_"]
    var values = fields["tree_values_"]
    var tree_coefs = fields["tree_coefs_"]
    return Python.list(PythonObject(K), PythonObject(model_generation(model)), trees, values, tree_coefs,
                       fields["init_score_"], PythonObject(len(trees)), PythonObject(len(values)),
                       PythonObject(len(tree_coefs)))


def _dart_freeze(fields: PythonObject, context: PythonObject) raises:
    """The read-only rule (p3nc_fitted_freeze) for every array the kept
    forest mirrors: each tree's node arrays and leaf values. Once per
    fitted model (the first predict)."""
    var trees = fields["trees_"]
    var values = fields["tree_values_"]
    var count = min(len(trees), len(values))
    for i in range(count):  # driver-loop: over the T fitted trees, once per fitted model
        var tf = py_fields(trees[i])
        _ = fitted_freeze(tf["_offsets"], context)
        _ = fitted_freeze(tf["_colid"], context)
        _ = fitted_freeze(tf["_quesval"], context)
        _ = fitted_freeze(tf["_left_child"], context)
        _ = fitted_freeze(values[i], context)


def _dart_forest(model: PythonObject, fields: PythonObject, K: Int) raises -> PythonObject:
    """`(key, [cols, ques, lefts, vals], sizes, coefs, starts)`: the per-tree
    address lists `x_trees_dart_predict` reads, built once per fitted state
    and kept in the native side table (lane perf-trees-owed, 2026-11-06;
    every predict rebuilt them, about 30 Python calls per tree). The key
    holds the arrays, so the cached addresses stay valid; a replaced tree,
    array or sequence item changes the key and the lists are rebuilt. The
    addresses and values are the ones the per-call build produced, so the
    launch and its bits are unchanged."""
    var key = _dart_key(model, fields, K)
    var held = trees_side_get(_DART_SLOT, model)
    if held is not None and trees_key_same(held[0], key):
        return held
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
    var record = Python.list(key, Python.list(cols, ques, lefts, vals), sizes, coefs, starts)
    _ = trees_side_put(_DART_SLOT, model, record)
    return record


def _dart_resident_predict(model: PythonObject, fields: PythonObject, forest: PythonObject, xt: PythonObject,
                           x_addr: Int, n: Int, d: Int, K: Int, score: PythonObject, context: PythonObject) raises:
    """DART's raw score over the device-kept forest (N039 / N041): the
    forest is uploaded once per fitted model (`x_trees_dart_forest_prepare`)
    and kept in the tree resident table under the model's identity and
    generation, the dimensions and the identities of the fitted sequences;
    a predict uploads X only. Same buffers, same launch: no bit change."""
    var prepare = _x(xt, "x_trees_dart_forest_prepare")
    var query = _x(xt, "x_trees_dart_forest_predict")
    var release = _x(xt, "x_trees_dart_forest_release")
    var mirrors = List[PythonObject]()
    mirrors.append(fields["trees_"])
    mirrors.append(fields["tree_values_"])
    mirrors.append(fields["tree_coefs_"])
    mirrors.append(fields["init_score_"])
    var words = List[Int]()
    words.append(K)
    words.append(d)
    words.append(len(forest[2]))
    var model_key = model_identity_key(model)
    var generation = model_generation(model)
    var handle = trees_resident_find(model_key, generation, TREES_RESIDENT_DART, xt, words, mirrors)
    var kept = handle > 0
    if not kept:
        _dart_freeze(fields, context)
        var made = py_call_typed(prepare, forest[1], forest[2], forest[3], forest[4], Python.list(_P(d), _P(K)))
        handle = Int(py=made[0])
        try:
            if trees_resident_watch(model, model_key):
                kept = trees_resident_put(model_key, generation, TREES_RESIDENT_DART, xt, release, handle,
                                          words^, mirrors^, Int(py=made[1]))
        except error:
            _ = py_call(release, PythonObject(handle))
            raise error^
    if kept:
        trees_resident_pin(handle)
    var use = ForestHandle(handle, kept, release)
    try:
        _ = py_call_typed(query, PythonObject(handle), _P(x_addr), _A(score), Python.list(_P(n), _P(d), _P(K)))
    finally:
        forest_handle_done(use)
    _ = score  # keep the Python object alive through the native call above


def _dart_raw(model: PythonObject, fields: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
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
    var score = admitted_vector(K * n, "<f8", context)
    if n == 0:
        return score
    var forest = _dart_forest(model, fields, K)
    var xt = _xt(fields, context)
    if trees_entry(xt, "x_trees_dart_forest_prepare") is not None:
        _dart_resident_predict(model, fields, forest, xt, x.address, n, d, K, score, context)
    else:
        _ = py_call_typed(_x(xt, "x_trees_dart_predict"), _P(x.address), forest[1], forest[2], forest[3],
                          forest[4], _A(score), Python.list(_P(n), _P(d), _P(K)))
    _ = x  # keep the Python object alive through the native call above
    _ = score  # keep the Python object alive through the native call above
    _ = x^
    return score


def _dart_raw_rows(model: PythonObject, fields: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_DARTBase._raw_rows`: (n, K) row-major raw scores."""
    var raw = _dart_raw(model, fields, X, context)
    var K = Int(py=_int(fields["n_classes_"]))
    var n = Int(py=raw.size) // K
    var out = admitted_vector(n * K, "<f8", context)
    _ = py_call_typed(_x(_xt(fields, context), "x_trees_transpose_f64"), _A(raw), _A(out), Python.list(_P(K), _P(n)))
    _ = out  # keep the Python object alive through the native call above
    _ = raw  # keep the Python object alive through the native call above
    return trees_reshape(out, Python.tuple(n, K), context)


def _multiclass(fields: PythonObject) raises -> Bool:
    return Int(py=_int(py_get(fields, "n_classes_", _P(1)))) > 1


def dart_codes(model: PythonObject, fields: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """DARTClassifier's predicted class codes (int32), undecoded."""
    var xt = _xt(fields, context)
    if _multiclass(fields):
        var acc = _dart_raw_rows(model, fields, X, context)
        var n = Int(py=acc.shape[0])
        var K = Int(py=acc.shape[1])
        return _argmax(xt, acc, n, K, context)
    var raw = _dart_raw(model, fields, X, context)
    var n = Int(py=raw.size)
    var codes = admitted_vector(n, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_positive_codes"), _A(raw), _A(codes), Python.list(_P(n)))
    _ = codes  # keep the Python object alive through the native call above
    _ = raw  # keep the Python object alive through the native call above
    return codes


def dart_predict(model: PythonObject, args: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not classifier:
        return _dart_raw(model, fields, args[0], context)
    return trees_decode_labels(fields["classes_"], dart_codes(model, fields, args[0], context), context)


def boost_predict_codes(model: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """A Bagging / AdaBoost / DART classifier's `(classes_, codes)`: the
    prediction before its label decode (N043, ne1-a-trees). `predict`
    decodes exactly these codes."""
    var fields = py_fields(model)
    if kind.startswith("bagging_"):
        return Python.tuple(fields["classes_"], bagging_codes(model, fields, X, context))
    if kind.startswith("ada_"):
        return Python.tuple(fields["classes_"], ada_codes(model, fields, X, context))
    if kind.startswith("dart_"):
        return Python.tuple(fields["classes_"], dart_codes(model, fields, X, context))
    raise Error("unknown tree ensemble classifier " + kind)


def dart_decision_function(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if _multiclass(fields):
        return _dart_raw_rows(model, fields, args[0], context)
    return _dart_raw(model, fields, args[0], context)


def dart_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var xt = _xt(fields, context)
    if _multiclass(fields):
        var acc = _dart_raw_rows(model, fields, args[0], context)
        var n = Int(py=acc.shape[0])
        var K = Int(py=acc.shape[1])
        var flat = trees_reshape(acc, Python.tuple(n * K), context)
        _ = py_call_typed(_x(xt, "x_trees_softmax_rows"), _A(flat), Python.list(_P(n), _P(K)))
        _ = flat  # keep the Python object alive through the native call above
        return trees_reshape(flat, Python.tuple(n, K), context)
    var raw = _dart_raw(model, fields, args[0], context)
    var n = Int(py=raw.size)
    var acc = admitted_vector(2 * n, "<f8", context)
    var zero = admitted_vector(n, "<f8", context)
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


def _rte_fit(model: PythonObject, fields: PythonObject, x: AdmittedX, context: PythonObject) raises:
    """`RandomTreesEmbedding.fit` on a C-order float32 X: uniform targets,
    one ExtraTreesRegressor fit, the leaf numbering."""
    var n = x.rows
    var d = x.cols
    var seed = _seed(fields["random_state"], context)
    var xt = _xt(fields, context)
    var u = admitted_vector(n, "<f8", context)
    _ = py_call_typed(_x(xt, "x_trees_uniform"), _A(u), Python.list(_P(n), _P(seed), _P(0)))
    _ = u  # keep the Python object alive through the native call above
    var yr = admit_f32(u, 1, "y", context, allow_nonfinite=True)
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
    _member_fit(forest, _as_array(x, Python.tuple(n, d), context), yr.array, PythonObject(None), context)
    py_setitem(fields, "forest_", forest)
    var ff = py_fields(forest)
    var offsets = ff["_offsets"]
    var n_trees = Int(py=offsets.size) - 1
    var nn = _i32_word(offsets, n_trees)
    var node_col = admitted_vector(nn, "<i4", context)
    var col = Int(py=py_call_typed(_x(xt, "x_trees_leaf_numbering"), _A(ff["_left_child"]), _A(node_col), Python.list(_P(nn))))
    _ = node_col  # keep the Python object alive through the native call above
    py_setitem(fields, "_tree_base", _copy_i32(offsets, 0, n_trees, context))
    py_setitem(fields, "_node_col", node_col)
    py_setitem(fields, "n_trees_", _P(n_trees))
    py_setitem(fields, "n_output_features_", _P(col))
    _ = node_col  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", _P(d))


def _rte_transform(fields: PythonObject, x: AdmittedX, context: PythonObject) raises -> PythonObject:
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
    var nodes = admitted_vector(n * trees, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_apply"), _A(ff["_offsets"]), _A(ff["_colid"]), _A(ff["_quesval"]),
                      _A(ff["_left_child"]), _P(x.address), _A(nodes), Python.list(_P(n), _P(d), _P(0), _P(trees)))
    _ = nodes  # keep the Python object alive through the native call above
    var out = admitted_vector(n * width, "<f8", context)
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
    return boost_operation_of(model, boost_kind(model, context), operation, args, context)


def boost_operation_of(model: PythonObject, kind: String, operation: String, args: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`boost_operation` with the schema `kind` its caller already resolved
    (the public door reads it once; lane perf-sweep-trees, 2026-11-05)."""
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
