# SPDX-License-Identifier: Apache-2.0
# Cross-validation of python/mojolearn/model_selection.py, natively (lane
# mcb2-msel): the estimator clone (`_clone`), the fold drawing and checks
# (`_folds`, `_prepare_folds`, `_cv_folds`, `_indices`, `_overlap`), the
# resident fold rows (`_FoldRows`: X, y and every fold's indices put ONCE
# into the `_Resident` store and each fold's rows one device gather out of
# it, or `_arena_io.DeviceRows` for an estimator that takes them), and
# `cross_val_score`, `cross_validate`, `cross_val_predict` and
# `split_descriptor`. The loops over folds are this owner's; the estimator,
# its fit / score / predict and a scorer are the caller's objects, called
# as the Python called them. Same helpers, same arguments, same order: the
# folds, the gathered rows and every score are unchanged.
from std.memory import memcpy
from std.time import perf_counter_ns
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.py_attr import py_attr
from bindings.native_buffer_owner import native_storage_allocate
from bindings.msel_estimator_native import msel_native_clone, msel_clone_parameter, msel_estimator_call, msel_estimator_plan
from bindings.family_common import (
    family_array, family_call, family_call_kw, family_isinstance, family_is_str, family_repr, family_str, family_type_name,
)
from bindings.msel_common import (
    msel_b, msel_n_samples, msel_admit, msel_entry, msel_try, msel_missed, msel_catch, msel_is_bool, msel_is_integral, msel_is_real, msel_int,
    msel_op, msel_cmp, msel_materialize, msel_try_materialize, msel_flatten, msel_is_array, msel_addr, msel_p64,
    msel_new, msel_f64_list, msel_index_i64, msel_resident_new, msel_resident_op, msel_resident_take,
    msel_resident_release, msel_resident_close, msel_get_scorer, msel_score, msel_lazy,
)
from bindings.msel_split import msel_split, msel_check_cv, msel_default_folds, msel_truth, msel_cls, msel_sabotage

#: model_selection's CV_FAST bits of the resample binding's define mask.
comptime MSEL_CV_SLICE_BIT = 32
comptime MSEL_CV_TRUST_FOLDS_BIT = 64


# ---------------------------------------------------------------- the estimator


def msel_clone(value: PythonObject, parameter: Bool, context: PythonObject) raises -> PythonObject:
    return msel_clone_parameter(value, context) if parameter else msel_native_clone(value, context)


def msel_classifier(estimator: PythonObject, pinned: Bool, context: PythonObject) raises -> Bool:
    return False if estimator is None else Bool(py=msel_estimator_plan(estimator, context)["classifier"])


def msel_fit(estimator: PythonObject, x: PythonObject, y: PythonObject, context: PythonObject) raises:
    _ = msel_estimator_call(estimator, "fit", Python.tuple(x, y), context)


def msel_seconds() -> Float64:
    """`time.perf_counter()`."""
    return Float64(perf_counter_ns()) / 1e9


# ---------------------------------------------------------------- folds


def msel_indices(value: PythonObject, n: Int, name: String, context: PythonObject) raises -> PythonObject:
    """`_indices(value, n, name)`: a nonempty 1-D integer index Array as
    int64, checked in range and duplicate free by `check_indices_i64`."""
    var v = msel_materialize(value, name, context)
    var dtype = String(py=v.dtype)
    var kind = 0
    if dtype.byte_length() >= 2:
        kind = Int(dtype.as_bytes()[1])
    if Int(py=v.ndim) != 1 or (kind != ord("i") and kind != ord("u")) or Int(py=v.size) == 0:
        raise Error(name + " must be a nonempty 1-D integer index array")
    var as_i64 = msel_index_i64(v, context)
    var status = msel_int(family_call(msel_entry("check_indices_i64", context), PythonObject(msel_addr(as_i64)),
                                      as_i64.size, PythonObject(n)))
    _ = as_i64  # keep the Python object alive through the native call above
    if status == 1:
        raise Error(name + " contains an out-of-range index")
    _ = as_i64  # keep the Python object alive through the native calls above
    if status == 2:
        raise Error(name + " contains duplicate indices")
    _ = as_i64  # keep the Python object alive through the native calls above
    return as_i64


def msel_overlap(train: PythonObject, test: PythonObject, n: Int, context: PythonObject) raises -> Bool:
    """`_overlap`: whether two accepted index Arrays share a row."""
    if Int(py=train.size) == 0 or Int(py=test.size) == 0 or n <= 0:
        return False
    return msel_truth(family_call(msel_entry("indices_overlap_i64", context), PythonObject(msel_addr(train)), train.size,
                                  PythonObject(msel_addr(test)), test.size, PythonObject(n)))


def msel_splits_of(splitter: PythonObject, x: PythonObject, y: PythonObject, groups: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`splitter.split(X, y, groups)`: a mojolearn splitter's splits natively
    (its `split` is the shell over the same call), anything else the
    caller's own `split`."""
    if py_type(py_type(splitter)) is msel_b().type and py_contains(context["family_msel_splitter_layouts"], py_type(splitter)):
        return msel_split(splitter, x, y, groups, context)
    raise Error("TypeError: Python splitter callbacks are not accepted; use a native splitter or explicit index pairs")


def msel_checked_folds(pairs: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """Every (train, test) of `pairs` checked by `_indices` and `_overlap`
    before the first fit; refused when there is none."""
    var b = msel_b()
    var it = family_call(b.iter, pairs)
    var folds = Python.list()
    while True:  # driver-loop: over folds
        var item = msel_try(b.next, Python.tuple(it), b.StopIteration, context)
        if msel_missed(item, context):
            break
        if py_type(item) is not b.list and py_type(item) is not b.tuple:
            raise Error("TypeError: each native fold must be an explicit list or tuple of index buffers")
        var pair = family_call(b.tuple, item)
        if len(pair) != 2:
            raise Error(String("too many values to unpack (expected 2)") if len(pair) > 2
                        else String("not enough values to unpack (expected 2, got ") + String(len(pair)) + ")")
        var train = msel_indices(pair[0], n, "train", context)
        var test = msel_indices(pair[1], n, "test", context)
        if msel_overlap(train, test, n, context):
            raise Error("train and test indices overlap")
        _ = py_call(folds.append, Python.tuple(train, test))
    if len(folds) == 0:
        raise Error("cv must produce at least one fold")
    return folds


def msel_materialize_y(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_materialize(y, 'y')[0]`, or `flatten_labels(y)` when that refuses
    with TypeError."""
    var got = msel_try_materialize(y, "y", msel_b().TypeError, context)
    if msel_missed(got, context):
        return msel_flatten(y, context)
    return got


def msel_cv_folds(estimator: PythonObject, x_in: PythonObject, y_in: PythonObject, cv: PythonObject,
                  groups: PythonObject, pinned: Bool, context: PythonObject) raises -> PythonObject:
    """`_cv_folds`: (X, y, folds), every fold validated before any fit."""
    var x = msel_materialize(x_in, "X", context)
    var y = PythonObject(None) if y_in is None else msel_materialize_y(y_in, context)
    var splitter = msel_check_cv(cv, y, msel_classifier(estimator, pinned, context), PythonObject(False), PythonObject(None), context)
    var n = msel_n_samples(x, context)
    var folds = msel_checked_folds(msel_splits_of(splitter, x, y, groups, context), n, context)
    return Python.tuple(x, y, folds)


def msel_fold_source(cv: PythonObject, estimator: PythonObject, x: PythonObject, y: PythonObject,
                     groups: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_folds(cv, estimator, X, y, groups)`: the default unshuffled folds
    for None or an integer, a splitter's split, or the caller's iterable."""
    var b = msel_b()
    if cv is None or msel_is_integral(cv, context):
        if groups is not None:
            raise Error("mojolearn cross_val_score: groups is read only by a splitter "
                        + "passed as cv; the default unshuffled folds would ignore it, "
                        + "so it is refused (pass cv=<splitter with .split(X, y, groups)>)")
        return msel_default_folds(y, PythonObject(5) if cv is None else cv, msel_classifier(estimator, False, context), context)
    if py_type(py_type(cv)) is b.type and py_contains(context["family_msel_splitter_layouts"], py_type(cv)):
        return msel_splits_of(cv, x, y, groups, context)
    if py_type(cv) is b.list or py_type(cv) is b.tuple:
        return cv
    raise Error("TypeError: Python fold iterator callbacks are not accepted; provide native splitters or explicit list/tuple index pairs")


def msel_fast_on(bit: Int, context: PythonObject) raises -> Bool:
    """`_cv_fast_on(bit)`: the resample binding's FAST define mask (0 off
    the FAST tier)."""
    var mode = context["family_msel_backend_fields"]["_DEFAULT_MODE"]
    var mask = Int(py=context["family_msel_fast_masks"][mode])
    return (mask & bit) != 0


def msel_native_default_cv(cv: PythonObject, groups: PythonObject, context: PythonObject) raises -> Bool:
    """`_native_default_cv`."""
    return (groups is None and (cv is None or msel_is_integral(cv, context))
            and not msel_is_bool(cv, context) and not msel_sabotage())


def msel_prepare_folds(estimator: PythonObject, x_in: PythonObject, y_in: PythonObject, cv: PythonObject,
                       scoring_in: PythonObject, groups: PythonObject, error_score: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`_prepare_folds`: (X, y, folds, scoring), every fold validated
    before the first fit."""
    var b = msel_b()
    if not family_is_str(error_score) or not msel_cmp("eq", error_score, PythonObject("raise")):
        raise Error("NotImplementedError: cross_val_score supports error_score='raise' only")
    var scoring = msel_get_scorer(scoring_in, context)
    var x = msel_materialize(x_in, "X", context)
    var y = msel_materialize_y(y_in, context)
    if msel_is_array(y, context) and Int(py=y.ndim) != 1:
        raise Error("y must be 1-D and match X rows")
    if Int(py=x.ndim) != 2 or Int(py=x.shape[0]) == 0 or Int(py=x.shape[1]) == 0:
        raise Error("X must be a nonempty dense 2-D buffer array")
    var n = msel_n_samples(x, context)
    if msel_n_samples(y, context) != n:
        raise Error("y must be a 1-D buffer array matching X rows")
    if groups is not None:
        var group_values = msel_materialize_y(groups, context)
        if (msel_is_array(group_values, context) and Int(py=group_values.ndim) != 1) or msel_n_samples(group_values, context) != n:
            raise Error("groups must be 1-D and match X rows")
    if msel_fast_on(MSEL_CV_TRUST_FOLDS_BIT, context) and msel_native_default_cv(cv, groups, context):
        var trusted = msel_default_folds(y, PythonObject(5) if cv is None else cv, msel_classifier(estimator, False, context), context)
        return Python.tuple(x, y, trusted, scoring)
    var folds = msel_checked_folds(msel_fold_source(cv, estimator, x, y, groups, context), n, context)
    return Python.tuple(x, y, folds, scoring)


# ---------------------------------------------------------------- fold rows


def msel_has_c(array: PythonObject) raises -> Bool:
    """`Array._has_order('C')`: C order, or at most one non-unit axis."""
    if String(py=array.order) == "C":
        return True
    var big = 0
    for k in range(Int(py=array.ndim)):  # driver-loop: over array rank metadata
        if Int(py=array.shape[k]) != 1:
            big += 1
    return big <= 1


def msel_device_rows_ok(estimator: PythonObject, scoring_in: PythonObject, context: PythonObject) raises -> Bool:
    if not Bool(py=msel_estimator_plan(estimator, context)["device_rows"]):
        return False
    var multi = msel_scorers(scoring_in, context)
    if multi is None:
        _ = msel_get_scorer(scoring_in, context)
    return True


def _arena_rows_call(obj: PythonObject, op: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """One `_arena_io` DeviceRows / resident-scope operation through its
    compiled door, as the Python methods call it."""
    var io = context["family_msel_arena_io"]
    return family_call(family_call(msel_b().getattr, io, PythonObject("_ROWS_CALL")), obj, PythonObject(op), args,
                       family_call(msel_b().getattr, io, PythonObject("_ROWS_CONTEXT")))


struct MselFoldRows(Movable):
    """`_FoldRows`: each fold's rows of X (and y) for the fits that share
    the folds, out of X, y and the fold indices resident on the device."""
    var x: PythonObject
    var y: PythonObject
    var folds: PythonObject
    var res: PythonObject
    var extra: PythonObject
    var dev: Bool
    var scope: PythonObject

    def __init__(out self, x: PythonObject, y: PythonObject, folds: PythonObject, device_rows: Bool,
                 context: PythonObject) raises:
        self.x = x
        self.y = y
        self.folds = folds
        self.res = msel_resident_new(context)
        self.extra = PythonObject(None)
        self.dev = (device_rows and msel_is_array(x, context) and String(py=x.dtype) == "<f4" and Int(py=x.ndim) == 2
                    and Int(py=x.size) > 0 and msel_has_c(x))
        self.scope = PythonObject(None)
        if self.dev:
            # only DeviceRows bases enter this scope (`rows_slot` ignores the
            # size floor); no other program input is kept across folds
            var scope = native_storage_allocate(family_call(msel_b().getattr, context["family_msel_arena_io"],
                                                            PythonObject("resident")))
            py_setattr(scope, "min_words", PythonObject(1 << 62))
            py_setattr(scope, "_scope", PythonObject(None))
            _ = _arena_rows_call(scope, "scope_enter", Python.tuple(), context)
            self.scope = scope

    def xrows(self, idx_in: PythonObject, context: PythonObject) raises -> PythonObject:
        if not self.dev:
            return self.rows(self.x, idx_in, context)
        var idx = msel_index_i64(idx_in, context)
        var dr = native_storage_allocate(family_call(msel_b().getattr, context["family_msel_arena_io"],
                                                     PythonObject("DeviceRows")))
        var take = family_call(msel_b().getattr, self.res, PythonObject("take"))
        _ = _arena_rows_call(dr, "init", Python.tuple(self.x, idx, take), context)
        return dr

    def rows(self, values: PythonObject, idx: PythonObject, context: PythonObject) raises -> PythonObject:
        var b = msel_b()
        if family_isinstance(values, b.list):
            return py_call(b.list, py_call(b.map, py_call(b.getattr, values, PythonObject("__getitem__")), idx._mv))
        return msel_resident_take(self.res, values, idx, context)

    def take(mut self, i: Int, y_in: PythonObject, context: PythonObject) raises -> PythonObject:
        """(X_train, y_train, X_test, y_test) of fold i; a `y_in` (a permuted
        y) stands in for the stored one."""
        var pair = self.folds[i]
        var train = pair[0]
        var test = pair[1]
        var xtr = self.xrows(train, context)
        var xte = self.xrows(test, context)
        var y = y_in
        if y is None:
            y = self.y
        elif y is not self.y and y is not self.extra:
            if self.extra is not None:
                msel_resident_release(self.res, self.extra, context)
            self.extra = y
        if y is None:
            return Python.tuple(xtr, PythonObject(None), xte, PythonObject(None))
        return Python.tuple(xtr, self.rows(y, train, context), xte, self.rows(y, test, context))

    def close(mut self, context: PythonObject) raises:
        self.extra = PythonObject(None)
        var scope = self.scope
        self.scope = PythonObject(None)
        try:
            if scope is not None:
                _ = _arena_rows_call(scope, "scope_exit", Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None)), context)
        finally:
            msel_resident_close(self.res, context)


# ---------------------------------------------------------------- cross-validation


def msel_scorers(scoring: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = msel_b()
    var kind = py_type(scoring)
    if kind is b.list or kind is b.tuple or kind is b.set:
        var out = Python.dict()
        var items = py_call(b.list, scoring)
        for i in range(len(items)):  # driver-loop: scorer names, never caller protocols
            if py_type(items[i]) is not b.str:
                raise Error("TypeError: multiple scoring names must be exact strings")
            py_setitem(out, items[i], msel_get_scorer(items[i], context))
        return out
    var mappings = context["mapping_types"]
    var mapping = False
    for i in range(len(mappings)):  # driver-loop: fixed native dict/OrderedDict/defaultdict types
        mapping = mapping or kind is mappings[i]
    if mapping:
        var out = Python.dict()
        var items = py_call(b.list, py_call(scoring.items))
        for i in range(len(items)):  # driver-loop: named native scorer records
            if py_type(items[i][0]) is not b.str:
                raise Error("TypeError: multiple scoring keys must be exact strings")
            py_setitem(out, items[i][0], msel_get_scorer(items[i][1], context))
        return out
    return PythonObject(None)


struct MselScores(Movable):
    """The per-fold results of one cross-validation (`_cross_validate_loop`'s
    `out`, `ests` and `idx`)."""
    var names: List[String]
    var keys: PythonObject
    var test: List[List[Float64]]
    var train: List[List[Float64]]
    var fit_time: List[Float64]
    var score_time: List[Float64]
    var estimators: PythonObject
    var train_idx: PythonObject
    var test_idx: PythonObject

    def __init__(out self, keys: PythonObject) raises:
        var names = List[String]()
        var test = List[List[Float64]]()
        var train = List[List[Float64]]()
        for i in range(len(keys)):  # driver-loop: over scorer names
            names.append(String(py=family_call(msel_b().str, keys[i])))
            test.append(List[Float64]())
            train.append(List[Float64]())
        self.keys = keys
        self.names = names^
        self.test = test^
        self.train = train^
        self.fit_time = List[Float64]()
        self.score_time = List[Float64]()
        self.estimators = Python.list()
        self.train_idx = Python.list()
        self.test_idx = Python.list()


def msel_error_value(error_score: PythonObject) raises -> Float64:
    return Float64(py=family_call(msel_b().float, error_score))


def msel_cross_validate_run(estimator: PythonObject, y: PythonObject, mut rows: MselFoldRows, scoring: PythonObject,
                            return_train: Bool, return_estimator: Bool, return_indices: Bool,
                            error_score: PythonObject, context: PythonObject) raises -> MselScores:
    """`_cross_validate_folds` on the folds of `rows`: one clone fit and
    scored per fold. `estimator` is cloned per fold (a search candidate:
    the configured estimator, `_Pinned`'s clone)."""
    var b = msel_b()
    var multi = msel_scorers(scoring, context)
    var single = PythonObject(None)
    var keys = Python.list()
    if multi is not None:
        keys = py_call(b.list, multi)
    else:
        single = msel_get_scorer(scoring, context)
        _ = py_call(keys.append, PythonObject("score"))
    var out = MselScores(keys)
    var raise_errors = family_is_str(error_score)
    var folds = rows.folds
    for i in range(len(folds)):  # driver-loop: over folds
        var est = msel_clone(estimator, False, context)
        var t0 = msel_seconds()
        var got = rows.take(i, PythonObject(None) if y is rows.y else y, context)
        var xtr = got[0]
        var ytr = got[1]
        var xte = got[2]
        var yte = got[3]
        var ok = True
        if raise_errors:
            msel_fit(est, xtr, ytr, context)
        else:
            try:
                msel_fit(est, xtr, ytr, context)
            except error:
                if not msel_catch(error, b.Exception):
                    raise error^
                ok = False
        var t1 = msel_seconds()
        for k in range(len(keys)):  # driver-loop: over scorer names
            var sc = multi[keys[k]] if multi is not None else single
            out.test[k].append(Float64(py=msel_score(est, xte, yte, sc, context)) if ok else msel_error_value(error_score))
            if return_train:
                out.train[k].append(Float64(py=msel_score(est, xtr, ytr, sc, context)) if ok else msel_error_value(error_score))
        out.fit_time.append(t1 - t0)
        out.score_time.append(msel_seconds() - t1)
        if return_estimator:
            _ = py_call(out.estimators.append, est)
        if return_indices:
            _ = py_call(out.train_idx.append, folds[i][0])
            _ = py_call(out.test_idx.append, folds[i][1])
    return out^


def msel_scores_dict(scores: MselScores, return_train: Bool, return_estimator: Bool, return_indices: Bool,
                     context: PythonObject) raises -> PythonObject:
    """The `cross_validate` result dict, keys in the Python's order."""
    var res = Python.dict()
    py_setitem(res, "fit_time", msel_f64_list(scores.fit_time, context))
    py_setitem(res, "score_time", msel_f64_list(scores.score_time, context))
    for k in range(len(scores.names)):  # driver-loop: over scorer names
        py_setitem(res, PythonObject("test_" + scores.names[k]), msel_f64_list(scores.test[k], context))
        if return_train:
            py_setitem(res, PythonObject("train_" + scores.names[k]), msel_f64_list(scores.train[k], context))
    if return_estimator:
        py_setitem(res, "estimator", scores.estimators)
    if return_indices:
        var idx = Python.dict()
        py_setitem(idx, "train", scores.train_idx)
        py_setitem(idx, "test", scores.test_idx)
        py_setitem(res, "indices", idx)
    return res


def msel_require_serial(n_jobs: PythonObject, error_score: PythonObject, caller: String,
                        context: PythonObject) raises:
    """`_require_serial`."""
    if not py_contains(Python.tuple(PythonObject(None), PythonObject(1)), n_jobs):
        raise Error("NotImplementedError: " + caller + " supports n_jobs=1 (or None) only")
    if not (family_is_str(error_score) and msel_cmp("eq", error_score, PythonObject("raise"))) and not msel_is_real(error_score, context):
        raise Error("error_score must be 'raise' or a number")


def msel_cross_validate(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cross_validate(estimator, X, y, groups, scoring, cv, n_jobs,
    return_train_score, return_estimator, return_indices, error_score)`."""
    var estimator = args[0]
    var error_score = args[10]
    msel_require_serial(args[6], error_score, "cross_validate", context)
    var got = msel_cv_folds(estimator, args[1], args[2], args[5], args[3], False, context)
    var x = got[0]
    var y = got[1]
    var folds = got[2]
    var scoring = args[4]
    var rt = msel_truth(args[7])
    var re = msel_truth(args[8])
    var ri = msel_truth(args[9])
    var rows = MselFoldRows(x, y, folds, msel_device_rows_ok(estimator, scoring, context), context)
    var result = PythonObject(None)
    try:
        var scores = msel_cross_validate_run(estimator, y, rows, scoring, rt, re, ri, error_score, context)
        result = msel_scores_dict(scores, rt, re, ri, context)
    finally:
        rows.close(context)
    return result


# ---------------------------------------------------------------- cross_val_score


def _row_view(arr: PythonObject, a: Int, b: Int, context: PythonObject) raises -> PythonObject:
    """`_row_range_view(arr, a, b)`: rows [a, b) of a C-order Array as a
    read-only view over the same memory (`arr` kept alive by `_base`)."""
    var n = Int(py=arr.shape[0])
    if a == 0 and b == n:
        return arr
    var bi = msel_b()
    var per = Int(py=arr.size) // n
    var item = Int(py=arr.itemsize)
    var view = native_storage_allocate(py_type(arr))
    py_setattr(view, "_store", arr._store)
    py_setattr(view, "_base", arr)
    py_setattr(view, "_pin", arr._pin)
    py_setattr(view, "_mv", py_call(py_attr(arr._mv, "__getitem__"), py_call(bi.slice, PythonObject(a * per), PythonObject(b * per))))
    py_setattr(view, "_addr", PythonObject(Int(py=arr._addr) + a * per * item))
    py_setattr(view, "_readonly", PythonObject(True))
    var shape = Python.list()
    _ = py_call(shape.append, PythonObject(b - a))
    var strides = Python.list()
    var rank = Int(py=arr.ndim)
    for k in range(1, rank):  # driver-loop: over array rank metadata
        _ = py_call(shape.append, arr.shape[k])
    var step = item
    var steps = List[Int](length=rank, fill=0)
    for k in range(rank):  # driver-loop: over array rank metadata
        var axis = rank - 1 - k
        steps[axis] = step
        step *= Int(py=shape[axis])
    for k in range(rank):  # driver-loop: over array rank metadata
        _ = py_call(strides.append, PythonObject(steps[k]))
    py_setattr(view, "shape", py_call(bi.tuple, shape))
    py_setattr(view, "dtype", arr.dtype)
    py_setattr(view, "ndim", PythonObject(rank))
    py_setattr(view, "size", PythonObject((b - a) * per))
    py_setattr(view, "itemsize", PythonObject(item))
    py_setattr(view, "nbytes", PythonObject((b - a) * per * item))
    py_setattr(view, "order", PythonObject("C"))
    py_setattr(view, "strides", py_call(bi.tuple, strides))
    return view


def _rows_outside(arr: PythonObject, a: Int, b: Int, context: PythonObject) raises -> PythonObject:
    """`_rows_outside`: rows [0, a) then [b, n), a view when one side is
    empty, else one new Array filled by two memcpys."""
    var n = Int(py=arr.shape[0])
    if a == 0:
        return _row_view(arr, b, n, context)
    if b == n:
        return _row_view(arr, 0, a, context)
    var bi = msel_b()
    var per = Int(py=arr.size) // n
    var item = Int(py=arr.itemsize)
    var shape = py_call(bi.list, arr.shape)
    py_setitem(shape, PythonObject(0), PythonObject(n - (b - a)))
    var out = family_array(py_call(bi.tuple, shape), String(py=arr.dtype), context)
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=arr._addr))
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    _ = out  # keep the Python object alive through the native call above
    memcpy(dest=dst, src=src, count=a * per * item)
    _ = out  # keep the Python object alive through the native call above
    memcpy(dest=dst + a * per * item, src=src + b * per * item, count=(n - b) * per * item)
    _ = out  # keep the Python object alive through the native call above
    return out


def _kfold_ranges(x: PythonObject, y: PythonObject, folds: PythonObject, context: PythonObject) raises -> List[Int]:
    """`_kfold_ranges`: each fold's test block as (a, b) pairs, or an empty
    list unless every fold is a contiguous block."""
    var none = List[Int]()
    var n = msel_n_samples(x, context)
    if (not msel_is_array(y, context) or String(py=x.order) != "C" or String(py=y.order) != "C" or n < 2):
        return none^
    var ranges = List[Int]()
    for i in range(len(folds)):  # driver-loop: over folds (two endpoints each)
        var train = folds[i][0]
        var test = folds[i][1]
        var size = Int(py=test.size)
        if size == 0 or Int(py=train.size) == 0:
            return List[Int]()
        var p = msel_p64(test)
        _ = test  # keep the Python object alive through the native call above
        var a = Int(p[0])
        _ = test  # keep the Python object alive through the native call above
        var b = Int(p[size - 1]) + 1
        _ = test  # keep the Python object alive through the native call above
        if b - a != size or Int(py=train.size) != n - size:
            return List[Int]()
        ranges.append(a)
        ranges.append(b)
    return ranges^


def msel_fit_score_fold(fitted: PythonObject, xtr: PythonObject, ytr: PythonObject, xte: PythonObject,
                        yte: PythonObject, scoring: PythonObject, context: PythonObject) raises -> Float64:
    """`_fit_score_fold`."""
    msel_fit(fitted, xtr, ytr, context)
    return Float64(py=msel_score(fitted, xte, yte, scoring, context))


def msel_cross_val_score(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cross_val_score(estimator, X, y, cv, scoring, groups, n_jobs, error_score)`."""
    var estimator = args[0]
    var cv = args[3]
    var groups = args[5]
    var n_jobs = args[6]
    if msel_is_bool(n_jobs, context) or not msel_is_integral(n_jobs, context) or msel_cmp("ne", n_jobs, PythonObject(1)):
        raise Error("NotImplementedError: cross_val_score supports n_jobs=1 only")
    var scoring = args[4]
    if family_is_str(scoring):
        scoring = msel_get_scorer(scoring, context)
    var got = msel_prepare_folds(estimator, args[1], args[2], cv, scoring, groups, args[7], context)
    var x = got[0]
    var y = got[1]
    var folds = got[2]
    scoring = got[3]
    var ranges = List[Int]()
    var sliced = False
    if msel_fast_on(MSEL_CV_SLICE_BIT, context) and msel_native_default_cv(cv, groups, context):
        ranges = _kfold_ranges(x, y, folds, context)
        sliced = len(ranges) > 0
    var scores = List[Float64]()
    var rows = MselFoldRows(x, y, folds, msel_device_rows_ok(estimator, scoring, context), context)
    try:
        for i in range(len(folds)):  # driver-loop: over folds
            var fitted = msel_clone(estimator, False, context)
            if sliced:
                var a = ranges[2 * i]
                var b = ranges[2 * i + 1]
                scores.append(msel_fit_score_fold(fitted, _rows_outside(x, a, b, context), _rows_outside(y, a, b, context),
                                                  _row_view(x, a, b, context), _row_view(y, a, b, context), scoring, context))
            else:
                var fr = rows.take(i, PythonObject(None), context)
                scores.append(msel_fit_score_fold(fitted, fr[0], fr[1], fr[2], fr[3], scoring, context))
            # each fold's estimator is released before the next is built
            _ = fitted^
    finally:
        rows.close(context)
    return msel_f64_list(scores, context)


# ---------------------------------------------------------------- cross_val_predict


def _numeric_dtype(dtype: String) -> Bool:
    return dtype == "<f4" or dtype == "<f8" or dtype == "<i4" or dtype == "<i8" or dtype == "<u4" or dtype == "<u1"


def _numeric_pred(pred: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_numeric_pred`: a numeric buffer or list as an Array, else `pred`."""
    if msel_is_array(pred, context):
        return pred
    var b = msel_b()
    var a = msel_try_materialize(pred, "predictions", Python.tuple(b.TypeError, b.ValueError, b.OverflowError), context)
    if msel_missed(a, context):
        return pred
    return a if _numeric_dtype(String(py=a.dtype)) else pred


def _place(preds: PythonObject, tests: PythonObject, n: Int, res: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_cross_val_predict_place`: each fold's predictions put back in row
    order on the device (`msel_scatter_rows` into one resident block), or
    the label-object route."""
    var b = msel_b()
    var count = len(preds)
    var numeric = True
    for f in range(count):  # driver-loop: over folds (dtype and shape checks)
        var p = preds[f]
        if not (msel_is_array(p, context) and _numeric_dtype(String(py=p.dtype)) and Int(py=p.ndim) >= 1
                and Int(py=p.shape[0]) == Int(py=tests[f].size)):
            numeric = False
    var width = Python.tuple()
    if numeric:
        width = py_call(b.tuple, py_call(py_attr(preds[count - 1].shape, "__getitem__"), py_call(b.slice, PythonObject(1), PythonObject(None))))
    var same = numeric
    var integral = numeric and len(width) == 0
    if numeric:
        for f in range(count):  # driver-loop: over folds (shape checks)
            var p = preds[f]
            if not msel_cmp("eq", py_call(b.tuple, py_call(py_attr(p.shape, "__getitem__"), py_call(b.slice, PythonObject(1), PythonObject(None)))), width):
                same = False
            var d = String(py=p.dtype)
            if not (d.startswith("<i") or d.startswith("<u")):
                integral = False
    var codes = context["family_msel_codes"]
    if same:
        var dtype = String("<i8") if integral else String("<f8")
        var shape = family_call(msel_op("add"), Python.tuple(n), width)
        var out = family_array(shape, dtype, context)
        var cols = 1
        for k in range(len(width)):  # driver-loop: over prediction shape metadata
            cols *= Int(py=width[k])
        if cols > 0 and n > 0:
            var block = msel_int(family_call(msel_entry("msel_alloc", context), PythonObject(n * cols * 8)))
            try:
                var scatter = msel_entry("msel_scatter_rows", context)
                for f in range(count):  # driver-loop: over folds (one device scatter each)
                    var t = tests[f]
                    if Int(py=t.size) > 0:
                        var src = preds[f]
                        if not msel_has_c(src):
                            src = msel_materialize(src, "predictions", context)
                        var sid = msel_resident_op(res, "msel_resident_entry", Python.tuple(t, PythonObject("<i8")), context)[2]
                        _ = family_call(scatter, PythonObject(block), PythonObject(n), codes[PythonObject(dtype)],
                                        PythonObject(msel_addr(src)), codes[src.dtype], t.size, PythonObject(cols), sid)
                        _ = src  # keep the Python object alive through the native call above
                _ = family_call(msel_entry("msel_read", context), PythonObject(block), PythonObject(msel_addr(out)),
                                PythonObject(n * cols * 8))
                _ = out  # keep the Python object alive through the native call above
            finally:
                _ = family_call(msel_entry("msel_free", context), PythonObject(block))
        return out
    # label objects (str / mixed): placed by list, the builtins' C loops
    var rows = family_call(msel_op("mul"), py_call(b.list, Python.tuple(PythonObject(None))), PythonObject(n))
    var last_width = Python.tuple()
    for f in range(count):  # driver-loop: over folds
        var pred = preds[f]
        var vals = PythonObject(None)
        if msel_truth(py_call(b.hasattr, pred, PythonObject("tolist"))):
            vals = family_call(family_call(b.getattr, pred, PythonObject("tolist")))
        else:
            vals = family_call(b.list, pred)
        _ = family_call(b.list, py_call(b.map, py_call(b.getattr, rows, PythonObject("__setitem__")),
                                        py_call(b.list, tests[f]._mv), vals))
        if msel_truth(py_call(b.hasattr, pred, PythonObject("shape"))):
            last_width = py_call(b.tuple, py_call(py_attr(pred.shape, "__getitem__"), py_call(b.slice, PythonObject(1), PythonObject(None))))
        else:
            last_width = Python.tuple()
    var from_list = family_call(b.getattr, msel_cls("Array", context), PythonObject("from_list"))
    var itertools = Python.import_module("itertools")
    if len(last_width) > 0:
        var flat = family_call(b.list, py_call(b.map, b.float, py_call(itertools.chain.from_iterable, rows)))
        var block = family_call(from_list, flat, PythonObject("<f8"))
        return family_call(block.reshape, family_call(msel_op("add"), Python.tuple(n), last_width))
    if msel_truth(family_call(b.all, py_call(b.map, b.isinstance, rows, py_call(itertools.repeat, context["family_msel_integral"])))):
        return family_call(from_list, rows, PythonObject("<i8"))
    if msel_truth(family_call(b.all, py_call(b.map, b.isinstance, rows, py_call(itertools.repeat, context["family_msel_real"])))):
        return family_call(from_list, rows, PythonObject("<f8"))
    return rows


def msel_cross_val_predict(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cross_val_predict(estimator, X, y, groups, cv, n_jobs, method)`."""
    var estimator = args[0]
    msel_require_serial(args[5], PythonObject("raise"), "cross_val_predict", context)
    var got = msel_cv_folds(estimator, args[1], args[2], args[4], args[3], False, context)
    var x = got[0]
    var y = got[1]
    var folds = got[2]
    var n = msel_n_samples(x, context)
    var total = 0
    for f in range(len(folds)):  # driver-loop: over folds
        total += Int(py=folds[f][1].size)
    if total != n:
        raise Error("cross_val_predict only works for partitions")
    var held = msel_new(n, "<i8", context)
    var at = 0
    var tests = Python.list()
    for f in range(len(folds)):  # driver-loop: over folds (one copy each)
        var t = folds[f][1]
        var size = Int(py=t.size)
        if size > 0:
            memcpy(dest=msel_p64(held) + at, src=msel_p64(t), count=size)
            _ = t  # keep the Python object alive through the native call above
            _ = held  # keep the Python object alive through the native call above
        at += size
        _ = py_call(tests.append, t)
    if msel_int(family_call(msel_entry("check_indices_i64", context), PythonObject(msel_addr(held)), PythonObject(n),
                            PythonObject(n))) != 0:
        raise Error("cross_val_predict only works for partitions")
    _ = held  # keep the Python object alive through the native calls above
    var method = args[6]
    var rows = MselFoldRows(x, y, folds, msel_device_rows_ok(estimator, PythonObject(None), context), context)
    var result = PythonObject(None)
    try:
        var preds = Python.list()
        for i in range(len(folds)):  # driver-loop: over folds
            var est = msel_clone(estimator, False, context)
            var fr = rows.take(i, PythonObject(None), context)
            msel_fit(est, fr[0], fr[1], context)
            if py_type(method) is not msel_b().str:
                raise Error("TypeError: native prediction method must be an exact string")
            var pred = msel_estimator_call(est, String(method), Python.tuple(fr[2]), context)
            _ = py_call(preds.append, _numeric_pred(pred, context))
        result = _place(preds, tests, n, rows.res, context)
    finally:
        rows.close(context)
    return result


# ---------------------------------------------------------------- split_descriptor


def _canonical(value: PythonObject) raises -> PythonObject:
    """`_canonical(value)`: sorted-key, compact, ASCII JSON bytes."""
    var keywords = Python.dict()
    py_setitem(keywords, "sort_keys", PythonObject(True))
    py_setitem(keywords, "separators", Python.tuple(",", ":"))
    py_setitem(keywords, "ensure_ascii", PythonObject(True))
    py_setitem(keywords, "allow_nan", PythonObject(False))
    var text = family_call_kw(Python.import_module("json").dumps, keywords, value)
    return py_call(text.encode, PythonObject("ascii"))


def _digest(chunks: PythonObject) raises -> PythonObject:
    """`_digest(*chunks)`: sha256 over length-prefixed chunks."""
    var m = py_call(Python.import_module("hashlib").sha256)
    for i in range(len(chunks)):  # driver-loop: over a few descriptor chunks
        _ = py_call(m.update, py_call(PythonObject(len(chunks[i])).to_bytes, PythonObject(8), PythonObject("little")))
        _ = py_call(m.update, chunks[i])
    return py_call(m.hexdigest)


def _ascii(text: String) raises -> PythonObject:
    return py_call(PythonObject(text).encode, PythonObject("ascii"))


def _order_digest(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_order_digest`: the values in arrival order with dtype and shape."""
    var b = msel_b()
    if family_isinstance(value, b.list):
        return _digest(Python.tuple(_ascii("list"), _canonical(value)))
    var arr = msel_materialize(value, name, context)
    if not msel_has_c(arr):
        arr = family_call(arr._as_c)
    var raw = py_call(b.bytes, arr._mv)
    return _digest(Python.tuple(_ascii("array"), py_call(arr.dtype.encode, PythonObject("ascii")),
                                _canonical(py_call(b.list, arr.shape)), raw))


def msel_split_descriptor(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`split_descriptor(X, y, estimator=..., cv=..., groups=...)`."""
    var b = msel_b()
    var estimator = args[2]
    var cv = args[3]
    var groups = args[4]
    if (cv is None or (not msel_is_bool(cv, context) and msel_is_integral(cv, context))) and estimator is None:
        raise Error("split_descriptor: the default folds are stratified for a classifier "
                    + "and plain KFold otherwise, so estimator is required when cv is None "
                    + "or an integer (pass the estimator cross_val_score was given)")
    var x = msel_materialize(args[0], "X", context)
    if Int(py=x.ndim) != 2 or Int(py=x.shape[0]) == 0 or Int(py=x.shape[1]) == 0:
        raise Error("X must be a nonempty dense 2-D buffer array")
    var y = msel_materialize_y(args[1], context)
    var n = msel_n_samples(x, context)
    if msel_n_samples(y, context) != n:
        raise Error("y must be a 1-D buffer array matching X rows")
    var it = family_call(b.iter, msel_fold_source(cv, estimator, x, y, groups, context))
    var folds = Python.list()
    var sizes = Python.list()
    while True:  # driver-loop: over folds
        var item = msel_try(b.next, Python.tuple(it), b.StopIteration, context)
        if msel_missed(item, context):
            break
        if py_type(item) is not b.list and py_type(item) is not b.tuple:
            raise Error("TypeError: each native fold must be an explicit list or tuple of index buffers")
        var pair = family_call(b.tuple, item)
        if len(pair) != 2:
            raise Error(String("too many values to unpack (expected 2)") if len(pair) > 2
                        else String("not enough values to unpack (expected 2, got ") + String(len(pair)) + ")")
        var train = py_call(b.list, msel_indices(pair[0], n, "train", context)._mv)
        var test = py_call(b.list, msel_indices(pair[1], n, "test", context)._mv)
        _ = py_call(folds.append, Python.tuple(train, test))
        _ = py_call(sizes.append, py_call(b.list, Python.tuple(len(train), len(test))))
    if len(folds) == 0:
        raise Error("cv must produce at least one fold")
    var descriptor = Python.dict()
    py_setitem(descriptor, "schema", context["family_msel_schema"])
    py_setitem(descriptor, "n_rows", PythonObject(n))
    py_setitem(descriptor, "n_features", x.shape[1])
    py_setitem(descriptor, "X_sha256", _order_digest(x, "X", context))
    py_setitem(descriptor, "y_sha256", _order_digest(y, "y", context))
    py_setitem(descriptor, "groups_sha256", PythonObject(None) if groups is None else _order_digest(groups, "groups", context))
    py_setitem(descriptor, "fold_assignment_sha256", _digest(Python.tuple(_ascii("folds"), _canonical(folds))))
    py_setitem(descriptor, "n_folds", PythonObject(len(folds)))
    py_setitem(descriptor, "fold_sizes", sizes)
    py_setitem(descriptor, "sha256", py_call(py_call(Python.import_module("hashlib").sha256, _canonical(descriptor)).hexdigest))
    return descriptor
