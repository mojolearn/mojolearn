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
from std.python._cpython import PyObjectPtr, GILReleased
from std.ffi import c_int, external_call
from std.sys.compile import is_defined
from bindings.cpython_table import cpython_table
from bindings.optimizer_public_words import stash_python_error
from bindings.native_unit_threads import (
    native_unit_threads_run, native_unit_gil_enter, native_unit_gil_leave, native_unit_cores, NativeUnitFailure,
    native_unit_gate, native_unit_gate_enter, native_unit_gate_leave,
)
from core.host_parallel import host_fp_env, host_fp_env_set
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.py_attr import py_attr
from bindings.py_slice_native import py_slice, py_slice_from
from bindings.native_buffer_owner import native_storage_allocate
from bindings.arena_resident_plan import ArenaResidentPlan
from bindings.admitted_registry import admitted_scope_begin, admitted_scope_end, admitted_hold_rows_of, admitted_is_held
from bindings.admitted_meta_scope import admitted_hold_array_f32_deferred
from bindings.metrics_native_support import metric_float
from bindings.msel_json import msel_canonical
from bindings.sha256_door import sha256_update, sha256_hexdigest
from core.sha256 import Sha256
from core.msel_fold_digest import msel_fold_digest_hex
from bindings.msel_estimator_native import (
    msel_native_clone, msel_native_clone_planned, msel_clone_parameter, msel_estimator_call, msel_estimator_plan,
    msel_planned_call, msel_native_set_params,
)
from bindings.family_common import (
    family_array, family_call, family_call_kw, family_isinstance, family_is_str, family_repr, family_str, family_type_name,
)
from bindings.msel_common import (
    msel_take_labels, msel_b, msel_n_samples, msel_admit, msel_entry, msel_try, msel_missed, msel_catch, msel_is_bool, msel_is_integral, msel_is_real, msel_int,
    msel_op, msel_cmp, msel_materialize, msel_try_materialize, msel_flatten, msel_is_array, msel_addr, msel_p64, msel_pf64,
    msel_new, msel_f64_list, msel_index_i64, msel_resident_new, msel_resident_op, msel_resident_take,
    msel_resident_release, msel_resident_close, msel_get_scorer, msel_score, msel_score_value, MselMetricPrepared, msel_lazy,
)
from bindings.msel_split import msel_split, msel_check_cv, msel_default_folds, msel_truth, msel_flag_truth, msel_cls, msel_sabotage

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


def msel_fit_planned(estimator: PythonObject, plan: PythonObject, x: PythonObject, y: PythonObject,
                     context: PythonObject) raises:
    """`msel_fit` of a fresh `msel_native_clone_planned` clone and its plan."""
    _ = msel_planned_call(estimator, plan, "fit", Python.tuple(x, y), context)


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
    var group_values = PythonObject(None) if groups is None else msel_materialize_y(groups, context)
    var splitter = msel_check_cv(cv, y, msel_classifier(estimator, pinned, context), PythonObject(False), PythonObject(None), context)
    var n = msel_n_samples(x, context)
    var folds = msel_checked_folds(msel_splits_of(splitter, x, y, group_values, context), n, context)
    return Python.tuple(x, y, folds)


def msel_cv_splitter(estimator: PythonObject, y: PythonObject, cv: PythonObject, pinned: Bool,
                     context: PythonObject) raises -> PythonObject:
    """The splitter `msel_cv_folds` resolves (`check_cv(cv, y, classifier)`)
    for an already materialized y. `check_cv` reads y only for its target
    type, so a caller that redraws folds for the same labels in another
    order (permutation_test_score) resolves it once."""
    return msel_check_cv(cv, y, msel_classifier(estimator, pinned, context), PythonObject(False), PythonObject(None),
                         context)


def msel_cv_folds_admitted(splitter: PythonObject, x: PythonObject, y: PythonObject, group_values: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    """`msel_cv_folds(...)[2]` over the X, y and groups the caller already
    materialized and the splitter it resolved once (`msel_cv_splitter`):
    the same split and the same fold checks, without copying X, re-reading
    y and the groups, or resolving the splitter again (lane ne2-b-members,
    N055)."""
    var n = msel_n_samples(x, context)
    return msel_checked_folds(msel_splits_of(splitter, x, y, group_values, context), n, context)


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
    if py_type(error_score) is not b.str or not msel_cmp("eq", error_score, PythonObject("raise")):
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
    var group_values = PythonObject(None)
    if groups is not None:
        group_values = msel_materialize_y(groups, context)
        if (msel_is_array(group_values, context) and Int(py=group_values.ndim) != 1) or msel_n_samples(group_values, context) != n:
            raise Error("groups must be 1-D and match X rows")
    if msel_fast_on(MSEL_CV_TRUST_FOLDS_BIT, context) and msel_native_default_cv(cv, groups, context):
        var trusted = msel_default_folds(y, PythonObject(5) if cv is None else cv, msel_classifier(estimator, False, context), context)
        return Python.tuple(x, y, trusted, scoring)
    var folds = msel_checked_folds(msel_fold_source(cv, estimator, x, y, group_values, context), n, context)
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
    var entry = context["family_msel_rows_entry"]
    if py_type(entry) is not py_type(msel_b().len):
        raise Error("ImportError: native cross-validation resident-row binding is unavailable")
    return family_call(entry, obj, PythonObject(op), args, context["family_msel_rows_context"])


def _msel_hold_rows_of(x: PythonObject, rows: PythonObject) raises:
    """Hold a fold's rows (an exact float32 C-order Array gathered out of the
    held X) with X's verdict."""
    if Int(py=rows.ndim) != 2 or String(py=rows.dtype) != "<f4" or String(py=rows.order) != "C":
        return
    var n = Int(py=rows.shape[0])
    var cols = Int(py=rows.shape[1])
    if n < 1 or cols < 1 or admitted_is_held(rows):
        return
    admitted_hold_rows_of(x, rows, Int(py=rows._addr), n, cols)


struct MselFoldRows(Movable):
    """`_FoldRows`: each fold's rows of X (and y) for the fits that share
    the folds, out of X, y and the fold indices resident on the device."""
    var x: PythonObject
    var y: PythonObject
    var folds: PythonObject
    var res: PythonObject
    var resident: ArenaResidentPlan
    var extra: PythonObject
    var dev: Bool
    var scope: PythonObject
    var rows_class: PythonObject
    var rows_take: PythonObject
    # N087 (lane ne1-b-door): one DeviceRows per fold side per CV call, not
    # one per fold side per candidate: {address of the fold's index object:
    # (that object, its DeviceRows)}. The index object is held, so its
    # address cannot be reused while the entry lives.
    var rows_cache: PythonObject
    var rows_dict: PythonObject
    # S4 (lane s1): the admitted-registry hold scope of the fold set. A host
    # float32 C-order X is held for the whole set, and each fold's X rows
    # (gathered out of it) are held in a nested scope while that fold runs:
    # every member door, in any binding (one registry per process), reads
    # them as admitted, with no view or check of its own. -1: no scope open.
    # Lane s2: X is held with its finite test DEFERRED, not scanned here (the
    # scan was one host pass over X per CV call, also on a GPU install whose
    # members test the words in their kernel entries). A member that tests in
    # its entry reads the rows untested; the first one that needs a host
    # verdict scans X once (bindings/admitted_meta_scope.mojo
    # admitted_held_verify, through the fold rows' parent link), so a host
    # column still scans X at most once per CV call.
    var scope_mark: Int
    var fold_mark: Int
    var x_held: Bool
    # Lane p3-msel-parallel-and-process-pool: True while units of one run
    # take folds on several threads (MselUnitRun). Each take is then made
    # alone (the run's gate), the per-fold registry hold is skipped (a hold
    # scope belongs to the thread that opened it; a member on another thread
    # re-scans its rows: lost reuse, never a wrong verdict), and a cached
    # DeviceRows is not handed to a second unit (two concurrent units could
    # materialize and write one host snapshot).
    var shared: Bool
    # cross_val_score's FAST slice route (MSEL_CV_SLICE_BIT): [a, b) of each
    # contiguous test fold, 2 words per fold; empty: every take gathers.
    var ranges: List[Int]

    def __init__(out self, x: PythonObject, y: PythonObject, folds: PythonObject, device_rows: Bool,
                 context: PythonObject) raises:
        self.x = x
        self.y = y
        self.folds = folds
        self.res = PythonObject(None)
        self.resident = ArenaResidentPlan(context["family_msel_resident_context"])
        self.extra = PythonObject(None)
        self.dev = (device_rows and msel_is_array(x, context) and String(py=x.dtype) == "<f4" and Int(py=x.ndim) == 2
                    and Int(py=x.size) > 0 and msel_has_c(x))
        self.scope = PythonObject(None)
        self.rows_class = PythonObject(None)
        self.rows_take = PythonObject(None)
        self.rows_cache = Python.dict()
        self.rows_dict = PythonObject(None)
        if self.dev:
            # DeviceRows exposes a bound take owner to its materializer. Keep
            # that separately mutable public boundary on its existing checks.
            self.res = msel_resident_new(context)
            # only DeviceRows bases enter this scope (`rows_slot` ignores the
            # size floor); no other program input is kept across folds
            var scope = native_storage_allocate(family_call(msel_b().getattr, context["family_msel_arena_io"],
                                                            PythonObject("resident")))
            py_setattr(scope, "min_words", PythonObject(1 << 62))
            py_setattr(scope, "_scope", PythonObject(None))
            _ = _arena_rows_call(scope, "scope_enter", Python.tuple(), context)
            self.scope = scope
            # perf2-msel: the DeviceRows class and the resident store's bound
            # `take` are read once per fold set, not once per fold side. The
            # store is this owner's own object; the class is the import-time
            # `_arena_io` connection read once at the start of the run.
            self.rows_class = family_call(msel_b().getattr, context["family_msel_arena_io"], PythonObject("DeviceRows"))
            self.rows_take = family_call(msel_b().getattr, self.res, PythonObject("take"))
            self.rows_dict = context["family_msel_rows_context"]["maxabs_rows_dict"]
        self.scope_mark = -1
        self.fold_mark = -1
        self.x_held = False
        self.shared = False
        self.ranges = List[Int]()
        if not self.dev and msel_is_array(x, context):
            self.scope_mark = admitted_scope_begin()
            try:
                self.x_held = admitted_hold_array_f32_deferred(x)
            except error:
                admitted_scope_end(self.scope_mark)
                self.scope_mark = -1
                raise error^

    def _hold_fold(mut self, xtr: PythonObject, xte: PythonObject, context: PythonObject) raises:
        """Hold fold rows gathered out of the held X for this fold only,
        with X's verdict (rows of a finite X are finite; rows of an untested
        X are linked to it): the previous fold's rows are dropped first, so
        at most one fold's arrays are held beyond their own use."""
        if not self.x_held or self.shared:
            return
        if self.fold_mark >= 0:
            admitted_scope_end(self.fold_mark)
        self.fold_mark = admitted_scope_begin()
        if msel_is_array(xtr, context):
            _msel_hold_rows_of(self.x, xtr)
        if msel_is_array(xte, context):
            _msel_hold_rows_of(self.x, xte)

    def _end_holds(mut self):
        if self.fold_mark >= 0:
            try:
                admitted_scope_end(self.fold_mark)
            except ignored:
                _ = ignored^
            self.fold_mark = -1
        if self.scope_mark >= 0:
            try:
                admitted_scope_end(self.scope_mark)
            except ignored:
                _ = ignored^
            self.scope_mark = -1

    def xrows(mut self, idx_in: PythonObject, context: PythonObject) raises -> PythonObject:
        if not self.dev:
            var x = self.x
            return self.rows(x, idx_in, context)
        var key = PythonObject(Int(idx_in._obj_ptr))
        var hit = py_get(self.rows_cache, key)
        if hit is not None and hit[0] is idx_in and not self.shared:
            # Reused only while no route has taken its host rows: a host
            # snapshot an estimator was handed may since have been written
            # (an in-place transform), so a materialized one is replaced.
            var dr = hit[1]
            var fields = py_call(self.rows_dict.__get__, dr, self.rows_class)
            if py_get(fields, "_host") is None:
                return dr
        var idx = msel_index_i64(idx_in, context)
        var dr = native_storage_allocate(self.rows_class)
        _ = _arena_rows_call(dr, "init", Python.tuple(self.x, idx, self.rows_take), context)
        py_setitem(self.rows_cache, key, Python.tuple(idx_in, dr))
        return dr

    def resident_take(mut self, values: PythonObject, idx: PythonObject, context: PythonObject) raises -> PythonObject:
        if self.dev:
            return msel_resident_take(self.res, values, idx, context)
        return self.resident.take(values, idx)

    def resident_store(mut self, values: PythonObject, indices: Bool, context: PythonObject) raises -> Int:
        if self.dev:
            var dtype = PythonObject("<i8") if indices else PythonObject(None)
            return Int(py=msel_resident_op(self.res, "msel_resident_entry", Python.tuple(values, dtype), context)[2])
        return self.resident.store(values, indices)

    def resident_release(mut self, values: PythonObject, context: PythonObject) raises:
        if self.dev:
            msel_resident_release(self.res, values, context)
        else:
            self.resident.release(values)

    def rows(mut self, values: PythonObject, idx: PythonObject, context: PythonObject) raises -> PythonObject:
        var b = msel_b()
        if family_isinstance(values, b.list):
            return msel_take_labels(values, idx, context)
        return self.resident_take(values, idx, context)

    def take(mut self, i: Int, y_in: PythonObject, context: PythonObject) raises -> PythonObject:
        """(X_train, y_train, X_test, y_test) of fold i; a `y_in` (a permuted
        y) stands in for the stored one."""
        if len(self.ranges) > 0 and y_in is None:
            # Views over contiguous rows, no gather (the slice route).
            var a = self.ranges[2 * i]
            var b = self.ranges[2 * i + 1]
            var xs = _rows_outside(self.x, a, b, context)
            var xv = _row_view(self.x, a, b, context)
            self._hold_fold(xs, xv, context)
            return Python.tuple(xs, _rows_outside(self.y, a, b, context), xv, _row_view(self.y, a, b, context))
        var pair = self.folds[i]
        var train = pair[0]
        var test = pair[1]
        var xtr = self.xrows(train, context)
        var xte = self.xrows(test, context)
        self._hold_fold(xtr, xte, context)
        var y = y_in
        if y is None:
            y = self.y
        elif y is not self.y and y is not self.extra:
            if self.extra is not None:
                var previous = self.extra
                self.resident_release(previous, context)
            self.extra = y
        if y is None:
            return Python.tuple(xtr, PythonObject(None), xte, PythonObject(None))
        return Python.tuple(xtr, self.rows(y, train, context), xte, self.rows(y, test, context))

    def close(mut self, context: PythonObject) raises:
        self._end_holds()
        self.extra = PythonObject(None)
        self.rows_cache = Python.dict()
        self.rows_take = PythonObject(None)
        var scope = self.scope
        self.scope = PythonObject(None)
        try:
            if scope is not None:
                _ = _arena_rows_call(scope, "scope_exit", Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None)), context)
        finally:
            if self.dev:
                msel_resident_close(self.res, context)
            else:
                self.resident.close()


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
                            error_score: PythonObject, context: PythonObject, workers: Int = 1) raises -> MselScores:
    """`_cross_validate_folds` on the folds of `rows`: one clone fit and
    scored per fold, the folds run as units on `workers` threads."""
    var run = msel_cv_units(estimator, PythonObject(None), y, rows, scoring, return_train, return_estimator,
                            error_score, workers, context)
    return run.candidate(0, return_indices, context)


# ---------------------------------------------------------------- units
#
# Lane p3-msel-parallel-and-process-pool (2026-11-09; audit B worst path 6:
# candidates x folds ran serially, every candidate cloned once and again per
# fold, n_jobs refused). A UNIT is one (candidate, fold): a clone of the base
# estimator (the only clone the unit makes), the candidate's parameters set on
# it (their values cloned, as scikit-learn's clone(base).set_params(**clone(
# params)) does), the fold's rows taken, the fit and every score. Units are
# independent; they are pulled in index order (candidate-major) by every
# thread of the run (bindings/native_unit_threads.mojo) and each writes only
# its own result slots, so the results, and the first error raised, are those
# of the serial order on any thread count. Each fit computes what it computed
# serially: no fold order, reduction or kernel changes, so no bits change.
#
# REMOVED by this change: the per-candidate clone of GridSearchCV,
# RandomizedSearchCV and validation_curve (msel_search.mojo, the clone before
# set_params; each unit clones the base once), the serial fold loop of
# msel_cross_validate_run (the units replace it; one thread is the serial
# run) and the n_jobs refusal `msel_require_serial` (n_jobs is the thread
# bound now, `msel_unit_jobs`).

#: Old form behind a switch until the A/B is measured (CLAUDE.md "remove what
#: a change made unnecessary"): -D MOJOLEARN_MSEL_UNITS_SERIAL=1 runs every
#: unit on the calling thread, whatever n_jobs says. RUN OWED: GridSearchCV /
#: cross_validate rows of paper_compare on NVIDIA and AMD, units on vs off.
comptime MSEL_UNITS_SERIAL = is_defined["MOJOLEARN_MSEL_UNITS_SERIAL"]()

#: Fraction of the device's free bytes the concurrent units may claim, by
#: the bytes the first unit kept (measured, below): half, because what a fit
#: keeps after it returns (its fitted buffers, cached workspaces) is less than
#: its peak. A cost rule over any shape, not a board size.
comptime MSEL_UNITS_FREE_DIVISOR = 2


def msel_unit_jobs(n_jobs: PythonObject, error_score: PythonObject, caller: String,
                   context: PythonObject) raises -> Int:
    """The thread bound of a CV operation from its `n_jobs` (joblib's
    meaning): a positive int is the bound; -1 all cores, -k all cores but
    k - 1; None all cores (the default is parallel: AGENTS.md "parallelize
    tests and experiments by default"; the result does not depend on it).
    Also checks `error_score` as `_require_serial` did."""
    msel_admit(n_jobs, context)
    msel_admit(error_score, context)
    if not (py_type(error_score) is msel_b().str and msel_cmp("eq", error_score, PythonObject("raise"))) and not msel_is_real(error_score, context):
        raise Error("error_score must be 'raise' or a number")
    var cores = native_unit_cores()
    if n_jobs is None:
        return cores
    if py_type(n_jobs) is msel_b().bool or not msel_is_integral(n_jobs, context):
        raise Error("TypeError: " + caller + ": n_jobs must be None or an integer")
    var k = Int(py=family_call(msel_b().int, n_jobs))
    if k == 0:
        raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
    if k < 0:
        return max(cores + 1 + k, 1)
    return k


def msel_unit_workers(jobs: Int, units: Int) -> Int:
    comptime if MSEL_UNITS_SERIAL:
        return 1
    return max(min(jobs, units), 1)


def _msel_device_free(rows: MselFoldRows, context: PythonObject) -> Int:
    """Free bytes of the device the fold rows live on (the base binding's
    `msel_free_bytes`), or -1 with no device rows or no entry."""
    if not rows.dev:
        return -1
    try:
        var rc = context["family_msel_rows_context"]
        var entry = py_get(rc, "arena_msel_free_bytes")
        if entry is None:
            return -1
        return Int(py=family_call(entry))
    except:
        return -1


def _msel_units_thread(arg: Int) abi("C") -> Int:
    """A unit thread (and the caller's share): take the GIL, run units in the
    caller's floating-point environment until none is left."""
    try:
        var state = native_unit_gil_enter()
        var run = MutPointer[MselUnitRun, MutUntrackedOrigin](unsafe_from_address=arg)
        var saved = host_fp_env()
        host_fp_env_set(run[].fp_env)
        run[].drain()
        host_fp_env_set(saved)
        native_unit_gil_leave(state)
    except:
        pass
    return 0


struct MselUnitRun(Movable):
    """The units of one CV operation and their result slots; every field is
    read and written with the GIL held."""
    var base: PythonObject
    var candidates: PythonObject
    var y: PythonObject
    var rows: MutPointer[MselFoldRows, MutUntrackedOrigin]
    var multi: PythonObject
    var single: PythonObject
    var keys: PythonObject
    var n_keys: Int
    var n_folds: Int
    var n_candidates: Int
    var n_units: Int
    var return_train: Bool
    var return_estimator: Bool
    var raise_errors: Bool
    var error_score: PythonObject
    var context: PythonObject
    var next: Int
    var stop: Bool
    var gate: List[Int64]
    var test: List[Float64]
    var train: List[Float64]
    var fit_time: List[Float64]
    var score_time: List[Float64]
    var estimators: PythonObject
    var failure: NativeUnitFailure
    var fp_env: UInt64
    # cross_val_predict: the method each unit calls on its test rows (an
    # exact str), its output in `preds[u]`; None: the units score.
    var method: PythonObject
    var preds: PythonObject

    def __init__(out self, base: PythonObject, candidates: PythonObject, y: PythonObject, mut rows: MselFoldRows,
                 scoring: PythonObject, return_train: Bool, return_estimator: Bool, error_score: PythonObject,
                 method: PythonObject, context: PythonObject) raises:
        var b = msel_b()
        self.base = base
        self.candidates = candidates
        self.y = PythonObject(None) if y is rows.y else y
        self.rows = MutPointer[MselFoldRows, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=rows)))
        self.multi = PythonObject(None)
        self.single = PythonObject(None)
        self.keys = Python.list()
        if method is None:
            self.multi = msel_scorers(scoring, context)
        if method is not None:
            pass  # predicting units score nothing: no scorer is admitted
        elif self.multi is not None:
            self.keys = py_call(b.list, self.multi)
        else:
            self.single = msel_get_scorer(scoring, context)
            _ = py_call(self.keys.append, PythonObject("score"))
        self.n_keys = len(self.keys)
        self.n_folds = len(rows.folds)
        self.n_candidates = 1 if candidates is None else len(candidates)
        self.n_units = self.n_candidates * self.n_folds
        self.return_train = return_train
        self.return_estimator = return_estimator
        self.raise_errors = family_is_str(error_score)
        self.error_score = error_score
        self.context = context
        self.next = 0
        self.stop = False
        self.gate = native_unit_gate()
        self.test = List[Float64](length=self.n_units * self.n_keys, fill=0.0)
        self.train = List[Float64](length=self.n_units * self.n_keys, fill=0.0)
        self.fit_time = List[Float64](length=self.n_units, fill=0.0)
        self.score_time = List[Float64](length=self.n_units, fill=0.0)
        self.estimators = py_call(b.list, Python.tuple())
        if return_estimator:
            self.estimators = family_call(msel_op("mul"), Python.list(PythonObject(None)), PythonObject(self.n_units))
        self.failure = NativeUnitFailure()
        self.fp_env = host_fp_env()
        self.method = method
        self.preds = py_call(b.list, Python.tuple())
        if method is not None:
            self.preds = family_call(msel_op("mul"), Python.list(PythonObject(None)), PythonObject(self.n_units))

    def _take(mut self, f: Int) raises -> PythonObject:
        """Fold f's rows, one take at a time across threads: a take can
        release the GIL inside its device gather."""
        native_unit_gate_enter(self.gate)
        try:
            var got = self.rows[].take(f, self.y, self.context)
            native_unit_gate_leave(self.gate)
            return got
        except error:
            native_unit_gate_leave(self.gate)
            raise error^

    def unit(mut self, u: Int) raises:
        var b = msel_b()
        var c = u // self.n_folds
        var f = u - c * self.n_folds
        var cloned = msel_native_clone_planned(self.base, self.context)
        var est = cloned[0]
        var plan = cloned[1]
        if self.candidates is not None:
            _ = msel_native_set_params(est, msel_clone_parameter(self.candidates[c], self.context), self.context)
        var t0 = msel_seconds()
        var got = self._take(f)
        var xtr = got[0]
        var ytr = got[1]
        var xte = got[2]
        var yte = got[3]
        var ok = True
        if self.raise_errors:
            msel_fit_planned(est, plan, xtr, ytr, self.context)
        else:
            try:
                msel_fit_planned(est, plan, xtr, ytr, self.context)
            except error:
                if not msel_catch(error, b.Exception):
                    raise error^
                ok = False
        var t1 = msel_seconds()
        if self.method is not None:
            # Fit is native: no Python ran since the clone's plan was made.
            var pred = msel_planned_call(est, plan, String(self.method), Python.tuple(xte), self.context)
            py_setitem(self.preds, PythonObject(u), _numeric_pred(pred, self.context))
            return
        # Only metric ABI/options live across this unit's scorers; nothing
        # is shared with another unit.
        var plans = List[MselMetricPrepared]()
        for k in range(self.n_keys):  # driver-loop: metadata-only template owner per scoring slot
            plans.append(MselMetricPrepared(True))
        var test_memo = PythonObject(None)
        var train_memo = PythonObject(None)
        if ok and self.multi is not None and self.n_keys > 1:
            test_memo = Python.dict()
            if self.return_train:
                train_memo = Python.dict()
        var failed_value = 0.0
        if not ok:
            failed_value = msel_error_value(self.error_score)
        for k in range(self.n_keys):  # driver-loop: over scorer names
            var sc = self.multi[self.keys[k]] if self.multi is not None else self.single
            self.test[u * self.n_keys + k] = msel_score_value(est, xte, yte, sc, self.context, test_memo, plans[k]) if ok else failed_value
            if self.return_train:
                self.train[u * self.n_keys + k] = msel_score_value(est, xtr, ytr, sc, self.context, train_memo, plans[k]) if ok else failed_value
        _ = test_memo^
        _ = train_memo^
        self.fit_time[u] = t1 - t0
        self.score_time[u] = msel_seconds() - t1
        if self.return_estimator:
            py_setitem(self.estimators, PythonObject(u), est)

    def fail(mut self, u: Int, error: Error):
        self.stop = True
        self.failure.record(u, error)

    def drain(mut self):
        while True:  # driver-loop: over units, pulled in index order under the GIL
            if self.stop or self.next >= self.n_units:
                return
            var u = self.next
            self.next += 1
            try:
                self.unit(u)
            except error:
                self.fail(u, error)

    def reraise(mut self) raises:
        self.failure.reraise()

    def candidate(self, c: Int, return_indices: Bool, context: PythonObject) raises -> MselScores:
        """Candidate c's per-fold results, in fold order."""
        var out = MselScores(self.keys)
        var folds = self.rows[].folds
        for f in range(self.n_folds):  # driver-loop: over folds
            var u = c * self.n_folds + f
            for k in range(self.n_keys):  # driver-loop: over scorer names
                out.test[k].append(self.test[u * self.n_keys + k])
                if self.return_train:
                    out.train[k].append(self.train[u * self.n_keys + k])
            out.fit_time.append(self.fit_time[u])
            out.score_time.append(self.score_time[u])
            if self.return_estimator:
                _ = py_call(out.estimators.append, self.estimators[u])
            if return_indices:
                _ = py_call(out.train_idx.append, folds[f][0])
                _ = py_call(out.test_idx.append, folds[f][1])
        return out^


def msel_cv_units(base: PythonObject, candidates: PythonObject, y: PythonObject, mut rows: MselFoldRows,
                  scoring: PythonObject, return_train: Bool, return_estimator: Bool, error_score: PythonObject,
                  jobs: Int, context: PythonObject) raises -> MselUnitRun:
    return msel_cv_units(base, candidates, y, rows, scoring, return_train, return_estimator, error_score, jobs,
                         PythonObject(None), context)


def msel_cv_units(base: PythonObject, candidates: PythonObject, y: PythonObject, mut rows: MselFoldRows,
                  scoring: PythonObject, return_train: Bool, return_estimator: Bool, error_score: PythonObject,
                  jobs: Int, method: PythonObject, context: PythonObject) raises -> MselUnitRun:
    """Run every (candidate, fold) unit of `rows` (candidates None: the base
    estimator itself, one candidate) on at most `jobs` threads.

    The first unit runs alone on the calling thread: it creates every lazy
    process state (contexts, compiled kernels, caches) single-threaded, and
    the device bytes it kept bound how many units run at once. The rest run
    on min(jobs, units, free bytes / MSEL_UNITS_FREE_DIVISOR / kept bytes)
    threads."""
    var run = MselUnitRun(base, candidates, y, rows, scoring, return_train, return_estimator, error_score, method, context)
    var workers = msel_unit_workers(jobs, run.n_units)
    if workers <= 1 or run.n_units <= 1:
        run.drain()
        run.reraise()
        return run^
    var free0 = _msel_device_free(rows, context)
    run.next = 1
    try:
        run.unit(0)
    except error:
        run.fail(0, error)
    if not run.stop:
        var free1 = _msel_device_free(rows, context)
        if free0 > 0 and free1 >= 0 and free0 > free1:
            var kept = free0 - free1
            workers = max(min(workers, free1 // MSEL_UNITS_FREE_DIVISOR // kept), 1)
        workers = min(workers, run.n_units - 1)
        rows.shared = workers > 1
        try:
            if workers > 1:
                native_unit_threads_run(_msel_units_thread, Int(UnsafePointer(to=run)), workers)
            else:
                run.drain()
        finally:
            rows.shared = False
    run.reraise()
    return run^


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


def msel_cross_validate(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cross_validate(estimator, X, y, groups, scoring, cv, n_jobs,
    return_train_score, return_estimator, return_indices, error_score)`."""
    var estimator = args[0]
    var error_score = args[10]
    var jobs = msel_unit_jobs(args[6], error_score, "cross_validate", context)
    var got = msel_cv_folds(estimator, args[1], args[2], args[5], args[3], False, context)
    var x = got[0]
    var y = got[1]
    var folds = got[2]
    var scoring = args[4]
    msel_admit(args[7], context)
    msel_admit(args[8], context)
    msel_admit(args[9], context)
    var rt = msel_flag_truth(args[7], context)
    var re = msel_flag_truth(args[8], context)
    var ri = msel_flag_truth(args[9], context)
    var rows = MselFoldRows(x, y, folds, msel_device_rows_ok(estimator, scoring, context), context)
    var result = PythonObject(None)
    try:
        var scores = msel_cross_validate_run(estimator, y, rows, scoring, rt, re, ri, error_score, context, jobs)
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


def msel_cross_val_score(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cross_val_score(estimator, X, y, cv, scoring, groups, n_jobs, error_score)`."""
    var estimator = args[0]
    var cv = args[3]
    var groups = args[5]
    var jobs = msel_unit_jobs(args[6], PythonObject("raise"), "cross_val_score", context)
    var scoring = args[4]
    if py_type(scoring) is msel_b().str:
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
    if sliced:
        rows.ranges = ranges^
    try:
        # The folds are units (msel_cv_units) on `jobs` threads; a fit error
        # raises, as the fold loop did.
        var r = msel_cross_validate_run(estimator, y, rows, scoring, False, False, False, PythonObject("raise"),
                                        context, jobs)
        for i in range(len(r.test[0])):  # driver-loop: over folds (results only)
            scores.append(r.test[0][i])
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


def _prediction_list(pred: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = msel_b()
    msel_admit(pred, context)
    var kind = py_type(pred)
    var width = Python.tuple()
    var values = PythonObject(None)
    if kind is context["array_class"]:
        var packed = msel_materialize(pred, "predictions", context)
        var rank = Int(py=packed.ndim)
        if rank == 0:
            raise Error("TypeError: iteration over a 0-d Array")
        width = py_slice_from(packed.shape, 1)
        values = py_call(packed._mv.tolist)
        for k in range(rank - 1):  # driver-loop: native rank metadata groups flat values into rows
            var axis = rank - 1 - k
            var step = Int(py=packed.shape[axis])
            var count = 1
            for a in range(axis):  # driver-loop: outer extents also preserve empty dimensions
                count *= Int(py=packed.shape[a])
            var groups = Python.list()
            for r in range(count):  # driver-loop: native list slices preserve the Array.tolist nesting
                var part = py_slice(values, r * step, (r + 1) * step)
                _ = py_call(groups.append, part)
            values = groups
        _ = packed  # retain the native buffer while reading its scalar values
    elif kind is py_get(context, "ndarray"):
        # Exact ndarray.tolist is its C extension implementation. Admit
        # object-storage leaves before any label hash/comparison/conversion.
        values = py_call(pred.tolist)
        width = py_slice_from(pred.shape, 1)
        msel_admit(values, context)
        if py_type(values) is not b.list:
            raise Error("TypeError: iteration over a 0-d array")
    elif kind is b.list or kind is b.tuple or kind is b.range or kind is context["array_type"] or kind is b.memoryview:
        values = py_call(b.list, pred)
    else:
        raise Error("TypeError: native predictions require ordinary label containers or buffers")
    return Python.tuple(values, width)


def _place_label_rows(rows: PythonObject, vals: PythonObject, test: PythonObject, n: Int) raises:
    """`for i, v in zip(test, vals): rows[i] = v` over the C storage: the
    fold's int64 test indices read through their pointer and each label
    reference copied with PyList_SetItem (pass 3, rank 38: no boxed index
    and no Python setitem call per row). `rows` and `vals` are exact lists."""
    ref cpy = Python().cpython()
    var get_item = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")
    var count = min(Int(py=test.size), len(vals))
    var words = msel_p64(test)
    for j in range(count):  # driver-loop: native equivalent of the existing zip placement
        var index = Int(words[j])
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise Error("IndexError: list assignment index out of range")
        var label = get_item(vals._obj_ptr, j)
        if not label:
            var failure = stash_python_error()
            raise failure^
        if cpython_table()[].PyList_SetItem(rows._obj_ptr, index, cpy.Py_NewRef(label)) != 0:
            var failure = stash_python_error()
            raise failure^
    _ = vals  # borrowed labels stay owned through the increfs above
    _ = test  # keep the index storage alive through the pointer reads above


def _place(preds: PythonObject, tests: PythonObject, n: Int, mut resident_rows: MselFoldRows, context: PythonObject) raises -> PythonObject:
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
                        var sid = resident_rows.resident_store(t, True, context)
                        _ = family_call(scatter, PythonObject(block), PythonObject(n), codes[PythonObject(dtype)],
                                        PythonObject(msel_addr(src)), codes[src.dtype], t.size, PythonObject(cols), PythonObject(sid))
                        _ = src  # keep the Python object alive through the native call above
                _ = family_call(msel_entry("msel_read", context), PythonObject(block), PythonObject(msel_addr(out)),
                                PythonObject(n * cols * 8))
                _ = out  # keep the Python object alive through the native call above
            finally:
                _ = family_call(msel_entry("msel_free", context), PythonObject(block))
        return out
    # Label objects (str / mixed) use admitted native lists; numeric fallback
    # retains the public int64/float64 promotion without Array Python methods.
    var rows = family_call(msel_op("mul"), py_call(b.list, Python.tuple(PythonObject(None))), PythonObject(n))
    var last_width = Python.tuple()
    for f in range(count):  # driver-loop: fold outputs placed in original row order
        var converted = _prediction_list(preds[f], context)
        var vals = converted[0]
        _place_label_rows(rows, vals, tests[f], n)
        last_width = converted[1]
    if len(last_width) > 0:
        var flat = Python.list()
        for k in range(n):  # driver-loop: one native container level, preserving the previous reshape contract
            if py_type(rows[k]) is not b.list and py_type(rows[k]) is not b.tuple:
                raise Error("TypeError: multioutput predictions require native rows")
            _ = py_call(flat.extend, rows[k])
        var shape = family_call(msel_op("add"), Python.tuple(n), last_width)
        var size = n
        for k in range(len(last_width)):  # driver-loop: prediction output extents
            size *= Int(py=last_width[k])
        if len(flat) != size:
            raise Error("ValueError: prediction values do not match their output shape")
        var out = family_array(shape, "<f8", context)
        var pointer = msel_pf64(out)
        for k in range(size):  # driver-loop: native numeric scalar output placement
            pointer[k] = metric_float(flat[k], context)
            _ = out  # retain the destination buffer through each store
        return out
    var integers = True
    var reals = True
    for k in range(n):  # driver-loop: exact scalar identity checks before numeric conversion
        integers = integers and msel_is_integral(rows[k], context)
        reals = reals and msel_is_real(rows[k], context)
    if integers:
        var out = family_array(Python.tuple(n), "<i8", context)
        var pointer = msel_p64(out)
        for k in range(n):  # driver-loop: native integer label output placement
            pointer[k] = Int64(msel_int(rows[k]))
            _ = out  # retain the destination buffer through each store
        return out
    if reals:
        var out = family_array(Python.tuple(n), "<f8", context)
        var pointer = msel_pf64(out)
        for k in range(n):  # driver-loop: native real label output placement
            pointer[k] = metric_float(rows[k], context)
            _ = out  # retain the destination buffer through each store
        return out
    return rows


def msel_cross_val_predict(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cross_val_predict(estimator, X, y, groups, cv, n_jobs, method)`."""
    var estimator = args[0]
    var jobs = msel_unit_jobs(args[5], PythonObject("raise"), "cross_val_predict", context)
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
    if py_type(method) is not msel_b().str:
        raise Error("TypeError: native prediction method must be an exact string")
    try:
        # The folds are units (msel_cv_units): one clone, fit and predict each.
        var run = msel_cv_units(estimator, PythonObject(None), y, rows, PythonObject(None), False, False,
                                PythonObject("raise"), jobs, method, context)
        result = _place(run.preds, tests, n, rows, context)
    finally:
        rows.close(context)
    return result


# ---------------------------------------------------------------- split_descriptor


def _canonical(value: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_canonical(value, context)


def _digest(chunks: PythonObject) raises -> PythonObject:
    var hasher = Sha256()
    for i in range(len(chunks)):  # driver-loop: descriptor chunks retain the historical 8-byte length prefixes
        sha256_update(hasher, py_call(PythonObject(len(chunks[i])).to_bytes, 8, "little"))
        sha256_update(hasher, chunks[i])
    return PythonObject(hasher.hexdigest())


def _ascii(text: String) raises -> PythonObject:
    return py_call(PythonObject(text).encode, PythonObject("ascii"))


def _folds_digest(trains: List[PythonObject], tests: List[PythonObject]) raises -> PythonObject:
    """`_digest(("folds", canonical(folds)))` written natively from the int64
    index buffers (core/msel_fold_digest.mojo): the same bytes, so the same
    digest, without building or walking Python int lists."""
    var train_addresses = List[Int]()
    var train_counts = List[Int]()
    var test_addresses = List[Int]()
    var test_counts = List[Int]()
    for f in range(len(trains)):  # driver-loop: over folds
        train_addresses.append(msel_addr(trains[f]))
        train_counts.append(Int(py=trains[f].size))
        test_addresses.append(msel_addr(tests[f]))
        test_counts.append(Int(py=tests[f].size))
    var hex = msel_fold_digest_hex(train_addresses, train_counts, test_addresses, test_counts)
    # The caller's `trains` / `tests` lists own the index buffers for this call.
    return PythonObject(hex)


def _order_digest(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_order_digest`: the values in arrival order with dtype and shape."""
    var b = msel_b()
    if py_type(value) is b.list:
        return _digest(Python.tuple(_ascii("list"), _canonical(value, context)))
    var arr = msel_materialize(value, name, context)
    # pass 3 (rank 580): the values are hashed in place from the admitted
    # C-order storage (the same bytes `bytes(arr._mv)` held, after the same
    # 8-byte little-endian length prefix), not copied into a Python bytes
    # object and then into the hasher's store.
    var hasher = Sha256()
    var head = Python.tuple(_ascii("array"), py_call(arr.dtype.encode, PythonObject("ascii")),
                            _canonical(py_call(b.list, arr.shape), context))
    for i in range(len(head)):  # small-loop(three descriptor chunks)
        sha256_update(hasher, py_call(PythonObject(len(head[i])).to_bytes, 8, "little"))
        sha256_update(hasher, head[i])
    var nbytes = Int(py=arr._mv.nbytes)
    var prefix = List[UInt8](length=8, fill=0)
    for k in range(8):  # small-loop(eight length bytes, little-endian)
        prefix[k] = UInt8((nbytes >> (8 * k)) & 0xFF)
    hasher.update_list(prefix)
    if nbytes > 0:
        hasher.update(msel_addr(arr), nbytes)
    _ = arr  # keep the storage alive through the hash above
    return PythonObject(hasher.hexdigest())


def msel_split_descriptor(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`split_descriptor(X, y, estimator=..., cv=..., groups=...)`."""
    var b = msel_b()
    var estimator = args[2]
    var cv = args[3]
    var groups = PythonObject(None) if args[4] is None else msel_materialize_y(args[4], context)
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
    # perf2-msel: the folds stay int64 index Arrays; their digest is written
    # from the buffers (it walked Python int lists through the JSON writer).
    var trains = List[PythonObject]()
    var tests = List[PythonObject]()
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
        var train = msel_indices(pair[0], n, "train", context)
        var test = msel_indices(pair[1], n, "test", context)
        var sized = Python.list()
        _ = py_call(sized.append, PythonObject(Int(py=train.size)))
        _ = py_call(sized.append, PythonObject(Int(py=test.size)))
        _ = py_call(sizes.append, sized)
        trains.append(train^)
        tests.append(test^)
    if len(trains) == 0:
        raise Error("cv must produce at least one fold")
    var descriptor = Python.dict()
    py_setitem(descriptor, "schema", context["family_msel_schema"])
    py_setitem(descriptor, "n_rows", PythonObject(n))
    py_setitem(descriptor, "n_features", x.shape[1])
    py_setitem(descriptor, "X_sha256", _order_digest(x, "X", context))
    py_setitem(descriptor, "y_sha256", _order_digest(y, "y", context))
    py_setitem(descriptor, "groups_sha256", PythonObject(None) if groups is None else _order_digest(groups, "groups", context))
    py_setitem(descriptor, "fold_assignment_sha256", _folds_digest(trains, tests))
    py_setitem(descriptor, "n_folds", PythonObject(len(trains)))
    py_setitem(descriptor, "fold_sizes", sizes)
    py_setitem(descriptor, "sha256", sha256_hexdigest(_canonical(descriptor, context)))
    return descriptor
