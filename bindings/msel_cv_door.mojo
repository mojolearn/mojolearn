# SPDX-License-Identifier: Apache-2.0
# Cross-validation of python/mojolearn/model_selection.py, natively (lane
# mcb2-msel): the estimator clone (`_clone`), the fold drawing and checks
# (`_folds`, `_prepare_folds`, `_cv_folds`, `_indices`), the
# resident fold rows (`_FoldRows`: X, y and every fold's indices put ONCE
# into the `_Resident` store and each fold's rows one device gather out of
# it, or `_arena_io.DeviceRows` for an estimator that takes them), and
# `cross_val_score`, `cross_validate`, `cross_val_predict` and
# `split_descriptor`. Lane mf-r9b: every cv tool's folds are fitted and
# scored by the search kind's typed RUN (`msel_cv_by_search`); this door
# draws and checks the folds (UNPACK) and builds the tool's result (PACK).
from bindings.member_scope_door import member_scope_word
from core.member_abi import MEMBER_SCOPE_NONE
from core.msel_refusal_words import msel_empty_scoring_words, msel_cvp_method_words
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.base_helpers_released_unpack import base_check_indices_i64
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.msel_helpers_key_unpack import msel_helpers_key
from bindings.msel_member_device_unpack import msel_member_host_folds
from bindings.msel_helpers_released_unpack import (
    msel_store_alloc, msel_store_read, msel_store_free, msel_scatter_rows,
)
from bindings.native_unit_os_threads_run import native_unit_cores
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.py_attr import py_attr
from bindings.py_slice_native import py_slice, py_slice_from
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.arena_resident_plan_unpack import ArenaResidentPlan
from bindings.py_admitted_registry import admitted_scope_begin, admitted_scope_end
from bindings.admitted_meta_scope_unpack import admitted_hold_array_f32_deferred
from bindings.metrics_native_support_door import metric_float
from bindings.msel_json_pack import msel_canonical
from bindings.sha256_door import sha256_update, sha256_hexdigest
from core.sha256 import Sha256
from core.msel_fold_digest import msel_fold_digest_hex
from bindings.msel_estimator_native_door import msel_estimator_call, msel_estimator_plan
from bindings.family_common_unpack import (
    family_isinstance,
    family_repr,
)
from bindings.family_common_pack import (
    family_array,
    family_call,
)
from bindings.msel_common_door import (
    msel_take_labels, msel_b, msel_n_samples, msel_admit, msel_try, msel_missed, msel_is_bool, msel_is_integral, msel_is_real, msel_int,
    msel_op, msel_cmp, msel_materialize, msel_try_materialize, msel_flatten, msel_is_array, msel_addr, msel_p64, msel_pf64,
    msel_new, msel_f64_list, msel_index_i64, msel_resident_new, msel_resident_op, msel_resident_take,
    msel_resident_release, msel_resident_close, msel_get_scorer,
)
from bindings.msel_split_door import msel_split, msel_check_cv, msel_default_folds, msel_flag_truth, msel_cls, msel_sabotage
from bindings.msel_cv_run import _numeric_dtype
from bindings.msel_member_units_door import msel_member_fit_in_place, msel_text_rows
from bindings.msel_search_state_door import msel_search_fields, msel_carrier_search

#: model_selection's CV_FAST bits of the resample binding's define mask.
# TOMBSTONE: MOJOLEARN_CV_FAST_SLICE (plumbing folded into the typed search RUN, rule 2026-11-22) deleted 2026-11-23 by lane/mf-msel5b; code recoverable at 6f0a3de6ed.
# Restore: git apply experiments/removed/MOJOLEARN_CV_FAST_SLICE.patch; record in docs/TOMBSTONES.md.
comptime MSEL_CV_TRUST_FOLDS_BIT = 64


# ---------------------------------------------------------------- the estimator


def msel_classifier(estimator: PythonObject, pinned: Bool, context: PythonObject) raises -> Bool:
    return False if estimator is None else Bool(py=msel_estimator_plan(estimator, context)["classifier"])


def msel_fit(estimator: PythonObject, x: PythonObject, y: PythonObject, context: PythonObject) raises:
    _ = msel_estimator_call(estimator, "fit", Python.tuple(x, y), context)


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
    var status = base_check_indices_i64(base_helpers_key(context["family_msel_entries"]), msel_addr(as_i64),
                                        Int(py=as_i64.size), n)
    _ = as_i64  # keep the Python object alive through the native call above
    if status == 1:
        raise Error(name + " contains an out-of-range index")
    _ = as_i64  # keep the Python object alive through the native calls above
    if status == 2:
        raise Error(name + " contains duplicate indices")
    _ = as_i64  # keep the Python object alive through the native calls above
    return as_i64


def msel_splits_of(splitter: PythonObject, x: PythonObject, y: PythonObject, groups: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`splitter.split(X, y, groups)`: a mojolearn splitter's splits natively
    (its `split` is the shell over the same call), anything else the
    caller's own `split`."""
    if py_type(py_type(splitter)) is msel_b().type and py_contains(context["family_msel_splitter_layouts"], py_type(splitter)):
        return msel_split(splitter, x, y, groups, context)
    raise Error("TypeError: Python splitter callbacks are not accepted; use a native splitter or explicit index pairs")


def msel_checked_folds(pairs: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """Every (train, test) of `pairs` checked by `_indices` before the first
    fit; refused when there is none. A train side may share rows with its
    test side, as scikit-learn accepts (`cross_val_predict` alone needs a
    partition and checks it itself)."""
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
    var plan = msel_estimator_plan(estimator, context)
    if not Bool(py=plan["device_rows"]):
        return False
    var multi = msel_scorers(scoring_in, context)
    if multi is None:
        _ = msel_get_scorer(scoring_in, context)
    # The fold rows follow the member's own device (device wave, 2026-11-21):
    # a member that says device="cpu" in a process with a GPU set takes its
    # folds gathered on the host (the `device_rows` False form every
    # non-opted class already takes); rows resident on the device are for
    # members that run there. Once per cross-validation call, after the
    # scorer refusals above (same order as before); the default member
    # (no `device`, or None) costs one dictionary read.
    if msel_member_host_folds(py_call(plan["fields_get"], estimator, py_type(estimator)), msel_b(), context):
        return False
    return True


def _arena_rows_call(obj: PythonObject, op: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """One `_arena_io` DeviceRows / resident-scope operation through its
    compiled door, as the Python methods call it."""
    var entry = context["family_msel_rows_entry"]
    if py_type(entry) is not py_type(msel_b().len):
        raise Error("ImportError: native cross-validation resident-row binding is unavailable")
    # call-table: member DeviceRows (arena rows door over the Python owner) has no native entry yet
    return family_call(entry, obj, PythonObject(op), args, context["family_msel_rows_context"])


def msel_rows_scope_open(context: PythonObject) raises -> PythonObject:
    """Model selection's own resident scope (`_arena_io.resident`), entered
    on this thread with the DeviceRows floor: only a row-indexed base (the
    fold rows taken on the device out of it) stays resident, no other program
    input across folds. The fold rows' set (`MselFoldRows`) and a search fit
    through its member kind (bindings/msel_search_door.mojo, lane mf-search5)
    open it; the caller closes it with `msel_rows_scope_close` on every
    path."""
    var scope = native_storage_allocate(family_call(msel_b().getattr, context["family_msel_arena_io"],
                                                    PythonObject("resident")))
    # the floor goes to the native scope state with the enter call; no
    # Python attribute of the `resident` object is written
    _ = _arena_rows_call(scope, "scope_enter", Python.tuple(PythonObject(1 << 62)), context)
    return scope


def msel_rows_scope_close(scope: PythonObject, context: PythonObject) raises:
    """Leave a scope `msel_rows_scope_open` entered: its caches and the
    references they took are released."""
    _ = _arena_rows_call(scope, "scope_exit", Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None)),
                         context)


#: `-D MOJOLEARN_MSEL_FAST_LC_TEST_REUSE` (lane fx-board-modelsel-metrics,
#: 2026-11-19; decided by A/B round 3, below; bench/results/paper_compare/owed-pre0867.json
#: `fx-lc-test-reuse`, docs/apple-fast/EXPERIMENTS.md). learning_curve's units are
#: (train size, fold): every size of a fold scores on the SAME test rows, and since
#: the units became independent (lane p3-msel-parallel-and-process-pool) those rows
#: are gathered out of host X once per unit, not once per fold (5 sizes x 5 folds
#: of 1,000,000 x 11: 25 test gathers where 5 do, about 180 MB of new pages). With
#: the define a fold set that asks for it (`reuse_test`, set by learning_curve
#: only) keeps each fold's host test rows of X and of the stored y and hands them
#: to that fold's later units. The same rows, so the same scores: no bit moves.
#: Why an arm and not the default: a member that transforms its test X in place
#: would be handed its own written rows again (as the serial loop before the
#: units did), and units on several threads read one array; that is to be
#: measured and reviewed with the A/B, not argued.
#: DEFAULT since 2026-11-20; its `_OFF` old path (a gather once per unit) was deleted 2026-11-23.
#: A/B round 3, 2026-11-20 (lane fx-verdicts5; NVIDIA L40S host 2ef132a6de04, wheel 28a780604c and its B set, base and
#: arm back to back in one pod job, tag nvc-ab-20261120T063324Z-c630ad, workers bound to the GPU's NUMA node 0; one
#: scored run per arm; scores, quality and digests equal). Base -> arm ms:
#: learning-curve istella (1,000,000 x 220) 17,979 -> 15,332 (0.853), taxi (1,000,000 x 11) 700.4 -> 669.6 (0.956).
#: PROMOTED: faster, the same scores. General: the saving is (sizes - 1) x folds test gathers at any shape. The
#: in-place-transform case above is the pre-units behaviour restored and is not a board row; no member of the
#: board's two rows writes its test rows (digests equal).
#: Lane mf-r9b (2026-11-24): the units and this fold-set cache are deleted; every learning_curve is the typed
#: search RUN, whose `_same_rows` hands a fold's one gathered test block to each of its sizes (the same reuse).
# TOMBSTONE: MOJOLEARN_MSEL_FAST_LC_TEST_REUSE_OFF (decided: winner promoted, A/B round 3) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_MSEL_FAST_LC_TEST_REUSE_OFF.patch; record in docs/TOMBSTONES.md.

#: Units over native members (lane ma-core, 2026-11-21) were deleted by lane mf-msel5c (2026-11-23): a member
#: whose family published its member entry is fitted and scored by the search kind's typed RUN
#: (core/msel_search_run.mojo); the Python units that served what that kind did not take are deleted too
#: (lane mf-r9b: the kind refuses those in their words).
# TOMBSTONE: MOJOLEARN_MSEL_MEMBER_UNITS_OFF (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_MSEL_MEMBER_UNITS_OFF.patch; record in docs/TOMBSTONES.md.


struct MselFoldRows(Movable):
    """`_FoldRows`: X and y resident for the calls that share them (lane
    mf-r9b: the cv tools' folds are the typed search RUN's; what is left is
    the resident store and scope a caller reads rows out of: the
    permutation test's permuted y and X scope, cross_val_predict's label
    placement, the process workers' fold rows)."""
    var x: PythonObject
    var y: PythonObject
    var folds: PythonObject
    var res: PythonObject
    var resident: ArenaResidentPlan
    var dev: Bool
    var scope: PythonObject
    # S4 (lane s1): the admitted-registry hold scope of the set. A host
    # float32 C-order X is held for the whole set: every member door, in any
    # binding (one registry per process), reads it as admitted, with no view
    # or check of its own. -1: no scope open.
    # Lane s2: X is held with its finite test DEFERRED, not scanned here (the
    # scan was one host pass over X per CV call, also on a GPU install whose
    # members test the words in their kernel entries); the first member that
    # needs a host verdict scans X once (bindings/admitted_meta_scope.mojo
    # admitted_held_verify).
    var scope_mark: Int
    var x_held: Bool
    # REMOVED (lane mf-r9b, with the units `msel_cv_units`): the per-fold
    # take (`take`, `xrows`, `yrows`, the MSEL_LC_TEST_REUSE test-row cache
    # `_take_reusing_test` / `reuse_test`, the DeviceRows cache, the per-fold
    # hold scope `_hold_fold` and the units' `shared` flag). The typed
    # search RUN takes the fold rows (its `_same_rows` reuses a fold's test
    # rows across sizes, as reuse_test did).

    def __init__(out self, x: PythonObject, y: PythonObject, folds: PythonObject, device_rows: Bool,
                 context: PythonObject) raises:
        self.x = x
        self.y = y
        self.folds = folds
        self.res = PythonObject(None)
        self.resident = ArenaResidentPlan(context["family_msel_resident_context"])
        self.dev = (device_rows and msel_is_array(x, context) and String(py=x.dtype) == "<f4" and Int(py=x.ndim) == 2
                    and Int(py=x.size) > 0 and msel_has_c(x))
        self.scope = PythonObject(None)
        if self.dev:
            self.res = msel_resident_new(context)
            # only DeviceRows bases enter this scope (`rows_slot` ignores the
            # size floor); no other program input is kept across folds
            self.scope = msel_rows_scope_open(context)
        self.scope_mark = -1
        self.x_held = False
        if not self.dev and msel_is_array(x, context):
            self.scope_mark = admitted_scope_begin()
            try:
                self.x_held = admitted_hold_array_f32_deferred(x)
            except error:
                admitted_scope_end(self.scope_mark)
                self.scope_mark = -1
                raise error^

    def _end_holds(mut self):
        if self.scope_mark >= 0:
            try:
                admitted_scope_end(self.scope_mark)
            except ignored:
                _ = ignored^
            self.scope_mark = -1

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

    def close(mut self, context: PythonObject) raises:
        self._end_holds()
        var scope = self.scope
        self.scope = PythonObject(None)
        try:
            if scope is not None:
                msel_rows_scope_close(scope, context)
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
        if len(items) == 0:
            # scikit-learn's `_check_multimetric_scoring` refusal (lane w3-fix-b
            # row 14; the search kind's words, core/msel_refusal_words.mojo)
            raise Error(msel_empty_scoring_words(False, String(py=py_call(b.repr, scoring))))
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
        if len(items) == 0:
            raise Error(msel_empty_scoring_words(True, String(py=py_call(b.repr, scoring))))
        return out
    return PythonObject(None)


# ---------------------------------------------------------------- units
#
# REMOVED (lane mf-r9b, 2026-11-24): the Python units of a CV operation
# (`msel_cv_units`, `MselUnitRun`, its thread `_msel_units_thread`, the
# device-bytes worker bound `msel_unit_workers` / `_msel_device_free` /
# MSEL_UNITS_FREE_DIVISOR, `msel_cross_validate_run`, `MselScores`,
# `msel_scores_dict` and `msel_error_value`; recoverable at ea2e0c0139).
# Every search and cv tool is the search kind's typed RUN
# (core/msel_search_run.mojo, `msel_cv_by_search` / the search door), whose
# folds run as units across the host cores; what it does not fit is refused
# natively in its words (bindings/msel_member_units_door.mojo THE REFUSAL
# RULE), so nothing reached the units any more.
# TOMBSTONE: MOJOLEARN_MSEL_UNITS_SERIAL (retired: redundant) deleted 2026-11-09 by integ-w6 (integ/pre-0867-w3); code recoverable at lane/p3-msel-parallel-and-process-pool@9d7f5a2af9.
# Restore: none (never reached main); the run-time rule native_unit_device_workers
# (bindings/native_unit_threads.mojo) runs GPU units one at a time, which is this
# switch's form on NVIDIA, AMD and Apple; the CPU is never timed.
# TOMBSTONE: MOJOLEARN_MSEL_MEMBER_UNITS_OFF (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.


def msel_unit_jobs(n_jobs: PythonObject, error_score: PythonObject, caller: String,
                   context: PythonObject) raises -> Int:
    """The thread bound of a CV operation from its `n_jobs` (joblib's
    meaning): a positive int is the bound; -1 all cores, -k all cores but
    k - 1; None all cores (the default is parallel: AGENTS.md "parallelize
    tests and experiments by default"; the result does not depend on it).
    Also checks `error_score` before any fold or fit, in scikit-learn's
    parameter validation words (`InvalidParameterError` of the constraints
    [StrOptions({'raise'}), Real], sklearn/utils/_param_validation.py; lane
    mf-msel5c: an error_score string other than 'raise' is refused here for
    every caller, so no unit route sees one)."""
    msel_admit(n_jobs, context)
    msel_admit(error_score, context)
    if not (py_type(error_score) is msel_b().str and msel_cmp("eq", error_score, PythonObject("raise"))) and not msel_is_real(error_score, context):
        raise Error("ValueError: The 'error_score' parameter of " + caller + " must be a str among {'raise'} or an "
                    + "instance of 'float'. Got " + family_repr(error_score) + " instead.")
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


def _tool_names(scoring: PythonObject, context: PythonObject) raises -> List[String]:
    """The cv tools' scorer names in the door's order (`msel_scorers`: a
    list, tuple, set or mapping keeps one of each, in order); one scorer:
    ["score"], as the search kind's PACK names one scorer."""
    var names = List[String]()
    var many = msel_scorers(scoring, context)
    if many is None:
        names.append(String("score"))
        return names^
    var given = py_call(msel_b().list, many)
    for i in range(len(given)):  # driver-loop: over the scorer names (once per call)
        names.append(String(py=given[i]))
    return names^


def msel_cv_by_search(estimator: PythonObject, x_in: PythonObject, y_in: PythonObject, folds: PythonObject,
                      scoring: PythonObject, n_jobs: PythonObject, error_score: PythonObject, return_train: Bool,
                      keep: Bool, method: PythonObject, indices: Bool, context: PythonObject) raises -> PythonObject:
    """THE CV TOOLS through THE ONE TYPED SEARCH RUN (lane mf-msel5;
    core/msel_search_run.mojo module note). UNPACK: a carrier GridSearchCV
    over ONE candidate ({}: the estimator's own parameters) with the tool's
    scoring, n_jobs, error_score, return_train_score and refit=False, its
    instance field `_msel_cv_tool` = (keep, method, indices, `folds`: the
    door's own checked folds), fitted in place by the search kind
    (`msel_member_fit_in_place`; the search's own resident scope opened as
    `msel_search_fit` opens it). PACK: a dict with `names` (the scorer
    names), `test` / `train` (per name, the folds' scores), `fit_time`,
    `score_time` and, as asked, `estimator`, `indices`, `predictions` (lane
    mf-msel5b: the result the RUN wrote at every row, `placed`; or a
    classifier's decoded labels, `placed` False, which the caller places).
    y=None (lane mf-msel5b): an unsupervised base, fitted without a target
    and scored by its own score. Never None (lane mf-r9b): a base, scorer
    or data the kind does not fit is refused natively before any fit, in
    the words of whoever refuses it (bindings/msel_member_units_door.mojo
    THE REFUSAL RULE), so every cv tool is this RUN and no unit route is
    left beside it. The callers hand X and y their own door already
    materialized (`msel_cv_folds`, `msel_prepare_folds`) and at least one
    fold (`msel_checked_folds`; learning_curve refuses empty train sizes
    first): the reads below raise their own words, never a fallback."""
    if len(folds) == 0:
        raise Error("RuntimeError: cv tool invariant: the door handed the search kind no fold"
                    + " (msel_checked_folds refuses an empty split)")
    var b = msel_b()
    var ya = PythonObject(None)
    # A TEXT 2-D y (lane mf-tail5b) is no Array: the kind's UNPACK reads its
    # rows once, per output (bindings/msel_member_units_door.mojo).
    var text_y = y_in is not None and msel_text_rows(y_in, context, b)
    var xa = msel_materialize(x_in, "X", context)
    if text_y:
        ya = y_in
    elif y_in is not None:
        ya = msel_materialize_y(y_in, context)
    # THE RESULT BUFFER's shape words (lane mf-msel5b): the target's outputs
    # (0: a 1-D y) and the rows; the kind's UNPACK places the Array in `holder`.
    var outputs = 0
    if text_y:
        outputs = len(py_call(b.list, py_call(b.list, y_in)[0]))
    elif ya is not None and msel_is_array(ya, context) and Int(py=ya.ndim) == 2:
        outputs = Int(py=ya.shape[1])
    var holder = Python.list(PythonObject(None))
    var names = _tool_names(scoring, context)
    var grid = Python.dict()
    var keywords = Python.dict()
    py_setitem(keywords, "scoring", scoring)
    py_setitem(keywords, "n_jobs", n_jobs)
    py_setitem(keywords, "error_score", error_score)
    py_setitem(keywords, "refit", PythonObject(False))
    py_setitem(keywords, "return_train_score", PythonObject(return_train))
    py_setitem(keywords, "estimator", estimator)
    py_setitem(keywords, "param_grid", grid)
    var search = msel_carrier_search(msel_cls("GridSearchCV", context), keywords, context)
    var fields = msel_search_fields(search, context)
    py_setitem(fields, "_msel_cv_tool", Python.tuple(PythonObject(keep), method, PythonObject(indices), folds,
                                                     holder, PythonObject(outputs), xa.shape[0]))
    # THE RESIDENT FOLD ROWS, as `msel_search_fit` reads them (the member's
    # own device; the carrier names no `device`).
    # (lane mf-tail5) Inside an enclosing scope of the same kind (the
    # permutation test's resident fold rows, `MselFoldRows` with `dev`, opened
    # by `msel_rows_scope_open` too) the RUN reads THAT scope: X stays
    # resident once across every permutation's carrier, no second scope.
    var scope = PythonObject(None)
    if (msel_device_rows_ok(estimator, scoring, context) and not msel_member_host_folds(fields, b, context)
            and member_scope_word(context["family_msel_rows_context"]) == MEMBER_SCOPE_NONE):
        scope = msel_rows_scope_open(context)
    try:
        msel_member_fit_in_place(search, ya, xa, context, b)  # raises every refusal (THE REFUSAL RULE)
    finally:
        if scope is not None:
            msel_rows_scope_close(scope, context)
    var state = msel_search_fields(search, context)
    var results = py_get(state, "cv_results_")
    var tool = py_get(state, "_msel_cv_result")
    var splits = py_get(state, "n_splits_")
    if results is None or tool is None or splits is None:
        raise Error("RuntimeError: cv tool invariant: the search kind's PACK wrote no cv_results_, n_splits_"
                    + " or _msel_cv_result")
    var k = Int(py=splits)
    var out = Python.dict()
    var name_list = Python.list()
    var tests = Python.list()
    var trains = Python.list()
    var many = len(names) != 1 or names[0] != "score" or msel_scorers(scoring, context) is not None
    for j in range(len(names)):  # driver-loop: over the scorer names (their fold columns)
        var suffix = String("_") + names[j] if many else String("")
        var te = List[Float64](capacity=k)
        var tr = List[Float64](capacity=k)
        for i in range(k):  # driver-loop: over the folds (one cv_results_ column each, one candidate)
            te.append(Float64(py=results["split" + String(i) + "_test_score" + suffix][0]))
            if return_train:
                tr.append(Float64(py=results["split" + String(i) + "_train_score" + suffix][0]))
        _ = py_call(name_list.append, PythonObject(names[j]))
        _ = py_call(tests.append, msel_f64_list(te, context))
        _ = py_call(trains.append, msel_f64_list(tr, context))
    py_setitem(out, "names", name_list)
    py_setitem(out, "test", tests)
    py_setitem(out, "train", trains)
    var keys: List[String] = ["fit_time", "score_time", "estimator", "indices", "predictions", "placed"]
    for i in range(len(keys)):  # small-loop(the tool's result keys)
        var value = py_get(tool, keys[i])
        if value is not None:
            py_setitem(out, keys[i], value)
    return out


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


def _tool_scores_dict(by: PythonObject, return_train: Bool, return_estimator: Bool, return_indices: Bool,
                      context: PythonObject) raises -> PythonObject:
    """The `cross_validate` result dict from `msel_cv_by_search`'s PACK,
    keys in `msel_scores_dict`'s order."""
    var res = Python.dict()
    py_setitem(res, "fit_time", by["fit_time"])
    py_setitem(res, "score_time", by["score_time"])
    var names = by["names"]
    for k in range(len(names)):  # driver-loop: over scorer names
        var name = String(py=names[k])
        py_setitem(res, PythonObject("test_" + name), by["test"][k])
        if return_train:
            py_setitem(res, PythonObject("train_" + name), by["train"][k])
    if return_estimator:
        py_setitem(res, "estimator", by["estimator"])
    if return_indices:
        py_setitem(res, "indices", by["indices"])
    return res


def msel_cross_validate(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cross_validate(estimator, X, y, groups, scoring, cv, n_jobs,
    return_train_score, return_estimator, return_indices, error_score)`."""
    var estimator = args[0]
    var error_score = args[10]
    _ = msel_unit_jobs(args[6], error_score, "cross_validate", context)  # n_jobs and error_score words, first
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
    # THE ONE TYPED SEARCH RUN (lane mf-msel5, `msel_cv_by_search`): the
    # folds above fitted and scored as one candidate's units, with the
    # units' times, kept estimators and rows. Lane mf-r9b: the only route;
    # what the kind does not fit is refused there in its words (the units
    # route that ran beside it is gone).
    var by = msel_cv_by_search(estimator, x, y, folds, scoring, args[6], error_score, rt, re, PythonObject(None), ri,
                               context)
    return _tool_scores_dict(by, rt, re, ri, context)


# ---------------------------------------------------------------- cross_val_score


def msel_cross_val_score(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cross_val_score(estimator, X, y, cv, scoring, groups, n_jobs, error_score)`."""
    var estimator = args[0]
    var cv = args[3]
    var groups = args[5]
    # n_jobs words, first; then error_score's ('raise' or a real number,
    # scikit-learn's words; lane w3f3-b: a number is served below)
    _ = msel_unit_jobs(args[6], args[7], "cross_val_score", context)
    var scoring = args[4]
    if py_type(scoring) is msel_b().str:
        scoring = msel_get_scorer(scoring, context)
    # error_score is checked above and handed to the RUN (lane w3f3-b: a
    # number scores a failing fit or score that value and warns as
    # scikit-learn's `_score` / `_warn_or_raise_about_fit_failures` do, at
    # the search PACK); the folds' preparation is the parallel route's too,
    # which keeps its 'raise'-only rule
    var got = msel_prepare_folds(estimator, args[1], args[2], cv, scoring, groups, PythonObject("raise"), context)
    var x = got[0]
    var y = got[1]
    var folds = got[2]
    scoring = got[3]
    # THE ONE TYPED SEARCH RUN (lane mf-msel5, `msel_cv_by_search`). The
    # FAST + Apple slice route's row-range views are the RUN's own views of a
    # side that is one run of rows (lane mf-msel5b; TOMBSTONE above). Lane
    # mf-r9b: the only route (its refusals raise there; no units beside it).
    var by = msel_cv_by_search(estimator, x, y, folds, scoring, args[6], args[7], False, False,
                               PythonObject(None), False, context)
    return by["test"][0]


# ---------------------------------------------------------------- cross_val_predict

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
        # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
        var grouped = Optional[PythonObject](py_call(packed._mv.tolist))
        for k in range(rank - 1):  # driver-loop: native rank metadata groups flat values into rows
            var axis = rank - 1 - k
            var step = Int(py=packed.shape[axis])
            var count = 1
            for a in range(axis):  # driver-loop: outer extents also preserve empty dimensions
                count *= Int(py=packed.shape[a])
            var groups = Python.list()
            for r in range(count):  # driver-loop: native list slices preserve the Array.tolist nesting
                var part = py_slice(grouped.value(), r * step, (r + 1) * step)
                _ = py_call(groups.append, part)
            grouped = Optional[PythonObject](groups)
        values = grouped.value()
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
            var key = msel_helpers_key(context)
            var block = msel_store_alloc(key, n * cols * 8)
            try:
                var dst_code = Int(py=codes[PythonObject(dtype)])
                for f in range(count):  # driver-loop: over folds (one device scatter each)
                    var t = tests[f]
                    if Int(py=t.size) > 0:
                        var src = preds[f]
                        if not msel_has_c(src):
                            src = msel_materialize(src, "predictions", context)
                        var sid = resident_rows.resident_store(t, True, context)
                        _ = msel_scatter_rows(key, block, n, dst_code, msel_addr(src), Int(py=codes[src.dtype]),
                                              Int(py=t.size), cols, sid)
                        _ = src  # keep the Python object alive through the native call above
                _ = msel_store_read(key, block, msel_addr(out), n * cols * 8)
                _ = out  # keep the Python object alive through the native call above
            finally:
                _ = msel_store_free(key, block)
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
        # TEXT MULTI-OUTPUT LABELS (lane mf-tail5b; was a float conversion
        # of every label, which failed on the first string): rows of labels
        # that are not all real numbers stay the row lists, each row its
        # outputs' labels in order, as a 1-D text result stays its list.
        var all_real = True
        for k in range(size):  # driver-loop: exact scalar identity checks before numeric conversion
            if not msel_is_real(flat[k], context):
                all_real = False
                break
        if not all_real:
            return rows
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
    # scikit-learn validates `method` before anything else (its
    # `validate_params` StrOptions; lane w3-fix-b row 14): one exact-str read.
    var method = args[6]
    if py_type(method) is not msel_b().str:
        raise Error("TypeError: native prediction method must be an exact string")
    var asked = String(py=method)
    if (asked != "predict" and asked != "predict_proba" and asked != "predict_log_proba"
            and asked != "decision_function"):
        raise Error(msel_cvp_method_words(String(py=py_call(msel_b().repr, method))))
    _ = msel_unit_jobs(args[5], PythonObject("raise"), "cross_val_predict", context)  # n_jobs words, first
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
    for f in range(len(folds)):  # driver-loop: over folds (one copy each)
        var t = folds[f][1]
        var size = Int(py=t.size)
        if size > 0:
            unsafe_memcpy(dest=msel_p64(held) + at, src=msel_p64(t), count=size)
            _ = t  # keep the Python object alive through the native call above
            _ = held  # keep the Python object alive through the native call above
        at += size
    if base_check_indices_i64(base_helpers_key(context["family_msel_entries"]), msel_addr(held), n, n) != 0:
        raise Error("cross_val_predict only works for partitions")
    _ = held  # keep the Python object alive through the native calls above
    # THE ONE TYPED SEARCH RUN (lane mf-msel5, `msel_cv_by_search`): each
    # fold's test-side `method` answered by its unit. Lane mf-r9b: the only
    # route (its refusals raise there; no units beside it).
    var by = msel_cv_by_search(estimator, x, y, folds, PythonObject(None), args[5], PythonObject("raise"), False,
                               False, method, False, context)
    # THE RESULT BUFFER (lane mf-msel5b): the RUN wrote every unit's
    # response at its rows; nothing is placed again. A classifier's
    # labels were decoded once for every row: an int64 / float64 Array
    # of them is the result; other labels take the label placement over
    # the identity rows (the old route's result types).
    var pred = by["predictions"]
    if Bool(py=py_call(msel_b().bool, by["placed"])):
        return pred
    if msel_is_array(pred, context) and Int(py=pred.ndim) == 1 and (String(py=pred.dtype) == "<i8"
                                                                      or String(py=pred.dtype) == "<f8"):
        return pred
    var ident = msel_new(n, "<i8", context)
    var words = msel_p64(ident)
    for i in range(n):  # driver-loop: over the rows (the identity placement of decoded labels)
        words[i] = Int64(i)
    _ = ident  # written through the pointer above
    var host_rows = MselFoldRows(x, y, folds, False, context)
    var placed = PythonObject(None)
    try:
        placed = _place(Python.list(pred), Python.list(ident), n, host_rows, context)
    finally:
        host_rows.close(context)
    return placed


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
