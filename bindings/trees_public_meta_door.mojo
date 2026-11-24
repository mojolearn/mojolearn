# SPDX-License-Identifier: Apache-2.0
# The meta-estimators' public operations in one native call each:
# VotingClassifier/VotingRegressor, StackingClassifier/StackingRegressor,
# MultiOutputRegressor/MultiOutputClassifier, OneVsRestClassifier and
# CalibratedClassifierCV (python/mojolearn/_expansion_trees.py). This is the
# Python orchestration those methods ran, step for step: the parameter checks
# with their wording, the member clones, the cv folds, the x_trees glue
# kernels in the same order with the same arguments, and the fitted state.
#
# Members are exact built-in mojolearn estimators with connected ABI plans.
# Clone, fit and prediction call those compiled operations directly; Python
# methods, constructors, property hooks and custom splitters never execute.
#
# Round trips removed against the Python path: X is converted to float32
# once per operation and handed to every member; the fold rows are copied
# out of the one device fold buffer as int32 Arrays (no memoryview list).
#
# CalibratedClassifierCV(GaussianNB) no longer takes the x_prep one-program
# fit (MOJOLEARN_CALIB_GNB_FOLDS, FAST + Apple only): that program was
# assembled by the prep family's Python program builder, so the native
# driver takes the reference route (one GaussianNB fit per fold through its
# native entry, the same calibrators). IDENTICAL never took it; the x_prep
# kernels stay in place for the prep family.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr, py_fields
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.py_attr import py_attr
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_wrap_array, _store_code
from bindings.py_admitted_registry import (
    admitted_scope_begin, admitted_scope_end, admitted_scope_open, admitted_hold, admitted_hold_deferred,
    admitted_hold_rows_of, admitted_is_held,
)
from bindings.py_admitted_registry import admitted_held_index
from bindings.admitted_meta_scope_unpack import admitted_held_verify
from bindings.trees_public_inputs_run import trees_finite_f32
from core.admitted_refusals import admitted_finite_refuse
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.family_common_pack import family_call
from bindings.trees_public_inputs_door import (
    TreesLabels,
    trees_builtins,
    trees_table,
    trees_reshape,
    trees_address,
    trees_itemsize,
    trees_encode_labels,
    trees_decode_labels,
    trees_cast,
)
from bindings.admitted_input_run import AdmittedXView
from bindings.admitted_input_unpack import (
    AdmittedX, admit_f32, admit_weight_words, admit_ids, admitted_words, admitted_array, admitted_vector,
    admit_c_view, admitted_dtype, admitted_unfilled,
)
from bindings.classical_arrays_door import classical_address
# TOMBSTONE: TreesView / _view_of_array / trees_view / trees_check_rank / trees_f32 / trees_check_sample_weight / trees_all_finite / trees_vector / trees_empty uses and _place_view (the trees admission aliases; internal Arrays are read as themselves) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from xtrees.trees_meta_run import CalibratedProgram, meta_binary_column_check
from xtrees.trees_forest_run import trees_check_features
from bindings.trees_public_forest_door import (
    _check_mode,
    _repr,
    _str,
    _truthy,
    _is_bool,
    _is_integral,
    forest_kind,
)
from xtrees.trees_forest_run import (
    _refuse,
    _classifier,
)
from bindings.trees_public_boost_door import (
    _mode, _xt, _xr, _x_f32, _as_array, _arange, _class_major, _gather,
    _argmax, _member_codes,
)
from bindings.trees_public_score_door import trees_r2, trees_class_codes
from bindings.classical_labels_door import labels_accuracy, _classification_encoded as score_classification_encoded
from bindings.member_set_door import (
    MemberSet, member_set_admit_or_raise, member_set_own_block, member_set_run, member_set_pack, member_set_release,
    member_set_notices, member_set_reads_given, member_set_given,
    member_set_resident, member_set_pins_take, member_set_pins_return,
)
from bindings.member_given_y_unpack import member_given_y
from core.member_unit_classes import member_unit_classes, member_unit_recode
from core.member_abi import (
    MemberMatrix, MEMBER_F64, MEMBER_I64,
    MEMBER_F32, MEMBER_I32, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION,
    MEMBER_CAN_DECISION, MEMBER_CAN_PREDICT_PROBA, member_kind_capabilities,
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE,
)
from bindings.member_scope_door import member_pin_take, member_pin_return, member_scope_word
from core.member_set_run import MEMBER_SET_TURN_HELD, member_set_first_failure
from core.member_families import MEMBER_FAMILY_GP, MEMBER_FAMILY_ENSEMBLE
from core.admitted_rows_plan import admitted_rows_shape
from xtrees.trees_calibrated_fit_run import (
    CalFitPlan, CalFitOut, CalPlace, CAL_ALL_ANSWERED, CAL_FIT_FAILED, calibrated_fit_run, cal_fold_place,
    cal_lacking_warning, cal_cv_check,
)
from bindings.msel_split_spec_unpack import msel_cv_fold_words
from std.memory import unsafe_memcpy
from bindings.member_set_door import (
    member_set_tests, member_held_open, member_held_run, member_held_close, MemberHeldCall,
)
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, VOTE_ALL_ANSWERED, vote_soft_run, vote_codes_run, vote_regress_run, vote_transform_run,
    ovr_assemble_run, multioutput_codes_run, multioutput_proba_run, stack_place_run, STACK_PREDICT, STACK_PROBA,
    STACK_DECISION,
    # nn-integrate (compile fix): lane bm-meta-3's calibrated predict RUN names, used below and never imported
    CalPair, CalFit, calibrated_proba_run,
)
from bindings.prep_native_estimator_door import (
    prep_estimator_admit, prep_estimator_clone, prep_estimator_call, prep_estimator_has,
    prep_estimator_plan, prep_native_context, prep_native_split, prep_estimator_predict_internal,
)
from bindings.classical_labels_door import _label_list, classical_score_from_codes
from bindings.member_owner_unpack import member_mark
from bindings.trees_meta_device_door import meta_members_device, meta_member_device_set, meta_device_origin
from providers.python.member_xneighbors_boundary_door import xneighbors_kind_reads_marker, xneighbors_marker_index
from core.xneighbors_member import LP_MARKER_WORD
from core.ensemble_member import META_CAL_BINARY_MARKER_WORDS
from bindings.p3_labels_metrics_native_door import p3_interleave_columns
from bindings.metrics_cls_native_unpack import metrics_cls_operation
from bindings.operation_names_mcls import MCLS_OPN_FAMILY_MCLS_R2
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.classical_pandas_door import native_pandas_exact
from bindings.native_unit_threads_unpack import native_unit_jobs, native_unit_device_workers
from core.host_parallel import host_parallelize
from bindings.device_rows_door import DoorRows, door_is_device_rows
from core.meta_fold_rows import meta_rows_index, meta_rows_compose, meta_rows_take_u32
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from std.python._cpython import GILReleased
from std.ffi import c_int, external_call
from bindings.native_array_storage_door import native_array_from_values
from bindings.msel_inputs_unpack import msel_admit, msel_input_count
from bindings.msel_common_door import msel_warn
from bindings.trees_meta_admission_door import meta_admit_configuration
from bindings.trees_staged_input_door import trees_stage_members, trees_stage_wanted, trees_stage_tables, trees_unstage
from bindings.trees_side_records_unpack import trees_side_get, trees_side_put
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.base_helpers_released_unpack import (
    base_check_indices_i64, base_bincount_i64, base_reduce_stat_int, base_equal_elements,
)
from core.linear_entry_table import LinearEntryKey
from core.entry_table import EntryRef
from core.linear_multi_calls import ridge_multi_ref, ridge_multi_available, rm_predict
from bindings.trees_meta_ridge_run import RidgeMultiFit, ridge_multi_fit_run
from bindings.py_optional import py_or_none
from xtrees.glue_calls import (
    xt_accumulate, xt_accumulate_cols, xt_scatter, xt_device_folds, xt_transpose_f64,
    xt_stack_w64, xt_rows_any_unset, xt_indicator_codes, xt_binary_proba, xt_normalize_rows,
    xt_platt_apply_strided, xt_isotonic_predict_strided, xt_complement_pairs,
    xt_ovr_assemble, xt_calibrated_proba, xt_scale,
)
from bindings.trees_keys_unpack import trees_keys
# Lane mf-refuse5: the constructor refusals, ONE body with the META kind's admission.
from bindings.trees_meta_validate_unpack import (
    meta_jobs as _jobs,
    meta_cv_value as _cv_value,
    meta_stacking_validate as _stacking_validate,
    meta_calibrated_validate as _calibrated_validate,
    meta_voting_validate,
    meta_voting_weights,
    MetaActive as Active,
    meta_active as _active,
)
from bindings.py_balanced import py_item


comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
# TOMBSTONE: MOJOLEARN_CALIB_FAST_HOST_FOLDS (NOISE: NV L40S 4 cells, median 1.00, 0.97..1.04) deleted 2026-11-24 by lane/ab-outcome-3 (o3-trees); code recoverable at c23cb55fda.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_CALIB_FAST_HOST_FOLDS.ab-outcome-3.patch
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


# ------------------------------------------------------------- utilities


comptime _META_KIND_SLOT = "trees_meta_kind_v1"


# ------------------------------------------------------------ member calls
# THE member call of the meta door (lane dw-trees-boost, 2026-11-21). A member
# of a meta-estimator is an estimator of any connected family: its fit and
# its predictions are that family's public operations, reached through the
# preprocessing door's estimator call. Every such call of this file passes
# through `meta_member_call` (21 sites: 17 method calls, 4 internal float32
# predicts), so the one shared member interface (core/, the shared-files
# lane: a member's device, its run context and its typed operations) replaces
# these three bodies and nothing else here. Until it exists a member runs
# where ITS OWN stored `device` says, which is why no meta class carries the
# `_mojolearn_device` marker: a device="cpu" meta-estimator must not report
# cpu while its members run on the GPU.
# Nothing is added to a call: each overload is the former call expression.


@always_inline
def meta_member_call(member: PythonObject, method: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`member.<method>(*args)` of a connected estimator."""
    return prep_estimator_call(member, method, args, context)


@always_inline
def meta_member_call(member: PythonObject, method: String, args: PythonObject, context: PythonObject,
                     keywords: PythonObject) raises -> PythonObject:
    """`member.<method>(*args, **keywords)` (the member fits)."""
    return prep_estimator_call(member, method, args, context, keywords)


@always_inline
def meta_member_call(member: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A regressor member's internal float32 predict
    (`prep_estimator_predict_internal`)."""
    return prep_estimator_predict_internal(member, args, context)


def meta_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The connected meta schema of the model's class (or an ancestor).
    Once per class (lane ne2-b-members, N072; the forest_kind scheme): every
    public call walked the class ancestry with one table probe per ancestor.
    The record, kept weakly on the class in the native side table, holds the
    connected table, the class's `__mro__` tuple and the answer; a
    reconnection (a new table) or a `__bases__` assignment (a new `__mro__`
    tuple) walks again."""
    var b = trees_builtins()
    var nk = trees_keys()
    var table = py_item(context, nk[].k_trees_meta_classes)
    var cls = py_type(model)
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    var held = trees_side_get(_META_KIND_SLOT, cls)
    if held is not None and held[0] is table and held[1] is mro:
        return String(held[2])
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        var kind = py_get(table, mro[i])
        if kind is not None:
            _ = trees_side_put(_META_KIND_SLOT, cls, Python.tuple(table, mro, kind))
            return String(kind)
    raise Error("TypeError: the native meta-estimator owner requires a connected meta-estimator class")


def _attr(owner: PythonObject, key: PythonObject, default: PythonObject) raises -> PythonObject:
    """`getattr(owner, name, default)` with the held key of the name
    (`nk[].k_<name>`; lane bw3-trees: a str was built from the name per call)."""
    return py_call(trees_builtins().getattr, owner, key, default)


def _has(owner: PythonObject, key: PythonObject) raises -> Bool:
    """`hasattr(owner, name)` with the held key of the name."""
    return Bool(py=py_call(trees_builtins().hasattr, owner, key))


def _call_kw(callable: PythonObject, args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
    """`callable(*args, **kwargs)` of a Python tuple and dict, borrowing both."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(callable._obj_ptr, args._obj_ptr, kwargs._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def _owner(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def _copy_block(array: PythonObject, start: Int, stop: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    """`array[start:stop]` of a flat Array as a new Array: a block copy in C."""
    var b = trees_builtins()
    var item = trees_itemsize(typestr)
    var view = py_call(b.memoryview, array._store)
    var raw = py_call(py_attr(py_call(view.cast, "B"), "__getitem__"), py_call(b.slice, PythonObject(start * item), PythonObject(stop * item)))
    var nk = trees_keys()
    var store = py_call(py_item(context, nk[].k_array_type), PythonObject(_store_code(typestr)))
    _ = py_call(store.frombytes, py_call(raw.tobytes))
    return classical_wrap_array(store, Python.tuple(stop - start), typestr, "C", context)


def _f64_word(array: PythonObject, index: Int) raises -> Float64:
    return _F64P(unsafe_from_address=trees_address(array))[index]


def _x_array(X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`as_f32_c(X, ndim=2, name="X")` as the Array the members read. Inside
    the operation's hold scope (meta_admitted_operation) the matrix is held
    once with its finite test deferred (bindings/py_admitted_registry.mojo
    `admitted_hold_deferred`): every member door reads it as admitted, with
    no second view or check; a door that tests in its kernel entry scans
    nothing, and the first door that needs a host verdict scans X once for
    the whole call (admitted_held_verify records it)."""
    if admitted_is_held(X):
        return X
    var m = _x_f32(X, context)
    # fb-boost-copy (2026-11-17): `_as_array` is now the zero-copy view this
    # door's `_x_view_array` (fb-infer/C, deleted) built; one helper for both doors.
    var Xa = _as_array(m, Python.tuple(m.rows, m.cols), context)
    _hold_deferred(Xa)
    return Xa


def _hold_deferred(Xa: PythonObject) raises:
    """Hold the float32 C-order Array `Xa` for the members with its finite
    test deferred (lane s2's form, as model selection holds its X).
    fb12-rows (2026-11-17): this scanned X on the host before every
    meta-estimator fit and prediction (`trees_finite_f32`, one pass over X
    per call that 0.8.55 never made) and held only a finite X; a member that
    tests the words it uploads (Ridge / LinearRegression predict,
    providers/python/linear_predict_boundary_unpack.mojo) then paid that
    pass for nothing. Now a deferring door reads X untested, a door that
    needs a host verdict proves it once for the whole call (and for every
    fold held as rows of X), and a non-finite X is refused by the first
    member door, in its own words, as before. The scan cost is per word, so
    the saving holds for every shape."""
    if not admitted_scope_open() or admitted_is_held(Xa):
        return
    var rows = _rows(Xa)
    var cols = _cols(Xa)
    if rows < 1 or cols < 1:
        return
    admitted_hold_deferred(Xa, trees_address(Xa), rows, cols, PythonObject(None))


def _rows(array: PythonObject) raises -> Int:
    return Int(py=array.shape[0])


def _cols(array: PythonObject) raises -> Int:
    return Int(py=array.shape[1])


def _i32_vec(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`as_i32_c(value, ndim=1, name=name)` of a member's predicted codes."""
    var view = admit_c_view(value, name, context)
    if Int(py=view.ndim) != 1:
        raise Error("mojolearn: " + name + " must be 1-D, got " + String(Int(py=view.ndim)) + "-D")
    var n = Int(py=view.shape[0])
    var out = admitted_vector(n, "<i4", context)
    trees_cast(Int(py=classical_address(view)), admitted_dtype(view, context), trees_address(out), "<i4", n, context)
    _ = out  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native call above
    _ = view^
    return out


def _f32_array(value: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    """`as_f32_c(value, ndim=ndim, name=name)` as an Array."""
    var m = admit_f32(value, ndim, name, context, finite=False)
    var shape = Python.tuple(m.rows, m.cols) if ndim == 2 else Python.tuple(m.rows)
    return _as_array(m, shape, context)


def _out2d(x: PythonObject, n: Int, context: PythonObject) raises -> AdmittedX:
    """`_trees_output_2d(x, n)`: a member's output as a float32 (n, c) block
    (a 1-D output is c = 1)."""
    var nk = trees_keys()
    var shape = _attr(x, nk[].k_shape, Python.tuple())
    if len(shape) == 2:
        return admit_f32(x, 2, "output", context, finite=False)
    var v = admit_f32(x, 1, "output", context, finite=False)
    if v.rows != n:
        raise Error("cannot reshape Array of size " + String(v.rows) + " into shape (" + String(n) + ", 1)")
    var _keep_ret_v = admitted_words(v.array, v.address, n, 1)
    _ = v  # keep the Python object alive through the native call above
    return _keep_ret_v^


def _int_list(values: PythonObject) raises -> PythonObject:
    return py_call(trees_builtins().list, py_call(trees_builtins().map, trees_builtins().int, values))


def _codes_array(var values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list([int(c) for c in values], "<i4")`."""
    var b = trees_builtins()
    # A fitted native classifier may expose ClassLabels. Read its verified
    # list storage instead of asking it for a buffer or invoking list hooks.
    values = classifier_label_list(values, context)
    var listed = values if py_type(values) is b.list or py_type(values) is b.tuple else _label_list(values, context)
    var nk = trees_keys()
    var store = py_call(py_item(context, nk[].k_array_type), PythonObject("i"), _int_list(listed))
    return classical_wrap_array(store, Python.tuple(len(store)), "<i4", "C", context)


comptime _MEMBER_CODES_SLOT = "trees_meta_member_codes_v1"


def _member_class_codes(member: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(classes_, listed classes, int32 code Array, index of 1 or -1,
    classes are 0..k-1)` of a
    fitted member, kept per member in the native side table (lane
    perf2-trees, 2026-11-06: every soft-vote, stacking, multi-output and
    calibrated predict rebuilt the int32 Array, one `int()` per class plus an
    `array('i')`, for every member on every call). The record is reused while
    the member's `classes_` is the recorded object and its listed labels are
    the recorded objects one by one (identity only), so a refit, a replaced
    or an in-place edited `classes_` rebuilds it. The Array is only read by
    the glue kernels (`x_trees_accumulate_cols`, the calibrated fusion), never
    written. Same values as before: no bits change."""
    var b = trees_builtins()
    var nk = trees_keys()
    var classes = _attr(member, nk[].k_classes_, PythonObject(None))
    var values = classifier_label_list(classes, context)
    var listed = values if py_type(values) is b.list or py_type(values) is b.tuple else _label_list(values, context)
    var held = trees_side_get(_MEMBER_CODES_SLOT, member)
    if held is not None and held[0] is classes and len(held[1]) == len(listed):
        var same = True
        for i in range(len(listed)):  # driver-loop: over the member's fitted classes (identity only)
            if not (held[1][i] is listed[i]):
                same = False
                break
        if same:
            return held
    var subs = _int_list(listed)
    var store = py_call(py_item(context, nk[].k_array_type), PythonObject("i"), subs)
    var array = classical_wrap_array(store, Python.tuple(len(store)), "<i4", "C", context)
    var at = -1
    for i in range(len(subs)):  # driver-loop: over the member's fitted classes
        if Int(py=subs[i]) == 1:
            at = i
            break
    # [4]: the classes are 0 .. k-1 in order (a member's code is the
    # ensemble's code).
    var identity = True
    for i in range(len(subs)):  # driver-loop: over the member's fitted classes
        if Int(py=subs[i]) != i:
            identity = False
            break
    var record = Python.tuple(classes, py_call(b.list, listed), array, PythonObject(at), PythonObject(identity))
    _ = trees_side_put(_MEMBER_CODES_SLOT, member, record)
    return record


def _sub_cols(member: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_sub_cols`: a member's classes (the ensemble's codes) as int32."""
    return _member_class_codes(member, context)[2]


def _index_of_one(member: PythonObject, context: PythonObject) raises -> Int:
    """`[int(c) for c in member.classes_].index(1)`, or -1 when 1 is absent."""
    return Int(py=_member_class_codes(member, context)[3])


def _one_col(xt: PythonObject, p: AdmittedXView, j: Int, context: PythonObject) raises -> PythonObject:
    """Column j of a float32 (n, c) block as an (n, 1) Array (`_gather`)."""
    var rows = _arange(xt, p.rows, context)
    var col = _codes_array(Python.list(PythonObject(j)), context)
    return _gather(xt, p.address, p.rows, p.cols, rows, col, context)


# ------------------------------------------------------------- x_trees glue


def _acc_cols_w(xt: PythonObject, acc: PythonObject, x: PythonObject, cols: PythonObject, n: Int, k: Int,
                weight: Float64, context: PythonObject) raises:
    var xa = admit_f32(x, 2, "sub-estimator output", context, finite=False)
    var glue = _xr(xt)
    _ = xt_accumulate_cols(glue, trees_address(acc), xa.address, trees_address(cols), n, k,
                           Int(py=cols.shape[0]), weight)
    _ = cols
    _ = xa  # keep the Python object alive through the native call above
    _ = xa^


def _acc_w(xt: PythonObject, acc: PythonObject, x: PythonObject, n: Int, weight: Float64,
           context: PythonObject) raises:
    var xa = admit_f32(x, 1, "sub-estimator output", context, finite=False)
    var glue = _xr(xt)
    _ = xt_accumulate(glue, trees_address(acc), xa.address, n, weight)
    _ = xa  # keep the Python object alive through the native call above
    _ = xa^


def _place(xt: PythonObject, dst: PythonObject, n_rows: Int, n_cols: Int, blk: AdmittedXView, rows: PythonObject,
           col0: Int) raises -> Int:
    """`_TreesWrapperBase._place`: the block's columns into dst (an Array)
    at col0, at the rows of `rows` (an int32 Array)."""
    var glue = _xr(xt)
    _ = xt_scatter(glue, trees_address(dst), blk.address, trees_address(rows), n_rows, n_cols,
                   Int(py=rows.shape[0]), blk.cols, col0)
    _ = dst
    _ = rows
    return blk.cols


def _zeros(count: Int, context: PythonObject) raises -> PythonObject:
    return admitted_vector(count, "<f8", context)


def _class_zero(xt: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_class_major([0.0], n)`: a zero-filled buffer (`trees_empty` fills
    zero bytes, +0.0), without the separate fill program it used to run
    (lane perf-sweep-trees, 2026-11-05). Same bits."""
    _ = xt
    return admitted_vector(n, "<f8", context)


# ---------------------------------------------------------------- members


def _require_member(est: PythonObject, owner: String, context: PythonObject) raises:
    prep_estimator_admit(est, owner, context)


def _clone(est: PythonObject, device: PythonObject, origin: String, context: PythonObject) raises -> PythonObject:
    """A member clone for one fit. `device` is the meta-estimator's stored
    word, read once per fit: None keeps the member's own device (today's
    route, one identity compare); an explicit word is stored on a clone that
    names none, with `origin`, the meta-estimator it came from
    (bindings/trees_meta_device_door.mojo; the members were admitted by
    `meta_members_device` first)."""
    var clone = prep_estimator_clone(est, context)
    if device is not None:
        meta_member_device_set(device, origin, clone)
    return clone


def _forest_classifier_member(member: PythonObject, context: PythonObject) raises -> Bool:
    """True when `member`'s class (or an ancestor) is a connected forest
    classifier (random forest, extra trees, decision tree): its predict is
    the forest owner's `forest_predict`, whose codes `forest_predict_codes`
    returns before the decode."""
    try:
        return _classifier(forest_kind(member, context))
    except:
        return False


def _member_label_codes(member: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """`as_i32_c(member.predict(Xa))` of a fitted member whose classes are
    the ensemble's int codes (a hard vote, a stacking final estimator, a
    binary OneVsRest member, a MultiOutput column). A forest member is asked
    for its codes (`forest_predict_codes` through trees_public_boost
    `_member_codes`, the AdaBoost member route): no label decode and int32
    re-cast per member per call (lane ne2-b-members, N075 / N076); the
    int32 values are the decoded labels' in every case, so no bits move.
    Every other member predicts through its compiled plan as before."""
    if _forest_classifier_member(member, context):
        return _member_codes(member, Xa, context)
    # A member of a published family: its codes from the state it holds
    # (`_member_native`: only when its classes are 0 .. k-1 in order, so its
    # code is the ensemble's). int32 codes are the answer; int64 ones are
    # narrowed once, as the door's labels were.
    var native = _member_native(member, Xa, MEMBER_OP_PREDICT, True, context)
    if native is not None:
        if String(native.dtype) == "<i4":
            return native
        return _i32_vec(native, "codes", context)
    return _i32_vec(meta_member_call(member, "predict", Python.tuple(Xa), context), "codes", context)


def _given_rows(view: MemberMatrix, rows_of: PythonObject, column: Int, unit_n: Int) raises -> List[Float64]:
    """THE GIVEN y of one unit (lane mf-tail5; bindings/member_given_y_unpack.mojo):
    the `unit_n` rows `rows_of` (int64 or int32 row numbers Array; None: every
    row) of column `column` (-1: a one-column view) of the given `view`, as
    float64 words (exact for float64, int32 and |int64| <= 2^53), the rows
    split across the host's workers. Empty when the rows do not match."""
    var out = List[Float64]()
    var src = view.data
    var dtype = view.dtype
    var width = view.cols
    var col = column if column >= 0 else 0
    if col >= width:
        return out^
    var rows_at = 0
    var wide = True
    if rows_of is not None:
        if Int(py=rows_of.shape[0]) != unit_n:
            return out^
        var typestr = String(py=rows_of.dtype)
        if typestr == "<i4" or typestr == "int32":
            wide = False
        elif not (typestr == "<i8" or typestr == "int64"):
            return out^
        rows_at = trees_address(rows_of)
    elif view.rows != unit_n:
        return out^
    out = List[Float64](length=max(unit_n, 1), fill=0.0)
    var dst = Int(out.unsafe_ptr())
    var tasks = host_predict_task_count_work(unit_n, unit_n)
    var chunk = host_predict_chunk(unit_n, tasks)

    def task(t: Int) {imm src, imm dtype, imm width, imm col, imm rows_at, imm wide, imm dst, imm chunk, imm unit_n}:
        var target = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=dst)
        var begin = t * chunk
        var end = min(begin + chunk, unit_n)
        for i in range(begin, end):
            var r = i
            if rows_at != 0:
                r = Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=rows_at)[i]) if wide else Int(
                    MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=rows_at)[i])
            var at = r * width + col
            if dtype == MEMBER_I64:
                target[i] = Float64(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=src)[at])
            elif dtype == MEMBER_I32:
                target[i] = Float64(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=src)[at])
            else:
                target[i] = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=src)[at]

    host_parallelize(task, max(tasks, 1))
    _ = rows_of  # read through its address above
    out.resize(unit_n, 0.0)
    return out^


def _unit_given(mut set: MemberSet, given: PythonObject, unit_n: Int, context: PythonObject,
                mut given_blocks: List[List[Float64]], mut given_rows_last: PythonObject) raises:
    """THE GIVEN y of one unit of `_fit_units` (lane mf-tail5): one
    gather per distinct row set (the members of one fold share it)."""
    var source = given[0]
    var rows_of = given[1]
    var column = Int(py=given[2])
    var array_class = context["array_class"]
    var view = MemberMatrix()
    if source is not None and py_type(source) is array_class:
        var ndim = Int(py=source.ndim)
        view = member_given_y(source, Int(py=source.shape[0]), 1 if ndim == 1 else Int(py=source.shape[1]),
                              array_class)
    if view.given():
        if rows_of is None and column < 0 and view.cols == 1 and view.rows == unit_n:
            member_set_given(set, view.data, view.dtype)  # a view: every row of a 1-D y
        elif column >= 0 or rows_of is not None:
            if rows_of is None or not (rows_of is given_rows_last):
                given_blocks.append(_given_rows(view, rows_of, column, unit_n))
                given_rows_last = rows_of
            var last = len(given_blocks) - 1
            if last >= 0 and len(given_blocks[last]) == unit_n:
                member_set_given(set, Int(given_blocks[last].unsafe_ptr()), MEMBER_F64)
    _ = source  # the view reads its buffer through member_set_given above


def _fit_units(clones: PythonObject, xs: PythonObject, d: Int, targets: PythonObject, float_targets: Bool,
               weights: PythonObject, counts: PythonObject, jobs: Int, context: PythonObject,
               givens: PythonObject) raises:
    """`_fit_units_drawn` with no row draws and no kind masks asked."""
    _ = _fit_units_drawn(clones, xs, d, targets, float_targets, weights, counts, jobs, context, givens,
                         PythonObject(None), False)


def _fit_units_drawn(clones: PythonObject, xs: PythonObject, d: Int, targets: PythonObject, float_targets: Bool,
                     weights: PythonObject, counts: PythonObject, jobs: Int, context: PythonObject,
                     givens: PythonObject, draws: PythonObject, ask_kinds: Bool) raises -> List[Int]:
    """Fit the unfitted `clones` as ONE NATIVE MEMBER SET (core/member_abi.mojo,
    bindings/member_set_door.mojo): every unit admitted once, the fits run
    with the GIL released on host workers (independent units in parallel,
    at most `jobs`; one worker when a member's route is a GPU), the fitted
    states written into the clones as their own fit writes them.

    NO FIT DOOR IS LEFT (lane mf-del8, 2026-11-23; deleted: `_fit_member`,
    the door loop this function fell back to, and `_fit_units_native`'s
    False answers). Every refusal is raised before any fit, in the words of
    whoever refuses: a class the set does not take (`member_set_admit_or_raise`:
    a foreign class, a family's refused admission in its own words, a
    configuration with no native member fit), a non-finite X or float target
    (scikit-learn's check_array words, `admitted_finite_refuse`), a resident
    scope that holds something else; and after it a member's refused fit or
    publication in its own words (`member_set_run` / `member_set_pack`).

    Unit i: `xs[i]` its float32 C-order X Array of `d` columns (read by view:
    no copy; members of one cv fold share the fold's block), `targets[i]`
    its target Array of as many values, float32 (`float_targets`) or the
    int32 class codes 0 .. counts[i]-1 (`counts`: one class count per unit;
    None for regressors), `weights[i]` its float32 sample weights Array or
    None. `givens` (lane mf-tail5; None: none): per regressor unit None or
    (the y the door was GIVEN, the int64 row numbers of the unit's targets
    in it, or None for every row): THE GIVEN y (core/member_abi.mojo
    `MemberCall.y_given`) handed only to a member whose kind reads it, its
    rows gathered once per distinct row set.

    THE UNIT CLASSES (core/member_unit_classes.mojo): a classifier unit whose
    rows lack a class of the ensemble (a cv fold) is fitted on the classes it
    holds, as its fit door fitted it: its codes re-coded once per distinct
    target onto them, its member packed labelled with the ensemble codes it
    holds, so the callers' own checks (stacking's output width between
    folds) still fire. That pass is the set's class-cover pass
    (`skip_lacking`: the set makes no second one).

    THE ROW DRAWS (lane vdf-meta3): `draws` None, or per unit None or the
    int32 row numbers Array of the unit's rows in `xs[i]`, which is then
    the set's X (`xs[0]`, every drawn unit reads the same block) and whose
    `targets[i]` / `weights[i]` / THE GIVEN y cover every row of it: the
    draw selects X, the targets and the weights alike. A member whose kind
    reads a row index gets the index (inside `resident()` its device takes
    the rows out of the one resident X), any other member's worker gathers
    its rows: the door makes no fold copy. `ask_kinds`: the answer is each
    unit's kind-exact capability mask (`member_kind_capabilities`, one
    state minted per unit: ask it for a set of one unit per member, once
    per fit); empty otherwise."""
    var b = trees_builtins()
    var m = len(clones)
    var kinds = List[Int]()
    if m == 0:
        return kinds^
    var set = MemberSet(trees_address(xs[0]), Int(py=xs[0].shape[0]), d, MEMBER_F32 if float_targets else MEMBER_I32,
                        MEMBER_I32)
    set.run.cap = jobs
    # The members keep their fitted states: this door's predict sites read
    # them (`_member_native`), so a member's first predict imports nothing.
    set.adopt = True
    # THE UNIT CLASSES below read every classifier unit's classes once.
    set.run.skip_lacking = counts is not None
    var given_blocks = List[List[Float64]]()  # THE GIVEN y's gathered rows (lane mf-tail5), FIT only
    var given_rows_last = PythonObject(None)
    # THE UNIT CLASSES: per distinct target object, its held classes (empty:
    # every class) and, for a lacking one, its re-coded codes.
    var recoded = List[List[Int32]]()
    var unit_labels = Python.list()  # per unit: the labels its codes stand for (None for regressors)
    # The distinct target is keyed by its codes' address and length (the same
    # words give the same classes), and its labels are kept in a list that is
    # never re-bound: a loop-carried PythonObject re-bound under an `is` test
    # beside this loop's raising calls failed LLVM lowering with "operand #1
    # does not dominate this use" (Mojo 1.2.0; w3-integrate-3 build b1dee7d88c,
    # the same bug as ~/mojolearn-evidence/upstream/mojo-return-in-except-loop-dominance/).
    var classes_last_address = -1
    var classes_last_n = -1
    var held_last = List[Int32]()
    var recoded_last = -1
    var labels_seen = Python.list()  # per distinct target: its labels (the last is the current one)
    var classes_last_draw = 0
    for i in range(m):  # driver-loop: over the units (admission, once each)
        var classes = 0
        var target_address = trees_address(targets[i])
        set.labels = PythonObject(None)
        var labels = PythonObject(None)
        # THE ROW DRAWS: the unit's int32 row numbers in the set's X (0: every row of its own block).
        var draw = 0
        var draw_n = 0
        if draws is not None and draws[i] is not None:
            draw = trees_address(draws[i])
            draw_n = Int(py=draws[i].shape[0])
        if counts is not None:
            classes = Int(py=counts[i])
            var target_n = Int(py=targets[i].shape[0])
            if (target_address != classes_last_address or target_n != classes_last_n
                    or draw != classes_last_draw):
                # A drawn unit's held classes are those of its drawn rows (int32 row numbers).
                held_last = member_unit_classes(target_address, draw, False, draw_n if draw != 0 else target_n,
                                                classes)
                classes_last_address = target_address
                classes_last_n = target_n
                classes_last_draw = draw
                recoded_last = -1
                if len(held_last) > 0:
                    recoded.append(member_unit_recode(target_address, target_n, classes, held_last))
                    recoded_last = len(recoded) - 1
                    var held_labels = Python.list()
                    for c in range(len(held_last)):  # driver-loop: over the unit's held classes (a few labels)
                        _ = py_call(held_labels.append, PythonObject(Int(held_last[c])))
                    _ = py_call(labels_seen.append, held_labels)
                else:
                    _ = py_call(labels_seen.append, py_call(b.list, py_call(b.range, PythonObject(classes))))
            labels = labels_seen[len(labels_seen) - 1]
            if recoded_last >= 0:
                target_address = Int(recoded[recoded_last].unsafe_ptr())
                classes = len(held_last)
                # A class_weight dict is read over the ensemble codes the unit holds (its fit door's labels).
                set.labels = labels
        _ = py_call(unit_labels.append, labels)
        var w_address = 0
        if weights[i] is not None:
            w_address = trees_address(weights[i])
        member_set_admit_or_raise(set, clones[i], target_address, w_address, draw, draw_n, 0, 0, classes, context, b)
        if draw == 0:
            member_set_own_block(set, trees_address(xs[i]), Int(py=xs[i].shape[0]))
        if ask_kinds:
            kinds.append(member_kind_capabilities(set.run.plans[len(set.run.plans) - 1]))
        if float_targets and givens is not None and givens[i] is not None and member_set_reads_given(set):
            _unit_given(set, givens[i], Int(py=targets[i].shape[0]), context, given_blocks, given_rows_last)
        # Inside `resident()` a member that reads a scope keeps its X block
        # on the device (bindings/member_set_door.mojo, THE RESIDENT SCOPE).
        # A scope read that raises raises its own doors' words (nothing fitted).
        member_set_resident(set, xs[i], context)
    set.labels = PythonObject(None)
    # WHO REFUSES A NON-FINITE X OR y. X is held with its finite test
    # deferred to the members (`_hold_deferred`). A member whose kind tests
    # its rows (`member_set_tests`, MEMBER_CAN_TEST_ROWS) refuses them in its
    # own fit, from the pass its kernel already makes: nothing is scanned
    # here for it. For every other member each distinct X block and float
    # target it reads is proven here (a held block's recorded verdict, else
    # one host scan) and a non-finite one is refused before any fit in
    # scikit-learn's check_array words (lane mf-del8: no fit door is left to
    # refuse it).
    var proved_x = PythonObject(None)
    var proved_y = PythonObject(None)
    for i in range(m):  # driver-loop: over the units (one verdict per distinct block a non-testing member reads)
        if member_set_tests(set, i):
            continue
        if not (xs[i] is proved_x):
            if not _x_finite(xs[i], d):
                member_set_release(set)
                admitted_finite_refuse(1, String(py_type(clones[i]).__name__), "X")
            proved_x = xs[i]
        if float_targets and not (targets[i] is proved_y):
            var finite_y = trees_finite_f32(trees_address(targets[i]), Int(py=targets[i].shape[0]))
            _ = targets  # keep the Python objects alive through the scan above
            if not finite_y:
                member_set_release(set)
                admitted_finite_refuse(1, String(py_type(clones[i]).__name__), "y")
            proved_y = targets[i]
    # A failing unit releases every state and raises its member's own words.
    member_set_run(set)
    _ = xs  # keep the Python objects alive through the member fits above
    _ = targets
    _ = weights
    _ = givens
    _ = len(given_blocks)  # read through their addresses by the fits above (FIT only)
    try:
        for i in range(m):  # driver-loop: over the units (publication)
            _ = member_set_pack(set, i, unit_labels[i], b)
    except error:
        member_set_release(set)  # a failing publication leaves no state behind; its family's words
        raise error^
    # Every member's fit notices, once the whole set is published.
    member_set_notices(set)
    member_set_release(set)
    # THE LIFETIME RULE (core/member_abi.mojo): every block, target and weight
    # a member's fit was handed lives until its state was packed (exported,
    # and made to own its rows when adopted) or released: to here. A
    # borrowing member read its re-coded codes again at its pack.
    _ = xs
    _ = targets
    _ = weights
    _ = draws  # the row draws are read through their addresses by the fits above
    _ = len(recoded)
    return kinds^


struct _MetaX(Movable):
    """THE FOLD IN PLACE (lane vdf-meta3): the X a meta fit's members read.
    `x`: the float32 C-order Array (X itself, or the WHOLE host base of a
    model-selection fold, `_arena_io.DeviceRows`); `draw`: None, or the
    fold's int32 row numbers in `x`; `inv`: None, or the int64 fold
    position of each row of `x` (0 for a row outside the fold, never
    drawn); `n` the operation's rows, `d` the columns, `base_rows` x's rows."""
    var x: PythonObject
    var draw: PythonObject
    var inv: PythonObject
    var n: Int
    var d: Int
    var base_rows: Int

    def __init__(out self, x: PythonObject, draw: PythonObject, inv: PythonObject, n: Int, d: Int,
                 base_rows: Int):
        self.x = x
        self.draw = draw
        self.inv = inv
        self.n = n
        self.d = d
        self.base_rows = base_rows


def _meta_x_fit(X: PythonObject, context: PythonObject) raises -> _MetaX:
    """UNPACK of a meta fit's X, once per fit. A DeviceRows fold on a GPU
    route (the meta classes opt in with `_mojolearn_device_rows`, as the
    estimators do) is read IN PLACE: its host base is the members' X (the
    one model selection keeps resident: inside its `resident()` scope a
    member that reads a row index takes the fold's rows on the device out
    of it) and its rows are every member's row draw; the fold's targets and
    weights are put at their base rows (`_meta_take`). Nothing of the fold
    is gathered here. Any other X (and a fold on the CPU run context, the
    device="cpu" convention: its host rows) is admitted as before
    (`_x_array`)."""
    var nk = trees_keys()
    if not (py_get(context, nk[].k_device_cpu_context) is context) and door_is_device_rows(X, context):
        var door = DoorRows(X, context)
        var base = _x_array(door.base, context)
        var base_rows = _rows(base)
        var n = door.rows
        if n > 0 and base_rows > 0 and base_rows <= 2147483647 and _cols(base) > 0:
            var wide = _index_i64(door.indices, context)
            var draw = admitted_vector(n, "<i4", context)
            var inv = admitted_vector(base_rows, "<i8", context)
            meta_rows_index(trees_address(wide), n, base_rows, trees_address(draw), trees_address(inv))
            _ = wide  # read through its address above
            return _MetaX(base, draw, inv, n, _cols(base), base_rows)
        _ = door^
    var Xa = _x_array(X, context)
    return _MetaX(Xa, PythonObject(None), PythonObject(None), _rows(Xa), _cols(Xa), _rows(Xa))


def _meta_take(mx: _MetaX, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`values` (one 4-byte word per fold row: int32 codes, float32 targets
    or weights; None stays None) at the base rows of `mx`'s fold
    (core/meta_fold_rows.mojo `meta_rows_take_u32`), so the members' draws
    read the fold's own words. `values` itself when `mx` holds no fold."""
    if values is None or mx.inv is None:
        return values
    var typestr = String(py=values.dtype)
    var out = admitted_vector(mx.base_rows, "<i4" if typestr == "<i4" or typestr == "int32" else "<f4", context)
    meta_rows_take_u32(trees_address(values), trees_address(mx.inv), mx.base_rows, trees_address(out))
    _ = values  # read through its address above
    _ = mx.inv
    return out


def _meta_compose(mx: _MetaX, rows: PythonObject, context: PythonObject) raises -> PythonObject:
    """THE FOLD IN PLACE: int32 fold positions `rows` (a cv split of the
    fold) as rows of `mx.x` (core/meta_fold_rows.mojo `meta_rows_compose`:
    the inner fold composed inside the outer one, index words only); `rows`
    itself when `mx` holds no fold."""
    if mx.draw is None:
        return rows
    var count = Int(py=rows.shape[0])
    var out = admitted_vector(count, "<i4", context)
    meta_rows_compose(trees_address(rows), count, trees_address(mx.draw), trees_address(out))
    _ = rows  # read through its address above
    _ = mx.draw
    return out


def _fit_members_x(clones: PythonObject, mx: _MetaX, targets: PythonObject, float_targets: Bool,
                   weights: PythonObject, counts: PythonObject, jobs: Int, context: PythonObject,
                   givens: PythonObject) raises:
    """`_fit_members` over `mx` (THE FOLD IN PLACE): every member reads
    `mx.x` with the fold's rows as its draw; `targets` (per member, fold
    rows), `weights` (fold rows, or None) and THE GIVEN y tuples of every
    fold row are put at the base rows once per distinct object."""
    var m = len(clones)
    if mx.draw is None:
        _fit_members(clones, mx.x, mx.n, mx.d, targets, float_targets, weights, counts, jobs, context, givens)
        return
    var base_targets = Python.list()
    var last = PythonObject(None)
    var last_base = PythonObject(None)
    for i in range(m):  # driver-loop: over members (one take per distinct target object)
        if not (targets[i] is last):
            last = targets[i]
            last_base = _meta_take(mx, last, context)
        _ = py_call(base_targets.append, last_base)
    var base_givens = PythonObject(None)
    if givens is not None:
        base_givens = Python.list()
        for i in range(m):  # driver-loop: over members
            var g = givens[i]
            if g is None:
                _ = py_call(base_givens.append, g)
            else:
                # THE GIVEN y of every fold row, read at the base rows through `inv`.
                _ = py_call(base_givens.append, Python.tuple(g[0], mx.inv, g[2]))
    _ = _fit_units_drawn(clones, _repeat(mx.x, m), mx.d, base_targets, float_targets,
                         _repeat(_meta_take(mx, weights, context), m), counts, jobs, context, base_givens,
                         _repeat(mx.draw, m), False)


def _fit_members(clones: PythonObject, X: PythonObject, n: Int, d: Int, targets: PythonObject, float_targets: Bool,
                 weights: PythonObject, counts: PythonObject, jobs: Int, context: PythonObject,
                 givens: PythonObject) raises:
    """`_fit_units` for members that all read the one X (`n` x `d`) and the
    one sample-weight Array (or None)."""
    var m = len(clones)
    _fit_units(clones, _repeat(X, m), d, targets, float_targets, _repeat(weights, m), counts, jobs, context, givens)


def _x_finite(Xa: PythonObject, d: Int) raises -> Bool:
    """The finite verdict of the float32 C-order Array `Xa` (`d` columns)
    for a native member call, which reads admitted words and runs no member
    door: the recorded verdict of a held block (proved once for the whole
    operation and for every fold held as rows of it,
    `admitted_held_verify`), else one host scan (SIMD tasks across the
    cores), after which a finite block is held for the rest of the open
    scope so that it is scanned once, not once per member."""
    var held = admitted_held_index(Xa)
    if held >= 0:
        return admitted_held_verify(held)
    var n = _rows(Xa)
    var finite = trees_finite_f32(trees_address(Xa), n * d)
    _ = Xa  # keep the Python object alive through the native call above
    if finite and n > 0 and d > 0 and admitted_scope_open():
        admitted_hold(Xa, trees_address(Xa), n, d)
    return finite


def meta_member_open(member: PythonObject, operation: Int, classifier: Bool, context: PythonObject,
                     regressor_width: Int, skip_trees: Bool) -> MemberHeldCall:
    """THE X-FREE MEMBER UNPACK (lane mf-shap3, 2026-11-23): one fitted member's `operation` opened on the state
    it holds (bindings/member_set_door.mojo `member_held_open`), reading no rows: `served` only when its class
    carries a member mark and its family serves the call at the width the meta-estimator reads (a classifier's
    predict: its int codes, served only when its classes are 0 .. k-1 in order, a member's code then being the
    ensemble's; its predict_proba: its class count; a regressor: `regressor_width`, 0 any). `skip_trees`: a tree,
    forest or extra-trees member is not served (a public predict stages one device copy of X for those; the
    explainers' synthetic rows have none and call every member). Never raises. The caller proves its rows
    (`opened.tests`, else a finite scan) and closes it with `member_held_close`. `_member_open` is this plus the
    rows part of a public predict; Kernel and Permutation SHAP call it as it is
    (bindings/trees_agnostic_native_door.mojo)."""
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        var plan = prep_estimator_plan(member, context)
        if plan is None or py_get(plan, "member") is None:
            return MemberHeldCall()
        var width = regressor_width
        if classifier:
            width = 0
            var record = _member_class_codes(member, context)
            if operation == MEMBER_OP_PREDICT:
                if not Bool(py=record[4]):
                    return MemberHeldCall()
                width = 1
            elif operation == MEMBER_OP_PREDICT_PROBA:
                width = len(record[1])
        return member_held_open(member, operation, classifier, width, skip_trees, context, trees_builtins())
    except:
        return MemberHeldCall()


def _member_open(member: PythonObject, X: PythonObject, operation: Int, classifier: Bool,
                 context: PythonObject, regressor_width: Int = 0) -> MemberHeldCall:
    """UNPACK of one fitted member's `operation` over X (`_member_native`):
    the X-free open (`meta_member_open`, trees not served), then the rows
    part: `served` only when X is proven for it (a member that tests its
    rows: nothing scanned; any other: `_x_finite`). An unserved answer is
    closed already. Never raises. `regressor_width`: the columns a regressor
    must write (0: any). The caller runs it (`member_held_run`, or a typed
    RUN with its plan and state) and closes it with `member_held_close`."""
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        # Only an exact mojolearn Array is served (its shape, dtype and order
        # are native reads). A resident DeviceRows fold side goes to the
        # member's door, which resolves it natively; reading its Python
        # properties here ran Python inside CalibratedClassifierCV.fit
        # (9b369cd34f made them properties; test_native_meta_public_boundary).
        var nk = trees_keys()
        if py_type(X) is not py_item(context, nk[].k_array_class):
            return MemberHeldCall()
        var n = _rows(X)
        var d = _cols(X)
        if n <= 0 or d <= 0 or String(py=X.dtype) != "<f4" or String(py=X.order) != "C":
            return MemberHeldCall()
        var opened = meta_member_open(member, operation, classifier, context, regressor_width, True)
        var proven = False
        if opened.served:
            try:
                proven = opened.tests or _x_finite(X, d)
            except:
                proven = False
        if not proven:
            member_held_close(opened)
        return opened^
    except:
        return MemberHeldCall()


def _member_native(member: PythonObject, X: PythonObject, operation: Int, classifier: Bool,
                   context: PythonObject) -> PythonObject:
    """A fitted member's `operation` (MEMBER_OP_PREDICT, _PREDICT_PROBA,
    _DECISION) over X through the member state it holds
    (bindings/member_set_door.mojo `member_held_open`: a hit after the
    first use, and the state a native fit left with it; one member call, the
    GIL released, written into a new Array), or None when it is not served
    and the caller runs the member's own door (the fallback rule). Never
    raises. X is the float32 C-order Array every caller of this door reads
    (`_x_array`, a fold gather, the stacked meta features).

    A regressor (`classifier` False): its float32 predictions, (n,) for one
    output, (n, outputs) for more. A classifier: predict = its int32 or int64
    class codes, served only when its classes are 0 .. k-1 in order (a
    member's code is then the ensemble's code, the value its door's decoded
    label had); predict_proba = (n, its class count) columns in its class
    order, as its door's; decision_function = (n,) for one column, else
    (n, columns). The element type is the one the member's OUTPUTS answers.

    A NON-FINITE X: a member whose kind tests its rows (MEMBER_CAN_TEST_ROWS)
    refuses it in its own kernel pass (no host scan here), the call answers
    None and the member's door raises the estimator's words. For any other
    member X's finite verdict is read first (`_x_finite`: the held block's
    recorded verdict, else one host scan, then held); not finite = None and
    the door refuses it.

    NOT SERVED HERE, by rule: a class without a member mark; a tree, forest
    or extra-trees member (its door predicts from the one device copy of X
    this door stages for all of them, bindings/trees_staged_input_door.mojo;
    a member call would upload X once per member, and its route is not known
    before the hold's import; its fitted state is not kept at the native
    fit either). Once per member per public operation."""
    var opened = _member_open(member, X, operation, classifier, context)
    var result = PythonObject(None)
    if opened.served:
        try:
            result = member_held_run(opened, trees_address(X), _rows(X), _cols(X), context)
        except:
            result = PythonObject(None)
    member_held_close(opened)
    _ = X  # the rows are read through the whole member call
    return result


def _member_scores(member: PythonObject, method: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """A fitted classifier member's `predict_proba` or `decision_function`
    of X: through its held native member state when its family serves it
    (`_member_native`), else its own door (`meta_member_call`), which also
    raises the estimator's own words for a method it does not have."""
    if method == "predict_proba" or method == "decision_function":
        var native = _member_native(member, X, MEMBER_OP_PREDICT_PROBA if method == "predict_proba"
                                    else MEMBER_OP_DECISION, True, context)
        if native is not None:
            return native
    return meta_member_call(member, method, Python.tuple(X), context)


def _predict32(member: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """A fitted regressor member's float32 predictions of X: through its held
    native member state when its family serves it, else its internal
    float32 predict door (`meta_member_call`)."""
    var native = _member_native(member, X, MEMBER_OP_PREDICT, False, context)
    if native is not None:
        return native
    return meta_member_call(member, Python.tuple(X), context)


def _repeat(value: PythonObject, m: Int) raises -> PythonObject:
    """`[value] * m`: one target (or class count) shared by m members."""
    var items = Python.list()
    for _ in range(m):  # driver-loop: over members
        _ = py_call(items.append, value)
    return items


def _takes_weights(est: PythonObject, context: PythonObject) raises -> Bool:
    var plan = prep_estimator_plan(est, context)
    if plan is None:
        return False
    var nk = trees_keys()
    var keywords = py_get(py_get(plan, nk[].k_keywords, Python.dict()), nk[].k_fit, Python.tuple())
    return py_contains(keywords, PythonObject("sample_weight"))


def _member_weights(names: PythonObject, ests: PythonObject, sample_weight: PythonObject, n: Int,
                    context: PythonObject) raises -> PythonObject:
    """`_trees_member_weights`: None stays None; otherwise every member must
    take `sample_weight`, and the weights are checked once against X."""
    if sample_weight is None:
        return PythonObject(None)
    for i in range(len(ests)):  # driver-loop: over members
        if not _takes_weights(ests[i], context):
            var who = String("") if names[i] is None else " " + _str(names[i])
            raise Error("Underlying estimator" + who + " does not support sample weights.")
    var w = admit_weight_words(sample_weight, n, context)
    return _as_array(w, Python.tuple(n), context)


def _named(names: PythonObject, fitted: PythonObject) raises -> PythonObject:
    var out = Python.dict()
    for i in range(len(fitted)):  # driver-loop: over members
        py_setitem(out, names[i], fitted[i])
    return out


def _set(model: PythonObject, name: String, value: PythonObject) raises:
    py_setattr(model, name, value)


def _check_x(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_TreesWrapperBase._check_X`."""
    var nk = trees_keys()
    if not _truthy(_attr(model, nk[].k__fitted, PythonObject(False))):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var Xa = _x_array(X, context)
    var fitted = Int(py=_attr(model, nk[].k_n_features_in_, PythonObject(0)))
    trees_check_features(_cols(Xa), fitted)
    return Xa


# ------------------------------------------------------------------ cv folds


def _index_i64(v: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_index_i64`: a cv index list as an int64 Array."""
    try:
        var view = admit_c_view(v, "cv rows", context)
        var n = Int(py=view.size)
        var out = admitted_vector(n, "<i8", context)
        trees_cast(Int(py=classical_address(view)), admitted_dtype(view, context), trees_address(out), "<i8", n,
                   context)
        _ = view  # keep the Python object alive through the native call above
        return out
    except:
        var nk = trees_keys()
        var store = py_call(py_item(context, nk[].k_array_type), PythonObject("q"), v)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)


def _native_folds(xt: PythonObject, n_splits: Int, n: Int, codes: PythonObject, n_classes: Int,
                  context: PythonObject) raises -> PythonObject:
    """`_trees_native_folds`: the fold rows the device made, each fold's
    (train rows, test rows) as int32 Arrays."""
    var counts = admitted_vector(n_splits + 1, "<i4", context)
    var rows = admitted_vector(n_splits * n, "<i4", context)
    var c32 = counts
    var k = 0
    if codes is not None:
        c32 = codes
        k = n_classes
    var glue = _xr(xt)
    var status = xt_device_folds(glue, trees_address(c32), trees_address(rows), trees_address(counts),
                                 n, n_splits, k)
    _ = c32  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above
    _ = counts  # keep the Python object alive through the native call above
    if status == 1:
        raise Error("n_splits=" + String(n_splits) + " cannot be greater than the number of members in each class")
    _ = c32  # keep the Python object alive through the native calls above
    _ = rows  # keep the Python object alive through the native calls above
    _ = counts  # keep the Python object alive through the native calls above
    if status != 0:
        raise Error("y codes outside [0, n_classes)")
    _ = c32  # keep the Python object alive through the native calls above
    _ = rows  # keep the Python object alive through the native calls above
    _ = counts  # keep the Python object alive through the native calls above
    var cnt = _I32P(unsafe_from_address=trees_address(counts))
    _ = counts  # keep the Python object alive through the native call above
    var used = 0
    for i in range(n_splits):  # driver-loop: over the cv folds
        if Int(cnt[i]) > 0:
            used = i
        _ = counts  # keep the Python object alive through the native calls above
    used += 1
    var out = Python.list()
    for i in range(used):  # driver-loop: over the cv folds
        var c = Int(cnt[i])
        _ = counts  # keep the Python object alive through the native call above
        var base = i * n
        var tr = _copy_block(rows, base, base + n - c, "<i4", context)
        var te = _copy_block(rows, base + n - c, base + n, "<i4", context)
        _ = py_call(out.append, Python.tuple(tr, te))
    return out


def _splits(xt: PythonObject, cv: PythonObject, X: PythonObject, y: PythonObject, n: Int, codes: PythonObject,
            partition: Bool, n_classes: Int, context: PythonObject) raises -> PythonObject:
    """`_trees_splits`: [(train rows, test rows)] as int32 Arrays.

    A DeviceRows X (a model-selection fold handed to an opted-in meta fit;
    lane vdf-meta4): its row count is X's (`DoorRows.rows`, no host word of
    the fold read), and a mojolearn splitter is handed `range(rows)` for X,
    which is what its split reads of X (bindings/msel_split_door.mojo
    `msel_split`: "Split algorithms consume only X's row count"; the
    stratified kinds read y alone). The splits are fold positions, which the
    meta fit composes inside the fold (`_meta_compose`). Before, the fold
    reached `msel_input_count` and the splitter, which admit no DeviceRows
    (a TypeError)."""
    var b = trees_builtins()
    var c = _cv_value(cv, context)
    var x_rows = X
    if door_is_device_rows(X, context):
        x_rows = py_call(b.range, PythonObject(DoorRows(X, context).rows))
    var n_x = msel_input_count(x_rows, prep_native_context(context))
    var n_y = n
    if codes is not None:
        var codes_view = codes
        n_y = Int(py=codes_view.shape[0])
    if n != n_x or n != n_y or n_x != n_y:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(n_x) + ", "
                    + String(n_y) + "]")
    if py_type(c) is b.int:
        return _native_folds(xt, Int(py=c), n, codes, n_classes, context)
    var pairs = c
    var nk = trees_keys()
    if py_contains(py_item(prep_native_context(context), nk[].k_family_msel_splitter_layouts), py_type(c)):
        pairs = prep_native_split(c, x_rows, y, context)
    var out = Python.list()
    var seen = admitted_vector(n if n > 1 else 1, "<i8", context)
    var seen_addr = trees_address(seen)
    var bkey = base_helpers_key(py_item(context, nk[].k_trees_helpers))
    for pair in pairs:  # api-bound-loop: over the folds the user's cv splitter yields (public type: iterable of (train, test) pairs); one tuple unpack and one native index read per fold
        var both = py_call(b.tuple, pair)
        if len(both) != 2:
            raise Error("each cv split must be a (train, test) pair")
        var tr = _index_i64(both[0], context)
        var te = _index_i64(both[1], context)
        var ntr = Int(py=tr.size)
        var nte = Int(py=te.size)
        if ntr == 0 or nte == 0:
            raise Error("every cv split needs train and test rows")
        var tr_addr = trees_address(tr)
        var te_addr = trees_address(te)
        if base_check_indices_i64(bkey, tr_addr, ntr, n) == 1:
            raise Error("a cv index is outside [0, " + String(n) + ")")
        if base_check_indices_i64(bkey, te_addr, nte, n) == 1:
            raise Error("a cv index is outside [0, " + String(n) + ")")
        _ = base_bincount_i64(bkey, te_addr, 3, nte, n, seen_addr, True)
        _ = tr  # keep the index store alive through the native calls above
        _ = te  # keep the index store alive through the native calls above
        _ = seen  # keep the count store alive through the native call above
        var tr32 = admitted_vector(ntr, "<i4", context)
        var te32 = admitted_vector(nte, "<i4", context)
        trees_cast(trees_address(tr), "<i8", trees_address(tr32), "<i4", ntr, context)
        _ = tr32  # keep the Python object alive through the native call above
        _ = tr  # keep the Python object alive through the native call above
        trees_cast(trees_address(te), "<i8", trees_address(te32), "<i4", nte, context)
        _ = te32  # keep the Python object alive through the native call above
        _ = te  # keep the Python object alive through the native call above
        _ = py_call(out.append, Python.tuple(tr32, te32))
    if len(out) == 0:
        raise Error("cv produced no splits")
    if partition:
        var m = n if n > 1 else 1
        var low = base_reduce_stat_int(bkey, seen_addr, 3, m, 0)
        var high = base_reduce_stat_int(bkey, seen_addr, 3, m, 1)
        if low != 1 or high != 1:
            raise Error("cross_val_predict only works for partitions: every row must be in exactly one test set")
        _ = seen  # keep the Python object alive through the native calls above
    return out


# ------------------------------------------------------------------ Voting


def _voting_weights(model: PythonObject) raises -> PythonObject:
    """`VotingClassifier._weights`: one float per fitted member."""
    var nk = trees_keys()
    var m = len(_attr(model, nk[].k_estimators_, Python.list()))
    # One body with the META kind's admission (lane mf-refuse5).
    return meta_voting_weights(_attr(model, nk[].k_weights, PythonObject(None)), m)


def _voting_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                classifier: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var owner = _owner(model)
    var nk = trees_keys()
    # The voting value, then n_jobs / verbose: one body with the META kind's admission (lane mf-refuse5).
    var jobs = meta_voting_validate(model, classifier)
    # THE FOLD IN PLACE (lane vdf-meta3): a DeviceRows fold is read out of its base (`_meta_x_fit`).
    var mx = _meta_x_fit(X, context)
    var n = mx.n
    var target = y
    if classifier:
        var labels = trees_encode_labels(y, labels_scope(model, context))
        _set(model, "classes_", labels.classes)
        _set(model, "n_classes_", PythonObject(len(labels.classes)))
        target = labels.codes
    var active = _active(_attr(model, nk[].k_estimators, PythonObject(None)), owner, context)
    # The members run on the ensemble's device (bindings/trees_meta_device_door.mojo): read once per fit.
    var device = py_get(fields, "device")
    var origin = meta_device_origin(model, device)
    meta_members_device(device, active.ests, owner, origin, context, trees_builtins())
    var weights = _member_weights(active.names, active.ests, sample_weight, n, context)
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    var m = len(active.ests)
    var clones = Python.list()
    for i in range(m):  # driver-loop: over members
        _ = py_call(clones.append, _clone(active.ests[i], device, origin, context))
    # Native members read typed targets: the class codes, or (regressors) y
    # as float32 words (lane mf-del8: no member fit door is left; a y the
    # float32 admission refuses raises its words, a y whose length is not
    # X's scikit-learn's check_consistent_length words, before any fit).
    if classifier:
        var k = len(_attr(model, PythonObject("classes_"), Python.list()))
        _fit_members_x(clones, mx, _repeat(target, m), False, weights, _repeat(PythonObject(k), m), jobs, context,
                       PythonObject(None))
    else:
        var y32 = _f32_array(y, 1, "y", context)
        if _rows(y32) != n:
            raise Error("ValueError: Found input variables with inconsistent numbers of samples: ["
                        + String(n) + ", " + String(_rows(y32)) + "]")
        # THE GIVEN y (lane mf-tail5): y as given, every row, to a member that reads it
        _fit_members_x(clones, mx, _repeat(y32, m), True, weights, PythonObject(None), jobs, context,
                       _repeat(Python.tuple(y, PythonObject(None), PythonObject(-1)), m))
    for i in range(m):  # driver-loop: over members
        _ = py_call(fitted.append, clones[i])
    _set(model, "named_estimators_", _named(active.names, fitted))
    _set(model, "n_features_in_", PythonObject(mx.d))
    _set(model, "_fitted", PythonObject(True))
    if classifier:
        _ = _voting_weights(model)
    _ = fields
    return model


# THE VOTE SPLIT (lane bm-meta, 2026-11-22). predict, predict_proba and
# transform of Voting* are UNPACK (`_vote_unpack`: the weights, the class
# count and one typed `VoteMember` per member, read once), RUN
# (xtrees/trees_meta_vote_run.mojo: the native members' calls and the fold,
# no Python) and PACK (the Arrays the caller returns, allocated before RUN so
# that RUN writes into them). Before the split the fold ran here over the
# members' Python outputs: one Array per native member's answer, the same
# glue ops in the same order. A member its family does not serve (a tree,
# forest or extra-trees member, a class without a member mark) answers
# through its door in UNPACK, as before.

comptime _VOTE_SOFT = 0       # predict_proba columns, placed at the member's classes
comptime _VOTE_HARD = 1       # int32 class codes
comptime _VOTE_REGRESS = 2    # one float32 prediction per row
comptime _VOTE_BLOCK = 3      # a transform block: float32 (n, c) predictions


struct _VoteUnpack(Movable):
    """What `_vote_unpack` read: the typed members, the held native states
    they point into (closed by `_vote_close`; every one may run with no
    GIL), and the Python owners of the door blocks. `blocks`: per member,
    the Python object of its door block (its codes Array for a hard vote),
    None for a native member."""
    var members: List[VoteMember]
    var opened: List[MemberHeldCall]
    var keep: PythonObject
    var blocks: PythonObject
    var native: Int

    def __init__(out self) raises:
        self.members = List[VoteMember]()
        self.opened = List[MemberHeldCall]()
        self.keep = Python.list()
        self.blocks = Python.list()
        self.native = 0


def _vote_door(member: PythonObject, kind: Int, Xa: PythonObject, n: Int, weight: Float64, keep: PythonObject,
               context: PythonObject, answer: PythonObject) raises -> VoteMember:
    """One member's block from its own door (the old per-member path, which
    raises the estimator's own words), its owners kept in `keep`. `answer`:
    the Array its held native state already wrote (`member_held_run`), else
    None and the door is called."""
    if kind == _VOTE_SOFT:
        var scores = answer if answer is not None else meta_member_call(member, "predict_proba", Python.tuple(Xa),
                                                                         context)
        var xa = admit_f32(scores, 2, "sub-estimator output", context, finite=False)
        var cols = _sub_cols(member, context)
        _ = py_call(keep.append, xa.array)
        _ = py_call(keep.append, cols)
        var out = VoteMember(xa.address, Int(py=cols.shape[0]), trees_address(cols), weight)
        _ = xa^
        return out^
    if kind == _VOTE_HARD:
        var codes: PythonObject
        if answer is not None:
            codes = answer if String(answer.dtype) == "<i4" else _i32_vec(answer, "codes", context)
        elif _forest_classifier_member(member, context):
            codes = _member_codes(member, Xa, context)
        else:
            codes = _i32_vec(meta_member_call(member, "predict", Python.tuple(Xa), context), "codes", context)
        _ = py_call(keep.append, codes)
        return VoteMember(trees_address(codes), 1, 0, weight)
    if kind == _VOTE_REGRESS:
        var xa = admit_f32(answer if answer is not None else meta_member_call(member, Python.tuple(Xa), context), 1,
                           "sub-estimator output", context, finite=False)
        _ = py_call(keep.append, xa.array)
        var out = VoteMember(xa.address, 1, 0, weight)
        _ = xa^
        return out^
    var blk = _out2d(answer if answer is not None else meta_member_call(member, Python.tuple(Xa), context), n,
                     context)
    _ = py_call(keep.append, blk.array)
    var out = VoteMember(blk.address, blk.cols, 0, weight)
    _ = blk^
    return out^


def _vote_unpack(model: PythonObject, Xa: PythonObject, kind: Int, classifier: Bool, weights: PythonObject,
                 context: PythonObject) raises -> _VoteUnpack:
    """UNPACK of a vote: per fitted member its held native state when its
    family serves the operation (`_member_open`), else its door's block."""
    var nk = trees_keys()
    return _vote_unpack_of(_attr(model, nk[].k_estimators_, Python.list()), Xa, kind, classifier, weights, context)


def _vote_unpack_of(members: PythonObject, Xa: PythonObject, kind: Int, classifier: Bool, weights: PythonObject,
                    context: PythonObject) raises -> _VoteUnpack:
    """`_vote_unpack` over the fitted `members` (a list)."""
    var n = _rows(Xa)
    var got = _VoteUnpack()
    try:
        _vote_unpack_members(got, members, Xa, n, kind, classifier, weights, context)
        # The native members' plans by address, now that `got.opened` is
        # final: RUN reads them in place.
        for i in range(len(got.members)):  # driver-loop: over members
            if got.members[i].native:
                got.members[i].plan = Int(UnsafePointer(to=got.opened[got.members[i].plan].held.plan))
    except e:
        _vote_close(got)
        raise e^
    return got^


def _vote_unpack_members(mut got: _VoteUnpack, members: PythonObject, Xa: PythonObject, n: Int, kind: Int,
                         classifier: Bool, weights: PythonObject, context: PythonObject) raises:
    var operation = MEMBER_OP_PREDICT_PROBA if kind == _VOTE_SOFT else MEMBER_OP_PREDICT
    for i in range(len(members)):  # driver-loop: over members (admission, once each)
        var e = members[i]
        var weight = Float64(py=weights[i]) if weights is not None else 1.0
        # A classifier's float32 transform block is its door's (its predict
        # labels as float32), as before.
        var native = not (kind == _VOTE_BLOCK and classifier)
        if native:
            var opened = _member_open(e, Xa, operation, classifier and kind != _VOTE_BLOCK, context,
                                      1 if kind == _VOTE_REGRESS else 0)
            if opened.served and not opened.gil_free:
                # A kind that must run with the GIL held answers here, on the
                # public thread with the GIL held, as before the split; RUN
                # then holds no lock any member call released before.
                var answer = PythonObject(None)
                try:
                    answer = member_held_run(opened, trees_address(Xa), n, _cols(Xa), context)
                finally:
                    member_held_close(opened)
                got.members.append(_vote_door(e, kind, Xa, n, weight, got.keep, context, answer))
                _ = py_call(got.blocks.append, got.keep[len(got.keep) - 1])
                continue
            if opened.served:
                var class_cols = 0
                if kind == _VOTE_SOFT:
                    var cols = _sub_cols(e, context)
                    _ = py_call(got.keep.append, cols)
                    class_cols = trees_address(cols)
                # The plan's address is set once every member is opened
                # (`_vote_plans`): `got.opened` may still grow here.
                got.members.append(VoteMember(len(got.opened), opened.held.state, operation, opened.cols,
                                              opened.dtype, class_cols, weight))
                got.native += 1
                got.opened.append(opened^)
                _ = py_call(got.blocks.append, PythonObject(None))
                continue
        got.members.append(_vote_door(e, kind, Xa, n, weight, got.keep, context, PythonObject(None)))
        _ = py_call(got.blocks.append, got.keep[len(got.keep) - 1])


def _vote_close(mut got: _VoteUnpack):
    for i in range(len(got.opened)):  # driver-loop: over the opened native members
        member_held_close(got.opened[i])


def _vote_run(model: PythonObject, Xa: PythonObject, kind: Int, classifier: Bool, soft: Bool, k: Int,
              weights: PythonObject, glue: EntryRef, acc: Int, codes: Int, width: Int,
              context: PythonObject) raises:
    """UNPACK, RUN with the GIL released when any member is native (each is
    of a kind that may run without it: the others answered in UNPACK), and
    the fallback rule: a native member that refuses is asked through its door
    (which raises the estimator's words) and the vote runs again."""
    var got = _vote_unpack(model, Xa, kind, classifier, weights, context)
    var rows = VoteRows(trees_address(Xa), _rows(Xa), _cols(Xa))
    var nk = trees_keys()
    var members = _attr(model, nk[].k_estimators_, Python.list())
    while True:  # driver-loop: over refusals (at most one round per native member)
        var refused = VOTE_ALL_ANSWERED
        try:
            if got.native > 0:
                with GILReleased(Python()):
                    refused = _vote_dispatch(glue, rows, got.members, kind, soft, k, acc, codes, width)
            else:
                refused = _vote_dispatch(glue, rows, got.members, kind, soft, k, acc, codes, width)
        except e:
            _vote_close(got)
            raise e^
        if refused == VOTE_ALL_ANSWERED:
            break
        var weight = got.members[refused].weight
        got.members[refused] = _vote_door(members[refused], kind, Xa, rows.n, weight, got.keep, context,
                                          PythonObject(None))
        got.native -= 1
        _vote_zero(acc, rows.n * (width if width > 0 else (1 if kind == _VOTE_REGRESS else k)))
    _vote_close(got)
    _ = Xa  # the rows are read through RUN
    _ = got.keep  # the door blocks are read through RUN


def _check_meta_x(model: PythonObject, X: PythonObject, context: PythonObject) raises -> _MetaX:
    """`_check_x` of a predict / transform / score whose X may be a
    model-selection fold (lane vdf-meta4): fitted check, then `_meta_x_fit`
    (a DeviceRows fold on a GPU route: its host base and its rows, nothing
    gathered), then the feature count."""
    var nk = trees_keys()
    if not _truthy(_attr(model, nk[].k__fitted, PythonObject(False))):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var mx = _meta_x_fit(X, context)
    trees_check_features(mx.d, Int(py=_attr(model, nk[].k_n_features_in_, PythonObject(0))))
    return mx^


def _fold_scope(mx: _MetaX, context: PythonObject) raises -> Int:
    """THE FOLD FROM THE STORE (lane vdf-meta5): the resident scope a
    predict / transform / score over `mx` reads its fold in, model
    selection's (`family_msel_rows_context`), or MEMBER_SCOPE_NONE when `mx`
    holds no fold (nothing is indexed). One slot read per public operation."""
    if mx.draw is None:
        return MEMBER_SCOPE_NONE
    return member_scope_word(py_item(prep_native_context(context), PythonObject("family_msel_rows_context")))


def _fold_open(mut got: _VoteUnpack, member: PythonObject, x: PythonObject, operation: Int, classifier: Bool,
               regressor_width: Int, class_cols: Int, weight: Float64, check_kind: Bool,
               context: PythonObject) -> Bool:
    """THE FOLD FROM THE STORE (lane vdf-meta5; the admission vdf-meta4's
    `_vote_run_indexed` made inline, now shared by every meta predict that
    reads a fold by number): one fitted member opened on its held native
    state over the fold's base `x` and appended to `got` as a native
    `VoteMember`, True; False, with nothing appended and the opened state
    closed, when its family does not serve it without the GIL or (when
    `check_kind`: once per member per public operation) its kind does not
    read a row index and a scope (`member_kind_capabilities`, kind-exact).
    The caller then admits the fold's rows and runs its gathered path.
    Never raises."""
    var opened = _member_open(member, x, operation, classifier, context, regressor_width)
    if not (opened.served and opened.gil_free):
        if opened.served:
            member_held_close(opened)
        return False
    if check_kind:
        var mask = 0
        try:
            mask = member_kind_capabilities(opened.held.plan)
        except:
            mask = 0
        if (mask & MEMBER_CAN_ROW_INDEX) == 0 or (mask & MEMBER_CAN_SCOPE) == 0:
            member_held_close(opened)
            return False
    got.members.append(VoteMember(len(got.opened), opened.held.state, operation, opened.cols, opened.dtype,
                                  class_cols, weight))
    got.native += 1
    got.opened.append(opened^)
    return True


def _fold_plans(mut got: _VoteUnpack):
    """The native members' plans by address, once `got.opened` is final
    (RUN reads them in place)."""
    for i in range(len(got.members)):  # driver-loop: over members
        if got.members[i].native:
            got.members[i].plan = Int(UnsafePointer(to=got.opened[got.members[i].plan].held.plan))


struct _FoldPin(Movable):
    """THE PIN RULE, ONE REFERENCE PER CALL (lane vdf-meta4's accounting,
    shared by every indexed meta RUN since lane vdf-meta5): `calls`
    references on the owner of X's host words, one per member call of the
    RUN, each member's answer in its own byte (`VoteRows.took_each`);
    `settle` returns the references no cache took, after the RUN, whether it
    returned or raised."""
    var took: List[UInt8]
    var pin: Int
    var calls: Int

    def __init__(out self, source: PythonObject, calls: Int):
        self.calls = calls
        self.took = List[UInt8](length=max(calls, 1), fill=UInt8(0))
        self.pin = member_pin_take(source, calls)

    def rows(self, x: Int, base_rows: Int, d: Int, index: Int, count: Int, scope: Int) -> VoteRows:
        """THE INDEXED ROWS: `count` int32 row numbers at `index` of the base
        at `x` (`base_rows` x `d`), the scope and this pin offered on every
        member call, one answer byte per member."""
        var rows = VoteRows(x, base_rows, d, index, count, scope, self.pin,
                            Int(self.took.unsafe_ptr()) if self.pin != 0 else 0)
        rows.took_each = True
        return rows^

    def settle(mut self):
        var taken = 0
        for j in range(self.calls):  # driver-loop: over the member calls (one answer byte each)
            if self.took[j] != UInt8(0):
                taken += 1
        member_pin_return(self.pin, self.calls - taken)
        self.pin = 0
        self.calls = 0


def _vote_run_indexed(model: PythonObject, mx: _MetaX, kind: Int, classifier: Bool, soft: Bool, k: Int,
                      weights: PythonObject, glue: EntryRef, acc: Int, codes: Int, width: Int, scope: Int,
                      context: PythonObject) raises -> Bool:
    """THE FOLD FROM THE STORE (lane vdf-meta4): a vote over a fold `mx`
    (its host base and int32 rows) with every member opened on its held
    native state over the base and handed the fold's row index, the
    resident `scope` and one pin reference per member call
    (xtrees/trees_meta_vote_run.mojo THE INDEXED ROWS, `took_each`), so its
    device takes the fold's rows out of the base model selection keeps
    resident: no row of the fold is gathered on the host. Asked only when
    every member is served natively without the GIL and its kind reads a row
    index and a scope (`_fold_open`); False, with `acc` zeroed and nothing
    else written, otherwise or when a member call refuses: the caller then
    admits the fold's rows (`_x_array`) and runs the vote as before (the
    fallback rule, whose doors raise the members' own words). Same rows in
    the same order as the gathered block: same bits."""
    if kind == _VOTE_BLOCK and classifier:
        return False  # a classifier's transform block is its door's
    var nk = trees_keys()
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var operation = MEMBER_OP_PREDICT_PROBA if kind == _VOTE_SOFT else MEMBER_OP_PREDICT
    var got = _VoteUnpack()
    var answered = False
    try:
        var opened_all = True
        for i in range(len(members)):  # driver-loop: over members (admission, once each)
            var e = members[i]
            var weight = Float64(py=weights[i]) if weights is not None else 1.0
            var class_cols = 0
            if kind == _VOTE_SOFT:
                var cols = _sub_cols(e, context)
                _ = py_call(got.keep.append, cols)
                class_cols = trees_address(cols)
            if not _fold_open(got, e, mx.x, operation, classifier and kind != _VOTE_BLOCK,
                              1 if kind == _VOTE_REGRESS else 0, class_cols, weight, True, context):
                opened_all = False
                break
        if opened_all:
            _fold_plans(got)
            var pin = _FoldPin(mx.x, len(got.members))
            var rows = pin.rows(trees_address(mx.x), mx.base_rows, mx.d, trees_address(mx.draw), mx.n, scope)
            var refused = VOTE_ALL_ANSWERED
            try:
                with GILReleased(Python()):
                    refused = _vote_dispatch(glue, rows, got.members, kind, soft, k, acc, codes, width)
            finally:
                pin.settle()
            answered = refused == VOTE_ALL_ANSWERED
    except e:
        _vote_close(got)
        raise e^
    _vote_close(got)
    _ = mx.x  # the rows are read through RUN
    _ = mx.draw
    _ = got.keep  # the class columns are read through RUN
    if not answered:
        _vote_zero(acc, mx.n * (width if width > 0 else (1 if kind == _VOTE_REGRESS else k)))
    return answered


def _vote_run_x(model: PythonObject, X: PythonObject, mx: _MetaX, kind: Int, classifier: Bool, soft: Bool, k: Int,
                weights: PythonObject, glue: EntryRef, acc: Int, codes: Int, width: Int,
                context: PythonObject) raises:
    """`_vote_run` over `_check_meta_x`'s `mx` (lane vdf-meta4): a fold
    inside model selection's resident scope is voted FROM THE STORE
    (`_vote_run_indexed`); a fold it does not serve has its rows admitted
    (`_x_array`, the host gather every opted-in meta predict made before);
    any other X is `mx.x` itself. `codes` (vote_codes_run) is written last by
    the RUN, so a refused indexed round leaves nothing there to undo."""
    if mx.draw is None:
        _vote_run(model, mx.x, kind, classifier, soft, k, weights, glue, acc, codes, width, context)
        return
    var scope = _fold_scope(mx, context)
    if scope != MEMBER_SCOPE_NONE and _vote_run_indexed(model, mx, kind, classifier, soft, k, weights, glue, acc,
                                                        codes, width, scope, context):
        return
    _vote_run(model, _x_array(X, context), kind, classifier, soft, k, weights, glue, acc, codes, width, context)


def _vote_zero(acc: Int, count: Int):
    """A refused round may have written part of the scratch: zero it again."""
    var p = _F64P(unsafe_from_address=acc)
    for i in range(count):  # small-loop(the refusal path only: one zero pass before the vote runs again)
        p[i] = 0.0


def _vote_dispatch(glue: EntryRef, rows: VoteRows, members: List[VoteMember], kind: Int, soft: Bool, k: Int,
                   acc: Int, codes: Int, width: Int) raises -> Int:
    if width > 0:
        return vote_transform_run(glue, rows, members, k, soft, width, acc)
    if kind == _VOTE_REGRESS:
        return vote_regress_run(glue, rows, members, acc)
    if codes != 0:
        return vote_codes_run(glue, rows, members, k, soft, acc, codes)
    return vote_soft_run(glue, rows, members, k, acc)


def _voting_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    if String(_attr(model, nk[].k_voting, PythonObject(""))) == "hard":
        raise Error("AttributeError: predict_proba is not available when voting='hard'")
    var mx = _check_meta_x(model, X, context)  # a fold from the store (lane vdf-meta4)
    var xt = _xt(py_fields(model), context)
    var k = Int(py=_attr(model, nk[].k_n_classes_, PythonObject(0)))
    var n = mx.n
    var acc = _class_zero(xt, n * k, context)
    _vote_run_x(model, X, mx, _VOTE_SOFT, True, True, k, _voting_weights(model), _xr(xt), trees_address(acc), 0, 0,
                context)
    return trees_reshape(acc, Python.tuple(n, k), context)


def _voting_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    return trees_decode_labels(_attr(model, nk[].k_classes_, PythonObject(None)), _voting_codes(model, X, context), context)


def _voting_codes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`VotingClassifier.predict` before its decode: int32 codes into
    `classes_`."""
    var mx = _check_meta_x(model, X, context)  # a fold from the store (lane vdf-meta4)
    var xt = _xt(py_fields(model), context)
    var n = mx.n
    var nk = trees_keys()
    var k = Int(py=_attr(model, nk[].k_n_classes_, PythonObject(0)))
    var soft = String(_attr(model, nk[].k_voting, PythonObject(""))) == "soft"
    var acc = _class_zero(xt, n * k, context)
    var out = admitted_vector(n, "<i4", context)
    _vote_run_x(model, X, mx, _VOTE_SOFT if soft else _VOTE_HARD, True, soft, k, _voting_weights(model), _xr(xt),
                trees_address(acc), trees_address(out), 0, context)
    _ = acc  # the scratch is written through RUN
    return out


def _voting_transform(model: PythonObject, X: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var mx = _check_meta_x(model, X, context)  # a fold from the store (lane vdf-meta4)
    var xt = _xt(py_fields(model), context)
    var n = mx.n
    var nk = trees_keys()
    var m = len(_attr(model, nk[].k_estimators_, Python.list()))
    var soft = classifier and String(_attr(model, nk[].k_voting, PythonObject(""))) == "soft"
    var k = Int(py=_attr(model, nk[].k_n_classes_, PythonObject(0))) if classifier else 1
    var width = m * k if soft else m
    var out = _zeros(n * width, context)
    _vote_run_x(model, X, mx, _VOTE_SOFT if soft else _VOTE_BLOCK, classifier, soft, k, PythonObject(None),
                _xr(xt), trees_address(out), 0, width, context)
    return trees_reshape(out, Python.tuple(n, width), context)


def _voting_regress(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`VotingRegressor.predict`: the weighted average of the members'
    predictions."""
    var mx = _check_meta_x(model, X, context)  # a fold from the store (lane vdf-meta4)
    var xt = _xt(py_fields(model), context)
    var n = mx.n
    var acc = _zeros(n, context)
    _vote_run_x(model, X, mx, _VOTE_REGRESS, False, False, 1, _voting_weights(model), _xr(xt), trees_address(acc), 0,
                0, context)
    return acc


# ---------------------------------------------------------------- Stacking


def _stack_method(model: PythonObject, est: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    """`_StackingBase._method`."""
    var nk = trees_keys()
    var method = _attr(model, nk[].k_stack_method, PythonObject("auto"))
    if String(method) != "auto":
        return method
    if classifier:
        if prep_estimator_has(est, "predict_proba", context):
            return PythonObject("predict_proba")
        if prep_estimator_has(est, "decision_function", context):
            return PythonObject("decision_function")
    return PythonObject("predict")


def _stack_kind(method: String) -> Int:
    if method == "predict_proba":
        return STACK_PROBA
    if method == "decision_function":
        return STACK_DECISION
    return STACK_PREDICT


def _stack_door(member: PythonObject, kind: Int, Xs: PythonObject, n: Int, class_cols: Int, keep: PythonObject,
                answer: PythonObject, context: PythonObject) raises -> VoteMember:
    """One member's float32 (n, c) block from its own door (`_StackingBase._out`
    before the column choice: the internal float32 predict door, or the
    named method's door, which raises the estimator's own words), or from
    the Array its held state already wrote (`answer`)."""
    var out = answer
    if out is None:
        if kind == STACK_PREDICT:
            out = meta_member_call(member, Python.tuple(Xs), context)
        else:
            out = meta_member_call(member, "predict_proba" if kind == STACK_PROBA else "decision_function",
                                   Python.tuple(Xs), context)
    var blk = _out2d(out, n, context)
    _ = py_call(keep.append, blk.array)
    var got = VoteMember(blk.address, blk.cols, class_cols, 1.0)
    _ = blk^
    return got


def _stack_place(glue: EntryRef, members: PythonObject, kinds: List[Int], Xs: PythonObject, k: Int, binary: Bool,
                 meta: Int, meta_rows: Int, width: Int, place: Int, col0: Int, with_classes: Bool,
                 context: PythonObject) raises -> List[Int]:
    """UNPACK of the stacked outputs of `members` over the rows `Xs` (a
    float32 C-order Array): per member its held native state when its family
    serves the call (`_member_open`: predict as a regressor's float32
    predict, `_predict32`; predict_proba / decision_function as a
    classifier's, `_member_scores`), else its door's block; a kind that
    needs the GIL answers here. RUN `stack_place_run` (the GIL released when
    a member is native) into `meta` at the rows `place`; the fallback rule
    on a refusal. `with_classes`: each member's class columns and count are
    read (a cv fold's members, which may lack a class); a member fitted on
    every row saw every class. Answers the columns each member took."""
    var n = _rows(Xs)
    var got = _VoteUnpack()
    var counts = List[Int](capacity=len(members))
    var widths = List[Int](capacity=len(members))
    try:
        for i in range(len(members)):  # driver-loop: over members (admission, once each)
            var e = members[i]
            var kind = kinds[i]
            var classifier = kind != STACK_PREDICT
            var class_cols = 0
            var count = 0
            if classifier and with_classes:
                var cols = _sub_cols(e, context)
                _ = py_call(got.keep.append, cols)
                class_cols = trees_address(cols)
                count = Int(py=cols.shape[0])
            counts.append(count)
            var operation = MEMBER_OP_PREDICT if kind == STACK_PREDICT else (
                MEMBER_OP_PREDICT_PROBA if kind == STACK_PROBA else MEMBER_OP_DECISION)
            var opened = _member_open(e, Xs, operation, classifier, context)
            if opened.served and opened.gil_free:
                got.members.append(VoteMember(len(got.opened), opened.held.state, operation, opened.cols,
                                              opened.dtype, class_cols, 1.0))
                got.native += 1
                got.opened.append(opened^)
                continue
            var answer = PythonObject(None)
            if opened.served:
                try:
                    answer = member_held_run(opened, trees_address(Xs), n, _cols(Xs), context)
                finally:
                    member_held_close(opened)
            got.members.append(_stack_door(e, kind, Xs, n, class_cols, got.keep, answer, context))
        for i in range(len(got.members)):  # driver-loop: over members
            if got.members[i].native:
                got.members[i].plan = Int(UnsafePointer(to=got.opened[got.members[i].plan].held.plan))
        var rows = VoteRows(trees_address(Xs), n, _cols(Xs))
        var start = 0
        while True:  # driver-loop: over refusals (at most one round per native member)
            var refused = VOTE_ALL_ANSWERED
            if got.native > 0:
                with GILReleased(Python()):
                    refused = stack_place_run(glue, rows, got.members, kinds, counts, k, binary, meta, meta_rows,
                                              width, place, col0, widths, start)
            else:
                refused = stack_place_run(glue, rows, got.members, kinds, counts, k, binary, meta, meta_rows,
                                          width, place, col0, widths, start)
            if refused == VOTE_ALL_ANSWERED:
                break
            got.members[refused] = _stack_door(members[refused], kinds[refused], Xs, n,
                                               got.members[refused].class_cols, got.keep, PythonObject(None),
                                               context)
            got.native -= 1
            start = refused
    except e:
        _vote_close(got)
        raise e^
    _vote_close(got)
    _ = Xs  # the rows are read through RUN
    _ = got.keep  # the door blocks and class columns are read through RUN
    return widths^


def _stack_place_indexed(glue: EntryRef, members: PythonObject, kinds: List[Int], Xa: PythonObject, n: Int, d: Int,
                         rows_at: PythonObject, place: PythonObject, scope: Int, k: Int, binary: Bool, meta: Int,
                         meta_rows: Int, width: Int, mut widths: List[Int], check_kind: Bool,
                         context: PythonObject) raises -> Bool:
    """THE FOLD ROWS BY NUMBER (lane vdf-meta3): `_stack_place` of the rows
    `rows_at` (int32 row numbers of the whole X `Xa`, `n` x `d`: a cv fold's
    test rows inside fit, a model-selection fold's rows at predict, lane
    vdf-meta5) without gathering them: every member is opened on its held
    native state and handed X, the row index, the resident `scope` and one
    pin reference per member call (xtrees/trees_meta_vote_run.mojo THE
    INDEXED ROWS, `_FoldPin`), so its device takes the rows out of the X the
    scope keeps resident. `check_kind`: each member's kind is asked whether
    it reads a row index and a scope (`_fold_open`); False when the caller
    already asked (fit: its mask, once per fit). False, with nothing placed
    that a gathered `_stack_place` does not place again, when a member is
    not served natively without the GIL or a member call refuses: the caller
    then gathers the rows and runs `_stack_place` (the fallback rule, whose
    doors raise the members' own words). Same rows in the same order as the
    gathered block: same bits. `place`: the int32 rows of the meta block the
    outputs go to (the fold's positions; `rows_at` is the same rows in X as
    the members read them)."""
    var count = Int(py=rows_at.shape[0])
    var got = _VoteUnpack()
    var counts = List[Int](capacity=len(members))
    var placed = False
    try:
        for i in range(len(members)):  # driver-loop: over members (admission, once each)
            var e = members[i]
            var kind = kinds[i]
            var classifier = kind != STACK_PREDICT
            var class_cols = 0
            var held = 0
            if classifier:
                var cols = _sub_cols(e, context)
                _ = py_call(got.keep.append, cols)
                class_cols = trees_address(cols)
                held = Int(py=cols.shape[0])
            counts.append(held)
            var operation = MEMBER_OP_PREDICT if kind == STACK_PREDICT else (
                MEMBER_OP_PREDICT_PROBA if kind == STACK_PROBA else MEMBER_OP_DECISION)
            if not _fold_open(got, e, Xa, operation, classifier, 0, class_cols, 1.0, check_kind, context):
                _vote_close(got)
                return False
        _fold_plans(got)
        var pin = _FoldPin(Xa, len(got.members))
        var rows = pin.rows(trees_address(Xa), n, d, trees_address(rows_at), count, scope)
        var place_at = trees_address(place)
        var refused = VOTE_ALL_ANSWERED
        try:
            with GILReleased(Python()):
                refused = stack_place_run(glue, rows, got.members, kinds, counts, k, binary, meta, meta_rows, width,
                                          place_at, 0, widths, 0)
        finally:
            pin.settle()
        placed = refused == VOTE_ALL_ANSWERED
    except e:
        _vote_close(got)
        raise e^
    _vote_close(got)
    _ = Xa  # the rows are read through RUN
    _ = rows_at
    _ = place
    _ = got.keep  # the class columns are read through RUN
    return placed


def _meta_rows_resident(Xa: PythonObject, n: Int, d: Int, context: PythonObject) raises -> Bool:
    """UNPACK, once per fit (lane vdf-meta3): whether a meta fit that reads
    its cv folds BY ROW NUMBER (`_fit_units_drawn`) runs inside a resident
    scope of its own, as CalibratedClassifierCV's (`_cal_rows_resident`): X
    a non-empty float32 C-order block and the fit on a GPU route. A member
    whose image reads a row index and keeps X resident then reads X once for
    the whole fit and takes each fold's rows on the device; a host image's
    member gathers its rows on its worker. device="cpu" beside a GPU set (the
    CPU run context) enters none: same rows, same order."""
    if n <= 0 or d <= 0 or String(py=Xa.dtype) != "<f4" or String(py=Xa.order) != "C":
        return False
    var nk = trees_keys()
    return not (py_get(context, nk[].k_device_cpu_context) is context)


def _stacking_fit(model: PythonObject, X: PythonObject, y_full: PythonObject, codes: Bool, splits: PythonObject,
                  final: PythonObject, sample_weight: PythonObject, classifier: Bool, y_given: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`_StackingBase._fit_stack`. `y_full` is the int32 codes (classifier)
    or the float32 targets (regressor). `y_given` (a regressor; None for a
    classifier): y as the door was given it, THE GIVEN y of a member that
    reads it (lane mf-tail5).

    THE FOLD ROWS BY NUMBER (lane vdf-meta3): the cv folds' train rows are
    the member set's row draws over the one X (`_fit_units_drawn`) and the
    test rows the stacking RUN's row index (`_stack_place_indexed`), inside
    one resident scope (`_meta_rows_resident`): X goes up once for the
    whole fit and nothing comes down between the folds. Before, each fold's
    train and test sides were `_gather` (all of X uploaded, the rows
    gathered and DOWNLOADED, per fold, then uploaded again by every member).
    Same rows in the same order: no bit change."""
    # THE FOLD IN PLACE (lane vdf-meta3): a DeviceRows fold is read out of its base (`_meta_x_fit`); the cv
    # folds' rows are then composed inside it (`_meta_compose`).
    var mx = _meta_x_fit(X, context)
    var mctx = PythonObject(None)
    var scope = PythonObject(None)
    var scoped = _meta_rows_resident(mx.x, mx.base_rows, mx.d, context)
    if scoped:
        mctx = prep_native_context(context)
        # Inside model selection's own `resident()` scope (a fold read in place: its base is already resident
        # there) no second scope is entered: the members find the base in the caller's.
        if member_scope_word(py_item(mctx, PythonObject("family_msel_rows_context"))) == MEMBER_SCOPE_NONE:
            scope = _cal_scope_open(mctx)
    try:
        _ = _stacking_fit_run(model, mx, y_full, codes, splits, final, sample_weight, classifier, y_given,
                              scoped, context)
    finally:
        if scope is not None:
            _cal_scope_close(scope, mctx)
    return model


def _stacking_fit_run(model: PythonObject, mx: _MetaX, y_full: PythonObject, codes: Bool, splits: PythonObject,
                      final: PythonObject, sample_weight: PythonObject, classifier: Bool, y_given: PythonObject,
                      scoped: Bool, context: PythonObject) raises -> PythonObject:
    """`_stacking_fit`'s body over the admitted X `mx` (X, or a fold's base
    with its rows); `scoped`: inside the resident scope `_stacking_fit`
    entered. The cv folds' rows (`splits`, fold positions) are composed
    into rows of `mx.x` for every read of X; the meta features, the final
    estimator's fit and the placement keep fold positions."""
    var owner = _owner(model)
    var xt = _xt(py_fields(model), context)
    # This call owns the admitted X, y and optional weight storage. Member
    # fits cannot replace these local owners. A subsequent fit admits afresh.
    var Xa = mx.x
    var x_view = Xa
    var n = mx.n
    var d = mx.d
    var nb = mx.base_rows  # rows of X as the members read it (a fold's whole base)
    var nk = trees_keys()
    var active = _active(_attr(model, nk[].k_estimators, PythonObject(None)), owner, context)
    _require_member(final, owner, context)
    var all_names = py_call(trees_builtins().list, active.names)
    var all_ests = py_call(trees_builtins().list, active.ests)
    _ = py_call(all_names.append, PythonObject("final_estimator"))
    _ = py_call(all_ests.append, final)
    # The members and the final estimator run on the ensemble's device
    # (bindings/trees_meta_device_door.mojo): read once per fit, checked before any fit.
    var device = py_get(py_fields(model), "device")
    var origin = meta_device_origin(model, device)
    meta_members_device(device, all_ests, owner, origin, context, trees_builtins())
    var sw = _member_weights(all_names, all_ests, sample_weight, n, context)
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    var n_members = len(active.ests)
    # Class counts of the code targets (None for regressors): the native
    # member sets fit every member on the ensemble's codes.
    var counts = PythonObject(None)
    var final_counts = PythonObject(None)
    if codes:
        var k_classes = PythonObject(len(_attr(model, PythonObject("classes_"), Python.list())))
        counts = _repeat(k_classes, n_members)
        final_counts = _repeat(k_classes, 1)
    var full = Python.list()
    for i in range(n_members):  # driver-loop: over members
        _ = py_call(full.append, _clone(active.ests[i], device, origin, context))
    var jobs = native_unit_jobs(_attr(model, nk[].k_n_jobs, PythonObject(None)), owner)
    var whole_given = PythonObject(None)  # THE GIVEN y of every row (lane mf-tail5)
    if not codes and y_given is not None:
        whole_given = Python.tuple(y_given, PythonObject(None), PythonObject(-1))
    # The members' targets, weights and THE GIVEN y at the rows of `mx.x` (the fold's own words for a fold
    # read in place; the same objects otherwise). The final estimator reads the fold's.
    var y_view = _meta_take(mx, y_full, context)
    var sw_rows = _meta_take(mx, sw, context)
    var given_rows = whole_given
    if whole_given is not None and mx.inv is not None:
        given_rows = Python.tuple(y_given, mx.inv, PythonObject(-1))
    var full_draws = PythonObject(None)
    if mx.draw is not None:
        full_draws = _repeat(mx.draw, n_members)
    # The full fit answers each member's kind-exact mask once per fit (one unit per member): whether every
    # member reads its test rows by number inside the scope (`indexable`).
    var member_kinds = _fit_units_drawn(full, _repeat(Xa, n_members), d, _repeat(y_view, n_members), not codes,
                                        _repeat(sw_rows, n_members), counts, jobs, context,
                                        _repeat(given_rows, n_members), full_draws, True)
    for i in range(n_members):  # driver-loop: over members
        _ = py_call(fitted.append, full[i])
    var scope_word = MEMBER_SCOPE_NONE
    if scoped:
        scope_word = member_scope_word(py_item(prep_native_context(context), PythonObject("family_msel_rows_context")))
    var indexable = scope_word != MEMBER_SCOPE_NONE and n_members > 0
    for i in range(len(member_kinds)):  # driver-loop: over members (one mask each)
        if (member_kinds[i] & MEMBER_CAN_ROW_INDEX) == 0 or (member_kinds[i] & MEMBER_CAN_SCOPE) == 0:
            indexable = False
    var methods = Python.list()
    var kinds = List[Int]()
    for i in range(len(fitted)):  # driver-loop: over members
        var method = _stack_method(model, fitted[i], classifier, context)
        _ = py_call(methods.append, method)
        kinds.append(_stack_kind(String(method)))
    _set(model, "stack_method_", methods)
    var binary = _truthy(_attr(model, nk[].k__binary, PythonObject(False)))
    var passthrough = _truthy(_attr(model, nk[].k_passthrough, PythonObject(False)))
    var head_n = n if n < 2 else 2
    var head = _gather(xt, trees_address(x_view), nb, d, _meta_compose(mx, _arange(xt, head_n, context), context),
                       _arange(xt, d, context), context)
    _ = Xa  # keep the Python object alive through the native call above
    var widths = Python.list()
    var k_classes = len(_attr(model, PythonObject("classes_"), Python.list())) if classifier else 0
    var glue = _xr(xt)
    # The width probe over the first rows: the stacking RUN with no meta
    # block (`stack_place_run`, meta 0) answers the columns each member takes.
    var native_widths = _stack_place(glue, fitted, kinds, head, k_classes, binary, 0, 0, 0, 0, 0, False, context)
    _ = Xa  # keep the Python object alive through the native call above
    var total = 0
    for i in range(len(native_widths)):  # driver-loop: over members
        _ = py_call(widths.append, PythonObject(native_widths[i]))
        total += native_widths[i]
    var width = total + (d if passthrough else 0)
    var meta = _zeros(n * width, context)
    var meta_view = meta
    var cols_view = _arange(xt, d, context)
    # The cv folds x members are independent fits: fitted as ONE unit set per
    # wave of folds (lane ma-callers round 2; was one fold after another,
    # only a fold's members together). A wave holds as many folds' train
    # blocks as its workers can use at once: ceil(workers / members) folds on
    # a CPU route (workers = the cores, bounded by n_jobs), one fold on a GPU
    # (its member fits share the one device), so the memory held is one train
    # block per fold of the wave and nothing is held that no worker reads.
    # The gathers and the members' test outputs stay one after another: they
    # share this owner's `xt` program state (and the one device).
    var n_folds = len(splits)
    var wave = max(1, (native_unit_device_workers(jobs, context) + n_members - 1) // max(n_members, 1))
    var f0 = 0
    while f0 < n_folds:  # driver-loop: over waves of cv folds
        var f1 = min(n_folds, f0 + wave)
        var unit_members = Python.list()
        var unit_x = Python.list()
        var unit_y = Python.list()
        var unit_w = Python.list()
        var unit_given = Python.list()
        var unit_draws = Python.list()
        for f in range(f0, f1):  # driver-loop: over the wave's cv folds (their train row numbers)
            # THE FOLD ROWS BY NUMBER (lane vdf-meta3): every member of the
            # fold reads X, y, the weights and THE GIVEN y of every row, its
            # draw the fold's int32 train rows (the set selects all of them
            # with it): no fold block is copied here. Members never write
            # their inputs (the scikit-learn fit contract).
            var tr_rows = _meta_compose(mx, splits[f][0], context)
            for i in range(n_members):  # driver-loop: over members
                _ = py_call(unit_members.append, _clone(active.ests[i], device, origin, context))
                _ = py_call(unit_x.append, Xa)
                _ = py_call(unit_y.append, y_view)
                _ = py_call(unit_w.append, sw_rows)
                _ = py_call(unit_given.append, given_rows)
                _ = py_call(unit_draws.append, tr_rows)
        var unit_counts = PythonObject(None)
        if codes:
            unit_counts = _repeat(final_counts[0], len(unit_members))
        _ = _fit_units_drawn(unit_members, unit_x, d, unit_y, not codes, unit_w, unit_counts, jobs, context,
                             unit_given, unit_draws, False)
        _ = py_call(unit_x.clear)
        _ = py_call(unit_y.clear)
        _ = py_call(unit_w.clear)
        _ = py_call(unit_draws.clear)
        for f in range(f0, f1):  # driver-loop: over the wave's cv folds (test outputs)
            var te_view = splits[f][1]
            var te_rows = _meta_compose(mx, te_view, context)  # the test rows in X as the members read it
            var fold_members = Python.list()
            for i in range(n_members):  # driver-loop: over members
                _ = py_call(fold_members.append, unit_members[(f - f0) * n_members + i])
            # The fold's stacked outputs, typed (`stack_place_run`): a fold
            # whose training rows lack a class places its members' columns
            # at their classes (scikit-learn `_enforce_prediction_order`).
            # THE FOLD ROWS BY NUMBER: every member reads its test rows by
            # number out of the resident X (`_stack_place_indexed`); else
            # (a member that does not, a door member, a refusal) the fold's
            # test block is gathered as before.
            var got = List[Int]()
            var placed = False
            if indexable:
                placed = _stack_place_indexed(glue, fold_members, kinds, Xa, nb, d, te_rows, te_view, scope_word,
                                              k_classes, binary, trees_address(meta_view), n, width, got, False, context)
            if not placed:
                var Xte = _gather(xt, trees_address(x_view), nb, d, te_rows, cols_view, context)
                _ = Xa  # keep the Python object alive through the native call above
                if admitted_is_held(Xa):
                    admitted_hold_rows_of(Xa, Xte, trees_address(Xte), Int(py=te_view.shape[0]), d)
                got = _stack_place(glue, fold_members, kinds, Xte, k_classes, binary, trees_address(meta_view), n,
                                   width, trees_address(te_view), 0, True, context)
            _ = meta_view  # written through the RUN above
            _ = te_view  # the rows are read through the RUN above
            _ = te_rows
            for i in range(n_members):  # driver-loop: over members
                if got[i] != native_widths[i]:
                    raise Error("a member's output width changed between folds")
        f0 = f1
    if passthrough:
        # The fold's own rows of X (a fold read in place: gathered once here, for the passthrough columns).
        var x_fold = Xa
        if mx.draw is not None:
            x_fold = _gather(xt, trees_address(x_view), nb, d, mx.draw, cols_view, context)
        var xb = admitted_words(x_fold, trees_address(x_fold), n, d)
        _ = Xa  # keep the Python object alive through the native call above
        _ = _place(xt, meta_view, n, width, xb.view(), _arange(xt, n, context), total)
        _ = x_fold
        _ = Xa  # keep the Python object alive through the native call above
        _ = xb^
        _ = Xa  # keep the Python object alive through the native call above
    _ = x_view
    _ = y_view
    _set(model, "_widths_", widths)
    _set(model, "named_estimators_", _named(active.names, fitted))
    var final_ = _clone(final, device, origin, context)
    _set(model, "final_estimator_", final_)
    var meta32 = _f32_array(trees_reshape(meta, Python.tuple(n, width), context), 2, "meta", context)
    _fit_members(_repeat(final_, 1), meta32, n, width, _repeat(y_full, 1), not codes, sw, final_counts, jobs,
                 context, _repeat(whole_given, 1))
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "_fitted", PythonObject(True))
    return model


struct StackLayout(Movable):
    """X-FREE UNPACK of a fitted Stacking estimator's meta features (lane mf-shap3, 2026-11-23;
    `_stacking_transform` and Kernel / Permutation SHAP): its members (`estimators_`), what each is asked for
    (`stack_method_` as STACK_*), the class count (0 for a regressor), `_binary`, `passthrough` and the meta
    features' width (the members' `_widths_`, then X's `d` columns under passthrough)."""
    var members: PythonObject
    var kinds: List[Int]
    var k: Int
    var binary: Bool
    var passthrough: Bool
    var width: Int

    def __init__(out self, members: PythonObject, var kinds: List[Int], k: Int, binary: Bool, passthrough: Bool,
                 width: Int):
        self.members = members
        self.kinds = kinds^
        self.k = k
        self.binary = binary
        self.passthrough = passthrough
        self.width = width


def stacking_layout(model: PythonObject, d: Int, context: PythonObject) raises -> StackLayout:
    """`StackLayout` of the fitted `model` over rows of `d` features."""
    var nk = trees_keys()
    var widths = _attr(model, nk[].k__widths_, Python.list())
    var total = 0
    for i in range(len(widths)):  # driver-loop: over members
        total += Int(py=widths[i])
    var passthrough = _truthy(_attr(model, nk[].k_passthrough, PythonObject(False)))
    var binary = _truthy(_attr(model, nk[].k__binary, PythonObject(False)))
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var methods = _attr(model, nk[].k_stack_method_, Python.list())
    var kinds = List[Int]()
    for i in range(len(members)):  # driver-loop: over members
        kinds.append(_stack_kind(String(methods[i])))
    var k = len(_attr(model, nk[].k_classes_, Python.list()))  # 0 for a regressor
    return StackLayout(members, kinds^, k, binary, passthrough, total + (d if passthrough else 0))


def _stacking_transform(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_StackingBase.transform`: the meta features (float64).

    THE FOLD FROM THE STORE (lane vdf-meta5; the classes opt in with
    `_mojolearn_device_rows`): a model-selection fold (`_check_meta_x`) is
    stacked by number (`_stack_place_indexed` over its host base and its
    rows, inside model selection's resident scope): each member's device
    takes the fold's rows out of the resident base, no row is gathered on
    the host. A member that is not served that way, or a refusal, admits
    the fold's rows (`_x_array`, the gather every call made before) and
    stacks them as before; passthrough reads the admitted rows (the meta
    block is host float64 by the members' output ABI). Same rows in the
    same order: same bits."""
    var mx = _check_meta_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = mx.n
    var d = mx.d
    var layout = stacking_layout(model, d, context)
    var passthrough = layout.passthrough
    var width = layout.width
    var meta = _zeros(n * width, context)
    var meta_view = meta
    var rows_view = _arange(xt, n, context)
    var binary = layout.binary
    var col0 = 0
    var members = layout.members
    var k = layout.k
    var indexed = False
    if len(members) > 0:
        var scope = _fold_scope(mx, context)
        if scope != MEMBER_SCOPE_NONE:
            var widths = List[Int]()
            indexed = _stack_place_indexed(_xr(xt), members, layout.kinds, mx.x, mx.base_rows, d, mx.draw,
                                           rows_view, scope, k, binary, trees_address(meta_view), n, width, widths,
                                           True, context)
            if indexed:
                for i in range(len(widths)):  # driver-loop: over members
                    col0 += widths[i]
    var Xa = mx.x
    if mx.draw is not None and ((len(members) > 0 and not indexed) or passthrough):
        Xa = _x_array(X, context)  # the fold's rows, admitted as before
    # Lane bm-meta-3: UNPACK + RUN `stack_place_run` (members fitted on
    # every row saw every class: no class columns read).
    if len(members) > 0 and not indexed:
        var got = _stack_place(_xr(xt), members, layout.kinds, Xa, k, binary, trees_address(meta_view), n,
                               width, trees_address(rows_view), 0, False, context)
        for i in range(len(got)):  # driver-loop: over members
            col0 += got[i]
    _ = meta_view  # written through the RUN above
    _ = rows_view  # read through the RUN above
    _ = mx.x
    _ = mx.draw
    if passthrough:
        var xb = admitted_words(Xa, trees_address(Xa), n, d)
        _ = Xa  # keep the Python object alive through the native call above
        _ = _place(xt, meta_view, n, width, xb.view(), rows_view, col0)
        _ = Xa  # keep the Python object alive through the native call above
        _ = xb^
        _ = Xa  # keep the Python object alive through the native call above
    return trees_reshape(meta, Python.tuple(n, width), context)


def _stacking_classifier_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    _stacking_validate(model, context)
    var labels = trees_encode_labels(y, labels_scope(model, context))
    _set(model, "classes_", labels.classes)
    var k = len(labels.classes)
    _set(model, "_binary", PythonObject(k == 2))
    var xt = _xt(py_fields(model), context)
    var nk = trees_keys()
    var splits = _splits(xt, _attr(model, nk[].k_cv, PythonObject(None)), X, y, labels.n, labels.codes, True, k, context)
    var final = _attr(model, nk[].k_final_estimator, PythonObject(None))
    if final is None:
        final = prep_estimator_clone(py_item(py_item(context, nk[].k_native_estimator_defaults), nk[].k_LogisticRegression), context)
    return _stacking_fit(model, X, labels.codes, True, splits, final, sample_weight, True, PythonObject(None), context)


def _stacking_regressor_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    _stacking_validate(model, context)
    var y32 = _f32_array(y, 1, "y", context)
    var xt = _xt(py_fields(model), context)
    var nk = trees_keys()
    var splits = _splits(xt, _attr(model, nk[].k_cv, PythonObject(None)), X, y, _rows(y32), PythonObject(None), True, 0,
                         context)
    var final = _attr(model, nk[].k_final_estimator, PythonObject(None))
    if final is None:
        final = prep_estimator_clone(py_item(py_item(context, nk[].k_native_estimator_defaults), nk[].k_Ridge), context)
    return _stacking_fit(model, X, y32, False, splits, final, sample_weight, False, y, context)


def _stacking_meta32(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    return _f32_array(_stacking_transform(model, X, context), 2, "meta", context)


def _stacking_predict(model: PythonObject, X: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    var final_ = _attr(model, nk[].k_final_estimator_, PythonObject(None))
    var Xm = _stacking_meta32(model, X, context)
    if not classifier:
        return meta_member_call(final_, "predict", Python.tuple(Xm), context)
    return trees_decode_labels(_attr(model, nk[].k_classes_, PythonObject(None)),
                               _codes_of(Python.list(final_), Xm, context)[0], context)


def _stacking_codes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`StackingClassifier.predict` before its decode: the final estimator's
    int32 codes into `classes_`."""
    var nk = trees_keys()
    var final_ = _attr(model, nk[].k_final_estimator_, PythonObject(None))
    return _codes_of(Python.list(final_), _stacking_meta32(model, X, context), context)[0]


def _stacking_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    var final_ = _attr(model, nk[].k_final_estimator_, PythonObject(None))
    var ks: List[Int] = [len(_attr(model, nk[].k_classes_, Python.list()))]
    var Xm = _stacking_meta32(model, X, context)
    return _probas_of(model, Python.list(final_), ks, Xm, context)[0]


# ------------------------------------------------------------- MultiOutput


def _multioutput_y(Y: PythonObject) raises:
    """`_trees_multioutput_y`: scikit-learn's refusal of a 1-D target."""
    var b = trees_builtins()
    var nk = trees_keys()
    var nd = _attr(Y, nk[].k_ndim, PythonObject(None))
    if nd is None and Bool(py=py_call(b.isinstance, Y, Python.tuple(b.list, b.tuple))):
        var nested = len(Y) > 0 and Bool(py=py_call(b.isinstance, Y[0], Python.tuple(b.list, b.tuple)))
        nd = PythonObject(2 if nested else 1)
    if nd is not None and Int(py=py_call(b.int, nd)) < 2:
        raise Error("y must have at least two dimensions for multi-output regression but has only one.")


def _ridge_multi_entry(model: PythonObject, est: PythonObject, sample_weight: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`_multiout_ridge_entry`: the estimators binding table when it has the
    one-program ridge (FAST + Apple builds) and `est` is a plain Ridge, else
    None."""
    var nk = trees_keys()
    var ridge = py_type(py_item(py_item(context, nk[].k_native_estimator_defaults), nk[].k_Ridge))
    var plain = py_type(est) is ridge
    if plain:
        # Ridge checks its parameters at fit; this route skips Ridge.fit.
        _ = meta_member_call(est, "validate_params", Python.tuple(), context)
    if sample_weight is not None or not plain or _truthy(_attr(est, nk[].k_normalize, PythonObject(False))):
        return PythonObject(None)
    # The one-program ridge runs on the process's device table. A member
    # constructed with device="cpu" keeps its own route: each output is then
    # fitted by the member's fit, through the classical owner's CPU table
    # (lane pilot-ridge-device; read once per fit, never per row).
    var member_device = _attr(est, PythonObject("device"), PythonObject(None))
    if member_device is not None and String(member_device) == "cpu":
        return PythonObject(None)
    try:
        var table = trees_table(context, _mode(py_fields(est), context), "_mojolearn_estimators")
        if not ridge_multi_available(_estimators_key(table)):
            return PythonObject(None)
        return table
    except:
        return PythonObject(None)


def _estimators_key(table: PythonObject) raises -> LinearEntryKey:
    """UNPACK: the connected estimators image's ridge multi table key, which
    the trees bridge recorded at import (`estimators_native_key`)."""
    var key = py_get(table, PythonObject("estimators_native_key"))
    if key is None:
        raise Error("RuntimeError: mojolearn: the estimators binding has no native table key")
    return LinearEntryKey(Int(py=key[0]), Int(py=key[1]))


def _multioutput_ridge_fit(model: PythonObject, table: PythonObject, Xa: PythonObject, Ya: PythonObject,
                           device: PythonObject, origin: String, context: PythonObject) raises -> PythonObject:
    """`MultiOutputRegressor._fit_ridge_multi`: every target in one ridge
    program (glm/impl/ridge_multi.mojo)."""
    # UNPACK: the dimensions, the member's two parameters, the image's entry
    # and the buffers RUN writes (allocated here, published in PACK).
    var n = _rows(Xa)
    var d = _cols(Xa)
    var m = _cols(Ya)
    var nk = trees_keys()
    var est = _attr(model, nk[].k_estimator, PythonObject(None))
    var intercept = _truthy(_attr(est, nk[].k_fit_intercept, PythonObject(True)))
    # the member's alpha as Python's float() reads it
    var alpha = Float64(py=py_call_typed(trees_builtins().float, _attr(est, nk[].k_alpha, PythonObject(1.0))))
    var key = ridge_multi_ref(_estimators_key(table))  # resolved once for the fit
    var request = RidgeMultiFit(trees_address(Xa), trees_address(Ya), n, d, m, alpha, intercept)
    var xm = admitted_vector(d, "<f4", context)
    var coef = admitted_vector(m * d, "<f4", context)
    var ymean = admitted_vector(m, "<f4", context)
    var icpt32 = admitted_vector(m, "<f4", context)
    request.x_mean = trees_address(xm)
    request.coef = trees_address(coef)
    request.y_mean = trees_address(ymean)
    request.icpt = trees_address(icpt32)
    # the centering scratch, only with an intercept (`_column_means`, `_center`)
    var sums = PythonObject(None)
    var m64 = PythonObject(None)
    var work_x = PythonObject(None)
    if intercept:
        sums = _zeros(d, context)
        m64 = _zeros(d, context)
        work_x = admitted_array(Python.tuple(n, d), "<f4", context)
        request.sums = trees_address(sums)
        request.means64 = trees_address(m64)
        request.work = trees_address(work_x)
    # RUN: the image's four programs on plain values, the GIL released once
    # for all of them (it was released and retaken around each)
    with GILReleased(Python()):
        ridge_multi_fit_run(key, request)
    _ = sums  # keep the stores alive through the native calls above
    _ = m64
    _ = work_x
    _ = xm
    _ = Xa
    _ = Ya
    _ = icpt32
    _ = ymean
    _ = coef
    # PACK: one fitted ridge per output column and the batched state
    var fitted = Python.list()
    var ym = _F32P(unsafe_from_address=trees_address(ymean))
    _ = ymean  # keep the Python object alive through the native call above
    var ic = _F32P(unsafe_from_address=trees_address(icpt32))
    _ = icpt32  # keep the Python object alive through the native call above
    for j in range(m):  # driver-loop: over the output columns (one fitted ridge each)
        var e = _clone(est, device, origin, context)
        _set(e, "solver_", PythonObject("eig"))
        _set(e, "coef_", _copy_block(coef, j * d, (j + 1) * d, "<f4", context))
        _set(e, "_x_mean", xm)
        _set(e, "_y_mean", PythonObject(Float64(ym[j]) if intercept else 0.0))
        _ = ymean  # keep the Python object alive through the native call above
        _set(e, "intercept_", PythonObject(Float64(ic[j]) if intercept else 0.0))
        _ = icpt32  # keep the Python object alive through the native call above
        _set(e, "n_features_in_", PythonObject(d))
        _ = py_call(fitted.append, e)
    _set(model, "estimators_", fitted)
    _set(model, "_multi_ridge", Python.tuple(coef, icpt32, m))
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _multioutput_regressor_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                               context: PythonObject) raises -> PythonObject:
    var jobs = _jobs(model, False)
    var xt = _xt(py_fields(model), context)
    # THE FOLD IN PLACE (lane vdf-meta4; the class opts in with `_mojolearn_device_rows`): a DeviceRows fold is
    # read out of its base (`_meta_x_fit`): the members draw the fold's rows (`_fit_members_x`), a row-index
    # member taking them on the device out of the base model selection keeps resident.
    var mx = _meta_x_fit(X, context)
    _multioutput_y(y)
    var Ya = _f32_array(y, 2, "y", context)
    if _rows(Ya) != mx.n:
        raise Error("y has " + String(_rows(Ya)) + " rows, X has " + String(mx.n))
    var nk = trees_keys()
    var est = _attr(model, nk[].k_estimator, PythonObject(None))
    _require_member(est, _owner(model), context)
    # The member runs on the ensemble's device (bindings/trees_meta_device_door.mojo): read once per fit.
    # The one-program ridge reads its table from this estimator's run context (`context`).
    var device = py_get(py_fields(model), "device")
    var origin = meta_device_origin(model, device)
    meta_members_device(device, Python.list(est), _owner(model), origin, context, trees_builtins())
    var weights = _member_weights(Python.list(PythonObject(None)), Python.list(est), sample_weight, mx.n, context)
    _set(model, "_multi_ridge", PythonObject(None))
    var table = _ridge_multi_entry(model, est, weights, context)
    if table is not None and _cols(Ya) > 0:
        # The one-program ridge (FAST + Apple only, glm/impl/ridge_multi.mojo MULTIOUT_RIDGE) reads dense host
        # words: a fold's rows are gathered ONCE here, as the admission gathered them before (owed: its four
        # programs reading the estimators store's slot of the base, the fold taken on the device).
        var Xr = mx.x
        if mx.draw is not None:
            Xr = _gather(xt, trees_address(mx.x), mx.base_rows, mx.d, mx.draw, _arange(xt, mx.d, context), context)
            _ = mx.x  # read through its address above
        return _multioutput_ridge_fit(model, table, Xr, Ya, device, origin, context)
    var n = mx.n
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    var rows = _arange(xt, n, context)
    var columns = Python.list()
    var clones = Python.list()
    var column_given = Python.list()  # THE GIVEN y's column j per member (lane mf-tail5)
    for j in range(_cols(Ya)):  # driver-loop: over the output columns (one member each)
        var col = _codes_array(Python.list(PythonObject(j)), context)
        var yj = trees_reshape(_gather(xt, trees_address(Ya), n, _cols(Ya), rows, col, context), Python.tuple(n), context)
        _ = Ya  # keep the Python object alive through the native call above
        _ = py_call(columns.append, yj)
        _ = py_call(clones.append, _clone(est, device, origin, context))
        _ = py_call(column_given.append, Python.tuple(y, PythonObject(None), PythonObject(j)))
    # The columns are independent members: one native member set (parallel;
    # a member the set does not take is refused in its refuser's words).
    _fit_members_x(clones, mx, columns, True, weights, PythonObject(None), jobs, context, column_given)
    for j in range(len(clones)):  # driver-loop: over the output columns (one member each)
        _ = py_call(fitted.append, clones[j])
    _set(model, "n_features_in_", PythonObject(mx.d))
    _set(model, "_fitted", PythonObject(True))
    return model


def multioutput_ridge_ref(members: PythonObject, context: PythonObject) raises -> EntryRef:
    """X-FREE UNPACK of a MultiOutputRegressor fitted as one multi-output ridge (`_multi_ridge`; lane mf-shap3,
    2026-11-23: `_multioutput_regressor_predict` and Kernel / Permutation SHAP): the ridge multi entry of the
    estimators image its first member's tier names, resolved once per operation."""
    var table = trees_table(context, _mode(py_fields(members[0]), context), "_mojolearn_estimators")
    return ridge_multi_ref(_estimators_key(table))


def _multioutput_regressor_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var mx = _check_meta_x(model, X, context)  # a fold from the store (lane vdf-meta4)
    var xt = _xt(py_fields(model), context)
    var n = mx.n
    var nk = trees_keys()
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var m = len(members)
    var mr = _attr(model, nk[].k__multi_ridge, PythonObject(None))
    if mr is not None:
        # The one-program ridge (FAST + Apple) reads dense host rows: a fold's are admitted (owed with its fit).
        var Xa = mx.x
        if mx.draw is not None:
            Xa = _x_array(X, context)
        var outputs = Int(py=mr[2])
        var out = admitted_array(Python.tuple(n, outputs), "<f4", context)
        var rm_key = multioutput_ridge_ref(members, context)
        var p_x = trees_address(Xa)
        var p_coef = trees_address(mr[0])
        var p_icpt = trees_address(mr[1])
        var p_out = trees_address(out)
        var p_cols = _cols(Xa)
        with GILReleased(Python()):  # the former ridge_predict_multi export released it
            rm_predict(rm_key, p_x, p_coef, p_icpt, p_out, n, p_cols, outputs)
        _ = mr  # keep the coef and intercept stores alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = Xa  # keep the Python object alive through the native call above
        return out
    # THE MULTIOUTPUT SPLIT (lane bm-meta-2, 2026-11-22): the members'
    # predictions side by side are VotingRegressor.transform's body, so this
    # is that UNPACK / RUN (`_vote_run`, kind _VOTE_BLOCK; RUN
    # xtrees/trees_meta_vote_run.mojo `vote_transform_run`): each member's
    # held native state (its float32 predict, the GIL released) or its door's
    # float32 block (`_out2d` of its internal predict, as before), placed at
    # column j by the same glue scatter over the same rows. Before the split
    # this loop asked each member through `_predict32`, made a new Array of
    # its answer and placed it here (fb12-rows, 2026-11-17: the float32
    # internal predict, never the public float64 widening): the same values
    # in the same order, so no bits change.
    var out = _zeros(n * m, context)
    _vote_run_x(model, X, mx, _VOTE_BLOCK, False, False, 1, PythonObject(None), _xr(xt), trees_address(out), 0, m,
                context)
    return trees_reshape(out, Python.tuple(n, m), context)


def _has_buffer(value: PythonObject) raises -> Bool:
    """`_buffer._has_buffer`: `memoryview(value)` succeeds."""
    try:
        var view = py_call(trees_builtins().memoryview, value)
        _ = py_call(view.release)
        return True
    except:
        return False


def _label_columns(Y: PythonObject, xt: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_label_words(Y)` then its native transpose: a numeric 2-D label
    buffer as `(T, n, m)`, T the (m, n) 8-byte words (float64 for a float
    dtype, int64 for an integer one) whose rows are the label columns; None
    for anything else (bool, str, object, nested lists, uint64), which keeps
    the Python rows."""
    var b = trees_builtins()
    var nk = trees_keys()
    var dtype = _attr(Y, nk[].k_dtype, PythonObject(None))
    var nd = _attr(Y, nk[].k_ndim, PythonObject(None))
    if nd is None or not Bool(py=nd == PythonObject(2)):
        return PythonObject(None)
    var kind = _attr(dtype, nk[].k_kind, PythonObject(None))
    if kind is None:
        if py_type(dtype) is not b.str:
            return PythonObject(None)
        kind = py_call(py_attr(py_call(dtype.lstrip, "<>=|"), "__getitem__"), py_call(b.slice, PythonObject(0), PythonObject(1)))
    var k = String(kind)
    if k != "f" and k != "i" and k != "u":
        return PythonObject(None)
    var target = String("<f8") if k == "f" else String("<i8")
    var glue = _xr(xt)
    try:
        # Y in C order (admit_c_view: an F-ordered or strided Y is one
        # C-order copy, so the column-major arm is gone).
        var view = admit_c_view(Y, "Y", context)
        var vt = admitted_dtype(view, context)
        if Int(py=view.ndim) != 2:
            return PythonObject(None)
        if k != "f" and vt != "<i1" and vt != "<i4" and vt != "<i8" and vt != "<u1" and vt != "<u2" and vt != "<u4":
            return PythonObject(None)
        var n = Int(py=view.shape[0])
        var m = Int(py=view.shape[1])
        var T = admitted_vector(m * n, target, context)
        if vt == target:
            # fb-boost-copy (2026-11-17): Y already holds the 8-byte words the
            # transpose reads; it was cast float64->float64 (int64->int64)
            # into a zero-filled n x m copy first on every fit (O(n m)).
            _ = xt_transpose_f64(glue, Int(py=classical_address(view)), trees_address(T), n, m)
        else:
            var W = admitted_vector(n * m, target, context)
            trees_cast(Int(py=classical_address(view)), vt, trees_address(W), target, n * m, context)
            _ = xt_transpose_f64(glue, trees_address(W), trees_address(T), n, m)
            _ = W  # keep the Python object alive through the native call above
        _ = view  # keep the Python object alive through the native calls above
        _ = T  # keep the Python object alive through the native call above
        _ = view^
        return Python.tuple(T, n, m)
    except:
        return PythonObject(None)


def _multioutput_classifier_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    var jobs = _jobs(model, False)
    # THE FOLD IN PLACE (lane vdf-meta3): a DeviceRows fold is read out of its base (`_meta_x_fit`).
    var mx = _meta_x_fit(X, context)
    var n = mx.n
    var target = y
    var listed = py_type(y) is b.list or py_type(y) is b.tuple
    if not listed and native_pandas_exact(y):
        target = _label_list(y, context)
    var nk = trees_keys()
    var nd = _attr(target, nk[].k_ndim, PythonObject(None))
    if nd is not None and Bool(py=nd == PythonObject(1)):
        raise Error("y must have at least two dimensions for multi-output classification but has only one.")
    _multioutput_y(target)
    var est = _attr(model, nk[].k_estimator, PythonObject(None))
    _require_member(est, _owner(model), context)
    # The member runs on the ensemble's device (bindings/trees_meta_device_door.mojo): read once per fit.
    var device = py_get(py_fields(model), "device")
    var origin = meta_device_origin(model, device)
    meta_members_device(device, Python.list(est), _owner(model), origin, context, b)
    var weights = _member_weights(Python.list(PythonObject(None)), Python.list(est), sample_weight, n, context)
    var fitted = Python.list()
    var classes = Python.list()
    # One clone, one encoded code column and one class count per output
    # column; the columns are independent members fitted together below.
    var clones = Python.list()
    var code_columns = Python.list()
    var counts = Python.list()
    # The label steps run on the estimator's device (labels_scope), read once.
    var label_context = labels_scope(model, context)
    var columns = _label_columns(target, _xt(py_fields(model), context), context)
    if columns is not None:
        var T = columns[0]
        var rows_n = Int(py=columns[1])
        var m = Int(py=columns[2])
        if rows_n != n:
            raise Error("Y has " + String(rows_n) + " rows, X has " + String(n))
        _set(model, "estimators_", fitted)
        _set(model, "classes_", classes)
        for j in range(m):  # driver-loop: over the output columns (one member each)
            var labels = trees_encode_labels(_copy_block(T, j * rows_n, (j + 1) * rows_n, String(T.dtype), context), label_context)
            _ = py_call(clones.append, _clone(est, device, origin, context))
            _ = py_call(code_columns.append, labels.codes)
            _ = py_call(counts.append, PythonObject(len(labels.classes)))
            _ = py_call(classes.append, labels.classes)
    else:
        var rows = Python.list()
        if _has(target, nk[].k_tolist):
            rows = _label_list(target, context)  # cpu-route: label objects (str/mixed) no Array holds, one output column at a time
        else:
            rows = py_call(b.list, py_call(b.map, b.list, target))  # cpu-route: label objects (str/mixed) no Array holds
        if len(rows) != n:
            raise Error("y has " + String(len(rows)) + " rows, X has " + String(n))
        if len(rows) > 0 and not _has(rows[0], nk[].k___len__):
            raise Error("y must have at least two dimensions for multi-output classification but has only one.")
        var m = len(rows[0])
        _set(model, "estimators_", fitted)
        _set(model, "classes_", classes)
        var getter = native_loaded_module("operator").itemgetter
        for j in range(m):  # driver-loop: over the output columns (one member each)
            var column = py_call(b.list, py_call(b.map, py_call(getter, PythonObject(j)), rows))  # cpu-route: label objects (str/mixed) no Array holds, one output column at a time
            var labels = trees_encode_labels(column, label_context)
            _ = py_call(clones.append, _clone(est, device, origin, context))
            _ = py_call(code_columns.append, labels.codes)
            _ = py_call(counts.append, PythonObject(len(labels.classes)))
            _ = py_call(classes.append, labels.classes)
    _fit_members_x(clones, mx, code_columns, False, weights, counts, jobs, context, PythonObject(None))
    for j in range(len(clones)):  # driver-loop: over the output columns (one member each)
        _ = py_call(fitted.append, clones[j])
    _set(model, "n_features_in_", PythonObject(mx.d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _multioutput_member_codes(model: PythonObject, X: PythonObject, mx: _MetaX,
                              context: PythonObject) raises -> PythonObject:
    """Each output's member codes (int32, into that output's `classes_[j]`;
    the members were fitted on the column codes), lane bm-meta-3: UNPACK
    `_vote_unpack` (kind _VOTE_HARD: per member its held native state, else
    its door's codes, `_member_label_codes`'s routes in its order), RUN
    `multioutput_codes_run` (no Python; the GIL released when a member is
    native) into one int32 Array per output allocated here (PACK), which the
    native call writes directly, as `member_held_run`'s new Array was.
    THE FOLD FROM THE STORE (lane vdf-meta5): over a model-selection fold
    `mx` the members read the fold by number (`_codes_fold`); a fold they
    do not serve is admitted (`_x_array`) as before."""
    var nk = trees_keys()
    return _codes_of_x(_attr(model, nk[].k_estimators_, Python.list()), X, mx, context)


def _codes_of_x(members: PythonObject, X: PythonObject, mx: _MetaX, context: PythonObject) raises -> PythonObject:
    """`_codes_of` over `_check_meta_x`'s `mx`: a fold inside model
    selection's resident scope FROM THE STORE (`_codes_fold`), else the
    fold's rows admitted (`_x_array`), else `mx.x` itself."""
    if mx.draw is None:
        return _codes_of(members, mx.x, context)
    var scope = _fold_scope(mx, context)
    if scope != MEMBER_SCOPE_NONE:
        var got = _codes_fold(members, mx, scope, context)
        if got is not None:
            return got
    return _codes_of(members, _x_array(X, context), context)


def _codes_fold(members: PythonObject, mx: _MetaX, scope: Int, context: PythonObject) raises -> PythonObject:
    """THE FOLD FROM THE STORE (lane vdf-meta5): `_codes_of` over a fold
    `mx` with every member opened on its held native state over the base
    (`_fold_open`) and handed the fold's row index, `scope` and one pin
    reference per call (`_FoldPin`): its device takes the fold's rows out of
    the resident base. None, nothing published, when a member is not served
    that way or a call refuses: the caller admits the fold's rows and asks
    `_codes_of` (whose doors raise the members' words). Same rows, same
    order: same codes."""
    var got = _VoteUnpack()
    var out = Python.list()
    var answered = False
    try:
        var opened_all = True
        for i in range(len(members)):  # driver-loop: over members (admission, once each)
            if not _fold_open(got, members[i], mx.x, MEMBER_OP_PREDICT, True, 0, 0, 1.0, True, context):
                opened_all = False
                break
        if opened_all:
            _fold_plans(got)
            var dst = List[Int](capacity=len(got.members))
            for j in range(len(got.members)):  # driver-loop: over members (outputs)
                var col = admitted_vector(mx.n, "<i4", context)
                _ = py_call(out.append, col)
                dst.append(trees_address(col))
            var pin = _FoldPin(mx.x, len(got.members))
            var rows = pin.rows(trees_address(mx.x), mx.base_rows, mx.d, trees_address(mx.draw), mx.n, scope)
            var refused = VOTE_ALL_ANSWERED
            try:
                with GILReleased(Python()):
                    refused = multioutput_codes_run(rows, got.members, dst, 0)
            finally:
                pin.settle()
            answered = refused == VOTE_ALL_ANSWERED
    except e:
        _vote_close(got)
        raise e^
    _vote_close(got)
    _ = mx.x  # the rows are read through RUN
    _ = mx.draw
    if not answered:
        return PythonObject(None)
    return out


def _codes_of(members: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """The int32 codes of each fitted classifier member of `members` (its
    classes are the ensemble's int codes): UNPACK `_vote_unpack_of`, RUN
    `multioutput_codes_run`; one int32 Array per member (a MultiOutput
    column, a Stacking final estimator)."""
    var got = _vote_unpack_of(members, Xa, _VOTE_HARD, True, PythonObject(None), context)
    var n = _rows(Xa)
    var out = Python.list()
    var dst = List[Int](capacity=len(got.members))
    try:
        for j in range(len(got.members)):  # driver-loop: over members (outputs)
            var col = got.blocks[j]
            if got.members[j].native:
                col = admitted_vector(n, "<i4", context)
            _ = py_call(out.append, col)
            dst.append(trees_address(col))
        var rows = VoteRows(trees_address(Xa), n, _cols(Xa))
        var start = 0
        while True:  # driver-loop: over refusals (at most one round per native member)
            var refused = VOTE_ALL_ANSWERED
            if got.native > 0:
                with GILReleased(Python()):
                    refused = multioutput_codes_run(rows, got.members, dst, start)
            else:
                refused = multioutput_codes_run(rows, got.members, dst, start)
            if refused == VOTE_ALL_ANSWERED:
                break
            # The fallback rule: the refusing member's door raises its words.
            got.members[refused] = _vote_door(members[refused], _VOTE_HARD, Xa, n, 1.0, got.keep, context,
                                              PythonObject(None))
            py_setitem(out, PythonObject(refused), got.keep[len(got.keep) - 1])
            dst[refused] = got.members[refused].address
            got.native -= 1
            start = refused + 1
    except e:
        _vote_close(got)
        raise e^
    _vote_close(got)
    _ = Xa  # the rows are read through RUN
    _ = got.keep  # the door blocks are read through RUN
    return out


def _multioutput_classifier_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var mx = _check_meta_x(model, X, context)  # a fold from the store (lane vdf-meta5)
    return _multioutput_assemble(model, mx.n, _multioutput_member_codes(model, X, mx, context), context)


def _multioutput_assemble(model: PythonObject, n: Int, codes: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """`MultiOutputClassifier.predict` from each output's member codes (`n`
    rows): the decoded columns stacked into one (n, m) Array, or per-row
    label lists for label objects no Array holds."""
    var b = trees_builtins()
    var xt = _xt(py_fields(model), context)
    var nk = trees_keys()
    var classes = _attr(model, nk[].k_classes_, Python.list())
    var cols = Python.list()
    var stacked = True
    var all_int = True
    for j in range(len(codes)):  # driver-loop: over members
        var col = trees_decode_labels(classes[j], codes[j], context)
        _ = py_call(cols.append, col)
        if py_type(col) is not py_item(context, nk[].k_array_class):
            stacked = False
        elif String(col.dtype) != "<i8":
            all_int = False
            if String(col.dtype) != "<f8":
                stacked = False
    var glue = _xr(xt)
    if stacked:
        var kind = String("<i8") if all_int else String("<f8")
        var addrs = List[Int]()
        var keep = Python.list()
        for j in range(len(cols)):  # driver-loop: over members
            var col = cols[j]
            if String(col.dtype) != kind:
                var wide = admitted_vector(n, kind, context)
                trees_cast(trees_address(col), String(col.dtype), trees_address(wide), kind, n, context)
                _ = wide  # keep the Python object alive through the native call above
                _ = col  # keep the Python object alive through the native call above
                col = wide
            _ = py_call(keep.append, col)
            addrs.append(trees_address(col))
            _ = col  # keep the Python object alive through the native call above
        var out = admitted_vector(n * len(cols), kind, context)
        var stack_addrs: List[Int] = [trees_address(out)]
        for j in range(len(addrs)):  # driver-loop: over the column address words
            stack_addrs.append(addrs[j])
        _ = xt_stack_w64(glue, stack_addrs, n)
        _ = out  # keep the Python object alive through the native call above
        _ = keep
        return trees_reshape(out, Python.tuple(n, len(cols)), context)
    # label objects (str, mixed) no Array holds: row tuples of them
    var lists = Python.list()
    for j in range(len(cols)):  # driver-loop: over members
        var col = cols[j]
        _ = py_call(lists.append, col if py_type(col) is b.list or py_type(col) is b.tuple else _label_list(col, context))
    var ints = True
    for j in range(len(lists)):  # driver-loop: over members (each column's first label)
        if len(lists[j]) > 0 and not Bool(py=py_call(b.isinstance, lists[j][0], b.int)):
            ints = False
    # Pack exact int/float/bool label objects (bool classes) into one numeric
    # store as before. Any other label object (str, mixed) cannot live in a
    # numeric Array: native_array_from_values refuses it. Those outputs are
    # returned as the per-row label lists (one list of m labels per row), the
    # same label objects the single-output classifiers return for str classes.
    # Lane p2-labels-lists-arrays: the row-major labels are filled in C
    # (p3_interleave_columns: one slice assignment per column); the extend
    # per row is gone. The per-row label lists are built only when the
    # numeric store refuses the labels, by C zip/map over the columns: one
    # new list per row is Python object output a kernel file cannot build,
    # main's recorded debt (tools/hooks/compute_in_shells_baseline.txt).
    var flat = p3_interleave_columns(lists, n)
    var store = PythonObject(None)
    try:
        store = native_array_from_values("q" if ints else "d", flat)
    except:
        var zipped = py_call(b.list, py_call(b.map, b.list, _call_kw(b.zip, py_call(b.tuple, lists), Python.dict())))  # cpu-route: label objects (str/mixed) no Array holds
        return zipped
    return classical_wrap_array(store, Python.tuple(n, len(cols)), "<i8" if ints else "<f8", "C", context)


def _multioutput_classifier_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MultiOutputClassifier.predict_proba` (lane bm-meta-3): UNPACK
    `_vote_unpack` (kind _VOTE_SOFT: per member its held native state, else
    its door's float32 probabilities and class columns, `_member_scores`'s
    routes), RUN `multioutput_proba_run` into one float64 zero block per
    output allocated here, PACK the (n, k_j) views of them."""
    var mx = _check_meta_x(model, X, context)  # a fold from the store (lane vdf-meta5)
    var nk = trees_keys()
    var ks = List[Int]()
    var classes = _attr(model, nk[].k_classes_, Python.list())
    for j in range(len(classes)):  # driver-loop: over members (outputs)
        ks.append(len(classes[j]))
    var members = _attr(model, nk[].k_estimators_, Python.list())
    if mx.draw is None:
        return _probas_of(model, members, ks, mx.x, context)
    var scope = _fold_scope(mx, context)
    if scope != MEMBER_SCOPE_NONE:
        var got = _probas_fold(model, members, ks, mx, scope, context)
        if got is not None:
            return got
    return _probas_of(model, members, ks, _x_array(X, context), context)


def _probas_fold(model: PythonObject, members: PythonObject, ks: List[Int], mx: _MetaX, scope: Int,
                 context: PythonObject) raises -> PythonObject:
    """THE FOLD FROM THE STORE (lane vdf-meta5): `_probas_of` over a fold
    `mx`, every member reading the fold by number out of the resident base
    (`_fold_open`, `_FoldPin`; RUN `multioutput_proba_run`). None, nothing
    published, when a member is not served that way or a call refuses (the
    caller admits the fold's rows and asks `_probas_of`). Same rows, same
    order, same accumulation: same bits."""
    var xt = _xt(py_fields(model), context)
    var n = mx.n
    var got = _VoteUnpack()
    var blocks = Python.list()
    var answered = False
    try:
        var opened_all = True
        for i in range(len(members)):  # driver-loop: over members (admission, once each)
            var cols = _sub_cols(members[i], context)
            _ = py_call(got.keep.append, cols)
            if not _fold_open(got, members[i], mx.x, MEMBER_OP_PREDICT_PROBA, True, 0, trees_address(cols), 1.0,
                              True, context):
                opened_all = False
                break
        if opened_all:
            _fold_plans(got)
            var accs = List[Int](capacity=len(got.members))
            for j in range(len(got.members)):  # driver-loop: over members (outputs)
                var acc = _zeros(n * ks[j], context)
                _ = py_call(blocks.append, acc)
                accs.append(trees_address(acc))
            var glue = _xr(xt)
            var pin = _FoldPin(mx.x, len(got.members))
            var rows = pin.rows(trees_address(mx.x), mx.base_rows, mx.d, trees_address(mx.draw), n, scope)
            var refused = VOTE_ALL_ANSWERED
            try:
                with GILReleased(Python()):
                    refused = multioutput_proba_run(glue, rows, got.members, ks, accs, 0)
            finally:
                pin.settle()
            answered = refused == VOTE_ALL_ANSWERED
    except e:
        _vote_close(got)
        raise e^
    _vote_close(got)
    _ = mx.x  # the rows are read through RUN
    _ = mx.draw
    _ = got.keep  # the class columns are read through RUN
    if not answered:
        return PythonObject(None)
    var out = Python.list()
    for j in range(len(blocks)):  # driver-loop: over members (outputs)
        _ = py_call(out.append, trees_reshape(blocks[j], Python.tuple(n, ks[j]), context))
    return out


def _probas_of(model: PythonObject, members: PythonObject, ks: List[Int], Xa: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """Each fitted classifier member's probabilities placed at its class
    columns of a float64 (n, ks[j]) block (a MultiOutput column, a
    Stacking final estimator): UNPACK `_vote_unpack_of`, RUN
    `multioutput_proba_run`."""
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var got = _vote_unpack_of(members, Xa, _VOTE_SOFT, True, PythonObject(None), context)
    var blocks = Python.list()
    var accs = List[Int](capacity=len(got.members))
    try:
        for j in range(len(got.members)):  # driver-loop: over members (outputs)
            var acc = _zeros(n * ks[j], context)
            _ = py_call(blocks.append, acc)
            accs.append(trees_address(acc))
        var rows = VoteRows(trees_address(Xa), n, _cols(Xa))
        var glue = _xr(xt)
        var start = 0
        while True:  # driver-loop: over refusals (at most one round per native member)
            var refused = VOTE_ALL_ANSWERED
            if got.native > 0:
                with GILReleased(Python()):
                    refused = multioutput_proba_run(glue, rows, got.members, ks, accs, start)
            else:
                refused = multioutput_proba_run(glue, rows, got.members, ks, accs, start)
            if refused == VOTE_ALL_ANSWERED:
                break
            # The fallback rule: the refusing member's door raises its words.
            got.members[refused] = _vote_door(members[refused], _VOTE_SOFT, Xa, n, 1.0, got.keep, context,
                                              PythonObject(None))
            got.native -= 1
            start = refused
    except e:
        _vote_close(got)
        raise e^
    _vote_close(got)
    _ = Xa  # the rows are read through RUN
    _ = got.keep  # the door blocks are read through RUN
    var out = Python.list()
    for j in range(len(blocks)):  # driver-loop: over members (outputs)
        _ = py_call(out.append, trees_reshape(blocks[j], Python.tuple(n, ks[j]), context))
    return out


def _multioutput_classifier_score(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                                  context: PythonObject) raises -> PythonObject:
    """`MultiOutputClassifier.score`: the (weighted) fraction of rows whose
    every output matches."""
    var b = trees_builtins()
    var mx = _check_meta_x(model, X, context)  # a fold from the store (lane vdf-meta5)
    var codes = _multioutput_member_codes(model, X, mx, context)
    var pred = _multioutput_assemble(model, mx.n, codes, context)
    var nk = trees_keys()
    if py_type(pred) is not py_item(context, nk[].k_array_class):
        # Label objects no Array holds (str classes): scored in codes.
        return _multioutput_score_codes(model, codes, mx.n, y, sample_weight, context)
    var truth = admit_c_view(y, "y", context)
    var shown = py_call(b.tuple, pred.shape)
    var truth_shape = Python.list()
    for k in range(Int(py=truth.ndim)):  # driver-loop: over array rank metadata
        _ = py_call(truth_shape.append, PythonObject(Int(py=truth.shape[k])))
    if not Bool(py=py_call(b.tuple, truth_shape) == shown):
        raise Error("y has shape " + _repr(py_call(b.tuple, truth_shape)) + ", the predictions have shape " + _repr(shown))
    var n = _rows(pred)
    var m = _cols(pred)
    # One equality flag per cell (the base binding's elementwise compare at
    # one exact dtype), one any-mismatch flag per row (`x_trees_rows_any_unset`),
    # then the (weighted) accuracy of those flags.
    var pdt = String(pred.dtype)
    var tdt = admitted_dtype(truth, context)
    var t_int = tdt == "<i4" or tdt == "<i8" or tdt == "<u4" or tdt == "<u1" or tdt == "<u2" or tdt == "<i1"
    var p_int = pdt == "<i8"
    var size = n * m
    var eq = py_call(py_item(context, nk[].k_array_type), PythonObject("B"), py_call(py_item(context, nk[].k_native_bytes_type), size))
    var eq_addr = Int(py=py_call(eq.buffer_info)[0])  # UNPACK: the flag store's address (Python array)
    var bkey = base_helpers_key(py_item(context, nk[].k_trees_helpers))
    var tc = admitted_vector(size, "<i8" if t_int else "<f8", context)
    trees_cast(Int(py=classical_address(truth)), tdt, trees_address(tc), "<i8" if t_int else "<f8", size, context)
    _ = tc  # keep the Python object alive through the native call above
    _ = truth  # keep the Python object alive through the native call above
    if t_int == p_int:
        _ = base_equal_elements(bkey, trees_address(tc), trees_address(pred), 3 if p_int else 1, size, eq_addr)
        _ = eq  # keep the flag store alive through the native call above
        _ = tc  # keep the Python object alive through the native call above
        _ = pred  # keep the Python object alive through the native call above
    else:
        var ints = tc if t_int else pred
        var flts = pred if t_int else tc
        _ = base_equal_elements(bkey, trees_address(ints), trees_address(flts), 9, size, eq_addr)
        _ = eq  # keep the flag store alive through the native call above
        _ = flts  # keep the Python object alive through the native call above
        _ = ints  # keep the Python object alive through the native call above
    _ = truth^
    return _multioutput_rows_score(model, eq, n, m, False, sample_weight, context)


def _multioutput_rows_score(model: PythonObject, eq: PythonObject, n: Int, m: Int, col_major: Bool,
                            sample_weight: PythonObject, context: PythonObject) raises -> PythonObject:
    """The (weighted) fraction of rows whose m equality flags (`eq`, an
    array('B') of n * m cells) are all set. One native pass writes each
    row's any-mismatch flag (`x_trees_rows_any_unset`; lane ne2-b-members,
    N081: before it a bytes translate to flip the flags, an int32 cast, the
    x_neighbors row test and an `array('i', [1]) * n` of fake labels). The
    accuracy of those flags against zeros counts exactly the rows the
    all-match flags against ones counted, with the same weights: the same
    score bits."""
    var xt = _xt(py_fields(model), context)
    var miss = admitted_vector(n, "<i4", context)
    var glue = _xr(xt)
    _ = xt_rows_any_unset(glue, Int(py=py_call(eq.buffer_info)[0]), trees_address(miss), n, m,
                          1 if col_major else 0)
    _ = eq  # keep the flag store alive through the native call above
    _ = miss  # keep the Python object alive through the native call above
    var zero = admitted_vector(n, "<i4", context)
    var nk = trees_keys()
    var mode = py_get(py_fields(model), nk[].k_numeric_mode)
    return PythonObject(labels_accuracy(zero, miss, sample_weight, mode, labels_scope(model, context)))


def _multioutput_score_codes(model: PythonObject, codes: PythonObject, n: Int, y: PythonObject,
                             sample_weight: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MultiOutputClassifier.score` for outputs whose classes are label
    objects (str): each truth column encoded once against that output's
    fitted classes (-1 for a label it never saw, which never matches) and
    compared with the member's codes (lane ne2-b-members, N084: this score
    raised before, the predicted label lists have no buffer). A label
    matches exactly when it is the predicted class."""
    var b = trees_builtins()
    var nk = trees_keys()
    var classes = _attr(model, nk[].k_classes_, Python.list())
    var m = len(codes)
    var rows = Python.list()
    if _has(y, nk[].k_tolist):
        rows = _label_list(y, context)  # cpu-route: label objects (str/mixed) no Array holds, the explicit label input step
    else:
        rows = py_call(b.list, py_call(b.map, b.list, y))  # cpu-route: label objects (str/mixed) no Array holds
    var width = len(rows[0]) if len(rows) > 0 and _has(rows[0], nk[].k___len__) else 0
    if len(rows) != n or width != m:
        raise Error("y has shape " + _repr(Python.tuple(len(rows), width)) + ", the predictions have shape "
                    + _repr(Python.tuple(n, m)))
    # The label steps run on the estimator's device (labels_scope), read once.
    var label_context = labels_scope(model, context)
    var eq = py_call(py_item(context, nk[].k_array_type), PythonObject("B"), py_call(py_item(context, nk[].k_native_bytes_type), n * m))
    var base = Int(py=py_call(eq.buffer_info)[0])
    var bkey = base_helpers_key(py_item(context, nk[].k_trees_helpers))
    var getter = native_loaded_module("operator").itemgetter
    for j in range(m):  # driver-loop: over the output columns (one member each)
        var column = py_call(b.list, py_call(b.map, py_call(getter, PythonObject(j)), rows))  # cpu-route: label objects (str/mixed) no Array holds, one output column at a time
        var listed = classifier_label_list(classes[j], context)
        if py_type(listed) is not b.list and py_type(listed) is not b.tuple:
            listed = _label_list(listed, context)
        var truth = trees_class_codes(score_classification_encoded(column, "y", context), listed, label_context)
        var pred = codes[j]
        _ = base_equal_elements(bkey, trees_address(truth), trees_address(pred), 2, n, base + j * n)
        _ = eq  # keep the flag store alive through the native call above
        _ = truth  # keep the Python object alive through the native call above
        _ = pred  # keep the Python object alive through the native call above
    return _multioutput_rows_score(model, eq, n, m, True, sample_weight, context)


# --------------------------------------------------------------- OneVsRest


def _indicator(xt: PythonObject, codes: PythonObject, j: Int, context: PythonObject) raises -> PythonObject:
    """`OneVsRestClassifier._indicator`: the 0/1 targets of class j (codes:
    the encoder's int32 Array)."""
    var n = Int(py=codes.shape[0])
    var yj = admitted_vector(n, "<i4", context)
    var glue = _xr(xt)
    _ = xt_indicator_codes(glue, trees_address(codes), trees_address(yj), 0, n, j)
    _ = codes
    _ = yj  # keep the owners alive through the native call above
    return yj


def _ovr_fit(model: PythonObject, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    var jobs = _jobs(model, True)
    var xt = _xt(py_fields(model), context)
    # THE FOLD IN PLACE (lane vdf-meta3): a DeviceRows fold is read out of its base (`_meta_x_fit`).
    var mx = _meta_x_fit(X, context)
    var labels = trees_encode_labels(y, labels_scope(model, context))
    _set(model, "classes_", labels.classes)
    var k = len(labels.classes)
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var nk = trees_keys()
    var est = _attr(model, nk[].k_estimator, PythonObject(None))
    _require_member(est, _owner(model), context)
    # The member runs on the ensemble's device (bindings/trees_meta_device_door.mojo): read once per fit.
    var device = py_get(py_fields(model), "device")
    var origin = meta_device_origin(model, device)
    meta_members_device(device, Python.list(est), _owner(model), origin, context, trees_builtins())
    var targets = Python.list()
    if k == 2:
        _ = py_call(targets.append, labels.codes)
    else:
        var codes_view = labels.codes
        for j in range(k):  # driver-loop: over the classes (one native indicator each)
            _ = py_call(targets.append, _indicator(xt, codes_view, j, context))
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    var clones = Python.list()
    for i in range(len(targets)):  # driver-loop: over members
        _ = py_call(clones.append, _clone(est, device, origin, context))
    # One binary member per target (0/1 codes, two classes): independent,
    # fitted together.
    _fit_members_x(clones, mx, targets, False, PythonObject(None), _repeat(PythonObject(2), len(targets)), jobs,
                   context, PythonObject(None))
    for i in range(len(clones)):  # driver-loop: over members
        _ = py_call(fitted.append, clones[i])
    _set(model, "n_features_in_", PythonObject(mx.d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _proba_positive(xt: PythonObject, e: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """`OneVsRestClassifier._proba_positive`: predict_proba's column for
    code 1 (zeros when the member never saw it), float32 (n,)."""
    var n = _rows(Xa)
    var p = _out2d(_member_scores(e, "predict_proba", Xa, context), n, context)
    var at = _index_of_one(e, context)
    if at < 0:
        _ = p^
        return admitted_vector(n, "<f4", context)
    var col = _one_col(xt, p.view(), at, context)
    _ = p^
    return trees_reshape(col, Python.tuple(n), context)


def _positive(xt: PythonObject, e: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """`OneVsRestClassifier._positive`: decision_function when the member
    has one, else predict_proba's column for code 1."""
    if prep_estimator_has(e, "decision_function", context):
        return _f32_array(_member_scores(e, "decision_function", Xa, context), 1, "score", context)
    return _proba_positive(xt, e, Xa, context)


def _ovr_door(member: PythonObject, decision: Bool, Xa: PythonObject, n: Int, keep: PythonObject,
              context: PythonObject, answer: PythonObject) raises -> VoteMember:
    """One OneVsRest member's float32 block from its own door (the path that
    raises the estimator's own words): its decision column, else its
    predict_proba block. `answer`: the Array its held native state already
    wrote (`member_held_run`), else None and the door is called."""
    if decision:
        var given = answer if answer is not None else meta_member_call(member, "decision_function",
                                                                       Python.tuple(Xa), context)
        var blk = _out2d(_f32_array(given, 1, "score", context), n, context)
        _ = py_call(keep.append, blk.array)
        var out = VoteMember(blk.address, blk.cols, 0, 1.0)
        _ = blk^
        return out^
    var p = _out2d(answer if answer is not None else meta_member_call(member, "predict_proba", Python.tuple(Xa),
                                                                      context), n, context)
    _ = py_call(keep.append, p.array)
    var out = VoteMember(p.address, p.cols, 0, 1.0)
    _ = p^
    return out^


def ovr_member_words(member: PythonObject, decision: Bool, context: PythonObject) raises -> Tuple[Bool, Int]:
    """X-FREE UNPACK of one OneVsRest member (lane mf-shap3, 2026-11-23; `_ovr_fused` and Kernel / Permutation
    SHAP): whether it scores by decision_function (`decision` asked and it has one) and its positive column (0
    for a decision column; else its predict_proba column of code 1, -1 when it never saw code 1)."""
    var by_decision = decision and prep_estimator_has(member, "decision_function", context)
    return (by_decision, 0 if by_decision else _index_of_one(member, context))


def _ovr_fused(xt: PythonObject, members: PythonObject, Xa: PythonObject, n: Int, k: Int,
               decision: Bool, finish: Int, context: PythonObject) raises -> PythonObject:
    """The k > 2 OneVsRest epilogue (THE OVR SPLIT, lane bm-meta-2,
    2026-11-22). UNPACK: per member whether it scores by decision_function
    (when `decision` and it has one) and its positive column, then its held
    native state when its family serves the operation (`_member_open`; a kind
    that must run with the GIL held answers here, as before), else its door's
    block (`_ovr_door`). RUN: xtrees/trees_meta_vote_run.mojo
    `ovr_assemble_run` (the native calls, then the one `x_trees_ovr_assemble`
    program), the GIL released when a member is native. PACK: `out`,
    allocated here. A native member that refuses is asked through its door
    and RUN runs again (the fallback rule). Before the split this loop made
    one Array per member answer: the same blocks, the same program."""
    var out = admitted_vector(n, "<i4", context) if finish == 2 else admitted_vector(n * k, "<f8", context)
    var got = _VoteUnpack()
    var at = List[Int]()
    var uses_decision = List[Bool]()
    try:
        for j in range(len(members)):  # driver-loop: over members (admission, once each)
            var member = members[j]
            var words = ovr_member_words(member, decision, context)
            var by_decision = words[0]
            uses_decision.append(by_decision)
            at.append(words[1])
            var operation = MEMBER_OP_DECISION if by_decision else MEMBER_OP_PREDICT_PROBA
            var opened = _member_open(member, Xa, operation, True, context)
            if opened.served and not opened.gil_free:
                var answer = PythonObject(None)
                try:
                    answer = member_held_run(opened, trees_address(Xa), n, _cols(Xa), context)
                finally:
                    member_held_close(opened)
                got.members.append(_ovr_door(member, by_decision, Xa, n, got.keep, context, answer))
                continue
            if opened.served:
                got.members.append(VoteMember(len(got.opened), opened.held.state, operation, opened.cols,
                                              opened.dtype, 0, 1.0))
                got.native += 1
                got.opened.append(opened^)
                continue
            got.members.append(_ovr_door(member, by_decision, Xa, n, got.keep, context, PythonObject(None)))
        for i in range(len(got.members)):  # driver-loop: over members (the native plans by address, now final)
            if got.members[i].native:
                got.members[i].plan = Int(UnsafePointer(to=got.opened[got.members[i].plan].held.plan))
    except e:
        _vote_close(got)
        raise e^
    var rows = VoteRows(trees_address(Xa), n, _cols(Xa))
    var glue = _xr(xt)
    var address = trees_address(out)
    while True:  # driver-loop: over refusals (at most one round per native member)
        var refused = VOTE_ALL_ANSWERED
        try:
            if got.native > 0:
                with GILReleased(Python()):
                    refused = ovr_assemble_run(glue, rows, got.members, at, k, finish, address)
            else:
                refused = ovr_assemble_run(glue, rows, got.members, at, k, finish, address)
        except e:
            _vote_close(got)
            raise e^
        if refused == VOTE_ALL_ANSWERED:
            break
        got.members[refused] = _ovr_door(members[refused], uses_decision[refused], Xa, n, got.keep, context,
                                         PythonObject(None))
        got.native -= 1
    _vote_close(got)
    _ = Xa  # the rows are read through RUN
    _ = got.keep  # the door blocks are read through RUN
    _ = out
    return out


def _ovr_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    return trees_decode_labels(_attr(model, nk[].k_classes_, PythonObject(None)), _ovr_codes(model, X, context), context)


def _ovr_codes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`OneVsRestClassifier.predict` before its decode: int32 codes into
    `classes_`."""
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var nk = trees_keys()
    var classes = _attr(model, nk[].k_classes_, PythonObject(None))
    var k = len(classes)
    var members = _attr(model, nk[].k_estimators_, Python.list())
    if k == 2:
        return _member_label_codes(members[0], Xa, context)
    if len(members) == k:
        return _ovr_fused(xt, members, Xa, n, k, True, 2, context)
    var acc = _class_zero(xt, n * k, context)
    var rows = _arange(xt, n, context)
    for j in range(len(members)):  # driver-loop: over members
        var blk = _out2d(_positive(xt, members[j], Xa, context), n, context)
        _ = _place(xt, acc, n, k, blk.view(), rows, j)
        _ = blk^
    return _argmax(xt, acc, n, k, context)


def _ovr_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var nk = trees_keys()
    var k = len(_attr(model, nk[].k_classes_, Python.list()))
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var glue = _xr(xt)
    if k == 2:
        # `_trees_binary_proba`: rows (1 - p, p) of the positive column
        var p = _f32_array(_proba_positive(xt, members[0], Xa, context), 1, "p", context)
        var out = admitted_vector(2 * n, "<f8", context)
        _ = xt_binary_proba(glue, trees_address(p), trees_address(out), n)
        _ = out  # keep the Python object alive through the native call above
        _ = p  # keep the Python object alive through the native call above
        return trees_reshape(out, Python.tuple(n, 2), context)
    if len(members) == k:
        return trees_reshape(_ovr_fused(xt, members, Xa, n, k, False, 1, context), Python.tuple(n, k), context)
    var acc = _class_zero(xt, n * k, context)
    var rows = _arange(xt, n, context)
    for j in range(len(members)):  # driver-loop: over members
        var blk = _out2d(_proba_positive(xt, members[j], Xa, context), n, context)
        _ = _place(xt, acc, n, k, blk.view(), rows, j)
        _ = blk^
    _ = xt_normalize_rows(glue, trees_address(acc), n, k)
    _ = acc  # keep the Python object alive through the native call above
    return trees_reshape(acc, Python.tuple(n, k), context)


# ------------------------------------------------------------- Calibration


def _cal_scores(model: PythonObject, xt: PythonObject, e: PythonObject, Xa: PythonObject,
                context: PythonObject, ensemble: Bool, mut calibrate: List[Int], rows: Int = -1) raises -> PythonObject:
    """`CalibratedClassifierCV._scores`: (n, c) float64 scores, c = 1 when
    binary, else one column per class, placed by THE PLACEMENT RULE
    (xtrees/trees_calibrated_fit_run.mojo `cal_fold_place`, the native RUN's
    own rule; lane mf-cal3b: scikit-learn's placement of a member that lacks
    a class, its refusals in its words); `calibrate` gets the columns that
    are calibrated (ensemble). `rows`: the row count when the caller knows it
    (a resident DeviceRows fold side, whose `shape` is a Python property of
    `_arena_io.DeviceRows` since 9b369cd34f: reading it from here ran Python
    inside the fit)."""
    var n = rows if rows >= 0 else _rows(Xa)
    var nk = trees_keys()
    var k = len(_attr(model, nk[].k_classes_, Python.list()))
    var decision = prep_estimator_has(e, "decision_function", context)
    var out: PythonObject
    if decision:
        out = _member_scores(e, "decision_function", Xa, context)
    else:
        out = _member_scores(e, "predict_proba", Xa, context)
    var blk = _out2d(out, n, context)
    var glue = _xr(xt)
    # The member's classes as ensemble codes (`_sub_cols`), then the rule.
    var sub_cols = _sub_cols(e, context)
    var ks = len(_attr(e, nk[].k_classes_, Python.list()))
    var codes = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(sub_cols))
    var held = List[Int32](capacity=ks)
    for c in range(ks):  # driver-loop: over the member's class codes (a few words)
        held.append(codes[c])
    _ = sub_cols  # read through `codes` above
    var place = cal_fold_place(k, held, MEMBER_OP_DECISION if decision else MEMBER_OP_PREDICT_PROBA, blk.cols,
                               ensemble, n)
    calibrate = place.calibrate.copy()
    if k == 2:
        var acc = _zeros(n, context)
        if place.pick == 0 and blk.cols == 1:
            _ = xt_accumulate(glue, trees_address(acc), blk.address, n, 1.0)
            _ = acc  # keep the Python object alive through the native call above
            _ = blk  # keep the Python object alive through the native call above
        elif place.pick >= 0:
            var col = _one_col(xt, blk.view(), place.pick, context)
            _ = xt_accumulate(glue, trees_address(acc), trees_address(col), n, 1.0)
            _ = col  # keep the Python object alive through the native call above
            _ = acc  # keep the Python object alive through the native call above
        # pick -1: zeros (a fold holding only code 0, ensemble=False)
        _ = blk^
        return trees_reshape(acc, Python.tuple(n, 1), context)
    var acc = _class_zero(xt, n * k, context)
    var acc_at = trees_address(acc)
    _ = xt_accumulate_cols(glue, acc_at, blk.address, Int(place.place.unsafe_ptr()), n, k, len(place.place), 1.0)
    if place.fill != 0.0:
        # `_enforce_prediction_order`'s default of the classes the fold lacks
        var mark = List[Bool](length=k, fill=True)
        for i in range(len(place.place)):  # driver-loop: over the placed columns
            mark[Int(place.place[i])] = False
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=acc_at)
        for r in range(n):  # native input marshaling: one fill pass over one fold's unplaced columns
            for c in range(k):
                if mark[c]:
                    words[r * k + c] = place.fill
    _ = len(place.place)  # read through the native call above
    _ = acc  # keep the Python object alive through the native calls above
    _ = blk^
    return trees_reshape(acc, Python.tuple(n, k), context)


def _calibrated_probs(model: PythonObject, xt: PythonObject, e: PythonObject, cals: PythonObject, Xa: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`CalibratedClassifierCV._calibrated`: the flat (n, k) probabilities of
    one (member, calibrators) pair."""
    var n = _rows(Xa)
    var nk = trees_keys()
    var k = len(_attr(model, nk[].k_classes_, Python.list()))
    var which = List[Int]()
    var S = _cal_scores(model, xt, e, Xa, context, True, which)
    var c = _cols(S)
    var w = 2 if k == 2 else k
    var acc = admitted_vector(n * w, "<f8", context)
    var glue = _xr(xt)
    for j in range(len(cals)):  # driver-loop: over the class columns (one calibrator each)
        var entry = cals[j]
        var col = 1 if k == 2 else j
        var par = entry[1]
        if String(entry[0]) == "sigmoid":
            _ = xt_platt_apply_strided(glue, trees_address(S), trees_address(acc), n, Float64(py=par[0]),
                                       Float64(py=par[1]), n * c, c, j, n * w, w, col)
            _ = acc  # keep the Python object alive through the native call above
            _ = S  # keep the Python object alive through the native call above
        else:
            var knots_x = par[0]
            var knots_y = par[1]
            _ = xt_isotonic_predict_strided(glue, trees_address(knots_x), trees_address(knots_y), trees_address(S),
                                            trees_address(acc), Int(py=par[2]), n, n * c, c, j, n * w, w, col)
            _ = knots_x
            _ = knots_y
            _ = acc  # keep the Python object alive through the native call above
            _ = S  # keep the Python object alive through the native call above
    if k == 2:
        _ = xt_complement_pairs(glue, trees_address(acc), n)
        _ = acc  # keep the Python object alive through the native call above
        return acc
    _ = xt_normalize_rows(glue, trees_address(acc), n, k)
    _ = acc  # keep the Python object alive through the native call above
    return acc


def _cal_rows_call(obj: PythonObject, op: String, args: PythonObject, mctx: PythonObject) raises -> PythonObject:
    """One `_arena_io` resident-scope operation through its compiled door
    (model selection's `_arena_rows_call`, bindings/msel_cv_door.mojo)."""
    # call-table: member DeviceRows (arena rows door over the Python owner) has no native entry yet
    var nk = trees_keys()
    return family_call(py_item(mctx, nk[].k_family_msel_rows_entry), obj, PythonObject(op), args, py_item(mctx, nk[].k_family_msel_rows_context))


def _cal_rows_resident(base: PythonObject, Xa: PythonObject, n: Int, d: Int, context: PythonObject) raises -> Bool:
    """UNPACK, once per fit (THE RESIDENT ROWS, lane mf-cal7): whether the
    fit runs inside a resident scope of its own: the base's class takes
    DeviceRows (its plan's `device_rows` word: the naive Bayes and
    discriminant classifiers) and X is a non-empty float32 C-order block.
    Whether the fold members then read X by row index is their member's
    answer on its image (`_cal_native`: MEMBER_CAN_ROW_INDEX and
    MEMBER_CAN_SCOPE); a host image's member keeps the gathered rows."""
    if n <= 0 or d <= 0 or String(py=Xa.dtype) != "<f4" or String(py=Xa.order) != "C":
        return False
    var nk = trees_keys()
    # device="cpu" beside a GPU set (lane vdf-meta): this fit runs in the CPU
    # run context (bindings/trees_route_boundary_door.mojo `trees_cpu_context`,
    # the context that names itself), its base on the host images, which keep
    # the gathered rows: no device scope is entered. Same rows, same order.
    if py_get(context, nk[].k_device_cpu_context) is context:
        return False
    var plan = prep_estimator_plan(base, context)
    return plan is not None and _truthy(py_get(plan, nk[].k_device_rows, PythonObject(False)))


def _cal_scope_open(mctx: PythonObject) raises -> PythonObject:
    """Enter a resident scope (`_arena_io.resident`) on this thread with the
    DeviceRows floor: only a row-indexed base (the fold rows taken on the
    device out of it) stays resident, no other program input across folds,
    as model selection's `msel_rows_scope_open` enters its own. The caller
    leaves it with `_cal_scope_close` on every path."""
    var nk = trees_keys()
    var b = trees_builtins()
    var opened = native_storage_allocate(family_call(b.getattr, py_item(mctx, nk[].k_family_msel_arena_io),
                                                     PythonObject("resident")))
    # the floor goes to the native scope state with the enter call
    _ = _cal_rows_call(opened, "scope_enter", Python.tuple(PythonObject(1 << 62)), mctx)
    return opened


def _cal_scope_close(scope: PythonObject, mctx: PythonObject) raises:
    """Leave a scope `_cal_scope_open` entered: its caches and the references
    they took are released."""
    _ = _cal_rows_call(scope, "scope_exit", Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None)),
                       mctx)


def _cal_f64(values: List[Float64], m: Int, context: PythonObject) raises -> PythonObject:
    """PACK: `values[0:m]` as a new float64 (m,) Array (the door's
    `_copy_block` of an isotonic knot vector)."""
    var buf = admitted_unfilled(List[Int](length=1, fill=m), "<f8", context)
    if m > 0:
        unsafe_memcpy(dest=_F64P(unsafe_from_address=buf.addr), src=values.unsafe_ptr(), count=m)
    return buf.keep


def _cal_pack(res: CalFitOut, first: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """PACK: calibrators `first .. first + count` of the RUN as the door's
    list: ("sigmoid", (a, b)) or ("isotonic", (knots_x, knots_y, m)), the
    knots float64 Arrays."""
    var cals = Python.list()
    for j in range(first, first + count):  # driver-loop: over the class columns (one calibrator each)
        if res.isotonic[j]:
            var m = res.m[j]
            _ = py_call(cals.append, Python.tuple(PythonObject("isotonic"),
                                                  Python.tuple(_cal_f64(res.knots_x[j], m, context),
                                                               _cal_f64(res.knots_y[j], m, context), PythonObject(m))))
        else:
            _ = py_call(cals.append, Python.tuple(PythonObject("sigmoid"),
                                                  Python.tuple(PythonObject(res.a[j]), PythonObject(res.b[j]))))
    return cals


# TOMBSTONE (plain door path, no define): `_CalUnits`, `_cal_units_thread`,
# `_cal_units_run`, `_CAL_DONE` and `_CAL_NOT_MEMBER` (CalibratedClassifierCV's
# door fold units and the answer that sent a base to them) deleted 2026-11-23
# by lane mf-del8: every base is the typed RUN's (`_cal_native`), or refused
# there in its refuser's words; code recoverable at f1385f8f1c.


def _cal_fit_words(set: MemberSet, plan: CalFitPlan, u: Int) -> String:
    """The words of member `u`'s refused fit, as its own fit door raises
    them for the rows it was given. One family's door words carry the label
    of a one-class y, which a member never sees (labels do not cross the
    ABI): GaussianProcessClassifier (bindings/gp_classifier_native_door.mojo,
    after scikit-learn's _gpc.py: "... got 1 class (only class <label> is
    present)"), whose member says "... got 1 class"; the label is the fold's
    one ensemble code, the y its fit door was given (the deleted door units
    passed the fold's codes)."""
    var words = set.run.errors[u].copy()
    if (u < plan.folds() and len(plan.fold_classes[u]) == 1 and set.run.plans[u].family == MEMBER_FAMILY_GP
            and words.endswith("got 1 class")):
        words += " (only class " + String(Int(plan.fold_classes[u][0])) + " is present)"
    return words^


def _cal_warn_lacking(plan: CalFitPlan, upto: Int, context: PythonObject) raises:
    """PACK, ensemble=False (lane mf-cal3b): scikit-learn's
    `_enforce_prediction_order` RuntimeWarning for each of the first `upto`
    folds whose train rows lack a class, in fold order."""
    for f in range(min(upto, plan.folds())):  # driver-loop: over the cv folds (one warning per lacking fold)
        if len(plan.fold_classes[f]) > 0:
            msel_warn(cal_lacking_warning(len(plan.fold_classes[f]), plan.k),
                      native_loaded_module("builtins").RuntimeWarning, prep_native_context(context))


def _cal_marker(base: PythonObject, classes: PythonObject, context: PythonObject, b: PythonObject) raises -> Int:
    """UNPACK (lane mf-meta5b): the code of the unlabeled marker among `classes` when the base is a
    LabelPropagation / LabelSpreading member (its fit reads -1 as unlabeled), else -1."""
    var plan = prep_estimator_plan(base, context)
    if plan is None or py_get(plan, "member") is None:
        return -1
    var mark = member_mark(plan, b)
    if not xneighbors_kind_reads_marker(mark.family, mark.kind):
        return -1
    return xneighbors_marker_index(classes)


def _cal_native(model: PythonObject, xt: PythonObject, base: PythonObject, mx: _MetaX,
                k: Int, codes: PythonObject, classes: PythonObject, splits: PythonObject, ensemble: Bool, jobs: Int,
                pairs: PythonObject, device: PythonObject, origin: String, context: PythonObject) raises:
    """CalibratedClassifierCV.fit as UNPACK / RUN / PACK over ONE native
    member set (lanes mf-cal, mf-cal2 and mf-cal3, 2026-11-23;
    xtrees/trees_calibrated_fit_run.mojo module note), `pairs` filled in
    fold order exactly as the door's fold units filled it. Every other
    outcome RAISES, nothing appended and no state left behind (lane mf-cal3;
    lane mf-del8 deleted the door fold units, so no base is left to them):
      * a base the set does not take (`member_set_admit_or_raise`: a foreign
        class, its family's refused admission in its own words, a
        configuration with no native member fit), and a base whose member
        kind has no native call for the scores its class answers: before any
        fit, in those words;
      * a fold with no train or no test rows (a user's cv), before any fit:
        scikit-learn's check_array words, the ones its fold's fit or
        predict raises ("Found array with 0 sample(s) ...");
      * a non-finite X: `admitted_finite_refuse`, the one wording of every
        admission (before any fit for a member that does not test its rows;
        after a failure for one that does, whose kernel refused it);
      * a member's refused fit (a fold whose train rows hold one class
        included: its fit decides as the estimator's own fit does): the
        first failing fold's words, in fold order, as scikit-learn's fold
        loop raises the first (`_cal_fit_words`); every family's member
        raises its fit door's words for a one-class y (read for every
        classifier family: logistic, linear SVM, SGD / passive-aggressive /
        ridge, kernel SVM, discriminant, gbdt, dart, mlp, recurrent; the
        forest families' "y has fewer than 2 classes"; GP's label suffix
        added here);
      * a refused scoring call, a refused publication, a resident-scope
        read that raises: their own words.

    UNPACK (GIL held): one clone per fold (plus one over every row for
    ensemble=False), admitted into the set over X with the fold's train
    rows as its row draw and the ensemble's codes with k classes; every
    address read here; THE PIN RULE's references taken. RUN
    (`calibrated_fit_run`, no Python): the fold classes (a fold that lacks a
    class is re-coded onto the classes it holds), the fits (the folds in
    parallel on the host cores, one at a time on a GPU), the scores and the
    calibrators; the GIL released ONCE around all of it when every member's
    kind may run without it (held, as the held member calls hold it, for
    any other: the RUN has no Python either way). PACK: the pins returned,
    `member_pack` into each clone (a re-coded fold's member labelled with
    the ensemble codes it holds: the classes_ its fit door wrote), the fit
    notices, the calibrator tuples.

    THE FOLD IN PLACE (lane vdf-meta4): `mx` is X, or a model-selection
    fold's whole host base with the fold's rows (`_meta_x_fit`). Over a
    fold the units read the base: their row draws are the cv splits'
    train rows composed inside the fold (`_meta_compose`, the full unit
    of ensemble=False the fold's own rows) and their targets the fold's
    codes put at the base rows (`_meta_take`); each fold's member reads its
    test rows composed the same way while its scores are placed, and its
    test codes read, at the fold positions (CalFitPlan's TWO INDEXES).
    `codes` are the fold's (n rows) either way. No row of the fold is
    gathered on the host: a member that reads a row index takes them on
    the device out of the base model selection keeps resident."""
    var nk = trees_keys()
    var b = trees_builtins()
    var Xa = mx.x
    var n = mx.n
    var d = mx.d
    var folds = len(splits)
    var sigmoid = String(_attr(model, nk[].k_method, PythonObject("sigmoid"))) == "sigmoid"
    var decision = prep_estimator_has(base, "decision_function", context)
    var units = folds if ensemble else folds + 1
    # UNPACK
    var set = MemberSet(trees_address(Xa), mx.base_rows, d, MEMBER_I32, MEMBER_I32)
    set.run.cap = jobs
    set.adopt = True
    # `classes_` in code order: a base whose class_weight is a dict reads its
    # table over them at admission (lane mf-adm3), as scikit-learn's fold fit
    # reads the dict against the fold's labels.
    set.labels = classes
    # THE UNLABELED MARKER (lane mf-meta5b): a LabelPropagation / LabelSpreading base reads the label -1 as
    # unlabeled. scikit-learn fits each fold (ensemble=True) and the full estimator (ensemble=False) on y as
    # given, so those members get the marker's code in their marker word (their classes_ exclude -1; the RUN
    # leaves it out of their held classes); cross_val_predict (ensemble=False's folds) encodes y first, so
    # its fold members read -1 as one more class. They are admitted over the codes (an empty label list:
    # these kinds read no class_weight). Two classes with ensemble=False: scikit-learn's own fit fails there
    # (core/ensemble_member.mojo META_CAL_BINARY_MARKER_WORDS cites the line), raised natively in its words
    # (lane mf-ens6; the door's fold units kept it before).
    var marker = _cal_marker(base, classes, context, b)
    if marker >= 0:
        if not ensemble and k == 2:
            raise Error(META_CAL_BINARY_MARKER_WORDS)
        set.labels = Python.list()
    var markers = List[Int]()
    var codes_address = trees_address(codes)
    # THE FOLD IN PLACE: the units' targets at the rows of `Xa` (the fold's codes at its base rows; `codes`
    # itself otherwise), and every composed index (train draws, test reads) kept to the end of PACK.
    var draw_codes = _meta_take(mx, codes, context)
    var draw_codes_address = trees_address(draw_codes)
    var composed = Python.list()
    var clones = Python.list()
    for u in range(units):  # driver-loop: over the cv folds (+ the full-X member), admission once each
        var e = _clone(base, device, origin, context)
        _ = py_call(clones.append, e)
        var draw = 0
        var count = 0
        if u < folds:
            var tr = _meta_compose(mx, splits[u][0], context)
            _ = py_call(composed.append, tr)
            draw = trees_address(tr)
            count = Int(py=tr.shape[0])
        elif mx.draw is not None:
            draw = trees_address(mx.draw)  # the full unit over a fold: every fold row, in the base
            count = n
        member_set_admit_or_raise(set, e, draw_codes_address, 0, draw, count, 0, 0, k, context, b)
        if marker >= 0:
            var reads = ensemble or u == folds
            if reads:
                set.run.plans[u].params[LP_MARKER_WORD] = Float64(marker)
            markers.append(marker if reads else -1)
        # A scope read that raises raises its own doors' words (nothing is
        # fitted yet; the set holds no state).
        member_set_resident(set, Xa, context)
    # KIND-EXACT (lane mf-meta3): the family's union mask holds bits another
    # kind of it has (a hard VotingClassifier, META kind, has no
    # predict_proba where a soft one has). One state minted, asked and
    # released per fit.
    var kind_mask = member_kind_capabilities(set.run.plans[0])
    # A MEMBER-ONLY plan (python/mojolearn/_native_msel_more.py
    # `connect_members`: no method recipes, e.g. the explicit Host* classes of
    # lane mf-cal3b) answers `prep_estimator_has` from its class (lane w3f2-cv:
    # scikit-learn's `hasattr(estimator, "decision_function")`), so HostSVC's
    # decision_function is read above. REMOVED: the member-kind DECISION-bit
    # answer this fit read for a member-only base after it (gate 5645984a8f
    # mf_cal svch rows), which only that one door had.
    var scoring = MEMBER_CAN_DECISION if decision else MEMBER_CAN_PREDICT_PROBA
    var first = set.run.plans[0].copy()
    if (first.family == MEMBER_FAMILY_ENSEMBLE and len(first.params) > 2 and Int(first.params[0]) == 0
            and first.params[2] == 0.0 and (kind_mask & (MEMBER_CAN_DECISION | MEMBER_CAN_PREDICT_PROBA)) == 0):
        # A hard VotingClassifier base (META kind 0, soft word 0; lane
        # mf-meta3b): it has neither decision_function nor predict_proba, so
        # scikit-learn 1.9's `_check_response_method` refuses it
        # (calibration.py, before any fit for ensemble=False, after the first
        # fold's fit for ensemble=True: the same words), in its words.
        raise Error("AttributeError: " + String(py_type(base).__name__)
                    + " has none of the following attributes: decision_function, predict_proba.")
    if not set.run.plans[0].member.can(scoring) or (kind_mask & scoring) == 0:
        if (not decision and (kind_mask & (MEMBER_CAN_DECISION | MEMBER_CAN_PREDICT_PROBA)) == 0
                and not prep_estimator_has(base, "predict_proba", context)):
            # Neither its member kind nor its estimator has decision_function
            # or predict_proba (lane mf-cal7): scikit-learn 1.9's
            # `_check_response_method` refuses it, as for the hard vote above,
            # in its words (the door units fitted every fold and then failed).
            raise Error("AttributeError: " + String(py_type(base).__name__)
                        + " has none of the following attributes: decision_function, predict_proba.")
        # Its class answers the scoring method but its member kind has no native
        # call for it (lane mf-del8: the door fold units that fitted and scored
        # such a base are deleted). No member kind of a classifier family is
        # known to lack it (every classifier family's mask carries
        # MEMBER_CAN_DECISION or MEMBER_CAN_PREDICT_PROBA as its class
        # answers): refused natively, naming the gap, never answered by a
        # Python call.
        raise Error("NotImplementedError: CalibratedClassifierCV: " + String(py_type(base).__name__) + "'s native member has no "
                    + ("decision_function" if decision else "predict_proba")
                    + " call, which the calibration scores read. Use a base whose member serves it.")
    # A fold with no train or no test rows (only a user's cv iterable makes
    # one): scikit-learn's check_array refusal, raised by that fold's
    # `estimator.fit` (no train rows) or its `decision_function` /
    # `predict_proba` (no test rows), in its words, before any fit. (A train
    # draw of 0 rows would read as "every row" in the set.)
    for f in range(folds):  # driver-loop: over the cv folds (two row counts each)
        if Int(py=splits[f][0].shape[0]) == 0 or Int(py=splits[f][1].shape[0]) == 0:
            admitted_rows_shape(0, d, String(py_type(base).__name__))
    # WHO REFUSES A NON-FINITE X (as in `_fit_units`): a member whose kind
    # tests its rows refuses them itself; for any other X's verdict is read
    # ONCE here (every fold's rows are rows of X; the door scanned each fold)
    # and a non-finite X is refused in its doors' words (mf-cal2: the door
    # units no longer run only to raise them; nothing is fitted yet).
    var scan = False
    for u in range(units):  # driver-loop: over the members (one flag each)
        if not member_set_tests(set, u):
            scan = True
    if scan and not _x_finite(Xa, d):
        admitted_finite_refuse(1, String(py_type(base).__name__), "X")
    var plan = CalFitPlan(trees_address(Xa), n, d, codes_address, k,
                          MEMBER_OP_DECISION if decision else MEMBER_OP_PREDICT_PROBA, sigmoid, ensemble)
    if mx.draw is not None:
        plan.in_base(mx.base_rows, draw_codes_address)
    for f in range(folds):  # driver-loop: over the cv folds (their test rows' addresses, read with the GIL held)
        var te = splits[f][1]
        var te_x = _meta_compose(mx, te, context)  # `te` itself without a fold
        _ = py_call(composed.append, te_x)
        plan.fold_in(trees_address(te), trees_address(te_x), Int(py=te.shape[0]))
        _ = te^  # `splits` owns the fold rows to the end of RUN; named past the call that stored its address
    plan.markers = markers^
    # THE RESIDENT ROWS (lane mf-cal7; xtrees/trees_calibrated_fit_run.mojo
    # module note): inside the scope `_calibrated_fit` opened for a DeviceRows
    # base, a member whose image reads a row index and keeps X resident
    # (every unit is a clone on one image: unit 0 answers) is scored over the
    # whole X plus each fold's test rows, X's pin offered on each scoring call:
    # one reference per fold, taken here with the GIL held and the ones no
    # cache took returned after the RUN. Its fits already read their train
    # rows by index (core/member_set_run.mojo, `member_set_resident` above).
    var score_pin = 0
    if (set.run.scope != MEMBER_SCOPE_NONE and (kind_mask & MEMBER_CAN_ROW_INDEX) != 0
            and set.run.plans[0].member.can(MEMBER_CAN_ROW_INDEX) and set.run.plans[0].member.can(MEMBER_CAN_SCOPE)):
        score_pin = member_pin_take(Xa, folds)
        if score_pin != 0:
            plan.resident(set.run.scope, score_pin)
    var glue = _xr(xt)
    var released = True
    for u in range(units):  # driver-loop: over the members (THE GIL RULE word of each)
        if set.run.gil_free[u] == MEMBER_SET_TURN_HELD:
            released = False
    # RUN: one region, nothing Python inside (the pins are taken before it
    # and returned after it, with the GIL held).
    var res = CalFitOut()
    var answered = CAL_ALL_ANSWERED
    var failure = String("")
    try:
        try:
            member_set_pins_take(set)
            if released:
                var py = Python()
                with GILReleased(py):
                    answered = calibrated_fit_run(glue, set.run, plan, res)
            else:
                answered = calibrated_fit_run(glue, set.run, plan, res)
        finally:
            member_set_pins_return(set)
            if score_pin != 0:
                member_pin_return(score_pin, folds - plan.pins_taken())
    except error:
        # A refused scoring call (its member's words), every state released.
        failure = String(error)
        answered = CAL_FIT_FAILED
    # THE LIFETIME RULE: X, the codes and the fold rows were read through the
    # fits and the RUN above (every address was read before the region).
    _ = Xa
    _ = codes
    _ = splits
    _ = draw_codes
    _ = composed
    if answered != CAL_ALL_ANSWERED:
        member_set_release(set)
        # A member whose kind tests its rows refused a non-finite X in its
        # kernel: the admission's words, as its doors raise them (one scan,
        # on this failure path only; `scan` proved X already for the rest).
        if not scan and not _x_finite(Xa, d):
            admitted_finite_refuse(1, String(py_type(base).__name__), "X")
        if failure.byte_length() == 0:
            failure = _cal_fit_words(set, plan, member_set_first_failure(set.run))
        if not ensemble and plan.failed_fold >= 0:
            # scikit-learn warned for each fold before the failing one (and for
            # the failing fold itself when its scores were refused)
            _cal_warn_lacking(plan, plan.failed_fold + 1 if plan.failed_scoring else plan.failed_fold, context)
        _ = plan.folds()  # its fold classes were read for the words above
        raise Error(failure)
    # PACK
    var code_list = py_call(b.list, py_call(b.range, PythonObject(k)))
    try:
        for u in range(units):  # driver-loop: over the members (publication)
            var labels = code_list
            if u < folds and len(plan.fold_classes[u]) > 0:
                # a re-coded fold: fold code c is the ensemble code fold_classes[u][c]
                labels = Python.list()
                for c in range(len(plan.fold_classes[u])):  # driver-loop: over the fold's classes (a few labels)
                    _ = py_call(labels.append, PythonObject(Int(plan.fold_classes[u][c])))
            _ = member_set_pack(set, u, labels, b)
    except error:
        # A refused publication: its family's pack words, no state left.
        member_set_release(set)
        raise error^
    member_set_notices(set)
    member_set_release(set)
    if not ensemble:
        _cal_warn_lacking(plan, plan.folds(), context)
    # THE LIFETIME RULE: a borrowing member read its re-coded targets (the
    # plan's) and its draw and targets (`draw_codes`, `composed`) again at its pack: they live to here.
    _ = plan.folds()
    _ = draw_codes
    _ = composed
    if ensemble:
        for f in range(folds):  # driver-loop: over the cv folds (the pairs, in fold order)
            _ = py_call(pairs.append, Python.tuple(clones[f], _cal_pack(res, res.first[f], res.count[f], context)))
        return
    _ = py_call(pairs.append, Python.tuple(clones[folds], _cal_pack(res, res.final_first, res.final_count, context)))


def _calibrated_fit(model: PythonObject, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    _calibrated_validate(model, context)
    var xt = _xt(py_fields(model), context)
    # THE FOLD IN PLACE (lane vdf-meta4): a DeviceRows fold (the class opts in with `_mojolearn_device_rows`)
    # is read out of its host base (`_meta_x_fit`); the cv folds are composed inside it (`_cal_native`).
    var mx = _meta_x_fit(X, context)
    var Xa = mx.x
    var n = mx.n
    var d = mx.d
    var labels = trees_encode_labels(y, labels_scope(model, context))
    _set(model, "classes_", labels.classes)
    var k = len(labels.classes)
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var nk = trees_keys()
    # scikit-learn's two cv refusals, after its label encode and before any
    # split or fit (lane mf-del5; the META kind's one body, `cal_cv_check`).
    # Inconsistent X and y rows are refused first, by `_splits`, as
    # scikit-learn's `indexable` refuses them first.
    var codes_view = labels.codes
    if Int(py=codes_view.shape[0]) == n:
        var cv_words = msel_cv_fold_words(_attr(model, nk[].k_cv, PythonObject(None)), prep_native_context(context))
        cal_cv_check(trees_address(codes_view), n, k, cv_words[0], cv_words[1])
        _ = codes_view  # keep the Python object alive through the native count above
    var base = _attr(model, nk[].k_estimator, PythonObject(None))
    if base is None:
        base = prep_estimator_clone(py_item(py_item(context, nk[].k_native_estimator_defaults), nk[].k_LinearSVC), context)
    _require_member(base, _owner(model), context)
    # The base runs on the calibration's device (bindings/trees_meta_device_door.mojo): read once per fit.
    var device = py_get(py_fields(model), "device")
    var origin = meta_device_origin(model, device)
    meta_members_device(device, Python.list(base), _owner(model), origin, context, trees_builtins())
    _set(model, "_cal_fast", PythonObject(None))
    var ensemble = _truthy(_attr(model, nk[].k_ensemble, PythonObject(True)))
    var splits = _splits(xt, _attr(model, nk[].k_cv, PythonObject(None)), X, y, n, labels.codes, not ensemble, k, context)
    var pairs = Python.list()
    _set(model, "calibrated_classifiers_", pairs)
    # The folds are independent units of ONE typed RUN (`_cal_native`,
    # xtrees/trees_calibrated_fit_run.mojo, lane mf-cal; lane mf-del8 deleted
    # the door fold units `_CalUnits`, so every base is fitted, scored and
    # calibrated there, or refused there in its refuser's words).
    var jobs = native_unit_jobs(_attr(model, nk[].k_n_jobs, PythonObject(None)), "CalibratedClassifierCV")
    # THE RESIDENT ROWS (lane mf-cal7; fb12-rows 2026-11-17, a measured
    # winner kept as a route of that RUN): for a base whose class takes
    # DeviceRows (the naive Bayes and discriminant classifiers, GaussianNB
    # the default board member) the fit runs inside one resident scope with
    # the DeviceRows floor (`_cal_scope_open`), so the RUN hands every fold
    # member's fit and scores the whole X plus the fold's row numbers: X goes
    # up ONCE for the whole fit, each member program gathers its rows on the
    # device (`x_prep_dev_take_rows`), nothing comes down, and X is tested
    # finite once (`_cal_native`). Same rows in the same order as the door's
    # DeviceRows fold units (deleted, lane mf-cal7): no bit changes. Before,
    # each fold side was `_gather` (xtrees gather_f32_device): all of X
    # uploaded, the rows gathered and DOWNLOADED, twice per fold, then each
    # member uploaded its rows again and scanned them on the host.
    var mctx = PythonObject(None)
    var scope = PythonObject(None)
    if _cal_rows_resident(base, Xa, mx.base_rows, d, context):
        mctx = prep_native_context(context)
        # Inside model selection's own `resident()` scope (a fold read in place: its base is resident there
        # already) no second scope is entered: the members find the base in the caller's (as stacking).
        if member_scope_word(py_item(mctx, PythonObject("family_msel_rows_context"))) == MEMBER_SCOPE_NONE:
            scope = _cal_scope_open(mctx)
    try:
        # Raises every refusal in its refuser's words (lanes mf-cal3, mf-del8).
        _cal_native(model, xt, base, mx, k, labels.codes, labels.classes, splits, ensemble, jobs, pairs, device,
                    origin, context)
    finally:
        if scope is not None:
            _cal_scope_close(scope, mctx)
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _cal_door(e: PythonObject, decision: Bool, Xa: PythonObject, n: Int, class_cols: Int, keep: PythonObject,
             answer: PythonObject, context: PythonObject) raises -> VoteMember:
    """One pair member's float32 score block from its door (or the Array its
    held state already wrote, `answer`), as `_out2d` admitted it."""
    var out = answer
    if out is None:
        out = meta_member_call(e, "decision_function" if decision else "predict_proba", Python.tuple(Xa), context)
    var blk = _out2d(out, n, context)
    _ = py_call(keep.append, blk.array)
    var got = VoteMember(blk.address, blk.cols, class_cols, 1.0)
    _ = blk^
    return got


def _cal_member_decision(e: PythonObject, context: PythonObject) -> Bool:
    """scikit-learn's `hasattr(e, "decision_function")` for a fitted pair
    member of CalibratedClassifierCV (its predict reads the scores the fit
    calibrated, `_cal_native`'s rule): `prep_estimator_has`, which answers
    a MEMBER-ONLY plan (no method recipes: the Host* classes of lane
    mf-cal3b) from its class, as the fit reads it (lane w3f2-cv; REMOVED: the
    X-free member probe this predict made for a member-only plan, which the
    fit no longer matches). Read once per pair per predict, at UNPACK. Never
    raises (False: the predict_proba scores)."""
    try:
        return prep_estimator_has(e, "decision_function", context)
    except:
        return False


def calibrated_pair_codes(e: PythonObject, k: Int, keep: PythonObject,
                          context: PythonObject) raises -> Tuple[Int, Int]:
    """X-FREE UNPACK of one calibrated pair's member (lane mf-shap3, 2026-11-23; `_calibrated_fused` and Kernel /
    Permutation SHAP): for k > 2 the address of its class codes (int32, `_sub_cols`, its owner kept in `keep`)
    and its class count; (0, 0) for two classes."""
    if k == 2:
        return (0, 0)
    var nk = trees_keys()
    var sub_cols = _sub_cols(e, context)
    _ = py_call(keep.append, sub_cols)
    return (trees_address(sub_cols), len(_attr(e, nk[].k_classes_, Python.list())))


def calibrated_pair_words(e: PythonObject, fitted: PythonObject, k: Int, member_cols: Int, var cols_addr: Int,
                          var ks: Int, keep: PythonObject, mut cal_pairs: List[CalPair], mut cals: List[CalFit],
                          context: PythonObject) raises:
    """X-FREE UNPACK of one calibrated pair's words once its member's score columns (`member_cols`) are known
    (lane mf-shap3, 2026-11-23; `_calibrated_fused` and Kernel / Permutation SHAP): its positive column, its
    class codes and count (`calibrated_pair_codes`), and its calibrators (Platt's a, b; isotonic knots kept in
    `keep`), appended as one `CalPair` and its `CalFit`s."""
    var at = 0
    var used = len(fitted)
    if k == 2 and member_cols != 1:
        at = _index_of_one(e, context)
        meta_binary_column_check(at)
    if k != 2 and ks != k:
        # lane mf-cal3b: a member that lacks a class is placed by THE
        # PLACEMENT RULE (xtrees/trees_calibrated_fit_run.mojo
        # `cal_fold_place`, the fit's own rule: scikit-learn's
        # `_CalibratedClassifier.predict_proba`): its placed columns as int32
        # codes (kept in `keep`), its calibrators the last of them. Only such
        # a member pays this (one small array per call); a member holding
        # every class reads its class codes as before.
        var codes = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=cols_addr)
        var held = List[Int32](capacity=ks)
        for c in range(ks):  # driver-loop: over the member's class codes (a few words)
            held.append(codes[c])
        var operation = MEMBER_OP_DECISION if member_cols == 1 else MEMBER_OP_PREDICT_PROBA
        var place = cal_fold_place(k, held, operation, member_cols, True, 0)
        # scikit-learn's predict zips the score columns with the calibrators: a pair fitted with more
        # calibrators than its member's columns place (ensemble=False's full fit with the unlabeled marker,
        # lane mf-meta5b) reads the first of them.
        if len(place.calibrate) > len(fitted):
            raise Error("ValueError: mojolearn: CalibratedClassifierCV: a pair holds " + String(len(fitted))
                        + " calibrators for a member placed at " + String(len(place.calibrate)) + " columns")
        used = len(place.calibrate)
        var placed = admitted_vector(len(place.place), "<i4", context)
        var to = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(placed))
        for i in range(len(place.place)):  # driver-loop: over the placed columns (a few words)
            to[i] = place.place[i]
        _ = py_call(keep.append, placed)
        cols_addr = trees_address(placed)
        ks = len(place.place)
    cal_pairs.append(CalPair(at, cols_addr, ks, len(cals), used))
    for j in range(used):  # driver-loop: over the class columns (one calibrator each)
        var par = fitted[j][1]
        if String(fitted[j][0]) == "sigmoid":
            cals.append(CalFit(False, Float64(py=par[0]), Float64(py=par[1]), 0, 0, 0))
        else:
            _ = py_call(keep.append, par[0])
            _ = py_call(keep.append, par[1])
            cals.append(CalFit(True, 0.0, 0.0, trees_address(par[0]), trees_address(par[1]), Int(py=par[2])))


def _calibrated_fused(model: PythonObject, xt: PythonObject, pairs: PythonObject,
                      Xa: PythonObject, n: Int, k: Int, context: PythonObject) raises -> PythonObject:
    """`_calibrated_proba` through `x_trees_calibrated_proba`: the members'
    scores, then the whole epilogue (scores, calibrators, complement or
    normalization, float32 narrowing, the mean) as one device program with
    one readback (lane perf-sweep-trees, 2026-11-05; xtrees/calibrated_apply.mojo).

    Lane bm-meta-3 (2026-11-23): UNPACK here (per pair its member opened on
    its held state when its family serves it, else its door's block; the
    positive column, class codes and calibrator parameters read once), RUN
    `calibrated_proba_run` (no Python; the GIL released when a member is
    native), the fallback rule on a refusal. Same words: no bit moves."""
    var acc = admitted_vector(n * k, "<f8", context)
    var nk = trees_keys()
    var got = _VoteUnpack()
    var cal_pairs = List[CalPair](capacity=len(pairs))
    var cals = List[CalFit]()
    var decisions = List[Bool](capacity=len(pairs))
    try:
        for i in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs (admission, once each)
            var e = pairs[i][0]
            var fitted = pairs[i][1]
            var decision = _cal_member_decision(e, context)
            decisions.append(decision)
            var codes = calibrated_pair_codes(e, k, got.keep, context)
            var operation = MEMBER_OP_DECISION if decision else MEMBER_OP_PREDICT_PROBA
            var opened = _member_open(e, Xa, operation, True, context)
            var member: VoteMember
            if opened.served and opened.gil_free:
                member = VoteMember(len(got.opened), opened.held.state, operation, opened.cols, opened.dtype,
                                    codes[0], 1.0)
                got.native += 1
                got.opened.append(opened^)
            else:
                var answer = PythonObject(None)
                if opened.served:
                    try:
                        answer = member_held_run(opened, trees_address(Xa), n, _cols(Xa), context)
                    finally:
                        member_held_close(opened)
                member = _cal_door(e, decision, Xa, n, codes[0], got.keep, answer, context)
            got.members.append(member)
            calibrated_pair_words(e, fitted, k, member.cols, codes[0], codes[1], got.keep, cal_pairs, cals, context)
        for i in range(len(got.members)):  # driver-loop: over members
            if got.members[i].native:
                got.members[i].plan = Int(UnsafePointer(to=got.opened[got.members[i].plan].held.plan))
        var glue = _xr(xt)
        var rows = VoteRows(trees_address(Xa), n, _cols(Xa))
        # the address is read here, with the GIL held: inside GILReleased the trees_address call ran Python
        # without the GIL (segfault in verify --self-test, CalibratedClassifierCV.score; nn-integrate, lane
        # bm-meta-3's RUN)
        var acc_at = trees_address(acc)
        while True:  # driver-loop: over refusals (at most one round per native member)
            var refused = VOTE_ALL_ANSWERED
            if got.native > 0:
                with GILReleased(Python()):
                    refused = calibrated_proba_run(glue, rows, got.members, cal_pairs, cals, k, acc_at)
            else:
                refused = calibrated_proba_run(glue, rows, got.members, cal_pairs, cals, k, acc_at)
            if refused == VOTE_ALL_ANSWERED:
                break
            got.members[refused] = _cal_door(pairs[refused][0], decisions[refused], Xa, n,
                                             got.members[refused].class_cols, got.keep, PythonObject(None), context)
            got.native -= 1
    except e:
        _vote_close(got)
        raise e^
    _vote_close(got)
    _ = acc  # keep the Python object alive through the native call above
    _ = Xa  # the rows are read through RUN
    _ = got.keep  # the members' blocks, class codes and knots are read through RUN
    return trees_reshape(acc, Python.tuple(n, k), context)


def _scale(xt: PythonObject, acc: PythonObject, divisor: Float64) raises:
    # The mean over the calibrated pairs (moved here from trees_public_boost_door when lane mf-oob3 deleted its
    # other users there; this is its one caller).
    var glue = _xr(xt)
    _ = xt_scale(glue, trees_address(acc), Int(py=acc.size), divisor)
    _ = acc


def _calibrated_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var nk = trees_keys()
    var k = len(_attr(model, nk[].k_classes_, Python.list()))
    var pairs = _attr(model, nk[].k_calibrated_classifiers_, Python.list())
    if k >= 2:
        return _calibrated_fused(model, xt, pairs, Xa, n, k, context)
    var acc = _class_zero(xt, n * k, context)
    for i in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
        var p = _calibrated_probs(model, xt, pairs[i][0], pairs[i][1], Xa, context)
        _acc_w(xt, acc, p, n * k, 1.0, context)
    _scale(xt, acc, Float64(len(pairs)))
    return trees_reshape(acc, Python.tuple(n, k), context)


def _calibrated_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    return trees_decode_labels(_attr(model, nk[].k_classes_, PythonObject(None)), _calibrated_codes(model, X, context),
                               context)


def _calibrated_codes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`CalibratedClassifierCV.predict` before its decode: the row argmax of
    the calibrated probabilities, int32 codes into `classes_`."""
    var p = _calibrated_proba(model, X, context)
    var xt = _xt(py_fields(model), context)
    var nk = trees_keys()
    var classes = _attr(model, nk[].k_classes_, PythonObject(None))
    return _argmax(xt, p, _rows(p), len(classes), context)


# ------------------------------------------------------------------- score


def _regressor_score(model: PythonObject, prediction: PythonObject, y: PythonObject, sample_weight: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`RegressorScoreMixin.score`: the float32 R^2 of the prediction."""
    var nk = trees_keys()
    var mode = py_get(py_fields(model), nk[].k_numeric_mode)
    var truth = admit_f32(y, -1, "y", context, finite=False)
    var pred = admit_f32(prediction, -1, "prediction", context, finite=False)
    if len(_attr(pred.array, nk[].k_shape, Python.tuple(0))) == 2 or len(_attr(truth.array, nk[].k_shape, Python.tuple(0))) == 2:
        # 2-D targets: the metrics family's public r2_score (its multioutput
        # program), on the same float32 blocks.
        var metrics = py_item(context, nk[].k_native_meta_metrics)
        var selected = mode if mode is not None else py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
        var binding = metrics[1][selected]
        if py_type(binding) is trees_builtins().str:
            raise Error(String(binding))
        # pyout-metrics-cnn (2026-11-11): the metrics owner's body, called natively (no CPython call of
        # its family door through `prep_entry_call`); metrics[0] is the family context that door received.
        var result = metrics_cls_operation(PythonObject(None), MCLS_OPN_FAMILY_MCLS_R2,
                                           Python.tuple(binding, truth.array, pred.array, sample_weight,
                                                        "uniform_average", True, mode), metrics[0])
        _ = truth^
        _ = pred^
        return result
    var result = trees_r2(truth.array, pred.array, sample_weight, mode, context)
    _ = truth^
    _ = pred^
    return PythonObject(result)


def _meta_class_codes(model: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """A single-output meta classifier's prediction before its decode:
    int32 codes into its `classes_` (`predict` decodes exactly these)."""
    if kind == "voting_classifier":
        return _voting_codes(model, X, context)
    if kind == "stacking_classifier":
        return _stacking_codes(model, X, context)
    if kind == "ovr":
        return _ovr_codes(model, X, context)
    if kind == "calibrated":
        return _calibrated_codes(model, X, context)
    raise Error("unknown meta classifier schema " + kind)


def _meta_predict(model: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "voting_classifier":
        return _voting_predict(model, X, context)
    if kind == "voting_regressor":
        return _voting_regress(model, X, context)
    if kind == "stacking_classifier":
        return _stacking_predict(model, X, True, context)
    if kind == "stacking_regressor":
        return _stacking_predict(model, X, False, context)
    if kind == "multioutput_regressor":
        return _multioutput_regressor_predict(model, X, context)
    if kind == "multioutput_classifier":
        return _multioutput_classifier_predict(model, X, context)
    if kind == "ovr":
        return _ovr_predict(model, X, context)
    if kind == "calibrated":
        return _calibrated_predict(model, X, context)
    raise Error("unknown meta-estimator schema " + kind)


def _meta_fit(model: PythonObject, kind: String, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
              context: PythonObject) raises -> PythonObject:
    if kind == "voting_classifier":
        return _voting_fit(model, X, y, sample_weight, True, context)
    if kind == "voting_regressor":
        return _voting_fit(model, X, y, sample_weight, False, context)
    if kind == "stacking_classifier":
        return _stacking_classifier_fit(model, X, y, sample_weight, context)
    if kind == "stacking_regressor":
        return _stacking_regressor_fit(model, X, y, sample_weight, context)
    if kind == "multioutput_regressor":
        return _multioutput_regressor_fit(model, X, y, sample_weight, context)
    if kind == "multioutput_classifier":
        return _multioutput_classifier_fit(model, X, y, sample_weight, context)
    if kind == "ovr":
        return _ovr_fit(model, X, y, context)
    if kind == "calibrated":
        return _calibrated_fit(model, X, y, context)
    raise Error("unknown meta-estimator schema " + kind)


def _meta_transform(model: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "voting_classifier":
        return _voting_transform(model, X, True, context)
    if kind == "voting_regressor":
        return _voting_transform(model, X, False, context)
    if kind == "stacking_classifier" or kind == "stacking_regressor":
        return _stacking_transform(model, X, context)
    raise Error("AttributeError: '" + _owner(model) + "' object has no attribute 'transform'")


def _fit_params_weight(fit_params: PythonObject) raises -> PythonObject:
    """`fit(X, y, **fit_params)`: only `sample_weight` is a fit parameter."""
    var b = trees_builtins()
    var keys = py_call(b.list, fit_params)
    for i in range(len(keys)):  # driver-loop: over fixed call structure, small-loop(keyword arguments): the caller's fit parameters
        if String(keys[i]) != "sample_weight":
            raise Error("TypeError: fit() got an unexpected keyword argument '" + String(keys[i]) + "'")
    return py_get(fit_params, PythonObject("sample_weight"))


def meta_operation(model: PythonObject, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """The meta-estimators' methods (`trees_meta_*`) reached directly (a
    composed estimator's plan, cross-validation): admit, then run."""
    prep_estimator_admit(model, "native meta-estimator", context)
    meta_admit_configuration(model, context)
    msel_admit(args, prep_native_context(context))
    return meta_admitted_operation(model, meta_kind(model, context), operation, args, context)


def _member_list(model: PythonObject, kind: String) raises -> PythonObject:
    """The fitted members a prediction of `model` runs on X itself."""
    var nk = trees_keys()
    if kind == "calibrated":
        var pairs = _attr(model, nk[].k_calibrated_classifiers_, Python.list())
        var members = Python.list()
        for i in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
            _ = py_call(members.append, pairs[i][0])
        return members
    return _attr(model, nk[].k_estimators_, Python.list())


def _staged_input(model: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(Xa, staged)`: X converted once (`_check_x`, which every prediction
    path runs first; on the converted Array it returns that Array) and staged
    as the one device copy its forest members share (bindings/trees_staged_input.mojo).
    Any refusal here leaves `(X, [])`: the operation then raises it itself,
    in its own order and words.

    THE FOLD FROM THE STORE (lane vdf-meta5): a model-selection fold
    (DeviceRows on a GPU route) is left UNREAD when no forest binary would
    be staged (`trees_stage_wanted`, at the fitted width): the operation
    then reads it out of its host base (`_check_meta_x`) and its native
    members take the fold's rows on the device. Before, this admission
    gathered every fold on the host first, so no meta predict ever saw the
    fold (vdf-meta4's Voting / MultiOutputRegressor predicts from the store
    included). A fold that two or more forest members predict through is
    admitted and staged once, as before."""
    var nk = trees_keys()
    if not (py_get(context, nk[].k_device_cpu_context) is context) and door_is_device_rows(X, context):
        var wanted = Python.list()
        try:
            wanted = trees_stage_wanted(_member_list(model, kind),
                                        Int(py=_attr(model, nk[].k_n_features_in_, PythonObject(0))), context)
        except:
            return Python.tuple(X, Python.list())
        if len(wanted) == 0:
            return Python.tuple(X, Python.list())
        var Xf: PythonObject
        try:
            Xf = _check_x(model, X, context)
        except:
            return Python.tuple(X, Python.list())
        return Python.tuple(Xf, trees_stage_tables(wanted, trees_address(Xf), _rows(Xf), _cols(Xf)))
    var Xa: PythonObject
    try:
        Xa = _check_x(model, X, context)
    except:
        return Python.tuple(X, Python.list())
    # Staging never leaves a binary staged on a refusal: a failed stage is
    # unstaged inside trees_stage_members, and the caller unstages the rest.
    return Python.tuple(Xa, trees_stage_members(_member_list(model, kind), trees_address(Xa), _rows(Xa), _cols(Xa),
                                                context))


def meta_admitted_operation(model: PythonObject, kind: String, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """`meta_operation` after its admission. The public door
    (bindings/trees_public_meta_lifecycle.mojo) has already admitted the
    model, its constructor configuration (a native clone of the meta-estimator
    and every member) and the arguments, and read `kind`; it calls this
    directly (lane perf-meta-estimators, 2026-11-05: every public call used
    to pay that admission twice, the second time inside meta_operation).
    A prediction (predict, predict_proba, transform, score) stages X once
    for its forest members (lane perf-sweep-trees, 2026-11-05)."""
    # X is admitted once for the whole operation (lane A1
    # admit-trees-labels): the matrices the meta-estimator admits inside this
    # scope are held for its members' doors and dropped when it ends.
    var mark = admitted_scope_begin()
    var result = PythonObject(None)
    try:
        if (operation == "trees_meta_predict" or operation == "trees_meta_predict_proba"
                or operation == "trees_meta_transform" or operation == "trees_meta_score"):
            var prepared = _staged_input(model, kind, args[0], context)
            var call = Python.list(prepared[0])
            for i in range(1, len(args)):  # driver-loop: over fixed call structure, small-loop(args): the call's remaining positional arguments
                _ = py_call(call.append, args[i])
            try:
                result = _meta_dispatch(model, kind, operation, py_call(trees_builtins().tuple, call), context)
            finally:
                trees_unstage(prepared[1])
        else:
            result = _meta_dispatch(model, kind, operation, args, context)
    finally:
        admitted_scope_end(mark)
    return result


def _meta_dispatch(model: PythonObject, kind: String, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    if operation == "trees_meta_fit":
        return _meta_fit(model, kind, args[0], args[1], args[2], context)
    if operation == "trees_meta_predict":
        return _meta_predict(model, kind, args[0], context)
    if operation == "trees_meta_predict_proba":
        if kind == "voting_classifier":
            return _voting_predict_proba(model, args[0], context)
        if kind == "stacking_classifier":
            return _stacking_predict_proba(model, args[0], context)
        if kind == "multioutput_classifier":
            return _multioutput_classifier_proba(model, args[0], context)
        if kind == "ovr":
            return _ovr_predict_proba(model, args[0], context)
        if kind == "calibrated":
            return _calibrated_proba(model, args[0], context)
        raise Error("AttributeError: '" + _owner(model) + "' object has no attribute 'predict_proba'")
    if operation == "trees_meta_transform":
        return _meta_transform(model, kind, args[0], context)
    if operation == "trees_meta_fit_transform":
        var fitted = _meta_fit(model, kind, args[0], args[1], _fit_params_weight(args[2]), context)
        _ = fitted
        # The transform half stages X for its forest members like transform.
        var prepared = _staged_input(model, kind, args[0], context)
        var result = PythonObject(None)
        try:
            result = _meta_transform(model, kind, prepared[0], context)
        finally:
            trees_unstage(prepared[1])
        return result
    if operation == "trees_meta_score":
        var X = args[0]
        var y = args[1]
        var sample_weight = args[2]
        if kind == "multioutput_classifier":
            return _multioutput_classifier_score(model, X, y, sample_weight, context)
        if kind.endswith("_regressor"):
            var prediction = _meta_predict(model, kind, X, context)
            return _regressor_score(model, prediction, y, sample_weight, context)
        # Scored in codes (lane ne2-b-members, N082): the predicted codes
        # and `classes_` go to the shared score from codes (y encoded once
        # against `classes_`, bindings/classical_labels_door.mojo
        # `classical_score_from_codes`, lane ne1-h-labels) instead of a decode
        # to labels (a Python list of n objects for str classes) that the
        # accuracy then re-encoded. Same refusals, same match count and
        # weight sums: the same bits.
        var nk = trees_keys()
        var mode = py_get(py_fields(model), nk[].k_numeric_mode)
        var codes = _meta_class_codes(model, kind, X, context)
        return PythonObject(classical_score_from_codes(y, _attr(model, nk[].k_classes_, PythonObject(None)), codes,
                                                       sample_weight, mode, labels_scope(model, context)))
    raise Error("unknown meta-estimator operation " + operation)
