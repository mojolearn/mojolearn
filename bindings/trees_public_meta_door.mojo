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
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_FINITE, ARG_OUT_OF_RANGE
from bindings.py_admitted_registry import admitted_held_index
from bindings.admitted_meta_scope_unpack import admitted_held_verify
from bindings.trees_public_inputs_run import trees_finite_f32
from core.admitted_refusals import admitted_finite_refuse
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.family_common_pack import family_call
from std.sys.compile import is_defined
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
    admit_c_view, admitted_dtype, admitted_storage, admitted_unfilled,
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
    _mode, _xt, _xr, _x_f32, _as_array, _fsum, _arange, _class_major, _gather, _gather_codes,
    _gather_vec, _argmax, _scale, _member_codes,
)
from bindings.trees_public_score_door import trees_r2, trees_class_codes
from bindings.classical_labels_door import labels_accuracy, _classification_encoded as score_classification_encoded
from bindings.member_set_door import (
    MemberSet, member_set_admit, member_set_own_block, member_set_run, member_set_pack, member_set_release,
)
from core.member_abi import (
    MEMBER_F32, MEMBER_I32, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION,
)
from bindings.member_set_door import MEMBER_SET_NATIVE, member_predict_held
from bindings.prep_native_estimator_door import (
    prep_estimator_admit, prep_estimator_clone, prep_estimator_call, prep_estimator_has,
    prep_estimator_plan, prep_native_context, prep_native_split, prep_estimator_predict_internal,
)
from bindings.classical_labels_door import _label_list, classical_score_from_codes
from bindings.p3_labels_metrics_native_door import p3_interleave_columns
from bindings.metrics_cls_native_unpack import metrics_cls_operation
from bindings.operation_names_mcls import MCLS_OPN_FAMILY_MCLS_R2
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.classical_pandas_door import native_pandas_exact
from bindings.native_unit_threads_unpack import (
    native_unit_threads_run, native_unit_gil_enter, native_unit_gil_leave, native_unit_jobs, NativeUnitFailure,
    native_unit_gate_enter, native_unit_device_workers,
)
from bindings.native_unit_os_threads_run import native_unit_gate, native_unit_gate_leave
from core.host_parallel import host_fp_env, host_fp_env_set
from std.python._cpython import GILReleased
from std.ffi import c_int, external_call
from bindings.native_array_storage_door import native_array_from_values
from bindings.msel_inputs_unpack import msel_admit, msel_input_count
from bindings.trees_meta_admission_door import meta_admit_configuration
from bindings.trees_staged_input_door import trees_stage_members, trees_unstage
from bindings.trees_side_records_unpack import trees_side_get, trees_side_put
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.base_helpers_released_unpack import (
    base_check_indices_i64, base_bincount_i64, base_reduce_stat_int, base_equal_elements,
)
from core.linear_entry_table import LinearEntryKey
from core.linear_multi_calls import ridge_multi_ref, ridge_multi_available, rm_predict
from bindings.trees_meta_ridge_run import RidgeMultiFit, ridge_multi_fit_run
from bindings.py_optional import py_or_none
from xtrees.glue_calls import (
    xt_accumulate, xt_accumulate_cols, xt_accumulate_onehot, xt_scatter, xt_device_folds, xt_transpose_f64,
    xt_stack_w64, xt_rows_any_unset, xt_indicator_codes, xt_binary_proba, xt_normalize_rows, xt_platt_fit,
    xt_isotonic_fit, xt_platt_apply_strided, xt_isotonic_predict_strided, xt_complement_pairs,
    xt_accumulate_cols_many, xt_ovr_assemble, xt_calibrated_proba,
)
from bindings.trees_keys_unpack import trees_keys
from bindings.py_balanced import py_item


comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
#: fb12-rows (2026-11-17), B arm of CalibratedClassifierCV's fold rows when
#: the member cannot take a resident DeviceRows fold (or the folds run on
#: several threads): the rows copied out of X on the host with the finite
#: test in the same pass (`_fold_rows`), instead of `_gather`'s device round
#: trip (X up, rows gathered, rows down) per fold side. Default off; owed A/B
#: in docs/apple-fast/EXPERIMENTS.md (calibrated / stacking fit, NV).
comptime CALIB_FAST_HOST_FOLDS = is_defined["MOJOLEARN_CALIB_FAST_HOST_FOLDS"]()
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
    classes are 0..k-1, native element types)` of a
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
    # ensemble's code). [5]: the element type the member's native entry wrote
    # for predict, predict_proba and decision_function (`_member_native`; -1
    # not known yet), written in place when learned.
    var identity = True
    for i in range(len(subs)):  # driver-loop: over the member's fitted classes
        if Int(py=subs[i]) != i:
            identity = False
            break
    var record = Python.tuple(classes, py_call(b.list, listed), array, PythonObject(at), PythonObject(identity),
                              Python.list(PythonObject(-1), PythonObject(-1), PythonObject(-1)))
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


def _fold_rows(xt: PythonObject, Xa: PythonObject, n: Int, d: Int, rows: PythonObject, cols: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """Rows `rows` (an int32 Array) of the float32 C-order n x d `Xa` at
    columns `cols`, which the caller makes as arange(d) (every column, the
    cv folds' gather), as a new C-order Array, held for the member doors of
    the open hold scope when finite.

    The B arm under -D MOJOLEARN_CALIB_FAST_HOST_FOLDS (default off; the
    default for a member that takes DeviceRows is the resident fold,
    `_CalUnits._dev_rows`). fb12-rows (2026-11-17): a cv fold's X was `_gather` (xtrees
    gather_f32_device on a GPU install): the WHOLE of X uploaded, the rows
    gathered on the device and downloaded into a zero-filled Array, twice
    per fold (train and test rows), so a k-fold CalibratedClassifierCV fit
    moved 2k uploads and k downloads of X beside the members' own uploads of
    their rows; each member door then scanned its fold on the host. The
    rows are now copied out of X on the host (core/admitted_rows_gather.mojo,
    row tasks over the cores, model selection's fold gather since R13) into
    uninitialised storage, the copy testing the words as it goes, so a
    finite fold is held and its members scan nothing. Same words in the same
    order: no bit changes. The cost moved is per word (X's bytes over PCIe
    twice and back once, against one host copy), so it holds for every shape.
    An empty fold, or a column subset, keeps `_gather`."""
    var r = Int(py=rows.shape[0])
    if r < 1 or d < 1 or Int(py=cols.shape[0]) != d:
        return _gather(xt, trees_address(Xa), n, d, rows, cols, context)
    var out = admitted_storage(Python.tuple(r, d), "<f4", context, "C", True)
    var dst = trees_address(out)
    var status = admitted_gather_rows_f32[DType.int32](trees_address(Xa), n, d, trees_address(rows), r, dst)
    _ = Xa  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above
    if status == ARG_OUT_OF_RANGE:
        raise Error("x_trees gather: row index out of range")
    if status == ARG_FINITE and admitted_scope_open():
        admitted_hold(out, dst, r, d)
    return out


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


def _acc_votes(xt: PythonObject, acc: PythonObject, codes: PythonObject, n: Int, k: Int, on: Float64) raises:
    var glue = _xr(xt)
    _ = xt_accumulate_onehot(glue, trees_address(acc), trees_address(codes), n, k, on, 0.0)
    _ = acc
    _ = codes


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


def _clone(est: PythonObject, context: PythonObject) raises -> PythonObject:
    return prep_estimator_clone(est, context)


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


def _fit_member(member: PythonObject, X: PythonObject, y: PythonObject, weights: PythonObject,
                context: PythonObject) raises:
    var keywords = Python.dict()
    if weights is not None:
        var nk = trees_keys()
        py_setitem(keywords, nk[].k_sample_weight, weights)
    _ = meta_member_call(member, "fit", Python.tuple(X, y), context, keywords)


def _fit_units_native(clones: PythonObject, xs: PythonObject, d: Int, targets: PythonObject,
                      float_targets: Bool, weights: PythonObject, counts: PythonObject, jobs: Int,
                      context: PythonObject) raises -> Bool:
    """Fit the unfitted `clones` as ONE NATIVE MEMBER SET (core/member_abi.mojo,
    bindings/member_set_door.mojo): every unit admitted once, the fits run
    with the GIL released on host workers (independent units in parallel,
    at most `jobs`; one worker when a member's route is a GPU), the fitted
    states written into the clones as their own fit writes them. False, with
    nothing published, when any clone is not a native member (its family has
    not published, or its admit keeps the estimator's own doors) or any
    member refuses its fit or its publication: the caller then runs its
    `_fit_member` loop for all of them, which raises the estimator's own
    words if the request is wrong (the fallback rule).

    Unit i: `xs[i]` its float32 C-order X Array of `d` columns (read by view:
    no copy; members of one cv fold share the fold's block), `targets[i]`
    its target Array of as many values, float32 (`float_targets`) or the
    int32 class codes 0 .. counts[i]-1 (`counts`: one class count per unit;
    None for regressors), `weights[i]` its float32 sample weights Array or
    None.

    A classifier unit whose rows lack a class of the ensemble (a cv fold) is
    never fitted here: the set refuses before any fit
    (bindings/member_set_door.mojo, the class-cover pass) and the door loop
    fits the members on the classes they are given, as before, so the
    callers' own checks (stacking's output width between folds) still fire."""
    var b = trees_builtins()
    var m = len(clones)
    if m == 0:
        return False
    var set = MemberSet(trees_address(xs[0]), Int(py=xs[0].shape[0]), d, MEMBER_F32 if float_targets else MEMBER_I32,
                        MEMBER_I32)
    set.run.cap = jobs
    # The members keep their fitted states: this door's predict sites read
    # them (`_member_native`), so a member's first predict imports nothing.
    set.adopt = True
    for i in range(m):  # driver-loop: over the units (admission, once each)
        var classes = 0
        if counts is not None:
            classes = Int(py=counts[i])
        var w_address = 0
        if weights[i] is not None:
            w_address = trees_address(weights[i])
        if not member_set_admit(set, clones[i], trees_address(targets[i]), w_address, 0, 0, 0, 0, classes, context, b):
            return False
        member_set_own_block(set, trees_address(xs[i]), Int(py=xs[i].shape[0]))
    # The member entries read admitted words. X is held with its finite test
    # deferred to the member doors (`_hold_deferred`), which this route does
    # not run: each distinct X block and float target is proven here (a held
    # block's recorded verdict, else one scan). Not finite: the door loop,
    # whose first door refuses in the estimator's own words.
    for i in range(m):  # driver-loop: over the units (one verdict per distinct block)
        if (i == 0 or not (xs[i] is xs[i - 1])) and not _x_finite(xs[i], d):
            return False
        if float_targets and (i == 0 or not (targets[i] is targets[i - 1])):
            var finite_y = trees_finite_f32(trees_address(targets[i]), Int(py=targets[i].shape[0]))
            _ = targets  # keep the Python objects alive through the scan above
            if not finite_y:
                return False
    var fitted = False
    try:
        member_set_run(set)   # a failing unit releases every state
        fitted = True
    except:
        fitted = False
    _ = xs  # keep the Python objects alive through the member fits above
    _ = targets
    _ = weights
    if not fitted:
        return False
    var packed = False
    try:
        for i in range(m):  # driver-loop: over the units (publication)
            var codes = PythonObject(None)
            if counts is not None:
                codes = py_call(b.list, py_call(b.range, counts[i]))
            _ = member_set_pack(set, i, codes, b)
        packed = True
    except:
        packed = False
    member_set_release(set)  # a failing publication leaves no state behind
    return packed


def _fit_units(clones: PythonObject, xs: PythonObject, d: Int, targets: PythonObject, float_targets: Bool,
               weights: PythonObject, counts: PythonObject, jobs: Int, context: PythonObject) raises:
    """Fit every unit: as one native member set when every member's family
    has published its member entry (`_fit_units_native`), else one member
    fit door call after the other (`_fit_member`)."""
    if _fit_units_native(clones, xs, d, targets, float_targets, weights, counts, jobs, context):
        return
    for i in range(len(clones)):  # driver-loop: over the units
        _fit_member(clones[i], xs[i], targets[i], weights[i], context)


def _fit_members_native(clones: PythonObject, X: PythonObject, n: Int, d: Int, targets: PythonObject,
                        float_targets: Bool, weights: PythonObject, counts: PythonObject, jobs: Int,
                        context: PythonObject) raises -> Bool:
    """`_fit_units_native` for members that all read the one X (`n` x `d`)
    and the one sample-weight Array (or None)."""
    var m = len(clones)
    return _fit_units_native(clones, _repeat(X, m), d, targets, float_targets, _repeat(weights, m), counts, jobs,
                             context)


def _fit_members(clones: PythonObject, X: PythonObject, n: Int, d: Int, targets: PythonObject, float_targets: Bool,
                 weights: PythonObject, counts: PythonObject, jobs: Int, context: PythonObject) raises:
    """`_fit_units` for members that all read the one X (`n` x `d`) and the
    one sample-weight Array (or None)."""
    var m = len(clones)
    _fit_units(clones, _repeat(X, m), d, targets, float_targets, _repeat(weights, m), counts, jobs, context)


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


def _member_native(member: PythonObject, X: PythonObject, operation: Int, classifier: Bool,
                   context: PythonObject) -> PythonObject:
    """A fitted member's `operation` (MEMBER_OP_PREDICT, _PREDICT_PROBA,
    _DECISION) over X through the member state it holds
    (bindings/member_set_door.mojo `member_predict_held`: a hit after the
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
    (n, columns).

    NOT SERVED HERE, by rule: a class without a member mark; a tree, forest
    or extra-trees member (its door predicts from the one device copy of X
    this door stages for all of them, bindings/trees_staged_input_door.mojo;
    a member call would upload X once per member); a non-finite X (the
    door refuses it). Once per member per public operation."""
    comptime if not MEMBER_SET_NATIVE:
        return PythonObject(None)
    try:
        var plan = prep_estimator_plan(member, context)
        if plan is None or py_get(plan, "member") is None:
            return PythonObject(None)
        var n = _rows(X)
        var d = _cols(X)
        if n <= 0 or d <= 0 or String(py=X.dtype) != "<f4" or String(py=X.order) != "C":
            return PythonObject(None)
        var width = 0
        var known = MEMBER_F32
        var slot = 0
        var record = PythonObject(None)
        if classifier:
            record = _member_class_codes(member, context)
            if operation == MEMBER_OP_PREDICT:
                if not Bool(py=record[4]):
                    return PythonObject(None)
                width = 1
            elif operation == MEMBER_OP_PREDICT_PROBA:
                slot = 1
                width = len(record[1])
            else:
                slot = 2
            known = Int(py=record[5][slot])
        if not _x_finite(X, d):
            return PythonObject(None)
        var out = member_predict_held(member, trees_address(X), n, d, operation, classifier, width, known, True,
                                      context, trees_builtins())
        _ = X  # the rows are read through the whole member call
        if classifier and out.dtype != known:
            py_setitem(record[5], PythonObject(slot), PythonObject(out.dtype))
        return out.array
    except:
        return PythonObject(None)


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


struct Active(Copyable, Movable):
    """`_trees_estimators(estimators)`: the active names and estimators."""
    var names: PythonObject
    var ests: PythonObject

    def __init__(out self, names: PythonObject, ests: PythonObject):
        self.names = names
        self.ests = ests


def _active(estimators: PythonObject, owner: String, context: PythonObject) raises -> Active:
    var b = trees_builtins()
    if not _truthy(estimators):
        raise Error("estimators must be a non-empty list of (name, estimator)")
    var items = py_call(b.list, estimators)
    var all_names = Python.list()
    var names = Python.list()
    var ests = Python.list()
    for i in range(len(items)):  # driver-loop: over the (name, estimator) entries
        var pair = py_call(b.tuple, items[i])
        if len(pair) != 2:
            raise Error("estimators must be a non-empty list of (name, estimator)")
        _ = py_call(all_names.append, pair[0])
    if len(py_call(b.set, all_names)) != len(all_names):
        raise Error("estimator names must be unique")
    for i in range(len(items)):  # driver-loop: over the (name, estimator) entries
        var pair = py_call(b.tuple, items[i])
        var est = pair[1]
        if py_type(est) is b.str and String(est) == "drop":
            continue
        _ = py_call(names.append, pair[0])
        _ = py_call(ests.append, est)
    if len(ests) == 0:
        raise Error("all estimators are 'drop'")
    for i in range(len(ests)):  # driver-loop: over members
        _require_member(ests[i], owner, context)
    return Active(names, ests)


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


def _jobs(model: PythonObject, verbose: Bool) raises -> Int:
    """The worker bound of a meta-estimator's member fits from its `n_jobs`
    (`native_unit_jobs`: None and -1 every core, k > 0 at most k), after the
    `verbose` refusal where the estimator has that parameter. Members of
    published families fit in parallel (bindings/member_set_door.mojo);
    members on their own doors fit one after another whatever the bound.
    The results do not depend on it."""
    # TOMBSTONE: _refuse_jobs (n_jobs refused: "the members / columns / classes fit one after another") deleted
    # 2026-11-21 by lane ma-callers round 2: published members fit in parallel, so n_jobs is the worker bound.
    # Restore: git revert of that commit (no experiment define; a refusal, not an arm).
    var nk = trees_keys()
    if verbose and _truthy(_attr(model, nk[].k_verbose, PythonObject(0))):
        _refuse("verbose", "nothing logs per member.")
    return native_unit_jobs(_attr(model, nk[].k_n_jobs, PythonObject(None)), _owner(model))


# ------------------------------------------------------------------ cv folds


def _cv_value(cv: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_cv(cv)`: an int number of folds (None: 5), or `cv` itself when
    it is a splitter or an iterable of (train, test) index pairs."""
    var b = trees_builtins()
    if cv is None:
        return PythonObject(5)
    var native = prep_native_context(context)
    var nk = trees_keys()
    if py_type(py_type(cv)) is b.type and py_contains(py_item(native, nk[].k_family_msel_splitter_layouts), py_type(cv)):
        return cv
    if py_type(cv) is b.list or py_type(cv) is b.tuple:
        msel_admit(cv, native)
        return cv
    if py_type(cv) is not b.int:
        raise Error("TypeError: cv requires an integer, native splitter or list of index pairs; Python callbacks cannot execute")
    if _is_bool(cv) or not _is_integral(cv, context) or Bool(py=cv < PythonObject(2)):
        raise Error("cv=" + _repr(cv) + ": an int >= 2, a splitter with split(X, y), or an iterable of"
                    + " (train, test) index pairs")
    return py_call(b.int, cv)


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
    """`_trees_splits`: [(train rows, test rows)] as int32 Arrays."""
    var b = trees_builtins()
    var c = _cv_value(cv, context)
    var n_x = msel_input_count(X, prep_native_context(context))
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
        pairs = prep_native_split(c, X, y, context)
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
    var b = trees_builtins()
    var nk = trees_keys()
    var m = len(_attr(model, nk[].k_estimators_, Python.list()))
    var given = _attr(model, nk[].k_weights, PythonObject(None))
    var w = Python.list()
    if given is None:
        for _ in range(m):  # driver-loop: over members
            _ = py_call(w.append, PythonObject(1.0))
    else:
        var items = py_call(b.list, given)
        for i in range(len(items)):  # driver-loop: over the user weights (one per member)
            _ = py_call(w.append, py_call_typed(b.float, items[i]))
    if len(w) != m:
        raise Error("weights has " + String(len(w)) + " entries, there are " + String(m) + " estimators")
    return w


def _voting_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                classifier: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var owner = _owner(model)
    var nk = trees_keys()
    if classifier:
        var voting = _attr(model, nk[].k_voting, PythonObject(None))
        if not py_contains(Python.tuple("hard", "soft"), voting):
            raise Error("voting must be 'hard' or 'soft'")
    var jobs = _jobs(model, True)
    var Xa = _x_array(X, context)
    var target = y
    if classifier:
        var labels = trees_encode_labels(y, labels_scope(model, context))
        _set(model, "classes_", labels.classes)
        _set(model, "n_classes_", PythonObject(len(labels.classes)))
        target = labels.codes
    var active = _active(_attr(model, nk[].k_estimators, PythonObject(None)), owner, context)
    var weights = _member_weights(active.names, active.ests, sample_weight, _rows(Xa), context)
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    var m = len(active.ests)
    var clones = Python.list()
    for i in range(m):  # driver-loop: over members
        _ = py_call(clones.append, _clone(active.ests[i], context))
    # Native members read typed targets: the class codes, or (regressors) y
    # as float32 words. A y the float32 admission refuses keeps the member
    # fit doors, which refuse it in their own words.
    var native = False
    if classifier:
        var k = len(_attr(model, PythonObject("classes_"), Python.list()))
        native = _fit_members_native(clones, Xa, _rows(Xa), _cols(Xa), _repeat(target, m), False, weights,
                                     _repeat(PythonObject(k), m), jobs, context)
    else:
        var y32 = PythonObject(None)
        try:
            y32 = _f32_array(y, 1, "y", context)
        except:
            y32 = PythonObject(None)
        if y32 is not None and _rows(y32) == _rows(Xa):
            native = _fit_members_native(clones, Xa, _rows(Xa), _cols(Xa), _repeat(y32, m), True, weights,
                                         PythonObject(None), jobs, context)
    for i in range(m):  # driver-loop: over members
        if not native:
            _fit_member(clones[i], Xa, target, weights, context)
        _ = py_call(fitted.append, clones[i])
    _set(model, "named_estimators_", _named(active.names, fitted))
    _set(model, "n_features_in_", PythonObject(_cols(Xa)))
    _set(model, "_fitted", PythonObject(True))
    if classifier:
        _ = _voting_weights(model)
    _ = fields
    return model


def _voting_soft(model: PythonObject, xt: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """`VotingClassifier._soft`: the weighted average of the members'
    probabilities, class-major (n * k)."""
    var n = _rows(Xa)
    var nk = trees_keys()
    var k = Int(py=_attr(model, nk[].k_n_classes_, PythonObject(0)))
    var acc = _class_zero(xt, n * k, context)
    var w = _voting_weights(model)
    var members = _attr(model, nk[].k_estimators_, Python.list())
    # Every member's probabilities into the vote in one call (lane
    # ne2-b-members, N083): the accumulator stays on the device across the
    # members. The same adds in member order: the same bits. The per-member
    # fallback (a binary without the fused entry) is gone: every published
    # glue table has it (lane calltable-trees).
    var keep = List[PythonObject]()
    var addrs: List[Int] = [trees_address(acc)]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    for i in range(len(members)):  # driver-loop: over members
        var e = members[i]
        var xa = admit_f32(_member_scores(e, "predict_proba", Xa, context), 2,
                           "sub-estimator output", context, finite=False)
        var cols = _sub_cols(e, context)
        keep.append(xa.array)
        keep.append(cols)
        addrs.append(xa.address)
        addrs.append(trees_address(cols))
        var ks = Int(py=cols.shape[0])
        ints.append(ks)
        floats.append(Float64(ks))
        ints.append(0)
        floats.append(Float64(py=w[i]))
        _ = xa^
    var glue = _xr(xt)
    _ = xt_accumulate_cols_many(glue, addrs, ints, floats)
    _ = keep  # keep the members' blocks alive through the native call above
    _ = acc  # keep the Python object alive through the native call above
    _scale(xt, acc, _fsum(w, context))
    return acc


def _voting_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    if String(_attr(model, nk[].k_voting, PythonObject(""))) == "hard":
        raise Error("AttributeError: predict_proba is not available when voting='hard'")
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var k = Int(py=_attr(model, nk[].k_n_classes_, PythonObject(0)))
    return trees_reshape(_voting_soft(model, xt, Xa, context), Python.tuple(_rows(Xa), k), context)


def _voting_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    return trees_decode_labels(_attr(model, nk[].k_classes_, PythonObject(None)), _voting_codes(model, X, context), context)


def _voting_codes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`VotingClassifier.predict` before its decode: int32 codes into
    `classes_`."""
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var nk = trees_keys()
    var k = Int(py=_attr(model, nk[].k_n_classes_, PythonObject(0)))
    var acc: PythonObject
    if String(_attr(model, nk[].k_voting, PythonObject(""))) == "soft":
        acc = _voting_soft(model, xt, Xa, context)
    else:
        acc = _zeros(n * k, context)
        var w = _voting_weights(model)
        var members = _attr(model, nk[].k_estimators_, Python.list())
        for i in range(len(members)):  # driver-loop: over members
            _acc_votes(xt, acc, _member_label_codes(members[i], Xa, context), n, k, Float64(py=w[i]))
    return _argmax(xt, acc, n, k, context)


def _voting_transform(model: PythonObject, X: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var nk = trees_keys()
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var m = len(members)
    var soft = classifier and String(_attr(model, nk[].k_voting, PythonObject(""))) == "soft"
    var k = Int(py=_attr(model, nk[].k_n_classes_, PythonObject(0))) if classifier else 1
    var width = m * k if soft else m
    var out = _zeros(n * width, context)
    var rows = _arange(xt, n, context)
    for j in range(m):  # driver-loop: over members
        var e = members[j]
        if soft:
            var p = _zeros(n * k, context)
            _acc_cols_w(xt, p, _member_scores(e, "predict_proba", Xa, context), _sub_cols(e, context), n, k, 1.0, context)
            var blk = admit_f32(trees_reshape(p, Python.tuple(n, k), context), 2, "p", context, finite=False)
            _ = _place(xt, out, n, width, blk.view(), rows, j * k)
            _ = blk^
        else:
            var blk = _out2d(_predict32(e, Xa, context), n, context)
            _ = _place(xt, out, n, width, blk.view(), rows, j)
            _ = blk^
    return trees_reshape(out, Python.tuple(n, width), context)


def _voting_regress(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`VotingRegressor.predict`: the weighted average of the members'
    predictions."""
    var b = trees_builtins()
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var nk = trees_keys()
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var m = len(members)
    var given = _attr(model, nk[].k_weights, PythonObject(None))
    var w = Python.list()
    if given is None:
        for _ in range(m):  # driver-loop: over members
            _ = py_call(w.append, PythonObject(1.0))
    else:
        var items = py_call(b.list, given)
        for i in range(len(items)):  # driver-loop: over the user weights (one per member)
            _ = py_call(w.append, py_call_typed(b.float, items[i]))
    if len(w) != m:
        raise Error("weights has " + String(len(w)) + " entries, there are " + String(m) + " estimators")
    var acc = _zeros(n, context)
    for i in range(m):  # driver-loop: over members
        _acc_w(xt, acc, _predict32(members[i], Xa, context), n, Float64(py=w[i]), context)
    _scale(xt, acc, _fsum(w, context))
    return acc


# ---------------------------------------------------------------- Stacking


def _stacking_validate(model: PythonObject, context: PythonObject) raises:
    _ = _jobs(model, True)
    var nk = trees_keys()
    var method = _attr(model, nk[].k_stack_method, PythonObject("auto"))
    if not py_contains(Python.tuple("auto", "predict_proba", "decision_function", "predict"), method):
        raise Error("stack_method must be auto, predict_proba, decision_function or predict")
    _ = _cv_value(_attr(model, nk[].k_cv, PythonObject(None)), context)


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


def _stack_out(xt: PythonObject, est: PythonObject, method: String, Xs: PythonObject, n: Int, binary: Bool,
               context: PythonObject) raises -> PythonObject:
    """`_StackingBase._out`: a binary predict_proba keeps its second column."""
    var out = (_predict32(est, Xs, context) if method == "predict"
               else _member_scores(est, method, Xs, context))
    if method == "predict_proba" and binary:
        var p = _out2d(out, n, context)
        var col = _one_col(xt, p.view(), 1, context)
        _ = p^
        return col
    return out


def _stacking_fit(model: PythonObject, X: PythonObject, y_full: PythonObject, codes: Bool, splits: PythonObject,
                  final: PythonObject, sample_weight: PythonObject, classifier: Bool,
                  context: PythonObject) raises -> PythonObject:
    """`_StackingBase._fit_stack`. `y_full` is the int32 codes (classifier)
    or the float32 targets (regressor); a fold's targets are gathered."""
    var owner = _owner(model)
    var xt = _xt(py_fields(model), context)
    var Xa = _x_array(X, context)
    # This call owns the admitted X, y and optional weight storage. Member
    # fits cannot replace these local owners. A subsequent fit admits afresh.
    var x_view = Xa
    var y_view = y_full
    var n = Int(py=x_view.shape[0])
    var d = Int(py=x_view.shape[1])
    var nk = trees_keys()
    var active = _active(_attr(model, nk[].k_estimators, PythonObject(None)), owner, context)
    _require_member(final, owner, context)
    var all_names = py_call(trees_builtins().list, active.names)
    var all_ests = py_call(trees_builtins().list, active.ests)
    _ = py_call(all_names.append, PythonObject("final_estimator"))
    _ = py_call(all_ests.append, final)
    var sw = _member_weights(all_names, all_ests, sample_weight, n, context)
    var sw_addr = 0
    if sw is not None:
        sw_addr = trees_address(sw)
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
        _ = py_call(full.append, _clone(active.ests[i], context))
    var jobs = native_unit_jobs(_attr(model, nk[].k_n_jobs, PythonObject(None)), owner)
    _fit_members(full, Xa, n, d, _repeat(y_full, n_members), not codes, sw, counts, jobs, context)
    for i in range(n_members):  # driver-loop: over members
        _ = py_call(fitted.append, full[i])
    var methods = Python.list()
    var native_methods = List[String]()
    for i in range(len(fitted)):  # driver-loop: over members
        var method = _stack_method(model, fitted[i], classifier, context)
        _ = py_call(methods.append, method)
        native_methods.append(String(method))
    _set(model, "stack_method_", methods)
    var binary = _truthy(_attr(model, nk[].k__binary, PythonObject(False)))
    var passthrough = _truthy(_attr(model, nk[].k_passthrough, PythonObject(False)))
    var head_n = n if n < 2 else 2
    var head = _gather(xt, trees_address(x_view), n, d, _arange(xt, head_n, context), _arange(xt, d, context), context)
    _ = Xa  # keep the Python object alive through the native call above
    var widths = Python.list()
    var native_widths = List[Int]()
    var total = 0
    for i in range(len(fitted)):  # driver-loop: over members
        var blk = _out2d(_stack_out(xt, fitted[i], native_methods[i], head, head_n, binary, context), 2, context)
        _ = Xa  # keep the Python object alive through the native call above
        _ = py_call(widths.append, PythonObject(blk.cols))
        native_widths.append(blk.cols)
        total += blk.cols
        _ = blk^
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
        for f in range(f0, f1):  # driver-loop: over the wave's cv folds (train gathers)
            # The fold's train gathers are the same for every member: made
            # once per fold, not once per member (p3-trees; folds x members host
            # copies of X, y and the weights before). Members never write their
            # inputs (the scikit-learn fit contract), so sharing them is safe.
            var tr_view = splits[f][0]
            var y_tr: PythonObject
            if codes:
                y_tr = _gather_codes(xt, y_view, tr_view, context)
            else:
                y_tr = _gather_vec(xt, trees_address(y_view), Int(py=y_view.shape[0]), tr_view, context)
            var w_tr = PythonObject(None)
            if sw is not None:
                w_tr = _gather_vec(xt, sw_addr, n, tr_view, context)
                _ = sw  # keep the Python object alive through the native call above
            var Xtr = _gather(xt, trees_address(x_view), n, d, tr_view, cols_view, context)
            _ = Xa  # keep the Python object alive through the native call above
            if admitted_is_held(Xa):
                # Rows of a held X carry its verdict (finite rows of a finite X;
                # rows of an X an outer cross-validation held with its finite
                # test deferred stay deferred and linked to it, lane s2): the
                # fold's members read the fold matrices as admitted too.
                admitted_hold_rows_of(Xa, Xtr, trees_address(Xtr), Int(py=tr_view.shape[0]), d)
            for i in range(n_members):  # driver-loop: over members
                _ = py_call(unit_members.append, _clone(active.ests[i], context))
                _ = py_call(unit_x.append, Xtr)
                _ = py_call(unit_y.append, y_tr)
                _ = py_call(unit_w.append, w_tr)
        var unit_counts = PythonObject(None)
        if codes:
            unit_counts = _repeat(final_counts[0], len(unit_members))
        _fit_units(unit_members, unit_x, d, unit_y, not codes, unit_w, unit_counts, jobs, context)
        # The wave's train blocks are dropped before its test blocks are made.
        _ = py_call(unit_x.clear)
        _ = py_call(unit_y.clear)
        _ = py_call(unit_w.clear)
        for f in range(f0, f1):  # driver-loop: over the wave's cv folds (test outputs)
            var te_view = splits[f][1]
            var col0 = 0
            var Xte = _gather(xt, trees_address(x_view), n, d, te_view, cols_view, context)
            _ = Xa  # keep the Python object alive through the native call above
            if admitted_is_held(Xa):
                admitted_hold_rows_of(Xa, Xte, trees_address(Xte), Int(py=te_view.shape[0]), d)
            for i in range(n_members):  # driver-loop: over members
                var e = unit_members[(f - f0) * n_members + i]
                var blk = _out2d(_stack_out(xt, e, native_methods[i], Xte, Int(py=te_view.shape[0]), binary, context), Int(py=te_view.shape[0]), context)
                _ = Xa  # keep the Python object alive through the native call above
                var got = _place(xt, meta_view, n, width, blk.view(), te_view, col0)
                _ = blk^
                if got != native_widths[i]:
                    raise Error("a member's output width changed between folds (a class missing from a fold)")
                col0 += got
        f0 = f1
    if passthrough:
        var xb = admitted_words(Xa, trees_address(x_view), n, d)
        _ = Xa  # keep the Python object alive through the native call above
        _ = _place(xt, meta_view, n, width, xb.view(), _arange(xt, n, context), total)
        _ = Xa  # keep the Python object alive through the native call above
        _ = xb^
        _ = Xa  # keep the Python object alive through the native call above
    _ = x_view
    _ = y_view
    _set(model, "_widths_", widths)
    _set(model, "named_estimators_", _named(active.names, fitted))
    var final_ = _clone(final, context)
    _set(model, "final_estimator_", final_)
    var meta32 = _f32_array(trees_reshape(meta, Python.tuple(n, width), context), 2, "meta", context)
    _fit_members(_repeat(final_, 1), meta32, n, width, _repeat(y_full, 1), not codes, sw, final_counts, jobs,
                 context)
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _stacking_transform(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_StackingBase.transform`: the meta features (float64)."""
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var x_view = Xa
    var n = Int(py=x_view.shape[0])
    var d = Int(py=x_view.shape[1])
    var nk = trees_keys()
    var widths = _attr(model, nk[].k__widths_, Python.list())
    var total = 0
    for i in range(len(widths)):  # driver-loop: over members
        total += Int(py=widths[i])
    var passthrough = _truthy(_attr(model, nk[].k_passthrough, PythonObject(False)))
    var width = total + (d if passthrough else 0)
    var meta = _zeros(n * width, context)
    var meta_view = meta
    var rows_view = _arange(xt, n, context)
    var binary = _truthy(_attr(model, nk[].k__binary, PythonObject(False)))
    var col0 = 0
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var methods = _attr(model, nk[].k_stack_method_, Python.list())
    for i in range(len(members)):  # driver-loop: over members
        var blk = _out2d(_stack_out(xt, members[i], String(methods[i]), Xa, n, binary, context), n, context)
        col0 += _place(xt, meta_view, n, width, blk.view(), rows_view, col0)
        _ = blk^
    if passthrough:
        var xb = admitted_words(Xa, trees_address(x_view), n, d)
        _ = Xa  # keep the Python object alive through the native call above
        _ = _place(xt, meta_view, n, width, xb.view(), rows_view, col0)
        _ = Xa  # keep the Python object alive through the native call above
        _ = xb^
        _ = Xa  # keep the Python object alive through the native call above
    _ = x_view
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
    return _stacking_fit(model, X, labels.codes, True, splits, final, sample_weight, True, context)


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
    return _stacking_fit(model, X, y32, False, splits, final, sample_weight, False, context)


def _stacking_meta32(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    return _f32_array(_stacking_transform(model, X, context), 2, "meta", context)


def _stacking_predict(model: PythonObject, X: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    var final_ = _attr(model, nk[].k_final_estimator_, PythonObject(None))
    var Xm = _stacking_meta32(model, X, context)
    if not classifier:
        return meta_member_call(final_, "predict", Python.tuple(Xm), context)
    return trees_decode_labels(_attr(model, nk[].k_classes_, PythonObject(None)), _member_label_codes(final_, Xm, context),
                               context)


def _stacking_codes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`StackingClassifier.predict` before its decode: the final estimator's
    int32 codes into `classes_`."""
    var nk = trees_keys()
    var final_ = _attr(model, nk[].k_final_estimator_, PythonObject(None))
    return _member_label_codes(final_, _stacking_meta32(model, X, context), context)


def _stacking_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = trees_keys()
    var final_ = _attr(model, nk[].k_final_estimator_, PythonObject(None))
    var k = len(_attr(model, nk[].k_classes_, Python.list()))
    var Xm = _stacking_meta32(model, X, context)
    var n = _rows(Xm)
    var xt = _xt(py_fields(model), context)
    var acc = _class_zero(xt, n * k, context)
    _acc_cols_w(xt, acc, _member_scores(final_, "predict_proba", Xm, context), _sub_cols(final_, context), n, k, 1.0, context)
    return trees_reshape(acc, Python.tuple(n, k), context)


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
                           context: PythonObject) raises -> PythonObject:
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
        var e = _clone(est, context)
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
    var Xa = _x_array(X, context)
    _multioutput_y(y)
    var Ya = _f32_array(y, 2, "y", context)
    if _rows(Ya) != _rows(Xa):
        raise Error("y has " + String(_rows(Ya)) + " rows, X has " + String(_rows(Xa)))
    var nk = trees_keys()
    var est = _attr(model, nk[].k_estimator, PythonObject(None))
    _require_member(est, _owner(model), context)
    var weights = _member_weights(Python.list(PythonObject(None)), Python.list(est), sample_weight, _rows(Xa), context)
    _set(model, "_multi_ridge", PythonObject(None))
    var table = _ridge_multi_entry(model, est, weights, context)
    if table is not None and _cols(Ya) > 0:
        return _multioutput_ridge_fit(model, table, Xa, Ya, context)
    var n = _rows(Xa)
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    var rows = _arange(xt, n, context)
    var columns = Python.list()
    var clones = Python.list()
    for j in range(_cols(Ya)):  # driver-loop: over the output columns (one member each)
        var col = _codes_array(Python.list(PythonObject(j)), context)
        var yj = trees_reshape(_gather(xt, trees_address(Ya), n, _cols(Ya), rows, col, context), Python.tuple(n), context)
        _ = Ya  # keep the Python object alive through the native call above
        _ = py_call(columns.append, yj)
        _ = py_call(clones.append, _clone(est, context))
    # The columns are independent members: one native member set (parallel)
    # when the family has published, else one fit door call per column.
    _fit_members(clones, Xa, n, _cols(Xa), columns, True, weights, PythonObject(None), jobs, context)
    for j in range(len(clones)):  # driver-loop: over the output columns (one member each)
        _ = py_call(fitted.append, clones[j])
    _set(model, "n_features_in_", PythonObject(_cols(Xa)))
    _set(model, "_fitted", PythonObject(True))
    return model


def _multioutput_regressor_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var nk = trees_keys()
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var m = len(members)
    var mr = _attr(model, nk[].k__multi_ridge, PythonObject(None))
    if mr is not None:
        var outputs = Int(py=mr[2])
        var out = admitted_array(Python.tuple(n, outputs), "<f4", context)
        var table = trees_table(context, _mode(py_fields(members[0]), context), "_mojolearn_estimators")
        var rm_key = ridge_multi_ref(_estimators_key(table))
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
    var out = _zeros(n * m, context)
    var rows = _arange(xt, n, context)
    # fb12-rows (2026-11-17): each member predicts through its plan's
    # internal float32 operation (prep_estimator_predict_internal), as
    # 0.8.55's `e.predict(Xa)` from inside the package did. The public
    # `predict` op saw the user's frame above this door and returned
    # scikit-learn's float64 widening for a linear member: a wide device
    # epilogue, twice the download, a float64 output allocation, then
    # `_out2d` cast it back to float32 (a second allocation and pass) per
    # member per call. The widening is exact, so the placed values (and
    # bits) are the same.
    for j in range(m):  # driver-loop: over members
        var blk = _out2d(_predict32(members[j], Xa, context), n, context)
        _ = _place(xt, out, n, m, blk.view(), rows, j)
        _ = blk^
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
    var Xa = _x_array(X, context)
    var n = _rows(Xa)
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
            _ = py_call(clones.append, _clone(est, context))
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
            _ = py_call(clones.append, _clone(est, context))
            _ = py_call(code_columns.append, labels.codes)
            _ = py_call(counts.append, PythonObject(len(labels.classes)))
            _ = py_call(classes.append, labels.classes)
    _fit_members(clones, Xa, n, _cols(Xa), code_columns, False, weights, counts, jobs, context)
    for j in range(len(clones)):  # driver-loop: over the output columns (one member each)
        _ = py_call(fitted.append, clones[j])
    _set(model, "n_features_in_", PythonObject(_cols(Xa)))
    _set(model, "_fitted", PythonObject(True))
    return model


def _multioutput_member_codes(model: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """Each output's member codes (int32, into that output's `classes_[j]`;
    the members were fitted on the column codes): `_member_label_codes`."""
    var nk = trees_keys()
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var out = Python.list()
    for j in range(len(members)):  # driver-loop: over members
        _ = py_call(out.append, _member_label_codes(members[j], Xa, context))
    return out


def _multioutput_classifier_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    return _multioutput_assemble(model, Xa, _multioutput_member_codes(model, Xa, context), context)


def _multioutput_assemble(model: PythonObject, Xa: PythonObject, codes: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """`MultiOutputClassifier.predict` from each output's member codes: the
    decoded columns stacked into one (n, m) Array, or per-row label lists
    for label objects no Array holds."""
    var b = trees_builtins()
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
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
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var nk = trees_keys()
    var members = _attr(model, nk[].k_estimators_, Python.list())
    var classes = _attr(model, nk[].k_classes_, Python.list())
    var out = Python.list()
    for j in range(len(members)):  # driver-loop: over members
        var k = len(classes[j])
        var acc = _zeros(n * k, context)
        _acc_cols_w(xt, acc, _member_scores(members[j], "predict_proba", Xa, context), _sub_cols(members[j], context), n, k, 1.0, context)
        _ = py_call(out.append, trees_reshape(acc, Python.tuple(n, k), context))
    return out


def _multioutput_classifier_score(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                                  context: PythonObject) raises -> PythonObject:
    """`MultiOutputClassifier.score`: the (weighted) fraction of rows whose
    every output matches."""
    var b = trees_builtins()
    var Xa = _check_x(model, X, context)
    var codes = _multioutput_member_codes(model, Xa, context)
    var pred = _multioutput_assemble(model, Xa, codes, context)
    var nk = trees_keys()
    if py_type(pred) is not py_item(context, nk[].k_array_class):
        # Label objects no Array holds (str classes): scored in codes.
        return _multioutput_score_codes(model, codes, _rows(Xa), y, sample_weight, context)
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
    var Xa = _x_array(X, context)
    var labels = trees_encode_labels(y, labels_scope(model, context))
    _set(model, "classes_", labels.classes)
    var k = len(labels.classes)
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var nk = trees_keys()
    var est = _attr(model, nk[].k_estimator, PythonObject(None))
    _require_member(est, _owner(model), context)
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
        _ = py_call(clones.append, _clone(est, context))
    # One binary member per target (0/1 codes, two classes): independent,
    # fitted together.
    _fit_members(clones, Xa, _rows(Xa), _cols(Xa), targets, False, PythonObject(None),
                 _repeat(PythonObject(2), len(targets)), jobs, context)
    for i in range(len(clones)):  # driver-loop: over members
        _ = py_call(fitted.append, clones[i])
    _set(model, "n_features_in_", PythonObject(_cols(Xa)))
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


def _ovr_fused(xt: PythonObject, members: PythonObject, Xa: PythonObject, n: Int, k: Int,
               decision: Bool, finish: Int, context: PythonObject) raises -> PythonObject:
    """The k > 2 OneVsRest epilogue through `x_trees_ovr_assemble`: the
    members' positive columns (`_positive` when `decision`, else
    `_proba_positive`) widened into the scores and finished (normalized, or
    the row argmax) as one device program with one readback (lane
    perf-sweep-trees, 2026-11-05; before it a gather and a scatter, each with
    its readbacks, per member). Same kernels and widening: no bit moves."""
    var keep = List[PythonObject]()
    var out = admitted_vector(n, "<i4", context) if finish == 2 else admitted_vector(n * k, "<f8", context)
    var addrs: List[Int] = [trees_address(out)]
    var ints: List[Int] = [n, k, finish]
    var floats: List[Float64] = [Float64(n), Float64(k), Float64(finish)]
    for j in range(len(members)):  # driver-loop: over members
        var e = members[j]
        var bc: Int
        var at: Int
        if decision and prep_estimator_has(e, "decision_function", context):
            var score = _f32_array(_member_scores(e, "decision_function", Xa, context), 1, "score", context)
            var blk = _out2d(score, n, context)
            keep.append(blk.array)
            addrs.append(blk.address)
            bc = blk.cols
            at = 0
            _ = blk^
        else:
            var p = _out2d(_member_scores(e, "predict_proba", Xa, context), n, context)
            at = _index_of_one(e, context)
            keep.append(p.array)
            addrs.append(p.address)
            bc = p.cols
            _ = p^
        ints.append(bc)
        floats.append(Float64(bc))
        ints.append(at)
        floats.append(Float64(at))
    var glue = _xr(xt)
    _ = xt_ovr_assemble(glue, addrs, ints, floats)
    _ = out  # keep the Python object alive through the native call above
    _ = keep  # keep the members' blocks alive through the native call above
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


def _calibrated_validate(model: PythonObject, context: PythonObject) raises:
    var nk = trees_keys()
    var method = _attr(model, nk[].k_method, PythonObject("sigmoid"))
    if not py_contains(Python.tuple("sigmoid", "isotonic"), method):
        raise Error("method must be 'sigmoid' or 'isotonic'")
    _ = native_unit_jobs(_attr(model, nk[].k_n_jobs, PythonObject(None)), "CalibratedClassifierCV")
    var ensemble = _attr(model, nk[].k_ensemble, PythonObject(True))
    if not py_contains(Python.tuple(True, False), ensemble):
        _refuse("ensemble=" + _repr(ensemble), "True or False.")
    _ = _cv_value(_attr(model, nk[].k_cv, PythonObject(None)), context)


def _cal_scores(model: PythonObject, xt: PythonObject, e: PythonObject, Xa: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`CalibratedClassifierCV._scores`: (n, c) float64 scores, c = 1 when
    binary, else one column per class."""
    var n = _rows(Xa)
    var nk = trees_keys()
    var k = len(_attr(model, nk[].k_classes_, Python.list()))
    var out: PythonObject
    if prep_estimator_has(e, "decision_function", context):
        out = _member_scores(e, "decision_function", Xa, context)
    else:
        out = _member_scores(e, "predict_proba", Xa, context)
    var blk = _out2d(out, n, context)
    var glue = _xr(xt)
    if k == 2:
        var acc = _zeros(n, context)
        if blk.cols == 1:
            _ = xt_accumulate(glue, trees_address(acc), blk.address, n, 1.0)
            _ = acc  # keep the Python object alive through the native call above
            _ = blk  # keep the Python object alive through the native call above
        else:
            var at = _index_of_one(e, context)
            if at < 0:
                raise Error("1 is not in list")
            var col = _one_col(xt, blk.view(), at, context)
            _ = xt_accumulate(glue, trees_address(acc), trees_address(col), n, 1.0)
            _ = col  # keep the Python object alive through the native call above
            _ = acc  # keep the Python object alive through the native call above
        _ = blk^
        return trees_reshape(acc, Python.tuple(n, 1), context)
    var acc = _class_zero(xt, n * k, context)
    # Keep the freshly allocated class-index array until accumulate_cols has
    # consumed it. _A(_sub_cols(...)) returned only an address to a temporary.
    var sub_cols = _sub_cols(e, context)
    _ = xt_accumulate_cols(glue, trees_address(acc), blk.address, trees_address(sub_cols), n, k,
                           len(_attr(e, nk[].k_classes_, Python.list())), 1.0)
    _ = sub_cols
    _ = acc  # keep the Python object alive through the native call above
    _ = blk^
    return trees_reshape(acc, Python.tuple(n, k), context)


def _fit_calibrators(model: PythonObject, xt: PythonObject, S: PythonObject, codes: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`CalibratedClassifierCV._fit_calibrators`: one calibrator per class
    column, the column and its 0/1 target made natively."""
    var n = _rows(S)
    var c = _cols(S)
    var nk = trees_keys()
    var sigmoid = String(_attr(model, nk[].k_method, PythonObject("sigmoid"))) == "sigmoid"
    var St = S
    var glue = _xr(xt)
    if c > 1:
        St = admitted_vector(c * n, "<f8", context)
        _ = xt_transpose_f64(glue, trees_address(S), trees_address(St), n, c)
        _ = St  # keep the Python object alive through the native call above
    var yj = admitted_vector(n, "<i4", context)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var y64 = Optional[PythonObject]()
    if not sigmoid:
        y64 = admitted_vector(n, "<f8", context)
    var cals = Python.list()
    for j in range(c):  # driver-loop: over the class columns (one calibrator each)
        var cls = 1 if c == 1 else j
        var f = trees_address(St) + 8 * j * n
        _ = St  # keep the Python object alive through the native call above
        _ = xt_indicator_codes(glue, trees_address(codes), trees_address(yj),
                               trees_address(y64.value()) if Bool(y64) else 0, n, cls)
        _ = yj  # keep the Python object alive through the native call above
        _ = y64  # keep the Python object alive through the native call above
        if sigmoid:
            var ab = admitted_vector(2, "<f8", context)
            _ = xt_platt_fit(glue, f, trees_address(yj), trees_address(ab), n)
            _ = ab  # keep the Python object alive through the native call above
            _ = yj  # keep the Python object alive through the native call above
            _ = St  # keep the Python object alive through the native call above
            _ = py_call(cals.append, Python.tuple(PythonObject("sigmoid"),
                                                  Python.tuple(PythonObject(_f64_word(ab, 0)), PythonObject(_f64_word(ab, 1)))))
        else:
            var kx = admitted_vector(n, "<f8", context)
            var ky = admitted_vector(n, "<f8", context)
            var m = xt_isotonic_fit(glue, f, trees_address(py_or_none(y64)), trees_address(kx), trees_address(ky), n)
            _ = ky  # keep the Python object alive through the native call above
            _ = kx  # keep the Python object alive through the native call above
            _ = y64  # keep the Python object alive through the native call above
            _ = St  # keep the Python object alive through the native call above
            _ = py_call(cals.append, Python.tuple(PythonObject("isotonic"),
                                                  Python.tuple(_copy_block(kx, 0, m, "<f8", context),
                                                               _copy_block(ky, 0, m, "<f8", context), PythonObject(m))))
            _ = ky  # keep the Python object alive through the native call above
            _ = kx  # keep the Python object alive through the native call above
            _ = y64  # keep the Python object alive through the native call above
    return cals


def _calibrated_probs(model: PythonObject, xt: PythonObject, e: PythonObject, cals: PythonObject, Xa: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`CalibratedClassifierCV._calibrated`: the flat (n, k) probabilities of
    one (member, calibrators) pair."""
    var n = _rows(Xa)
    var nk = trees_keys()
    var k = len(_attr(model, nk[].k_classes_, Python.list()))
    var S = _cal_scores(model, xt, e, Xa, context)
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


struct _CalUnits(Movable):
    """CalibratedClassifierCV's folds as units: fold f's member clone, fit,
    test scores and (ensemble) calibrators, written to `results[f]`. Every
    field is read and written with the GIL held."""
    var model: PythonObject
    var xt: PythonObject
    var base: PythonObject
    var Xa: PythonObject
    var splits: PythonObject
    var codes: PythonObject
    var cols: PythonObject
    var n: Int
    var d: Int
    var ensemble: Bool
    var context: PythonObject
    var n_units: Int
    var next: Int
    var stop: Bool
    var results: PythonObject
    var failure: NativeUnitFailure
    var fp_env: UInt64
    # The x_trees gathers, scores and calibrators of a unit run one unit at
    # a time (they share the owner's `xt` program state and can release the
    # GIL inside); only the member fits overlap.
    var gate: List[Int64]
    # The resident-fold route (fb12-rows, 2026-11-17; set by _calibrated_fit
    # when the member takes `_arena_io.DeviceRows` and the folds run on the
    # calling thread): each fold side is a DeviceRows over the one resident X
    # (`mctx`: model selection's context, which owns the DeviceRows class and
    # the resident store whose `take` a host route would materialize with).
    var dev: Bool
    var mctx: PythonObject
    var rows_class: PythonObject
    var rows_take: PythonObject
    # The ensemble's class count (set by _calibrated_fit; 0: the member
    # doors fit every fold): what a fold's native member is fitted with.
    var k: Int

    def __init__(out self, model: PythonObject, xt: PythonObject, base: PythonObject, Xa: PythonObject,
                 splits: PythonObject, codes: PythonObject, cols: PythonObject, n: Int, d: Int, ensemble: Bool,
                 context: PythonObject) raises:
        self.model = model
        self.xt = xt
        self.base = base
        self.Xa = Xa
        self.splits = splits
        self.codes = codes
        self.cols = cols
        self.n = n
        self.d = d
        self.ensemble = ensemble
        self.context = context
        self.n_units = len(splits)
        self.next = 0
        self.stop = False
        self.results = Python.list()
        for _ in range(self.n_units):  # driver-loop: over the cv folds (result slots)
            _ = py_call(self.results.append, PythonObject(None))
        self.failure = NativeUnitFailure()
        self.fp_env = host_fp_env()
        self.gate = native_unit_gate()
        self.dev = False
        self.mctx = PythonObject(None)
        self.rows_class = PythonObject(None)
        self.rows_take = PythonObject(None)
        self.k = 0

    def fold_side(self, idx: PythonObject) raises -> PythonObject:
        """Fold side `idx` (cv row indices) of X for the member: a resident
        DeviceRows (no host words, no upload: the member's program gathers
        the rows on the device out of X, put once for the whole fit by the
        resident scope), else the host copy (B arm) or `_gather`."""
        if self.dev and Int(py=idx.shape[0]) > 0:
            return self._dev_rows(idx)
        comptime if CALIB_FAST_HOST_FOLDS:
            return _fold_rows(self.xt, self.Xa, self.n, self.d, idx, self.cols, self.context)
        return _gather(self.xt, trees_address(self.Xa), self.n, self.d, idx, self.cols, self.context)

    def _dev_rows(self, idx: PythonObject) raises -> PythonObject:
        """`_arena_io.DeviceRows(Xa, idx as int64, take)`, made by its door
        as model selection's `MselFoldRows.xrows` makes it."""
        var r = Int(py=idx.shape[0])
        var idx64 = idx
        var kind = String(py=idx.dtype)
        if kind != "<i8":
            idx64 = admitted_storage(Python.tuple(r), "<i8", self.context, "C", True)
            trees_cast(trees_address(idx), kind, trees_address(idx64), "<i8", r, self.context)
            _ = idx  # keep the Python object alive through the native call above
        var dr = native_storage_allocate(self.rows_class)
        _ = _cal_rows_call(dr, "init", Python.tuple(self.Xa, idx64, self.rows_take), self.mctx)
        return dr

    def _enter(mut self):
        native_unit_gate_enter(self.gate)

    def unit(mut self, f: Int) raises:
        var tr = self.splits[f][0]
        var te = self.splits[f][1]
        var e = _clone(self.base, self.context)
        # The fold's rows are held for its member doors in a scope of this
        # unit (dropped when the unit ends, so no fold's matrices outlive
        # it); a hold belongs to the thread that opened the scope.
        var mark = admitted_scope_begin()
        try:
            self._enter()
            var xtr = PythonObject(None)
            var ytr = PythonObject(None)
            try:
                xtr = self.fold_side(tr)
                ytr = _gather_codes(self.xt, self.codes, tr, self.context)
            finally:
                native_unit_gate_leave(self.gate)
            # A member of a published family is fitted through its member
            # entry on the fold's host rows (one unit: admitted once, the GIL
            # released for the fit, its fitted state left with the clone for
            # the scores below), with the ensemble's class count. A fold that
            # lacks a class is refused by the set before any fit
            # (bindings/member_set_door.mojo, the class-cover pass: one pass
            # over the fold's codes) and takes the member's fit door, which
            # fits the classes present, as does every unpublished family and
            # the resident DeviceRows route (the member program gathers its
            # rows on the device: tuned, untouched). Outside the gate: the
            # fits of different folds overlap, as the door fits do.
            var native = False
            if not self.dev and self.k > 0:
                native = _fit_units_native(_repeat(e, 1), _repeat(xtr, 1), self.d, _repeat(ytr, 1), False,
                                           _repeat(PythonObject(None), 1), _repeat(PythonObject(self.k), 1), 1,
                                           self.context)
            if not native:
                _fit_member(e, xtr, ytr, PythonObject(None), self.context)
            self._enter()
            try:
                var Xte = self.fold_side(te)
                var scores = _cal_scores(self.model, self.xt, e, Xte, self.context)
                if self.ensemble:
                    var cals = _fit_calibrators(self.model, self.xt, scores,
                                                _gather_codes(self.xt, self.codes, te, self.context), self.context)
                    py_setitem(self.results, PythonObject(f), Python.tuple(e, cals))
                else:
                    py_setitem(self.results, PythonObject(f), scores)
            finally:
                native_unit_gate_leave(self.gate)
        finally:
            admitted_scope_end(mark)
        _ = self.Xa  # keep the Python object alive through the native calls above

    def drain(mut self):
        while True:  # driver-loop: over the cv folds, pulled in order under the GIL
            if self.stop or self.next >= self.n_units:
                return
            var u = self.next
            self.next += 1
            try:
                self.unit(u)
            except error:
                self.stop = True
                self.failure.record(u, error)


def _cal_units_thread(arg: Int) abi("C") -> Int:
    try:
        var state = native_unit_gil_enter()
        var run = MutPointer[_CalUnits, MutUntrackedOrigin](unsafe_from_address=arg)
        var saved = host_fp_env()
        host_fp_env_set(run[].fp_env)
        run[].drain()
        host_fp_env_set(saved)
        native_unit_gil_leave(state)
    except:
        pass
    return 0


def _cal_units_run(mut units: _CalUnits, jobs: Int) raises:
    """Fold 0 alone on the calling thread (lazy process state is created
    single-threaded), the rest on min(jobs, folds - 1) threads; one at a time
    when the member fits run on a GPU (`native_unit_device_workers`)."""
    var workers = native_unit_device_workers(min(jobs, units.n_units - 1), units.context)
    if units.n_units > 1 and workers > 1:
        units.next = 1
        try:
            units.unit(0)
        except error:
            units.stop = True
            units.failure.record(0, error)
        if not units.stop:
            native_unit_threads_run(_cal_units_thread, Int(UnsafePointer(to=units)), workers)
    else:
        units.drain()
    units.failure.reraise()


def _cal_rows_call(obj: PythonObject, op: String, args: PythonObject, mctx: PythonObject) raises -> PythonObject:
    """One `_arena_io` DeviceRows / resident-scope operation through its
    compiled door (model selection's `_arena_rows_call`, bindings/msel_cv_door.mojo)."""
    # call-table: member DeviceRows (arena rows door over the Python owner) has no native entry yet
    var nk = trees_keys()
    return family_call(py_item(mctx, nk[].k_family_msel_rows_entry), obj, PythonObject(op), args, py_item(mctx, nk[].k_family_msel_rows_context))


def _cal_resident_op(res: PythonObject, op: String, mctx: PythonObject) raises:
    """One `_Resident` lifecycle operation (msel_common_door `msel_resident_op`)."""
    # call-table: member resident rows (arena_resident_native door over the Python _Resident) has no native entry yet
    var nk = trees_keys()
    _ = family_call(py_item(mctx, nk[].k_family_msel_resident_call), res, PythonObject(op), Python.tuple(),
                    py_item(mctx, nk[].k_family_msel_resident_context))


def _cal_admit_x_once(base: PythonObject, Xa: PythonObject, n: Int, d: Int) raises:
    """X's finite test, once for the fit, on the resident-fold route (the
    fold rows reach the member as DeviceRows, which its door does not scan):
    the held row's verdict (one scan, recorded for the rest of the call),
    or one scan of an unheld X; scikit-learn's refusal in the member's name."""
    var held = admitted_held_index(Xa)
    var finite = admitted_held_verify(held) if held >= 0 else trees_finite_f32(trees_address(Xa), n * d)
    _ = Xa  # keep the Python object alive through the native call above
    if not finite:
        admitted_finite_refuse(1, String(py_type(base).__name__), "X")


def _calibrated_fit(model: PythonObject, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    _calibrated_validate(model, context)
    var xt = _xt(py_fields(model), context)
    var Xa = _x_array(X, context)
    var n = _rows(Xa)
    var d = _cols(Xa)
    var labels = trees_encode_labels(y, labels_scope(model, context))
    _set(model, "classes_", labels.classes)
    var k = len(labels.classes)
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var nk = trees_keys()
    var base = _attr(model, nk[].k_estimator, PythonObject(None))
    if base is None:
        base = prep_estimator_clone(py_item(py_item(context, nk[].k_native_estimator_defaults), nk[].k_LinearSVC), context)
    _require_member(base, _owner(model), context)
    _set(model, "_cal_fast", PythonObject(None))
    var ensemble = _truthy(_attr(model, nk[].k_ensemble, PythonObject(True)))
    var splits = _splits(xt, _attr(model, nk[].k_cv, PythonObject(None)), X, y, n, labels.codes, not ensemble, k, context)
    var cols = _arange(xt, d, context)
    var pairs = Python.list()
    _set(model, "calibrated_classifiers_", pairs)
    # Lane p3-msel-parallel-and-process-pool: the folds are independent
    # units (_CalUnits) on n_jobs threads; each writes its own slot, so the
    # pairs and the out-of-fold scores keep the fold order. REMOVED: the two
    # serial fold loops and the n_jobs refusal. Same member fits, gathers and
    # calibrators per fold: no bits change.
    var jobs = native_unit_jobs(_attr(model, nk[].k_n_jobs, PythonObject(None)), "CalibratedClassifierCV")
    var units = _CalUnits(model, xt, base, Xa, splits, labels.codes, cols, n, d, ensemble, context)
    units.k = k
    # fb12-rows (2026-11-17): the folds' X stays on the GPU. Each fold side
    # was `_gather` (xtrees gather_f32_device): all of X uploaded, the rows
    # gathered and DOWNLOADED, twice per fold, then each member uploaded its
    # rows again and scanned them on the host (5 folds of a 1M x 220 X: 10
    # uploads + 5 downloads of X beside 5 more uploads, and 5 X-sized scans).
    # For a member whose class takes `_arena_io.DeviceRows` (the naive Bayes
    # and discriminant classifiers, GaussianNB the default board member) the
    # fold sides are now DeviceRows over X inside one resident scope, as
    # model selection hands its folds (MselFoldRows): X goes up ONCE for the
    # whole fit, each member program gathers its rows on the device
    # (`x_prep_dev_take_rows`), nothing comes down, and X is tested finite
    # once here. Same rows in the same order: no bit changes. Only on the
    # calling thread (a resident scope belongs to the thread that opened it):
    # parallel CPU units keep the per-fold rows.
    var plan = prep_estimator_plan(base, context)
    var dev = (plan is not None and _truthy(py_get(plan, nk[].k_device_rows, PythonObject(False))) and n > 0 and d > 0
               and String(py=Xa.dtype) == "<f4" and String(py=Xa.order) == "C"
               and native_unit_device_workers(min(jobs, len(splits) - 1), context) <= 1)
    var mctx = PythonObject(None)
    var res = PythonObject(None)
    var scope = PythonObject(None)
    if dev:
        _cal_admit_x_once(base, Xa, n, d)
        mctx = prep_native_context(context)
        var b = trees_builtins()
        res = native_storage_allocate(py_item(mctx, nk[].k_family_msel_resident_class))
        _cal_resident_op(res, "msel_resident_init", mctx)
        try:
            # only DeviceRows bases enter this scope (`rows_slot` ignores the
            # size floor), as in MselFoldRows
            var opened = native_storage_allocate(family_call(b.getattr, py_item(mctx, nk[].k_family_msel_arena_io),
                                                             PythonObject("resident")))
            # the floor goes to the native scope state with the enter call
            _ = _cal_rows_call(opened, "scope_enter", Python.tuple(PythonObject(1 << 62)), mctx)
            scope = opened
            units.rows_class = family_call(b.getattr, py_item(mctx, nk[].k_family_msel_arena_io), PythonObject("DeviceRows"))
            units.rows_take = family_call(b.getattr, res, PythonObject("take"))
        except error:
            # lane fb16-audit: a scope already entered is exited before the
            # store closes (a getattr after `scope_enter` raising left the
            # thread inside the resident scope)
            try:
                if scope is not None:
                    _ = _cal_rows_call(scope, "scope_exit",
                                       Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None)), mctx)
            finally:
                _cal_resident_op(res, "msel_resident_close", mctx)
            raise error^
        units.mctx = mctx
        units.dev = True
    try:
        _cal_units_run(units, jobs)
    finally:
        if dev:
            units.rows_take = PythonObject(None)
            try:
                _ = _cal_rows_call(scope, "scope_exit",
                                   Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None)), mctx)
            finally:
                _cal_resident_op(res, "msel_resident_close", mctx)
    if ensemble:
        for f in range(len(splits)):  # driver-loop: over the cv folds (results only)
            _ = py_call(pairs.append, units.results[f])
    else:
        var c = 1 if k == 2 else k
        var S = _zeros(n * c, context)
        for f in range(len(splits)):  # driver-loop: over the cv folds (placing each fold's scores)
            var blk = admit_f32(units.results[f], 2, "scores", context, finite=False)
            _ = _place(xt, S, n, c, blk.view(), splits[f][1], 0)
            _ = blk^
        var e = _clone(base, context)
        # The one member over every row sees every class (the codes were
        # encoded from this y): a native member fit when its family has
        # published, else its fit door. (A fold of `_CalUnits` may lack a
        # class: that fold's member takes its fit door, `_CalUnits.unit`.)
        _fit_members(_repeat(e, 1), Xa, n, d, _repeat(labels.codes, 1), False, PythonObject(None),
                     _repeat(PythonObject(k), 1), jobs, context)
        _ = py_call(pairs.append, Python.tuple(e, _fit_calibrators(model, xt, trees_reshape(S, Python.tuple(n, c), context),
                                                                   labels.codes, context)))
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _calibrated_fused(model: PythonObject, xt: PythonObject, pairs: PythonObject,
                      Xa: PythonObject, n: Int, k: Int, context: PythonObject) raises -> PythonObject:
    """`_calibrated_proba` through `x_trees_calibrated_proba`: the members'
    scores, then the whole epilogue (scores, calibrators, complement or
    normalization, float32 narrowing, the mean) as one device program with
    one readback (lane perf-sweep-trees, 2026-11-05; before it about twenty
    glue round trips per pair). Same kernels in the same order: no bit moves
    (xtrees/calibrated_apply.mojo)."""
    var keep = List[PythonObject]()
    var acc = admitted_vector(n * k, "<f8", context)
    var program = CalibratedProgram(trees_address(acc), n, k, len(pairs))
    var nk = trees_keys()
    for i in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
        var e = pairs[i][0]
        var cals = pairs[i][1]
        var out: PythonObject
        if prep_estimator_has(e, "decision_function", context):
            out = _member_scores(e, "decision_function", Xa, context)
        else:
            out = _member_scores(e, "predict_proba", Xa, context)
        var blk = _out2d(out, n, context)
        keep.append(blk.array)
        var at = 0
        var cols_addr = 0
        var ks = 0
        if k == 2:
            if blk.cols != 1:
                at = _index_of_one(e, context)
                meta_binary_column_check(at)
        else:
            var sub_cols = _sub_cols(e, context)
            keep.append(sub_cols)
            cols_addr = trees_address(sub_cols)
            ks = len(_attr(e, nk[].k_classes_, Python.list()))
        program.pair(blk.address, cols_addr, blk.cols, at, ks, len(cals))
        for j in range(len(cals)):  # driver-loop: over the class columns (one calibrator each)
            var entry = cals[j]
            var par = entry[1]
            if String(entry[0]) == "sigmoid":
                program.sigmoid(Float64(py=par[0]), Float64(py=par[1]))
            else:
                var knots_x = par[0]
                var knots_y = par[1]
                keep.append(knots_x)
                keep.append(knots_y)
                program.isotonic(trees_address(knots_x), trees_address(knots_y), Int(py=par[2]))
        _ = blk^
    var glue = _xr(xt)
    _ = xt_calibrated_proba(glue, program.addrs, program.ints, program.floats)
    _ = acc  # keep the Python object alive through the native call above
    _ = keep  # keep the members' blocks, class codes and knots alive through the native call above
    return trees_reshape(acc, Python.tuple(n, k), context)


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
    in its own order and words."""
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
