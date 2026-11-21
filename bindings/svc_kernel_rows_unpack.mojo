# SPDX-License-Identifier: Apache-2.0
# UNPACK pieces shared by the kernel SVM doors (SVC and SVR;
# bindings/svc_kernel_native_door.mojo sequences them): the admission of a
# fit X and of a query X with their owners (a host buffer, or a
# model-selection fold that stays on the device), the refusals of the status
# words a device entry returns, the entry lookups, and the row glue of the
# one-vs-one and Platt drivers (gathers and pair selection run by the
# binding's `svc_pair_epilogue`). Moved here unchanged from the door.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family_meta, admit_family_fold_meta
from bindings.py_balanced import py_call, py_item
from bindings.boundary_read_door import buffer_view
from providers.python.borrowed_buffer_pyapi_unpack import admit_native_buffer_exporter
from bindings.classical_arrays_door import classical_address, classical_is_array, classical_wrap_array
from bindings.family_common_pack import family_call, family_entry, family_array
from bindings.svc_common_unpack import svc_builtins, svc_field, svc_has, svc_int
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.device_rows_door_run import DoorRowsStore
from bindings.admitted_input_unpack import admit_x_unrecorded, admitted_finite_refuse, admit_rows_host_unchecked
from core.host_finite_scan import host_all_finite_f32_strided
from providers.python.svc_descriptor_pyapi_door import svc_epilogue_execute
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from bindings.py_optional import py_or_none
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from bindings.svm_keys_unpack import svm_keys

# `svc_pair_epilogue` modes (svm/impl/svc_rows.mojo), `_svm_impl._EPI_*` and `_GLUE_*`.
comptime _EPI_OVO = 0
comptime _EPI_OVR = 1
comptime _EPI_VOTES = 2
comptime _EPI_PROBA = 3
comptime _EPI_LOG_PROBA = 4
comptime _EPI_BINARY = 5
comptime _GLUE_GATHER = 6
comptime _GLUE_SELECT = 7
comptime _GLUE_C_ROWS = 8
comptime _GLUE_FOLD = 9
comptime _GLUE_FOLD_FINISH = 10


def _addr(value: PythonObject) raises -> Int:
    return Int(py=classical_address(value))


# The prepared-owner slots of the kernel SVMs' device-resident fitted state
# (lane ne1-a-gp-svm).
comptime _SVC_RESIDENT_SLOT = "svc_resident_fitted_v1"
comptime _SVR_RESIDENT_SLOT = "svr_resident_fitted_v1"


def _entry(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    return family_entry(binding, name, context)


# ---------------------------------------------------------------- the row glue


def _gather(binding: PythonObject, x_addr: Int, n_rows: Int, n_cols: Int, rows_addr: Int, nor: Int,
            cols_addr: Int, noc: Int, context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_svm_impl._gather`: `x[rows][:, cols]` (an absent index list, address
    0, keeps that axis whole), an exact copy, gathered in the binding."""
    var out = family_array(Python.tuple(PythonObject(nor), PythonObject(noc)), "<f4", context)
    if nor > 0 and noc > 0:
        _ = svc_epilogue_execute(binding, native, PythonObject(x_addr), PythonObject(rows_addr),
                        PythonObject(cols_addr), classical_address(out),
                        Python.list(PythonObject(_GLUE_GATHER), PythonObject(n_rows), PythonObject(n_cols),
                                    PythonObject(nor), PythonObject(noc)))
        _ = out  # keep the Python object alive through the native call above
    return out


def _gather_i32(binding: PythonObject, values_addr: Int, m: Int, rows_addr: Int, count: Int,
                context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_svm_impl._gather_i32`: `values[rows]`, the binding's 4-byte gather."""
    var out = family_array(Python.tuple(PythonObject(count)), "<i4", context)
    if count > 0:
        _ = svc_epilogue_execute(binding, native, PythonObject(values_addr), PythonObject(rows_addr),
                        PythonObject(0), classical_address(out),
                        Python.list(PythonObject(_GLUE_GATHER), PythonObject(m), PythonObject(1), PythonObject(count),
                                    PythonObject(1)))
        _ = out  # keep the Python object alive through the native call above
    return out


def _i32_array(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_svm_impl._i32`: an int32 Array as is, a sequence of ints packed."""
    if classical_is_array(values, context) and String(values.dtype) == "<i4":
        return values
    var b = svc_builtins()
    var store = PythonObject(None)
    var nk = svm_keys()
    try:
        store = py_call(py_item(context, nk[].k_array_type), PythonObject("i"), values)
    except:
        store = family_call(py_item(context, nk[].k_array_type), PythonObject("i"), family_call(b.map, b.int, values))
    return classical_wrap_array(store, Python.tuple(PythonObject(len(store))), "<i4", "C", context)


def _native_exporter(value: PythonObject) -> Bool:
    """True for an exact native buffer exporter (ndarray, array.array,
    bytes, ...): identity reads only, never a buffer hook."""
    try:
        admit_native_buffer_exporter(value)
        return True
    except:
        return False


def _gather_support_columns(binding: PythonObject, x_addr: Int, n_rows: Int, n_cols: Int, support: PythonObject,
                            count: Int, context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_gather(x, cols=support_)` for kernel='precomputed': the cross-kernel's
    columns at the support vectors. `count` < 0 takes the index count from
    `support` itself. A C-order int32 exporter (a fitted ndarray or
    array('i')) is read in place through ONE buffer view, released after the
    gather: no per-element read and no packed copy. An int32 Array is used as
    is; any other sequence is packed once (`_i32_array`), as before."""
    if not classical_is_array(support, context) and _native_exporter(support):
        var view = buffer_view(support)
        if view.ndim == 1 and view.matches("<i4"):
            var n = count if count >= 0 else view.shape[0]
            var out = _gather(binding, x_addr, n_rows, n_cols, 0, n_rows, view.address(), n, context, native)
            view.release()
            return out
        view.release()
    var held = _i32_array(support, context)
    var n = count if count >= 0 else Int(py=held.shape[0])
    var out = _gather(binding, x_addr, n_rows, n_cols, 0, n_rows, _addr(held), n, context, native)
    _ = held  # keep the Python object alive through the native call above
    return out


def _pair_select(binding: PythonObject, codes: PythonObject, ci: Int, cj: Int, c_rows: PythonObject,
                 context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_svm_impl._pair_select`: (row indices, 0/1 labels, bounds or None, m),
    the buffers uncut: their first m entries are the pair's."""
    var n = Int(py=codes.shape[0])
    var idx = family_array(Python.tuple(PythonObject(n)), "<i4", context)
    var lab = family_array(Python.tuple(PythonObject(n)), "<f4", context)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var cout = Optional[PythonObject]()
    if c_rows is not None:
        cout = family_array(Python.tuple(PythonObject(n)), "<f4", context)
    var m = 0
    if n > 0:
        m = svc_int(svc_epilogue_execute(binding, native, classical_address(codes),
                                PythonObject(0) if c_rows is None else classical_address(c_rows),
                                classical_address(lab), classical_address(idx),
                                Python.list(PythonObject(_GLUE_SELECT), PythonObject(n), PythonObject(ci), PythonObject(cj),
                                            PythonObject(0) if not cout else classical_address(cout.value()))))
        _ = lab  # keep the Python object alive through the native call above
        _ = idx  # keep the Python object alive through the native call above
        _ = cout  # keep the Python object alive through the native call above
    return Python.tuple(idx, lab, py_or_none(cout), PythonObject(m))



def _svm_fit_refuse(finite_word: Int, range_word: Int, fold: DoorRowsStore, est: String) raises:
    """The refusal of a fit X the device entry flagged (lane a8a), in
    admit_x's wording: a fold through `door_rows_refuse` (an index out of
    range or a NaN / infinity), host words through `admitted_finite_refuse`."""
    if fold.active():
        door_rows_refuse(finite_word, range_word, est, fold.base_rows)
    else:
        admitted_finite_refuse(finite_word, est)


def _svm_fit_entry(binding: PythonObject, native: PythonObject) raises -> Bool:
    """The GPU binding's fit entries take the device-X form (lane a8a): the
    binding that registers `svc_predict_resident` (a host build and the
    descriptor route have none; they take host words tested at the door)."""
    return _resident_entry(binding, native) is not None


def _svm_fit_admit(source: PythonObject, est: String, entry: Bool, context: PythonObject,
                   mut fold: DoorRowsStore, mut pin: ArrayPin) raises -> ArrayMetadata:
    """The fit X of a kernel SVM (lane a8a). With the device-X entry
    (`entry`) a model-selection fold stays on the device (admit_x
    `device_x`: `fold` receives its store ids, the record has no host
    words) and a host X is admitted untested (the entry tests the words it
    copies). Otherwise the float32 C-order host words, untested here (the
    door tests them with `_svm_x_scan` once the route is known)."""
    fold = DoorRowsStore.none()
    if entry and door_is_device_rows(source, context):
        var fx = admit_x_unrecorded(source, est, context, device_x=True, finite=False)
        if fx.store.active():
            fold = fx.store
            return admit_family_fold_meta(fx.rows, fx.cols)
        return admit_family_meta(fx.array, 2, "X", context, pin, True)
    return admit_family_meta(source, 2, "X", context, pin, True)


def _svm_fit_host_rows(source: PythonObject, context: PythonObject, mut pin: ArrayPin) raises -> ArrayMetadata:
    """A fold's host rows for a fit route that reads host words (one-vs-one,
    Platt, kernel='precomputed'), gathered in parallel out of model
    selection's host base; untested (the door tests them)."""
    return admit_family_meta(admit_rows_host_unchecked(source, context), 2, "X", context, pin, True)


def _svm_x_scan(x_meta: ArrayMetadata, est: String) raises:
    """The finite test of a fit X whose entry returns no status word (the
    host column, the descriptor route, the one-vs-one, Platt and
    precomputed routes): one parallel host pass over the admitted float32
    words, admit_x's wording (lane a8a)."""
    var src = x_meta.source
    if src.rows > 0 and src.cols > 0 and not host_all_finite_f32_strided(src.addr, src.s0, src.s1, src.rows, src.cols):
        admitted_finite_refuse(1, est)



struct _SvmQuery(Movable):
    """The admitted query X of a kernel SVM's decision (lane a7a): host
    float32 C-order words (`keep` owns them, `address` the first word), or a
    cross-validation fold the resident entry gathers on the device
    (`store.active()`, `address` 0). No host pass tests X: the entry tests
    the words it uploads or gathers and returns the status words."""
    var keep: PythonObject
    var owner: PythonObject
    var address: Int
    var rows: Int
    var cols: Int
    var store: DoorRowsStore

    def __init__(out self, keep: PythonObject, owner: PythonObject, address: Int, rows: Int, cols: Int,
                 store: DoorRowsStore):
        self.keep = keep
        self.owner = owner
        self.address = address
        self.rows = rows
        self.cols = cols
        self.store = store


def _svm_query(source: PythonObject, est: String, context: PythonObject, entry_tests: Bool) raises -> _SvmQuery:
    """The query X. `entry_tests` (the resident entry runs a computed
    kernel on support rows, and returns the finite word of the words it
    uploads or gathers): a model-selection fold stays on the device
    (admit_x `device_x`: no host gather, no download), and a host X takes
    no host finite pass (a borrowed float32 C-order X is read in place).
    Otherwise (the host column, the host-list route of an empty machine or
    kernel='precomputed', the descriptor route) the admission tests X, one
    parallel pass or the cast's own verdict, so every route refuses in the
    one wording 'Input X contains NaN or infinity (<est>).'"""
    if entry_tests and door_is_device_rows(source, context):
        var x = admit_x_unrecorded(source, est, context, device_x=True, finite=False)
        return _SvmQuery(x.array, x.source, x.address, x.rows, x.cols, x.store)
    var q_pin = ArrayPin()
    var q = admit_family_meta(source, 2, "X", context, q_pin, True, finite=not entry_tests,
                              refusal="Input X contains NaN or infinity (" + est + ").")
    # _SvmQuery is the door's record: it takes over the pin's two owners.
    return _SvmQuery(q_pin.keep, q_pin.owner, q.source.addr, q.shape[0], q.shape[1], DoorRowsStore.none())


def _svm_query_refuse(finite: Int, range_word: Int, q: _SvmQuery, est: String) raises:
    if q.store.active():
        door_rows_refuse(finite, range_word, est, q.store.base_rows)
    else:
        admitted_finite_refuse(finite, est)


def _svm_query_refuse(words: PythonObject, q: _SvmQuery, est: String) raises:
    """The refusal of a query the resident entry flagged (its [finite,
    range] words; None: another route, which tests the query itself), in
    scikit-learn's wording 'Input X contains NaN or infinity (<est>).'"""
    if words is None:
        return
    if q.store.active():
        door_rows_refuse(Int(py=words[0]), Int(py=words[1]), est, q.store.base_rows)
    else:
        admitted_finite_refuse(Int(py=words[0]), est)


def _svm_device_ok(resident_entry: PythonObject, kernel: Int, n_supports: List[Int]) raises -> Bool:
    """The resident entry runs (and tests the query of) every machine: a
    computed kernel (precomputed reads the cross-kernel's columns on the
    host) and support rows in every machine (an empty machine takes the
    host-list route). Then a fold may stay on the device."""
    if resident_entry is None or kernel == 4 or len(n_supports) == 0:
        return False
    for n in n_supports:  # small-loop(one word per fitted machine)
        if n <= 0:
            return False
    return True


def _query(fields: PythonObject, source: PythonObject, context: PythonObject, entry_tests: Bool) raises -> _SvmQuery:
    """`SVC._query`."""
    var nk = svm_keys()
    if not svc_has(fields, nk[].k_dual_coef_):
        raise Error("mojolearn SVC: call fit() first")
    var q = _svm_query(source, "SVC", context, entry_tests)
    var d = svc_int(svc_field(fields, nk[].k_n_features_in_, "n_features_in_"))
    if q.cols != d:
        raise Error("mojolearn SVC: X has " + String(q.cols) + " features, fit saw " + String(d))
    return q^


def _resident_entry(binding: PythonObject, native: PythonObject) raises -> PythonObject:
    """The binding's `svc_predict_resident` (lane ne1-a-gp-svm), or None:
    the descriptor route runs, or the binding (a host build) has none."""
    if native is not None:
        return PythonObject(None)
    return neural_descriptor_entry(binding, "svc_predict_resident", False)
