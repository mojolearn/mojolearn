# SPDX-License-Identifier: Apache-2.0
# The classical estimators' shared native label, weight and score steps:
# `_labels.encode_labels` / `decode_labels` / `argmax_rows`,
# `_sample_weight.check_sample_weight`, `_class_weights._class_weight_rows`,
# `_buffer.shape_of` / `_as_typed`, and `_expansion_metrics.accuracy_fraction`
# (scikit-learn's ClassifierMixin.score), with their refusals and wording and
# the same native helpers in the same order (the base binding's casts,
# `prepare_targets`, gathers and argmax; the x_trees weight kernels; the
# x_metrics COUNT program run by the shared native program runner), so every
# byte a kernel reads is unchanged.
#
# Derived from the tree owner's input and score steps (lane migrate-trees,
# bindings/trees_public_inputs.mojo and trees_public_score.mojo) for the
# linear, discriminant and naive Bayes classifiers; fold the two copies into
# one module once both lanes are on main.
#
# No Python function runs here: inputs are read through the buffer protocol
# and the Array slots, outputs are mojolearn Arrays assembled from their
# slots (classical_wrap_array), and the only callables invoked are compiled
# entry points and C builtins connected at import
# (python/mojolearn/_native_labels_bridge.py `label_context`).
from bindings.native_array_storage import native_array_from_bytes, native_array_from_values, native_array_zeros, native_array_empty, native_scratch_bytes, native_scratch_address
from bindings.native_loaded_module import native_loaded_module
from std.math import isfinite
from std.memory import bitcast, memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr, py_fields
from bindings.py_attr import py_attr
from std.python._cpython import PyObjectPtr
from bindings.py_typed_call import py_call_typed, typed_python_error
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.classical_arrays import classical_wrap_array, _store_code, _buffer_typestr
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, _pd_fields, _pd_get, classical_admit_value
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_inputs import sequential_sequence
from bindings.arena_prog_native import _metrics_run
from bindings.classical_attributes import classical_connect_numpy
from bindings.label_admission_native import labels_primitive_kind, labels_admit_primitive, classifier_label_list, LabelTypeKinds, labels_refusal
from bindings.label_snapshot_native import label_array_list
from bindings.label_integer_storage import _ndarray_buffer_address
from bindings.label_input_sources import (
    labels_list_scalar_kind, labels_series_values, labels_text_storage,
    LABELS_LIST_OTHER, LABELS_LIST_BOOL, LABELS_LIST_FLOAT)
from bindings.str_labels_native import StrTable, _read, _sorted
from bindings.cpython_table import cpython_table
from core.shared_slot import SharedSlot
from core.label_weights_native_api import LabelWeightsRequest, LABEL_WEIGHTS_COUNTS, LABEL_WEIGHTS_ROWS
from bindings.label_weights_boundary import label_weights_run
from bindings.kit_call import kit_connect, kit_provider, kit_capabilities, kit_call
from bindings.labels_native_abi import (
    LABELS_NATIVE_ABI_NAME, LABELS_NATIVE_FACTORY, LABELS_NATIVE_OPERATION_COUNT, LABELS_CAP_DEVICE,
    LABELS_OP_MODE, LABELS_OP_CAPABILITIES, LABELS_OP_PREPARE_TARGETS, LABELS_OP_REDUCE_STAT,
    LABELS_OP_ARGMAX_F32, LABELS_OP_ARGMAX_F64, LABELS_OP_GATHER_I32, LABELS_OP_GATHER_I64, LABELS_OP_GATHER_F64,
    LABELS_OP_CLASS_RATIO_F64, labels_reduce_stat_decode,
)
from bindings.labels_host_ops import prepare_targets_binding as _host_prepare_targets
from bindings.hotpath_helpers import reduce_stat_binding as _host_reduce_stat, gather_i32_binding as _host_gather_i32
from bindings.host_helpers import (
    argmax_rows_f32_binding as _host_argmax_rows_f32, argmax_rows_f64_binding as _host_argmax_rows_f64,
    gather_i64_binding as _host_gather_i64, gather_f64_binding as _host_gather_f64,
)
from bindings.array_helpers import (
    class_ratio_f64_host_binding as _host_class_ratio_f64, fsum_f64_binding as _host_fsum_f64,
)
from bindings.perf3_label_doors_text import labels_all_exact_str
from bindings.label_input_sources import LABELS_LIST_INT
from core.admitted_cast_elements import admitted_cast_elements
from bindings.admitted_input import (
    AdmittedX, admit_f32, admit_weight_words, admitted_words, admitted_array, admitted_vector, admit_c_view,
    admitted_dtype,
)
from bindings.classical_arrays import classical_address, classical_is_array, classical_dtype_str
from bindings.p3_labels_metrics_native import (
    p3_call_object, p3_decode_list, p3_text_codes, p3_text_classes, p3_text_lookup, p3_dict_codes,
    p3_integral_list, p3_item, p3_type_word, P3_TEXT_OK, P3_TEXT_NOT_STR)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]



# ------------------------------------------------------------ connection


def labels_builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def labels_type_name(t: PythonObject) raises -> String:
    """A type's name through type's own C descriptor: no metaclass
    attribute hook runs (error messages name refused label types)."""
    var b = labels_builtins()
    return String(py_call(b.type.__dict__["__name__"].__get__, t))


def _labels_one_of(t: PythonObject, types: PythonObject) raises -> Bool:
    for i in range(len(types)):  # driver-loop: over a fixed tuple of connected exact scalar types
        if t is types[i]:
            return True
    return False


def _labels_connect(context: PythonObject) raises:
    """Connect NumPy's exact scalar types once NumPy is loaded, so label
    admission recognizes them by identity (never numbers ABC callbacks)."""
    if (py_get(context, "ndarray") is None and py_contains(context, "modules")
            and py_contains(context, "native_module_type")):
        classical_connect_numpy(context)


def labels_bool_type(t: PythonObject, context: PythonObject) raises -> Bool:
    """Exactly `bool` or NumPy's `bool_` (labels_primitive_kind 2)."""
    _labels_connect(context)
    return labels_primitive_kind(t, context) == 2


def labels_real_type(t: PythonObject, context: PythonObject) raises -> Bool:
    """An exact native bool, integer or real scalar class: the former
    numbers.Real test (plus NumPy bool_) by identity."""
    _labels_connect(context)
    return labels_primitive_kind(t, context) >= 2


def labels_integral_type(t: PythonObject, context: PythonObject) raises -> Bool:
    """An exact native integer class (bool excluded): the former
    numbers.Integral test by identity."""
    _labels_connect(context)
    return labels_primitive_kind(t, context) == 3


def labels_text_type(t: PythonObject, context: PythonObject) raises -> Bool:
    """Exactly str or NumPy's str_."""
    _labels_connect(context)
    return labels_primitive_kind(t, context) == 1


def labels_table(context: PythonObject, mode: String, name: String) raises -> PythonObject:
    """The connected entry table of binding `name` in `mode`, or the
    refusal recorded at import ("<Class>: <message>")."""
    var tables = py_get(context["labels_bindings"], PythonObject(mode))
    if tables is None:
        raise Error("numeric_mode must be fast, identical or None")
    var table = py_get(tables, PythonObject(name))
    if table is None:
        raise Error("ImportError: mojolearn: " + name + " is not connected to the tree owner")
    if py_type(table) is py_type(PythonObject("")):
        raise Error(String(table))
    return table


def labels_entry(table: PythonObject, name: String) raises -> PythonObject:
    """A connected compiled entry point, or None when the binary has none."""
    return py_get(table, PythonObject(name))


def labels_need(table: PythonObject, name: String, message: String, kind: String = "RuntimeError") raises -> PythonObject:
    """A connected entry point the operation requires, refused with the
    Python wording when the binary predates it."""
    var entry = py_get(table, PythonObject(name))
    if entry is None:
        raise Error(kind + ": " + message)
    return entry


# TOMBSTONE: labels_helper / labels_call (the base binding's `_buffer._native` callables in the `labels_helpers` context table, one PyObject_CallObject per step) deleted 2026-11-09 by lane s1-shared-state-hardening-and-selector; code recoverable at 91863cfb35^ (244c15e707).
# Restore: experiments/removed/S1_LABELS_PY_CALL_SELECTOR.patch.
#
# THE NATIVE DEVICE-OR-HOST SELECTOR (lane s1, S3; bindings/labels_native_abi.mojo).
# The label steps below run on the GPU base binding's device helpers through
# its `labels_native_api_v1` provider, connected once per process and kept in
# the named slot `MojolearnLabelsProviderV2`, or, on an install with no GPU
# base binding (the label context's `labels_base_binding` is False), on the
# same host bodies in-process. A context without the label context at all
# (no `labels_base_binding` key) takes the label input step only
# (classical_encode_labels).

#: Layout version of `_LabelsProvider` (core/shared_slot.mojo rule).
comptime _LABELS_PROVIDER_LAYOUT = 2
comptime _LABELS_PROVIDER_NAME = "MojolearnLabelsProviderV2"


struct _LabelsProvider(Defaultable, Movable):
    # 0 not connected yet, 1 the device provider, 2 the host column
    var state: Int
    var provider: PythonObject

    def __init__(out self):
        self.state = 0
        self.provider = PythonObject(None)


comptime _LABELS_PROVIDER = SharedSlot[_LabelsProvider, _LABELS_PROVIDER_NAME, _LABELS_PROVIDER_LAYOUT]

comptime _LABELS_DEVICE = 1
comptime _LABELS_HOST = 2


def _labels_route(context: PythonObject) raises -> Int:
    """`_LABELS_DEVICE` or `_LABELS_HOST`: connect the GPU base binding's
    provider on first use (under the GIL, so one connection per process)."""
    var slot = _LABELS_PROVIDER.get_or_create_ptr()
    if slot[].state != 0:
        return slot[].state
    var base = py_get(context, "labels_base_binding")
    if base is None:
        raise Error("ImportError: mojolearn: the label steps need the label context (labels_base_binding)")
    if py_type(base) is py_type(PythonObject(False)):
        slot[].state = _LABELS_HOST
        return _LABELS_HOST
    if py_type(base) is py_type(PythonObject("")):
        # the refusal the import glue recorded ("<Class>: <message>")
        raise Error(String(base))
    var snapshot = kit_connect(base, LABELS_NATIVE_FACTORY, LABELS_NATIVE_ABI_NAME, LABELS_NATIVE_OPERATION_COUNT,
                               LABELS_OP_MODE, LABELS_OP_CAPABILITIES)
    if kit_capabilities(snapshot) != LABELS_CAP_DEVICE:
        raise Error("ImportError: mojolearn: the base binding's labels provider is not the device provider")
    slot[].provider = kit_provider(snapshot)
    slot[].state = _LABELS_DEVICE
    return _LABELS_DEVICE


def _labels_device_call(op: Int, addresses: List[Int], ints: List[Int]) raises -> Int:
    var slot = _LABELS_PROVIDER.get_or_create_ptr()
    # GIL kept: the provider's entries release it around their own kernels
    return kit_call(slot[].provider, op, addresses, ints, List[Float64](), False)


def labels_prepare_targets(context: PythonObject, y_addr: Int, y_code: Int, n: Int, classes_addr: Int,
                           codes_addr: Int, w_addr: Int, w_code: Int, w_flags: Int) raises -> Int:
    """The base binding's `prepare_targets` (bindings/labels_host_ops.mojo
    contract): k, or a negative weight status."""
    if _labels_route(context) == _LABELS_DEVICE:
        return _labels_device_call(LABELS_OP_PREPARE_TARGETS, [y_addr, classes_addr, codes_addr, w_addr],
                                   [y_code, n, w_code, w_flags])
    return Int(py=_host_prepare_targets(PythonObject(y_addr), PythonObject(y_code), PythonObject(n),
                                        PythonObject(classes_addr), PythonObject(codes_addr), PythonObject(w_addr),
                                        PythonObject(w_code), PythonObject(w_flags)))


def labels_reduce_stat(context: PythonObject, addr: Int, code: Int, n: Int, what: Int) raises -> PythonObject:
    """The base binding's `reduce_stat` (min 0, max 1, integral test 4, ...):
    the same Python answer either way."""
    if _labels_route(context) == _LABELS_DEVICE:
        var out = List[Int64](length=3, fill=Int64(0))
        var tag = _labels_device_call(LABELS_OP_REDUCE_STAT, [addr, Int(out.unsafe_ptr())], [code, n, what])
        var value = labels_reduce_stat_decode(tag, Int(out.unsafe_ptr()))
        _ = out^
        return value
    return _host_reduce_stat(PythonObject(addr), PythonObject(code), PythonObject(n), PythonObject(what))


def labels_argmax_rows(context: PythonObject, wide: Bool, scores_addr: Int, rows: Int, cols: Int,
                       dst_addr: Int) raises:
    """First-max-wins row argmax into int64 codes (float64 scores when
    `wide`)."""
    if _labels_route(context) == _LABELS_DEVICE:
        _ = _labels_device_call(LABELS_OP_ARGMAX_F64 if wide else LABELS_OP_ARGMAX_F32, [scores_addr, dst_addr],
                                [rows, cols])
        return
    if wide:
        _ = _host_argmax_rows_f64(PythonObject(scores_addr), PythonObject(rows), PythonObject(cols),
                                  PythonObject(dst_addr))
    else:
        _ = _host_argmax_rows_f32(PythonObject(scores_addr), PythonObject(rows), PythonObject(cols),
                                  PythonObject(dst_addr))


comptime LABELS_GATHER_I32 = 0
comptime LABELS_GATHER_I64 = 1
comptime LABELS_GATHER_F64 = 2


def labels_gather(context: PythonObject, kind: Int, table_addr: Int, n_table: Int, codes_addr: Int, n: Int,
                  dst_addr: Int) raises:
    """`dst[i] = table[codes[i]]` (kind LABELS_GATHER_*); a code out of range
    raises before any write."""
    if _labels_route(context) == _LABELS_DEVICE:
        var op = LABELS_OP_GATHER_I32 if kind == LABELS_GATHER_I32 else (
            LABELS_OP_GATHER_I64 if kind == LABELS_GATHER_I64 else LABELS_OP_GATHER_F64)
        _ = _labels_device_call(op, [table_addr, codes_addr, dst_addr], [n_table, n])
        return
    var t = PythonObject(table_addr)
    var k = PythonObject(n_table)
    var c = PythonObject(codes_addr)
    var m = PythonObject(n)
    var d = PythonObject(dst_addr)
    if kind == LABELS_GATHER_I32:
        _ = _host_gather_i32(t, k, c, m, d)
    elif kind == LABELS_GATHER_I64:
        _ = _host_gather_i64(t, k, c, m, d)
    else:
        _ = _host_gather_f64(t, k, c, m, d)


def labels_class_ratio_f64(context: PythonObject, counts_addr: Int, k: Int, n: Int, mode: Int,
                           out_addr: Int) raises -> Int:
    """Per-class ratios (mode 0 balanced, 1 prior) of k int64 counts;
    nonzero when a mode-0 count is zero."""
    if _labels_route(context) == _LABELS_DEVICE:
        return _labels_device_call(LABELS_OP_CLASS_RATIO_F64, [counts_addr, out_addr], [k, n, mode])
    return Int(py=_host_class_ratio_f64(PythonObject(counts_addr), PythonObject(k), PythonObject(n),
                                        PythonObject(mode), PythonObject(out_addr)))


def labels_fsum_f64(src_addr: Int, n: Int, out_addr: Int) raises -> Int:
    """`fsum` of n float64 terms into out[0] (bindings/array_helpers.mojo
    contract): a host helper in every base binding, so it runs in-process."""
    return Int(py=_host_fsum_f64(PythonObject(src_addr), PythonObject(n), PythonObject(out_addr)))


def labels_env(name: String, default: String) -> String:
    var value = getenv(name)
    return value if value else default


# --------------------------------------------------------------- outputs


# The array admission copy (lane consol-a0-admission merged the trees copy
# into this one) is gone: X, targets and weights are admitted by
# bindings/admitted_input.mojo (admit_x / admit_f32 / admit_weight_words /
# admit_c_view / admit_ids, admitted_array / admitted_vector), the one path.
# TOMBSTONE: labels_empty / labels_vector / LabelsView / labels_view / labels_check_rank / labels_shape_repr / _view_of_array / _view_of_buffer / _any_contiguous_address / LabelsMatrix / labels_f32 / labels_f32_view / labels_forest_layout / labels_all_finite / labels_check_sample_weight / labels_prepare_weights (the second admission path, and its trees_* aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
def labels_itemsize(typestr: String) raises -> Int:
    if typestr == "<f8" or typestr == "<i8":
        return 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<f2" or typestr == "<u2":
        return 2
    if typestr == "<u1" or typestr == "<i1":
        return 1
    raise Error("TypeError: " + typestr + " is not an Array dtype")


def labels_full_f32(count: Int, value: Float64, context: PythonObject) raises -> PythonObject:
    """`_buffer.full((count,), value, '<f4')`: one float32 conversion of
    `value`, the store repeated in C (`array.array.__mul__`)."""
    var one = native_array_from_values(String("f"), Python.list(PythonObject(value)))
    var store = py_call(py_attr(one, "__mul__"), PythonObject(count))
    return classical_wrap_array(store, Python.tuple(count), "<f4", "C", context)


def labels_reshape(array: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`array.reshape(shape)` of a fresh C-order Array: the same store under
    a new shape (the Array the Python reshape returned)."""
    # The store may be an `array.array` (classical_wrap_array) or the
    # uninitialised bytearray of admitted_family.admitted_array, which has no
    # `itemsize`: the new Array shares the source's store, typed view and
    # address, and only the shape, strides and rank are new.
    var b = native_loaded_module("builtins")
    var rank = len(shape)
    var size = 1
    for k in range(rank):  # driver-loop: over array rank metadata
        size *= Int(py=shape[k])
    if size != Int(py=array.size):
        raise Error("RuntimeError: an array store does not match its shape")
    var item = Int(py=array.itemsize)
    var steps = List[Int](length=rank, fill=0)
    var step = item
    for k in range(rank):  # driver-loop: over array rank metadata
        steps[rank - 1 - k] = step
        step *= Int(py=shape[rank - 1 - k])
    var strides = Python.list()
    for k in range(rank):  # driver-loop: over array rank metadata
        _ = py_call(strides.append, steps[k])
    var out = native_storage_allocate(context["array_class"])
    py_setattr(out, "_store", array._store)
    py_setattr(out, "_base", PythonObject(None))
    py_setattr(out, "_pin", PythonObject(None))
    py_setattr(out, "_mv", array._mv)
    py_setattr(out, "_addr", array._addr)
    py_setattr(out, "_readonly", PythonObject(False))
    py_setattr(out, "shape", py_call(b.tuple, shape))
    py_setattr(out, "dtype", array.dtype)
    py_setattr(out, "ndim", PythonObject(rank))
    py_setattr(out, "size", PythonObject(size))
    py_setattr(out, "nbytes", PythonObject(size * item))
    py_setattr(out, "itemsize", PythonObject(item))
    py_setattr(out, "strides", py_call(b.tuple, strides))
    py_setattr(out, "order", PythonObject("C"))
    return out


def labels_freeze(array: PythonObject) raises:
    """A fitted model array the caller cannot write through (the copy
    `_mojolearn_after_fit` took): the fresh export's own storage, marked
    read-only."""
    py_setattr(array, "_readonly", PythonObject(True))


def labels_address(array: PythonObject) raises -> Int:
    return Int(py=array._addr)


# ----------------------------------------------------------------- inputs


def _pandas_values(source: PythonObject) raises -> PythonObject:
    """The storage array of an exact single-block pandas Series (what its
    `tolist()` lists), read through pandas' native slots."""
    var manager = py_get(_pd_fields(source), "_mgr")
    var blocks = py_get(_pd_fields(manager), "blocks")
    if py_type(blocks) is not labels_builtins().tuple or len(blocks) != 1:
        raise Error("TypeError: pandas Series requires native single-block storage")
    return _pd_get("pandas._libs.internals", "Block", "values", blocks[0])


def _native_code(typestr: String) raises -> Int:
    """`_array._NATIVE_CODE`: the core helpers' dtype numbering."""
    if typestr == "<f4":
        return 0
    if typestr == "<f8":
        return 1
    if typestr == "<i4":
        return 2
    if typestr == "<i8":
        return 3
    if typestr == "<u4":
        return 4
    if typestr == "<u1":
        return 5
    if typestr == "<u2":
        return 6
    if typestr == "<i1":
        return 7
    if typestr == "<f2":
        return 8
    raise Error("TypeError: " + typestr + " has no native element code")


def labels_cast(src: Int, source: String, dst: Int, target: String, count: Int, context: PythonObject) raises:
    """`Array.astype(target)` of a contiguous block (`_native_astype`): the
    core cast `cast_elements` in process, one conversion per element in
    storage order, element tasks across the host's cores."""
    # Lane A1 admit-trees-labels: this was a Python call into the base
    # binding's `cast_elements` (one task, `_tasks = 1`); the same typed
    # kernel now runs in process (core/admitted_cast_elements.mojo), split
    # across cores. Same conversion per element: no bit change.
    # TOMBSTONE: labels_cast / trees_cast py_call into the base binding's cast_elements deleted 2026-11-08 by lane a1-admit-trees-labels; code recoverable at f7d10d2b09.
    # Restore: experiments/removed/A1_ADMIT_TREES_LABELS.patch.
    if count == 0:
        return
    var refused = admitted_cast_elements(src, _native_code(source), dst, _native_code(target), count)
    if refused != 0:
        raise Error("TypeError: mojolearn: an element does not fit " + target)


# ----------------------------------------------------------------- labels


def _encode_code(typestr: String) -> Int:
    """`_labels._NATIVE_ENCODE`, or -1."""
    if typestr == "<f4":
        return 0
    if typestr == "<f8":
        return 1
    if typestr == "<i4":
        return 2
    if typestr == "<i8":
        return 3
    if typestr == "<u4":
        return 4
    if typestr == "<u1":
        return 5
    return -1


struct ClassicalLabels(Copyable, Movable):
    """`(classes, codes)`: the class list (Python label objects, the
    `classes_` attribute) and one int32 code per row (an Array)."""
    var classes: PythonObject
    var codes: PythonObject
    var n: Int

    def __init__(out self, classes: PythonObject, codes: PythonObject, n: Int):
        self.classes = classes
        self.codes = codes
        self.n = n


def _native_labels(view: PythonObject, bool_source: Bool, context: PythonObject) raises -> ClassicalLabels:
    """`_labels._encode_labels_native` over a C-ordered numeric label block
    (admit_c_view's view, or a fresh Array)."""
    return classical_encode_words(view, Int(py=classical_address(view)), Int(py=view.size),
                                  admitted_dtype(view, context), bool_source, context)


def classical_encode_words(keep: PythonObject, address: Int, n: Int, typestr: String, bool_source: Bool,
                           context: PythonObject) raises -> ClassicalLabels:
    """The one label encoder over `n` contiguous numeric labels of
    `typestr` at `address` (`keep` owns them): one `prepare_targets` call,
    the k class values as Python numbers (bools when `bool_source`)."""
    var wide_float = typestr == "<f4" or typestr == "<f8"
    var classes_store = admitted_vector(n, "<f8" if wide_float else "<i8", context)
    var codes = admitted_vector(n, "<i4", context)
    var y_addr = address
    var k = 0
    try:
        k = labels_prepare_targets(context, y_addr, _encode_code(typestr), n, labels_address(classes_store),
                                   labels_address(codes), 0, 0, 0)
        _ = codes  # keep the Python object alive through the native call above
        _ = classes_store  # keep the Python object alive through the native call above
    except error:
        var message = String(error)
        if "NaN label" in message:
            if message.startswith("ValueError: "):
                raise error^
            raise Error("ValueError: " + message)
        raise error^
    _ = keep
    var b = labels_builtins()
    var classes = Python.list()
    for i in range(k):  # driver-loop: over the k distinct classes
        if wide_float:
            _ = py_call(classes.append, PythonObject(_F64P(unsafe_from_address=labels_address(classes_store))[i]))
            _ = classes_store  # keep the Python object alive through the native call above
        else:
            var v = Int(_I64P(unsafe_from_address=labels_address(classes_store))[i])
            _ = classes_store  # keep the Python object alive through the native call above
            if bool_source:
                _ = py_call(classes.append, PythonObject(v != 0))
                _ = classes_store  # keep the Python object alive through the native call above
            else:
                _ = py_call(classes.append, PythonObject(v))
                _ = classes_store  # keep the Python object alive through the native call above
    var _keep_ret_classes_store = ClassicalLabels(classes, codes, n)
    _ = classes_store  # keep the Python object alive through the native call above
    return _keep_ret_classes_store^


def _label_list(var y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`y.tolist()` of a label container that has one (an ndarray, an
    `array.array`, a pandas Series, an Array): the C methods of the exact
    native classes; any other buffer through its memoryview."""
    y = classifier_label_list(y, context)
    classical_admit_value(y, context)
    var b = labels_builtins()
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple:
        return py_call(b.list, y)
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None and cls is ndarray:
        return py_call(ndarray.tolist, y)
    if cls is context["array_type"]:
        return py_call(context["array_type"].tolist, y)
    if native_pandas_exact(y):
        return _label_list(_pandas_values(y), context)
    if cls is context["array_class"]:
        return label_array_list(y, context)
    # Only connected C buffer exporters: a Python class's __buffer__ hook
    # would run inside PyMemoryView_FromObject.
    var owners = py_get(context, "buffer_owner_types")
    if cls is not b.memoryview and (owners is None or not _labels_one_of(cls, owners)):
        raise Error("TypeError: mojolearn: y must be a list, tuple, array or native buffer of labels, not "
                    + labels_type_name(cls))
    return py_call(native_buffer_view(y).tolist)


def _flatten_labels(var y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.flatten_labels`: a flat Python list of the label objects,
    nested lists and tuples flattened depth first."""
    var b = labels_builtins()
    y = classifier_label_list(y, context)
    var item = y
    var cls = py_type(y)
    if cls is not b.list and cls is not b.tuple and cls is not b.str and cls is not b.bytes:
        item = _label_list(y, context)
        cls = py_type(item)
    if cls is b.str or cls is b.bytes:
        raise Error("mojolearn: y must be a sequence of labels")
    var out = Python.list()
    var kinds = LabelTypeKinds()
    # Exact container class words are stable throughout this recursive walk;
    # capture them once, not per node/leaf.
    _flatten_into(item, out, Int(b.list._obj_ptr), Int(b.tuple._obj_ptr), context, kinds)
    _ = kinds^
    return out


def _flatten_into(item: PythonObject, out_list: PythonObject, list_word: Int, tuple_word: Int,
                  context: PythonObject, mut kinds: LabelTypeKinds, depth: Int = 0) raises:
    # p3-labels-metrics: items are read borrowed from the exact container's C
    # storage and their classes compared by pointer; a container holding no
    # nested container is admitted label by label (same order, same
    # refusals) and copied with ONE list.extend; leaves of a mixed container
    # go in by PyList_Append. No boxed index, no Python append call per label.
    if depth > 64:
        raise Error("TypeError: label nesting exceeds the supported native depth")
    var cls = p3_type_word(item._obj_ptr)
    if cls == list_word or cls == tuple_word:
        var is_tuple = cls == tuple_word
        var n = len(item)
        var nested = False
        for k in range(n):  # driver-loop: exact native container slots by pointer, stops at the first nested one
            var child = p3_type_word(p3_item(item, k, is_tuple))
            if child == list_word or child == tuple_word:
                nested = True
                break
        if not nested:
            for k in range(n):  # driver-loop: exact native container slots, one memoized class admission per label
                _ = kinds.admit_raw(p3_item(item, k, is_tuple), context)
            _ = py_call(out_list.extend, item)
            return
        ref cpy = Python().cpython()
        for k in range(n):  # driver-loop: exact native container slots, never subclass iteration/indexing hooks
            var raw = p3_item(item, k, is_tuple)
            var child = p3_type_word(raw)
            if child == list_word or child == tuple_word:
                _flatten_into(PythonObject(from_borrowed=raw), out_list, list_word, tuple_word, context, kinds, depth + 1)
            else:
                _ = kinds.admit_raw(raw, context)
                if Int(cpython_table()[].PyList_Append(out_list._obj_ptr, raw)) != 0:
                    raise cpy.unsafe_get_error()
        _ = item
        return
    _ = kinds.admit(item, context)
    _ = py_call(out_list.append, item)


def _label_kind(t: PythonObject, b: PythonObject, context: PythonObject) raises -> Int:
    """`_labels._kind_of_type`: 1 str, 2 number, 0 refused. Exact native
    types only: a subclass's Python comparison/hash hooks never run."""
    _labels_connect(context)
    var kind = labels_primitive_kind(t, context)
    return 1 if kind == 1 else 2 if kind > 1 else 0


def _sorted_classes(var labels: PythonObject, context: PythonObject) raises -> ClassicalLabels:
    """`_labels.sorted_classes`: the ORDER RULE over Python label objects
    (numbers by value, strings by code point, the first of equal labels
    kept), its refusals in walk order, one dense code per label."""
    var b = labels_builtins()
    labels = classifier_label_list(labels, context)
    if py_type(labels) is not b.list and py_type(labels) is not b.tuple:
        raise Error("TypeError: sorted labels require an exact native list or tuple")
    var n = len(labels)
    if n == 0:
        raise Error("mojolearn: y is empty")
    var seen_str = False
    var seen_number = False
    _labels_connect(context)
    var kinds = LabelTypeKinds()
    var is_tuple = py_type(labels) is b.tuple
    var int_word = Int(b.int._obj_ptr)
    var bool_word = Int(b.bool._obj_ptr)
    for i in range(n):  # driver-loop: borrowed label reads by class pointer (p3-labels-metrics), the explicit label input step
        var raw = p3_item(labels, i, is_tuple)
        # Inspect identities before hashing a class: an application metaclass
        # can define Python __hash__, even when the label itself is untouched.
        var primitive = kinds.kind_raw(raw, context)
        var kind = 1 if primitive == 1 else 2 if primitive > 1 else 0
        if kind == 0:
            if PythonObject(from_borrowed=raw) is None:
                var _refusal = labels_refusal(PythonObject(None))
                raise _refusal^
            raise Error("TypeError: labels require exact native strings or real numeric scalars without Python protocols, got " + labels_type_name(py_type(PythonObject(from_borrowed=raw))))
        if kind == 2:
            # An exact int or bool is never NaN (`v != v` is False for it).
            var word = p3_type_word(raw)
            if word != int_word and word != bool_word:
                var item = PythonObject(from_borrowed=raw)
                if Bool(py=item != item):
                    raise Error("mojolearn: y contains a NaN label; NaN is not a class")
        seen_str = seen_str or kind == 1
        seen_number = seen_number or kind == 2
    _ = kinds^  # end admission reuse before sorting or numerical helpers
    if seen_str and seen_number:
        raise Error("mojolearn: y mixes numeric and str labels; encode one kind before fitting")
    var classes = py_call(b.sorted, py_call(b.dict.fromkeys, labels))
    var index = Python.dict()
    for i in range(len(classes)):  # driver-loop: over class index table (classes-sized: distinct classes)
        py_setitem(index, classes[i], PythonObject(i))
    var codes = admitted_vector(n, "<i4", context)
    # One borrowed read and one C dict probe per label (the label's own C
    # hash and equality, as `index[label]` ran them); every label is a key.
    p3_dict_codes(index, labels, labels_address(codes), -1)
    _ = codes  # keep the Python object alive through the native writes above
    return ClassicalLabels(classes, codes, n)


def _numeric_label_view(y: PythonObject, context: PythonObject, mut bool_source: Bool) raises -> Optional[PythonObject]:
    """The numeric label vector `_labels._encode_labels_native` encodes (a
    C-ordered view of its own dtype, admit_c_view), or None for what only
    the label input step reads (lists, str or object buffers, a pandas
    Series, a matrix, an empty vector)."""
    classical_admit_value(y, context)
    var b = labels_builtins()
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes or native_pandas_exact(y):
        return None
    var view = PythonObject(None)
    var typestr = String("")
    try:
        if cls is not context["array_class"]:
            bool_source = String(native_buffer_view(y).format) == "?"
        elif String(y.dtype).endswith("b1"):
            return None  # a bool Array takes the label input step, as before
        view = admit_c_view(y, "y", context, True)
        typestr = admitted_dtype(view, context)
    except:
        return None
    var n = Int(py=view.size)
    var big = 0
    for k in range(Int(py=view.ndim)):  # driver-loop: over array rank metadata
        if Int(py=view.shape[k]) > big:
            big = Int(py=view.shape[k])
    if _encode_code(typestr) < 0 or n == 0 or n != big:
        return None
    return view


def _encode_label_view(y: PythonObject, context: PythonObject, mut bool_source: Bool) raises -> Optional[PythonObject]:
    """The numeric label block `classical_encode_labels` encodes with one
    native `prepare_targets` call (a C-ordered view or Array of a native
    encode dtype), or None for what only the label input step reads. Admits
    `y` (`classical_admit_value`) exactly once.

    Takes, each with exactly the classes and codes the label input step
    gives it (that step lists the same values in the same C order, then
    encodes the flat list natively or by the ORDER RULE, whose sorted
    distinct values and dense codes are the same): a numeric buffer or
    Array of any rank (read in C order: an F-ordered or strided one is one
    C-order copy, admit_c_view), a numpy-backed pandas Series (its stored
    ndarray, `labels_series_values`), a bool Array or buffer (bytes 0/1 read
    as uint8, the classes returned as bools: `bool_source`), and int8 /
    uint16 / float16 storage widened once to int64 / float64 by the native
    cast (the exact Python ints / floats `tolist()` gives)."""
    # Lane a1b-doors-opt-in-trees-linear: the block is read through
    # admit_c_view (the LabelsView copy is gone); an F-ordered matrix,
    # which took the label input step, is now its C-order copy: the same
    # C-order values, so the same classes and codes.
    classical_admit_value(y, context)
    var b = labels_builtins()
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes:
        return None
    var source = y
    if native_pandas_exact(y):
        source = labels_series_values(y, context)
        if source is None:
            return None
        cls = py_type(source)
    var view = PythonObject(None)
    var typestr = String("")
    try:
        if cls is context["array_class"]:
            bool_source = String(source.dtype).endswith("b1")
        else:
            bool_source = String(native_buffer_view(source).format) == "?"
        view = admit_c_view(source, "y", context, True)
        typestr = admitted_dtype(view, context)
    except:
        bool_source = False
        return None
    var n = Int(py=view.size)
    if n == 0:
        bool_source = False
        return None
    if _encode_code(typestr) < 0:
        if typestr != "<i1" and typestr != "<u2" and typestr != "<f2":
            bool_source = False
            return None
        # Every element of this C-order block widens exactly.
        var target = String("<f8") if typestr == "<f2" else String("<i8")
        var wide = admitted_vector(n, target, context)
        labels_cast(Int(py=classical_address(view)), typestr, labels_address(wide), target, n, context)
        _ = view
        _ = source
        return wide
    _ = source  # the Series' stored ndarray is kept alive by the view
    return view


def _flat_list_labels(y: PythonObject, context: PythonObject) raises -> Optional[ClassicalLabels]:
    """An exact flat list or tuple of exact ints, exact floats or exact
    bools, encoded by one native `prepare_targets` call without the label
    input step's walks (flatten + per-label admission, then the
    `_list_labels` type walk): the item classes are read once through the
    container's C storage (`labels_list_scalar_kind`). Ints and floats get
    exactly `_list_labels`' result (the same native array of the same
    values). Bools: the ORDER RULE sorts them False < True with dense codes,
    which is what the native encoder gives their 0/1 values with the classes
    returned as bools. None for anything else, and for an int outside int64
    (the label input step then refuses or encodes it as before)."""
    var kind = labels_list_scalar_kind(y)
    if kind == LABELS_LIST_OTHER:
        return None
    var n = len(y)
    var integral = kind != LABELS_LIST_FLOAT
    var store = PythonObject(None)
    try:
        store = native_array_empty(String("q" if integral else "d"))
        # array.extend over an exact list or tuple whose items were all
        # checked exact int / float / bool above: C conversion per item, no
        # Python hook (native_array_from_values' own rule).
        _ = py_call(store.extend, y)
    except:
        return None  # an int outside int64: a class only Python can hold
    var array = classical_wrap_array(store, Python.tuple(n), "<i8" if integral else "<f8", "C", context)
    return _native_labels(array, kind == LABELS_LIST_BOOL, context)


def _text_buffer_labels(y: PythonObject, context: PythonObject) raises -> Optional[ClassicalLabels]:
    """An exact ndarray of native unicode ('<U', rank >= 1, nonempty)
    encoded from its buffer: each element's UCS4 code points are read in C
    order (trailing NULs dropped, as `tolist()` drops them), the distinct
    labels ordered by code point (Python's str order) with the stable merge
    sort of `bindings/str_labels_native.mojo`, and the dense codes written
    natively. Only the k class strings are made, by `ndarray.item` at each
    class's first C-order index (the str `tolist()` gives there). The
    ORDER RULE's `sorted(dict.fromkeys(labels))` over the listed strs gives
    the same classes and codes. None for anything else."""
    if not labels_text_storage(y, context):
        return None
    var ndarray = context["ndarray"]
    var rank = Int(py=y.ndim)
    var pyshape = y.shape
    var pystrides = y.strides
    var shape = List[Int]()
    var strides = List[Int]()
    var n = 1
    for k in range(rank):  # driver-loop: over array rank metadata
        shape.append(Int(py=pyshape[k]))
        strides.append(Int(py=pystrides[k]))
        n *= shape[k]
    if n == 0:
        return None
    var width = Int(py=y.itemsize) // 4
    var address = _ndarray_buffer_address(y)
    var table = StrTable()
    var index = List[Int](length=rank, fill=0)
    var offset = 0
    for _ in range(n):  # driver-loop: the explicit label input step, one native read per label
        var source = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address + offset)
        var count = width
        while count > 0 and source[count - 1] == 0:  # driver-loop: over one label's trailing NUL padding
            count -= 1
        table.starts.append(len(table.points))
        table.lengths.append(count)
        for c in range(count):  # driver-loop: over one label's code points
            table.points.append(source[c])
        var axis = rank - 1
        while axis >= 0:  # driver-loop: over array rank metadata (C-order index step)
            index[axis] += 1
            offset += strides[axis]
            if index[axis] < shape[axis]:
                break
            offset -= strides[axis] * shape[axis]
            index[axis] = 0
            axis -= 1
    _ = y  # the caller's ndarray owns the buffer read above
    var order = _sorted(table, n)
    var classes = Python.list()
    var codes = admitted_vector(n, "<i4", context)
    var out = _I32P(unsafe_from_address=labels_address(codes))
    var item = ndarray.item
    var current = -1
    for t in range(n):  # driver-loop: over the sorted labels; a Python str only per distinct class
        var i = order[t]
        if t == 0 or table.compare(order[t - 1], i) != 0:
            current += 1
            _ = py_call(classes.append, py_call(item, y, PythonObject(i)))
        out[i] = Int32(current)
    _ = codes  # keep the Python object alive through the native writes above
    return ClassicalLabels(classes, codes, n)


def _text_list_labels(flat: PythonObject, context: PythonObject) raises -> Optional[ClassicalLabels]:
    """A flat list of admitted str labels (exact str or NumPy's str scalar,
    the only text kinds the flatten admits) encoded by the native str-label
    reader: code points read once per label through the C API, ordered by
    code point with a stable merge sort, so each class is the first label
    of its run, as `dict.fromkeys` keeps the first. Same classes and codes
    as the ORDER RULE; None when a label is not a str (numbers and mixed
    lists keep the ORDER RULE's own refusals and encodings)."""
    var n = len(flat)
    if n == 0:
        return None
    # p3-labels-metrics: hash to the k distinct labels by their UTF-8 bytes,
    # sort only the k classes (bindings/p3_labels_metrics_native.mojo): the
    # same classes (first label of each), order and codes. A str with no
    # UTF-8 form takes the code-point path below.
    var fast = p3_text_codes(flat, True)
    if fast.status == P3_TEXT_NOT_STR:
        return None
    if fast.status == P3_TEXT_OK:
        var fast_codes = admitted_vector(n, "<i4", context)
        var target = _I32P(unsafe_from_address=labels_address(fast_codes))
        for i in range(n):  # driver-loop: native copy of the n codes
            target[i] = fast.codes[i]
        _ = fast_codes  # keep the Python object alive through the native writes above
        return ClassicalLabels(p3_text_classes(flat, fast.first), fast_codes, n)
    var table = StrTable()
    if not _read(flat, table):
        return None
    var order = _sorted(table, n)
    var classes = Python.list()
    var codes = admitted_vector(n, "<i4", context)
    var out = _I32P(unsafe_from_address=labels_address(codes))
    var current = -1
    for t in range(n):  # driver-loop: over the sorted labels
        var i = order[t]
        if t == 0 or table.compare(order[t - 1], i) != 0:
            current += 1
            _ = py_call(classes.append, flat[i])
        out[i] = Int32(current)
    _ = codes  # keep the Python object alive through the native writes above
    return ClassicalLabels(classes, codes, n)


def _list_labels(y: PythonObject, context: PythonObject) raises -> Optional[ClassicalLabels]:
    """`_labels._encode_label_list_native`: a flat list or tuple whose every
    label is exactly an int, or exactly a float, through the native encoder;
    None for anything else (and for an int outside int64)."""
    var b = labels_builtins()
    var n = len(y)
    if n == 0:
        return None
    # One borrowed class-pointer read per label through the container's C
    # storage, stopping at the first other label (p3-labels-metrics).
    var scalar_kind = labels_list_scalar_kind(y)
    var all_int = scalar_kind == LABELS_LIST_INT
    var all_float = scalar_kind == LABELS_LIST_FLOAT
    if not all_int and not all_float:
        return None
    var store = PythonObject(None)
    try:
        store = native_array_from_values(String("q" if all_int else "d"), y)
    except:
        return None  # an int outside int64: a class only Python can hold
    var array = classical_wrap_array(store, Python.tuple(n), "<i8" if all_int else "<f8", "C", context)
    return _native_labels(array, False, context)


def classical_encode_labels(var y: PythonObject, context: PythonObject) raises -> ClassicalLabels:
    """`_labels.encode_labels(y)`: `classes` the Python class list and one
    int32 code per row (an Array). One native `prepare_targets` call for a
    numeric label vector (a buffer or a flat list of exact ints or exact
    floats); the ORDER RULE over Python objects for str and object labels.
    THE one label encoder (lane a2-admit-family: labels_encode_labels and the
    SVM's svc_encode_1d were second names for it)."""
    # A context without the base binding's label helpers (an owner context
    # that connects none) takes the label input step only, as svc_encode_1d
    # did: the same sorted classes and codes.
    if py_get(context, "labels_base_binding") is None:
        return _sorted_classes(_flatten_labels(y, context), context)
    var b = labels_builtins()
    y = classifier_label_list(y, context)
    var bool_source = False
    # perf2-labels: a numpy-backed Series, a C-ordered matrix, a bool Array
    # and int8/uint16/float16 storage join the numeric buffers here; flat
    # lists of exact ints/floats/bools and '<U' ndarrays skip the label input
    # step's per-label walks. Each gives the classes and codes that step
    # gives (see each helper); everything else takes that step unchanged.
    var numeric = _encode_label_view(y, context, bool_source)
    if numeric:
        return _native_labels(numeric.value(), bool_source, context)
    var direct = _flat_list_labels(y, context)
    if direct:
        return direct.value().copy()
    var text = _text_buffer_labels(y, context)
    if text:
        return text.value().copy()
    var flat = _flatten_labels(y, context)
    var listed = _list_labels(flat, context)
    if listed:
        return listed.value().copy()
    var named = _text_list_labels(flat, context)
    if named:
        return named.value().copy()
    return _sorted_classes(flat, context)


def labels_label_kind(var classes: PythonObject, context: PythonObject) raises -> String:
    classes = classifier_label_list(classes, context)
    var b = labels_builtins()
    if py_type(classes) is not b.list and py_type(classes) is not b.tuple:
        raise Error("TypeError: class labels require an exact native list or tuple")
    var all_int = True
    var all_real = True
    var kinds = LabelTypeKinds()
    var is_tuple = py_type(classes) is b.tuple
    for i in range(len(classes)):  # driver-loop: exact primitive kinds preserve bool/string object-label output
        var kind = kinds.admit_raw(p3_item(classes, i, is_tuple), context)
        all_int = all_int and kind == 3
        all_real = all_real and (kind == 3 or kind == 4)
    _ = kinds^
    return String("int") if all_int else String("float") if all_real else String("object")


def labels_decode_labels(var classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for int64 or int32 codes:
    integer classes an int64 Array, real classes a float64 Array (the base
    binding's gathers), str or bool classes a Python list."""
    var b = labels_builtins()
    classes = classifier_label_list(classes, context)
    var kind = labels_label_kind(classes, context)
    classical_admit_value(codes, context)
    var n = Int(py=codes.size)
    var k = len(classes)
    var code_dtype = String(codes.dtype)
    if kind != "int" and kind != "float" and (code_dtype == "<i4" or code_dtype == "<i8"):
        # p3-labels-metrics: str / bool / object classes decode as ONE
        # presized list filled in C from the code words (no widening copy,
        # no Python append per row). An out-of-range code takes the loop
        # below, which raises Python's IndexError as before.
        var listed = p3_decode_list(classes, labels_address(codes), 4 if code_dtype == "<i4" else 8, n)
        _ = codes  # keep the Python object alive through the native reads above
        if listed is not None:
            return listed
    var wide_codes = codes
    if code_dtype == "<i4":
        wide_codes = admitted_vector(n, "<i8", context)
        labels_cast(labels_address(codes), "<i4", labels_address(wide_codes), "<i8", n, context)
        _ = wide_codes  # keep the Python object alive through the native call above
    if (kind == "int" or kind == "float") and k > 0:
        var table = admitted_vector(k, "<i8" if kind == "int" else "<f8", context)
        var fits = True
        for i in range(k):  # driver-loop: over the k class values into the gather table
            if kind == "int":
                var v = classes[i]
                if Bool(py=v < PythonObject(-9223372036854775808)) or Bool(py=v > PythonObject(9223372036854775807)):
                    fits = False
                    break
                _I64P(unsafe_from_address=labels_address(table))[i] = Int64(Int(py=py_call(b.int, v)))
                _ = table  # keep the Python object alive through the native call above
            else:
                _F64P(unsafe_from_address=labels_address(table))[i] = Float64(py=py_call(b.float, classes[i]))
                _ = table  # keep the Python object alive through the native call above
        if fits:
            var out = admitted_vector(n, "<i8" if kind == "int" else "<f8", context)
            labels_gather(context, LABELS_GATHER_I64 if kind == "int" else LABELS_GATHER_F64, labels_address(table), k,
                          labels_address(wide_codes), n, labels_address(out))
            _ = table  # keep the Python object alive through the native call above
            _ = out  # keep the Python object alive through the native call above
            _ = wide_codes  # keep the Python object alive through the native call above
            return out
    # TOMBSTONE: labels_decode_labels per-row Python append loop (removed: one native list fill) deleted 2026-11-09 by lane p2-labels-lists-arrays; code recoverable at e68654a757.
    # Restore: git apply experiments/removed/P2_LABELS_LISTS_ARRAYS.patch
    # Classes the gathers cannot carry (str / bool / object, or ints outside
    # int64): ONE presized list filled in C from the int64 code words
    # (p3_decode_list). Lane p2-labels-lists-arrays: the per-row Python
    # append loop (cpu-route) is gone. Python's list index rule holds
    # (negative codes index from the end); an out-of-range code raises
    # Python's IndexError, as the subscript did.
    if code_dtype != "<i4" and code_dtype != "<i8":
        raise Error("TypeError: mojolearn: label codes must be int32 or int64, not " + code_dtype)
    var listed = p3_decode_list(classes, labels_address(wide_codes), 8, n)
    _ = wide_codes  # keep the Python object alive through the native reads above
    if listed is None:
        raise Error("IndexError: list index out of range")
    return listed


def labels_argmax_rows(scores: PythonObject, rows: Int, cols: Int, wide: Bool, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows` of a C-order float32 or float64 block: the
    base binding's first-max scan, int64 codes."""
    var out = admitted_vector(rows, "<i8", context)
    if rows == 0:
        return out
    if cols == 0:
        raise Error("argmax_rows: scores have no columns")
    labels_argmax_rows(context, wide, labels_address(scores), rows, cols, labels_address(out))
    _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------- class weights


def labels_class_weight_rows[NativeWeights: Bool = False](class_weight: PythonObject, labels: ClassicalLabels, xtrees: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """`_class_weights._class_weight_rows(class_weight, classes, codes,
    xbind)` with int32 codes on the x_trees binding: per-row float32
    weights from a dict or 'balanced', or None when every class weight is
    1. The counts and the per-row gather are the binding's kernels; only
    the k class weights are read here."""
    var b = labels_builtins()
    if class_weight is None:
        return PythonObject(None)
    if py_type(class_weight) is not b.str and py_type(class_weight) is not b.dict:
        raise Error("TypeError: class_weight requires an exact native string or dictionary")
    if py_type(class_weight) is b.dict:
        var items = py_call(b.list, class_weight.items())
        for i in range(len(items)):  # driver-loop: admit keys before native hashing and values before numeric conversion
            _ = labels_admit_primitive(items[i][0], context)
            _ = labels_admit_primitive(items[i][1], context)
    var classes = classifier_label_list(labels.classes, context)
    _ = labels_label_kind(classes, context)
    var k = len(classes)
    var n = labels.n
    var objects = Python.list()
    if py_type(class_weight) is b.str:
        var counts = admitted_vector(k, "<i4", context)
        comptime if NativeWeights:
            _ = label_weights_run(xtrees, LABEL_WEIGHTS_COUNTS,
                LabelWeightsRequest(labels_address(labels.codes) if n else 0, labels_address(counts), 0, n, k))
        else:
            # Other unmigrated classifier owners still connect these public
            # legacy entries. Logistic's specialization has no Python call.
            var count_entry = labels_need(xtrees, "x_trees_code_counts", "the x_trees binding has no x_trees_code_counts",
                                         "ImportError")
            _ = py_call_typed(count_entry, PythonObject(labels_address(labels.codes) if n else 0),
                              PythonObject(labels_address(counts)), Python.list(n, k))
        _ = counts  # keep the Python object alive through the native call above
        var wide = admitted_vector(k, "<i8", context)
        labels_cast(labels_address(counts), "<i4", labels_address(wide), "<i8", k, context)
        _ = wide  # keep the Python object alive through the native call above
        _ = counts  # keep the Python object alive through the native call above
        var w64 = admitted_vector(k, "<f8", context)
        if k and labels_class_ratio_f64(context, labels_address(wide), k, n, 0, labels_address(w64)) != 0:
            raise Error("ZeroDivisionError: division by zero")
        _ = wide  # keep the Python object alive through the native calls above
        _ = w64  # keep the Python object alive through the native calls above
        var p = _F64P(unsafe_from_address=labels_address(w64))
        _ = w64  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the k class weights for the argument checks below
            _ = py_call(objects.append, PythonObject(p[i]))
            _ = w64  # keep the Python object alive through the native call above
    else:
        if py_type(class_weight) is not b.dict:
            raise Error("TypeError: class_weight must be 'balanced', None or a dict")
        var keys = py_call(b.list, class_weight)
        for i in range(len(keys)):  # driver-loop: over user class_weight keys, admitted before hashing/comparison
            var kt = py_type(keys[i])
            if not labels_text_type(kt, context) and not labels_real_type(kt, context):
                raise Error("TypeError: class_weight keys must be native numbers or strings")
        var unknown = py_call(py_call(b.set, keys).difference, classes)
        if len(unknown) > 0:
            raise Error("class_weight contains labels absent from y")
        for i in range(k):  # driver-loop: over user class_weight dict per class
            _ = py_call(objects.append, py_call(class_weight.get, classes[i], PythonObject(1.0)))
    var values = List[Float64]()
    var all_one = True
    for i in range(k):  # driver-loop: over class weight argument checks
        var v = objects[i]
        var t = py_type(v)
        var kind = labels_admit_primitive(v, context)
        if kind != 3 and kind != 4:
            raise Error("class weights must be finite nonnegative real numbers")
        var f = Float64(py=py_call(b.float, v))
        if not isfinite(f) or f < 0:
            raise Error("class weights must be finite nonnegative real numbers")
        values.append(f)
        all_one = all_one and Bool(py=v == PythonObject(1))
    if all_one:
        return PythonObject(None)
    var narrowed = admitted_vector(k, "<f4", context)
    var q = _F32P(unsafe_from_address=labels_address(narrowed))
    _ = narrowed  # keep the Python object alive through the native call above
    var top = Float32(0)
    for i in range(k):  # driver-loop: over the k class weights narrowed to float32
        var f32 = Float32(values[i])
        if not isfinite(f32) or (values[i] > 0 and f32 == 0):
            raise Error("class weights must remain finite and nonzero when positive in Float32")
        q[i] = f32
        _ = narrowed  # keep the Python object alive through the native call above
        if i == 0 or f32 > top:
            top = f32
    if top <= 0:
        raise Error("class weights must have positive total")
    var out = admitted_vector(n, "<f4", context)
    if n:
        comptime if NativeWeights:
            _ = label_weights_run(xtrees, LABEL_WEIGHTS_ROWS,
                LabelWeightsRequest(labels_address(labels.codes), labels_address(narrowed), labels_address(out), n, k))
        else:
            var rows_entry = labels_need(xtrees, "x_trees_class_rows", "the x_trees binding has no x_trees_class_rows",
                                        "ImportError")
            _ = py_call_typed(rows_entry, PythonObject(labels_address(labels.codes)), PythonObject(labels_address(narrowed)),
                              PythonObject(labels_address(out)), Python.list(n, k))
        _ = narrowed  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    return out


def labels_weighted_rows(sample_weight: PythonObject, class_weight: PythonObject, labels: ClassicalLabels,
                        xtrees: PythonObject, context: PythonObject) raises -> AdmittedX:
    """`_expansion_trees._trees_weighted_rows`: the checked sample weights
    times the class_weight row weights, one correctly rounded float32
    product per row (`x_trees_mul_f32`); the sample weights alone when no
    class weight differs from 1."""
    var sw = admit_weight_words(sample_weight, labels.n, context)
    if class_weight is None:
        return sw^
    var rows = labels_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return sw^
    var out = admitted_vector(labels.n, "<f4", context)
    var mul = labels_need(xtrees, "x_trees_mul_f32", "the x_trees binding has no x_trees_mul_f32", "ImportError")
    _ = py_call_typed(mul, PythonObject(sw.address), PythonObject(labels_address(rows)), PythonObject(labels_address(out)),
                      Python.list(labels.n))
    _ = out  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above
    _ = sw^
    return admitted_words(out, labels_address(out), labels.n, 1)


def labels_class_rows_only(class_weight: PythonObject, labels: ClassicalLabels, xtrees: PythonObject,
                          context: PythonObject) raises -> Optional[AdmittedX]:
    """The class_weight row weights without sample weights, or None."""
    var rows = labels_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return None
    return admitted_words(rows, labels_address(rows), labels.n, 1)


# x_metrics stage ops (`_expansion_metrics._OPS`) and cm_epi kinds.
comptime _OP_GROUP_SORT = 0
comptime _OP_GROUP_SUM = 1
comptime _OP_PAIR_KEY = 2
comptime _OP_OFF_DIFF = 56
comptime _OP_CM_EPI = 59
comptime _CM_COUNT = 2
comptime _CM_WSUM_LEAF = 7
comptime _CM_WSUM_FOLD = 8
comptime _WSUM_BLOCK = 256
comptime _NONE = -1
comptime _PARAMS = 14
comptime _INT32_MAX = 2147483647



# ------------------------------------------------------------ shapes


def labels_shape_of(var value: PythonObject, context: PythonObject) raises -> List[Int]:
    """Shape from admitted native storage, without Array properties or protocols."""
    value = classifier_label_list(value, context)
    classical_admit_value(value, context)
    var b = labels_builtins()
    var cls = py_type(value)
    var shape = List[Int]()
    if cls is context["array_class"]:
        var dimensions = py_call(context["array_fields"]["shape"], value)
        for k in range(len(dimensions)):  # driver-loop: native Array shape slots
            shape.append(Int(py=dimensions[k]))
        return shape^
    if native_pandas_exact(value):
        var storage = native_pandas_f32(value, context, False, True)
        var dimensions = storage.shape
        for k in range(len(dimensions)):  # driver-loop: admitted native pandas storage dimensions
            shape.append(Int(py=dimensions[k]))
        return shape^
    if cls is b.list or cls is b.tuple:
        shape.append(len(value))
        if len(value) > 0:
            var first = value[0]
            if py_type(first) is b.list or py_type(first) is b.tuple:
                shape.append(len(first))
        return shape^
    var storage = native_torch_numpy(value) if _ordinary_torch(cls, context) else value
    var view = native_buffer_view(storage)
    for k in range(Int(py=view.ndim)):  # driver-loop: native buffer dimensions
        shape.append(Int(py=view.shape[k]))
    _ = storage^
    return shape^


def _shape_text(shape: List[Int]) raises -> String:
    """The Python repr of a shape tuple."""
    var b = labels_builtins()
    var items = Python.list()
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        _ = py_call(items.append, shape[k])
    return String(py_call(b.repr, py_call(b.tuple, items)))


def _buffer_format(value: PythonObject, context: PythonObject) raises -> String:
    """`linear_model._buffer_format`: the struct format code, byte order
    stripped; "" when the value exports no buffer."""
    var b = labels_builtins()
    if py_type(value) is context["array_class"]:
        var dtype = String(value.dtype)
        if dtype == "<f4":
            return "f"
        if dtype == "<f8":
            return "d"
        if dtype == "<i4":
            return "i"
        if dtype == "<i8":
            return "q"
        if dtype == "<u4":
            return "I"
        if dtype == "<u1":
            return "B"
        if dtype == "<f2":
            return "e"
        return ""
    try:
        return String(py_call(py_call(b.memoryview, value).format.lstrip, "<>=@!|"))
    except:
        return ""


def _has_float_leaf(value: PythonObject) raises -> Bool:
    """A Python float among the leaves of a nested list or tuple."""
    var b = labels_builtins()
    if py_type(value) is b.float:
        return True
    if py_type(value) is b.list or py_type(value) is b.tuple:
        for k in range(len(value)):  # cpu-route: the leaf types of a user Python list, the explicit input step
            if _has_float_leaf(value[k]):
                return True
    return False


# ------------------------------------------------------------ r2


def _as_f32_1d(source: PythonObject, name: String, context: PythonObject) raises -> AdmittedX:
    """`_metrics_impl._as_f32_1d`: a 1-D (or (n, 1) column) float32 vector,
    float64 refused by name, NaN or infinity refused."""
    var shape = labels_shape_of(source, context)
    var column = len(shape) == 2 and shape[1] == 1
    if len(shape) != 1 and not column:
        raise Error("mojolearn metrics: " + name + " must be 1-D, got shape " + _shape_text(shape))
    if shape[0] == 0:
        raise Error("mojolearn metrics: " + name + " is empty")
    var fmt = _buffer_format(source, context)
    var float64 = fmt == "d" if fmt.byte_length() > 0 else _has_float_leaf(source)
    if float64:
        raise Error("TypeError: mojolearn metrics: " + name + " is float64; the float64 overloads of "
                    + "r2_score, kl_divergence and silhouette_score are NOT implemented "
                    + "(no float64 on this GPU). Cast to float32 yourself so the "
                    + "precision you run at is the one you chose.")
    # The one admission (admit_f32): the finite test rides in its cast or
    # its one parallel scan; the metrics wording is kept.
    try:
        return admit_f32(source, len(shape), name, context)
    except error:
        if "contains NaN or infinity" in String(error):
            raise Error("mojolearn metrics: " + name + " contains NaN or infinity; refused "
                        + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                        + "reach a kernel whose answer for it is a washed sentinel")
        raise error^


def _reduce(addr: Int, count: Int, what: Int, context: PythonObject) raises -> PythonObject:
    """`Array.min()` (0) / `max()` (1) of a float32 vector: `reduce_stat`."""
    return labels_reduce_stat(context, addr, 0, count, what)


def labels_sample_weight_f32(source: PythonObject, n: Int, caller: String, context: PythonObject) raises -> AdmittedX:
    """`_metrics_impl._sample_weight_f32`: one finite, non-negative float32
    weight per sample with a positive total."""
    var shape = labels_shape_of(source, context)
    if len(shape) != 1:
        raise Error("mojolearn " + caller + ": sample_weight must be 1-D, got shape " + _shape_text(shape))
    if shape[0] != n:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(shape[0]) + " entries for "
                    + String(n) + " samples")
    var refusal = "mojolearn " + caller + ": sample_weight must have finite nonnegative entries"
    var w = AdmittedX(PythonObject(None), PythonObject(None), 0, 0, 0, 1, False)
    try:
        w = admit_f32(source, 1, "sample_weight", context)
    except error:
        if "contains NaN or infinity" in String(error):
            raise Error(refusal)
        raise error^
    if Bool(py=_reduce(w.address, n, 0, context) < PythonObject(0)):
        raise Error(refusal)
    _ = w  # keep the Python object alive through the native calls above
    if not Bool(py=_reduce(w.address, n, 1, context) > PythonObject(0)):
        raise Error("mojolearn " + caller + ": sample_weight must have positive total weight")
    _ = w  # keep the Python object alive through the native calls above
    return w^


def _metrics_mode(mode: PythonObject, context: PythonObject) raises -> String:
    """`_metrics_impl._get_binding`'s mode: the requested one or the
    package default."""
    var b = labels_builtins()
    var selected = mode
    if selected is not None:
        if py_type(selected) is not b.str:
            raise Error("numeric_mode must be 'fast' or 'identical'")
        var clean = String(py_call(b.str.lower, py_call(b.str.strip, selected)))
        if clean != "fast" and clean != "identical":
            raise Error("numeric_mode must be 'fast' or 'identical'")
        return clean
    return String(py_call(py_call(context["backend_fields"]["_DEFAULT_MODE"].strip).lower))


def labels_r2(y_true: PythonObject, prediction: PythonObject, sample_weight: PythonObject, mode: PythonObject,
             context: PythonObject) raises -> Float64:
    """`metrics.r2_score(y_true, prediction, sample_weight=..., numeric_mode=mode)`
    for 1-D targets (the default multioutput): the float32 R^2 kernel."""
    var y = _as_f32_1d(y_true, "y_true", context)
    var yh = _as_f32_1d(prediction, "y_pred", context)
    if y.rows != yh.rows:
        raise Error("mojolearn metrics: y_true has " + String(y.rows) + " entries and "
                    + "y_pred has " + String(yh.rows) + "; they must be the same length")
    var selected = _metrics_mode(mode, context)
    var metrics = labels_table(context, selected, "_mojolearn_metrics")
    var read_mode = labels_entry(metrics, "metrics_numeric_mode")
    if read_mode is None:
        read_mode = labels_entry(metrics, "umap_numeric_mode")
    if read_mode is not None:
        var code = Int(py=py_call_typed(read_mode))
        var actual = String("fast") if code == 0 else (String("identical") if code == 1 else String("None"))
        if actual != selected:
            raise Error("RuntimeError: mojolearn metrics: requested " + selected + ", binary reports "
                        + actual + "; rebuild it")
    var result = PythonObject(None)
    if sample_weight is not None:
        var w = labels_sample_weight_f32(sample_weight, y.rows, "r2_score", context)
        result = py_call_typed(metrics["r2_score_weighted"], PythonObject(y.address), PythonObject(yh.address),
                               PythonObject(w.address), Python.list(PythonObject(y.rows)))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = w^
    else:
        result = py_call_typed(metrics["r2_score"], PythonObject(y.address), PythonObject(yh.address),
                               Python.list(PythonObject(y.rows)))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
    _ = y^
    _ = yh^
    return Float64(py=py_call(labels_builtins().float, result))


# ------------------------------------------------------------ accuracy


struct ScoreLabels(Copyable, Movable):
    """`_classification_encoded`'s answer: encoded integer labels (sorted
    distinct Python ints and one int32 code per row) or a list of Python
    str or int labels; `kind` 'integer' or 'string'."""
    var encoded: Bool
    var classes: PythonObject
    var codes: PythonObject
    var values: PythonObject
    var n: Int
    var kind: String

    def __init__(out self, encoded: Bool, classes: PythonObject, codes: PythonObject, values: PythonObject,
                 n: Int, kind: String):
        self.encoded = encoded
        self.classes = classes
        self.codes = codes
        self.values = values
        self.n = n
        self.kind = kind


def _native_classification(var values: PythonObject, context: PythonObject) raises -> Optional[ScoreLabels]:
    """`_metrics_impl._native_classification_labels`: an integer or bool
    buffer (or a flat list of exact ints) through the native encoder."""
    values = classifier_label_list(values, context)
    classical_admit_value(values, context)
    var b = labels_builtins()
    var cls = py_type(values)
    if cls is b.str or cls is b.bytes:
        return None
    var view = PythonObject(None)
    if cls is b.list or cls is b.tuple:
        var n = len(values)
        if n == 0:
            return None
        # Every label exactly an int, read by class pointer through the
        # container's C storage (p3-labels-metrics).
        if labels_list_scalar_kind(values) != LABELS_LIST_INT:
            return None
        var store = PythonObject(None)
        try:
            store = native_array_from_values(String("q"), values)
        except:
            return None
        view = classical_wrap_array(store, Python.tuple(n), "<i8", "C", context)
    else:
        if native_pandas_exact(values):
            return None
        if len(labels_shape_of(values, context)) != 1:
            return None
        try:
            view = admit_c_view(values, "input", context, True)
            _ = admitted_dtype(view, context)
        except:
            return None
    if Int(py=view.ndim) != 1 or Int(py=view.size) < 1:
        return None
    var typestr = admitted_dtype(view, context)
    if typestr == "<u2" or typestr == "<i1":
        # exact; the encoder has no 8- or 16-bit arm (`astype('<i4')`)
        var count = Int(py=view.size)
        var wide = admitted_vector(count, "<i4", context)
        labels_cast(Int(py=classical_address(view)), typestr, labels_address(wide), "<i4", count, context)
        view = wide
        typestr = "<i4"
    if typestr != "<i4" and typestr != "<i8" and typestr != "<u4" and typestr != "<u1":
        return None
    # A bool buffer's classes are read as the ints 0 and 1 (`int(c)`).
    var labels = _native_labels(view, False, context)
    return ScoreLabels(True, labels.classes, labels.codes, PythonObject(None), labels.n, "integer")


def _classification_labels(values: PythonObject, name: String, context: PythonObject) raises -> ScoreLabels:
    """`_metrics_impl._classification_labels`: one-dimensional single-label
    data of str or integer labels (bools read as ints)."""
    var b = labels_builtins()
    var shape = labels_shape_of(values, context)
    if len(shape) != 1:
        raise Error(name + " must be one-dimensional single-label data")
    var labels = _flatten_labels(values, context)
    var n = len(labels)
    if n == 0:
        raise Error(name + " must contain at least one label")
    var all_str = True
    var all_int = True
    var kinds = LabelTypeKinds()
    for i in range(n):  # driver-loop: borrowed label reads, one memoized class admission per label
        var kind = kinds.admit_raw(p3_item(labels, i, False), context)
        all_str = all_str and kind == 1
        all_int = all_int and (kind == 2 or kind == 3)
    _ = kinds^  # no cached classification spans the following conversions
    var out = Python.list()
    if all_str:
        # `str(v)` of an exact str is v itself: the flat list (a fresh list)
        # is the converted one (p3-labels-metrics).
        if labels_all_exact_str(labels):
            return ScoreLabels(False, PythonObject(None), PythonObject(None), labels, n, "string")
        for i in range(n):  # cpu-route: NumPy str_ labels to exact str, the explicit label input step
            _ = py_call(out.append, py_call(b.str, labels[i]))
        return ScoreLabels(False, PythonObject(None), PythonObject(None), out, n, "string")
    if all_int:
        # Exact ints kept, bools 0 / 1, NumPy integer scalars their exact int
        # (C calls into one presized list; p3-labels-metrics).
        var bool_types = Python.tuple()
        if py_get(context, "ndarray") is not None:
            bool_types = Python.tuple(context["numeric_dtypes"][Python.tuple("b", 1)].type)
        return ScoreLabels(False, PythonObject(None), PythonObject(None), p3_integral_list(labels, bool_types),
                           n, "integer")
    raise Error("TypeError: " + name + " must contain only strings or only integers; "
                + "floating labels, missing labels and mixed types are unsupported")


def _classification_encoded(values: PythonObject, name: String, context: PythonObject) raises -> ScoreLabels:
    var fast = _native_classification(values, context)
    if fast:
        return fast.value().copy()
    return _classification_labels(values, name, context)


def _label_set(labels: ScoreLabels) raises -> PythonObject:
    var b = labels_builtins()
    return py_call(b.set, labels.classes if labels.encoded else labels.values)


def _codes(labels: ScoreLabels, order: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_metrics._codes`: each label's index in `order` (-1 when
    absent) as an int32 Array; encoded labels map their class table once and
    gather per row (`gather_i32`)."""
    var index = Python.dict()
    for i in range(len(order)):  # driver-loop: over the distinct class labels
        py_setitem(index, order[i], PythonObject(i))
    var out = admitted_vector(labels.n, "<i4", context)
    if labels.encoded:
        var k = len(labels.classes)
        var table = admitted_vector(k, "<i4", context)
        var t = _I32P(unsafe_from_address=labels_address(table))
        _ = table  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct class labels
            t[i] = Int32(Int(py=py_get(index, labels.classes[i], PythonObject(-1))))
            _ = table  # keep the Python object alive through the native call above
        labels_gather(context, LABELS_GATHER_I32, labels_address(table), k, labels_address(labels.codes), labels.n,
                      labels_address(out))
        _ = table  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        return out
    # One borrowed read and one C dict probe per row (p3-labels-metrics).
    p3_dict_codes(index, labels.values, labels_address(out), -1)
    _ = out  # keep the Python object alive through the native writes above
    return out


def _refuse_multilabel(y: PythonObject, name: String, caller: String, context: PythonObject) raises:
    var shape = labels_shape_of(y, context)
    if len(shape) == 2 and shape[1] != 1:
        raise Error("NotImplementedError: mojolearn " + caller + ": " + name + " is 2-D; multilabel-indicator targets are NOT "
                    + "IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only binary and multiclass 1-D labels")


struct _Program(Movable):
    """The fields of an `_expansion_metrics._Prog` the native runner reads:
    its size, input records, stage records and skipped ranges."""
    var size: Int
    var inputs: PythonObject
    var stages: PythonObject
    var skip: PythonObject

    def __init__(out self) raises:
        self.size = 0
        self.inputs = Python.list()
        self.stages = Python.list()
        self.skip = Python.list()

    def alloc(mut self, n: Int) raises -> Int:
        var off = self.size
        self.size += max(n, 0)
        if self.size > _INT32_MAX:
            raise Error("mojolearn metrics: the problem exceeds the Int32 arena bound")
        return off

    def put(mut self, array: PythonObject, count: Int) raises -> Int:
        """`put` / `put_i32` of a C-order float32 or int32 Array."""
        var off = self.alloc(count)
        _ = py_call(self.inputs.append, Python.tuple(PythonObject(off), array))
        _ = py_call(self.skip.append, Python.tuple(PythonObject(off), PythonObject(off + count)))
        return off

    def scratch(mut self, n: Int) raises -> Int:
        var off = self.alloc(n)
        if n > 0:
            _ = py_call(self.skip.append, Python.tuple(PythonObject(off), PythonObject(off + n)))
        return off

    def stage(mut self, op: Int, total: Int, params: List[Int]) raises:
        if len(params) > _PARAMS:
            raise Error("x_metrics: too many stage parameters")
        var record = Python.list(PythonObject(op), PythonObject(total))
        for i in range(_PARAMS):  # small-loop(params): the fourteen fixed parameter words of one stage
            _ = py_call(record.append, PythonObject(params[i] if i < len(params) else 0))
        _ = py_call(self.stages.append, record)

    def group(mut self, key: Int, n: Int, m: Int, weights: Int) raises -> Tuple[Int, Int]:
        """`_expansion_metrics._group` (values=_NONE, vstride=1, width=1)."""
        var off = self.alloc(m + 1)
        var order = self.scratch(n)
        var out = self.alloc(m)
        self.stage(_OP_GROUP_SORT, 1, [key, n, m, off, order])
        if m:
            self.stage(_OP_GROUP_SUM, m, [off, order, _NONE, 1, weights, out, 1])
        return (off, out)

    def counts(mut self, off: Int, m: Int) raises -> Int:
        """`_expansion_metrics._counts`."""
        var out = self.alloc(m)
        if m:
            self.stage(_OP_OFF_DIFF, m, [off, m, out])
        return out


def _run(var program: _Program, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_run(prog, numeric_mode)`: the shared native program runner over a
    program object holding these fields. Returns the arena store."""
    var b = labels_builtins()
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, "mojolearn._expansion_metrics")
    if module is None:
        raise Error("ImportError: mojolearn: the metrics programs are not connected")
    var prog_context = py_get(py_fields(module), "_PROG_CONTEXT")
    if prog_context is None:
        raise Error("ImportError: mojolearn: the metrics programs are not connected")
    var prog = py_call(context["labels_namespace"])
    var fields = py_fields(prog)
    py_setitem(fields, "size", PythonObject(program.size))
    py_setitem(fields, "_inputs", program.inputs)
    py_setitem(fields, "_stages", program.stages)
    py_setitem(fields, "_skip", program.skip)
    py_setitem(fields, "_outs", PythonObject(None))
    py_setitem(fields, "arena", PythonObject(None))
    _ = _metrics_run(prog, mode, prog_context)
    return fields["arena"]


def _arena_address(arena: PythonObject) raises -> Int:
    return Int(py=py_call(arena.buffer_info)[0])


def labels_accuracy(y_true: PythonObject, prediction: PythonObject, sample_weight: PythonObject, mode: PythonObject,
                   context: PythonObject) raises -> Float64:
    """`_expansion_metrics.accuracy_fraction(y_true, prediction,
    sample_weight, mode)`: the (weighted) matched fraction from the device
    COUNT tail."""
    var b = labels_builtins()
    var caller = String("accuracy_score")
    _refuse_multilabel(y_true, "y_true", caller, context)
    _refuse_multilabel(prediction, "y_pred", caller, context)
    var true = _classification_encoded(y_true, "y_true", context)
    var pred = _classification_encoded(prediction, "y_pred", context)
    if true.n != pred.n:
        raise Error("mojolearn " + caller + ": y_true and y_pred lengths differ")
    if true.kind != pred.kind:
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if true.n > _INT32_MAX:
        raise Error("mojolearn " + caller + ": at most INT32_MAX rows")
    var n = true.n
    var weighted = sample_weight is not None
    var w: Optional[AdmittedX] = None
    if weighted:
        w = labels_sample_weight_f32(sample_weight, n, caller, context)
    var present = py_call(b.sorted, py_call(_label_set(true).union, _label_set(pred)))
    var L = len(present)
    var yt = _codes(true, present, context)
    var yp = _codes(pred, present, context)
    return _accuracy_codes(yt, yp, n, L, w^, mode, context)


def _accuracy_codes(yt: PythonObject, yp: PythonObject, n: Int, L: Int, var w: Optional[AdmittedX],
                    mode: PythonObject, context: PythonObject) raises -> Float64:
    """The (weighted) matched fraction of two int32 code vectors over `L`
    labels (a code outside [0, L) never matches), from the device COUNT
    tail. The hit count (unweighted) and the soft binary64 weight sums over
    row blocks (weighted) do not depend on how the labels are numbered, so
    any numbering of the same labels gives the same bits (x_metrics
    cm_epi.mojo `_count`, group.mojo `pair_key_unit`)."""
    var weighted = False
    if w:
        weighted = True
    # `_Sums(true, pred, w, present, mode, tail=_cm_stage(..., COUNT, 11))`.
    var prog = _Program()
    var a = prog.put(yt, n)
    var bb = prog.put(yp, n)
    var matched = prog.scratch(n)
    prog.stage(_OP_PAIR_KEY, n, [a, bb, matched, L, 1])
    var W = _NONE
    var w_array = PythonObject(None)
    if weighted:
        w_array = w.value().array if py_type(w.value().array) is context["array_class"] else _owned_f32(w.value(), context)
        W = prog.put(w_array, n)
    var g0 = prog.group(matched, n, L, W)
    var g1 = prog.group(a, n, L, W)
    var g2 = prog.group(bb, n, L, W)
    if weighted:
        # every row in group 0: the total weight grouping
        var zero = prog.scratch(n)
        prog.stage(_OP_PAIR_KEY, n, [a, a, zero, 1, 2])
        _ = prog.group(zero, n, 1, W)
    if not weighted:
        _ = prog.counts(g0[0], L)
        _ = prog.counts(g1[0], L)
        _ = prog.counts(g2[0], L)
    # `_cm_stage(prog, groups, w, n, L, _CM_COUNT, 11)`.
    var out = prog.alloc(11)
    var tp = g0[1] if weighted else g0[0]
    var tot = n
    if weighted:
        # `_wsum_sf64(prog, L)`: the soft binary64 hit and weight total.
        var nb = max(1, (n + _WSUM_BLOCK - 1) // _WSUM_BLOCK)
        var s = prog.scratch(4 * nb)
        prog.stage(_OP_CM_EPI, nb, [_CM_WSUM_LEAF, matched, W, n, s, L])
        var stride = 1
        while stride < nb:  # driver-loop: over the fold levels (log2 of the block count)
            prog.stage(_OP_CM_EPI, (nb - stride + 2 * stride - 1) // (2 * stride), [_CM_WSUM_FOLD, s, stride, nb])
            stride *= 2
        tot = s
    prog.stage(_OP_CM_EPI, 1, [_CM_COUNT, tp, L, 1 if weighted else 0, tot, out])
    var arena = _run(prog^, mode, context)
    var base = _arena_address(arena)
    _ = arena  # keep the Python object alive through the native call above
    _ = yt  # the code vectors stay alive through the program run above
    _ = yp
    _ = w^
    _ = w_array^
    if weighted and Int(_I32P(unsafe_from_address=base + 4 * out)[0]) != 0:
        raise Error("ZeroDivisionError: float division by zero")
    _ = arena  # keep the Python object alive through the native calls above
    var fraction = _F64P(unsafe_from_address=base + 4 * (out + 8))[0]
    _ = arena  # keep the Python object alive through the native call above
    _ = arena^
    return fraction


# ------------------------------------------- union encode, score from codes
# Lane ne1-h-labels (plan NATIVE_ENTRY_POINTS_AFTER_0866 section H): the
# metric doors encode both label vectors in ONE native call, and a
# classifier's score compares its predicted codes with y encoded against
# `classes_` (no decode of the prediction to label objects and no second
# encode of it). The metric kernels and their inputs per row are the ones
# `labels_accuracy` runs: no bit changes.


def _encoded_side(labels: ScoreLabels, context: PythonObject) raises -> ClassicalLabels:
    """One side of a metric pair as (its sorted distinct labels, one int32
    code per row into them): encoded labels already are; a value list maps
    through the sorted set of its values (one C dict probe per row)."""
    if labels.encoded:
        return ClassicalLabels(labels.classes, labels.codes, labels.n)
    var b = labels_builtins()
    var classes = py_call(b.sorted, py_call(b.set, labels.values))
    return ClassicalLabels(classes, _codes(labels, classes, context), labels.n)


def labels_union_encode(y_true: PythonObject, y_pred: PythonObject, caller: String,
                        context: PythonObject) raises -> PythonObject:
    """`_expansion_metrics._pair` without the weights, in one native call:
    the multilabel refusals, both sides read as str or integer labels
    (`_classification_encoded`), the length, kind and INT32 checks in that
    order and with that wording, then `(true_classes, true_codes,
    pred_classes, pred_codes, kind, present)`: each side's sorted distinct
    labels and int32 codes into them, the kind ('string' or 'integer') and
    the sorted union of the two label sets."""
    var b = labels_builtins()
    _refuse_multilabel(y_true, "y_true", caller, context)
    _refuse_multilabel(y_pred, "y_pred", caller, context)
    var true = _classification_encoded(y_true, "y_true", context)
    var pred = _classification_encoded(y_pred, "y_pred", context)
    if true.n != pred.n:
        raise Error("ValueError: mojolearn " + caller + ": y_true and y_pred lengths differ")
    if true.kind != pred.kind:
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if true.n > _INT32_MAX:
        raise Error("ValueError: mojolearn " + caller + ": at most INT32_MAX rows")
    var present = py_call(b.sorted, py_call(_label_set(true).union, _label_set(pred)))
    var t = _encoded_side(true, context)
    var p = _encoded_side(pred, context)
    return Python.tuple(t.classes, t.codes, p.classes, p.codes, PythonObject(true.kind), present)


def _classes_score_kind(classes: PythonObject, context: PythonObject) raises -> String:
    """'string' when every class is exactly str (or NumPy str_), 'integer'
    when every class is an exact integer scalar (bool excluded): the kind
    the decoded prediction would have been read as. '' for anything else
    (bool, float or object classes, an empty table), which keeps the decode
    route."""
    var b = labels_builtins()
    var cls = py_type(classes)
    if cls is not b.list and cls is not b.tuple:
        return String("")
    var k = len(classes)
    if k == 0:
        return String("")
    var is_tuple = cls is b.tuple
    var all_str = True
    var all_int = True
    var kinds = LabelTypeKinds()
    for i in range(k):  # driver-loop: over the k fitted classes
        var kind = kinds.admit_raw(p3_item(classes, i, is_tuple), context)
        all_str = all_str and kind == 1
        all_int = all_int and kind == 3
    _ = kinds^
    _ = classes  # keep the container alive through the borrowed item reads above
    if all_str:
        return String("string")
    if all_int:
        return String("integer")
    return String("")


def _codes_in_range(codes: PythonObject, code_dtype: String, n: Int, k: Int,
                    context: PythonObject) raises -> Bool:
    """Every code in [0, k): the base binding's `reduce_stat` min and max
    over the code vector (a device tile reduction on a GPU install, the core
    host helper on a CPU-only one); no per-code Python or host loop here."""
    if n == 0:
        return True
    var address = labels_address(codes)
    var native = _native_code(code_dtype)
    var low = Int(py=labels_reduce_stat(context, address, native, n, 0))
    var high = Int(py=labels_reduce_stat(context, address, native, n, 1))
    _ = codes  # keep the Python object alive through the native calls above
    return low >= 0 and high < k


def labels_score_from_codes(y_true: PythonObject, var classes: PythonObject, codes: PythonObject,
                            sample_weight: PythonObject, mode: PythonObject,
                            context: PythonObject) raises -> Float64:
    """`accuracy_fraction(y_true, classes[codes], sample_weight, mode)` from
    a classifier's predicted codes (an int32 or int64 vector Array into its
    fitted `classes`): y is encoded once against `classes` (a label not in
    it never matches) and the predicted codes go to the COUNT program as
    they are. Same refusals in the same order as the decode route; classes
    that are neither all str nor all integers, and any code outside
    [0, len(classes)) (checked by one native min/max reduction), take that
    route unchanged, so a bad code raises as the decode raised."""
    classes = classifier_label_list(classes, context)
    var kind = _classes_score_kind(classes, context)
    classical_admit_value(codes, context)
    var code_dtype = String(codes.dtype)
    if kind == "" or Int(py=codes.ndim) != 1 or (code_dtype != "<i4" and code_dtype != "<i8"):
        return labels_accuracy(y_true, labels_decode_labels(classes, codes, context), sample_weight, mode, context)
    var n = Int(py=codes.size)
    if not _codes_in_range(codes, code_dtype, n, len(classes), context):
        # The decode route's contract for a code outside [0, len(classes)):
        # the decode raises (IndexError) before y is read, exactly as
        # before; a bad code is never scored as a miss.
        return labels_accuracy(y_true, labels_decode_labels(classes, codes, context), sample_weight, mode, context)
    var caller = String("accuracy_score")
    _refuse_multilabel(y_true, "y_true", caller, context)
    var true = _classification_encoded(y_true, "y_true", context)
    if true.n != n:
        raise Error("mojolearn " + caller + ": y_true and y_pred lengths differ")
    if true.kind != kind:
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if n > _INT32_MAX:
        raise Error("mojolearn " + caller + ": at most INT32_MAX rows")
    var w: Optional[AdmittedX] = None
    if sample_weight is not None:
        w = labels_sample_weight_f32(sample_weight, n, caller, context)
    var yt = _codes(true, classes, context)
    var yp = codes
    if code_dtype == "<i8":
        yp = admitted_vector(n, "<i4", context)
        labels_cast(labels_address(codes), "<i8", labels_address(yp), "<i4", n, context)
        _ = codes  # keep the Python object alive through the native call above
        _ = yp  # keep the Python object alive through the native call above
    return _accuracy_codes(yt, yp, n, len(classes), w^, mode, context)


def _owned_f32(w: AdmittedX, context: PythonObject) raises -> PythonObject:
    """A float32 Array copy of weights that borrow a foreign buffer (the
    program runner copies Arrays into its arena; `put` took `as_f32_c`'s
    Array). Copy the already-admitted contiguous float32 words natively."""
    var count = w.rows * w.cols
    var out = admitted_vector(count, "<f4", context)
    # `array` owns the admitted storage; it need not itself export bytes.
    memcpy(dest=_F32P(unsafe_from_address=labels_address(out)),
           src=_F32P(unsafe_from_address=w.address), count=count)
    _ = w  # retain the source export until the copy finishes
    return out




# ------------------------------------------------------------ public API
# The names the classical owners call.


# TOMBSTONE: labels_encode_labels (a second name of the one label encoder) deleted 2026-11-08 by lane a2-admit-family; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A2_FAMILY_ADMISSION_HELPERS.patch. Its body is
# classical_encode_labels above.


def classical_decode_labels(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for an int32 or int64 codes
    Array."""
    return labels_decode_labels(classes, codes, context)


def classical_union_encode(y_true: PythonObject, y_pred: PythonObject, caller: String,
                           context: PythonObject) raises -> PythonObject:
    """`labels_union_encode`: both metric label vectors in one call."""
    return labels_union_encode(y_true, y_pred, caller, context)


def classical_score_from_codes(y_true: PythonObject, classes: PythonObject, codes: PythonObject,
                               sample_weight: PythonObject, mode: PythonObject,
                               context: PythonObject) raises -> Float64:
    """`labels_score_from_codes`: a classifier's accuracy from its
    predicted codes and fitted classes."""
    return labels_score_from_codes(y_true, classes, codes, sample_weight, mode, context)


def classical_accuracy(y_true: PythonObject, prediction: PythonObject, sample_weight: PythonObject,
                       mode: PythonObject, context: PythonObject) raises -> Float64:
    """`_expansion_metrics.accuracy_fraction(y_true, prediction,
    sample_weight, mode)`."""
    return labels_accuracy(y_true, prediction, sample_weight, mode, context)
