# SPDX-License-Identifier: Apache-2.0
# StandardScaler.fit owns admission, buffers, route selection, launches and
# fitted-state publication here. The existing statistics kernels are unchanged.
from std.memory import unsafe_memcpy
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.os import getenv
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes_door import classical_connect_numpy, classical_check_layout
from bindings.classical_arrays_door import classical_input, classical_address, classical_typed, classical_is_array, classical_is_f32, classical_cast_kind, classical_c_order
from providers.python.preprocessing_input_pyapi_door import ScalerInputSnapshot, scaler_operation_input
from core.scaler_fit_impl import scaler_cast, scaler_fill, scaler_counts
from std.python._cpython import GILReleased
from core.entry_table import entry_ref_call, entry_ref_unresolved
from core.prep_entry_table import PrepEntryKey, scaler_ref, xp_transport_ref
from core.scaler_native_api import (
    ScalerFitRequest, ScalerTransformRequest, MinMaxFitRequest, MinMaxTransformRequest, SCALER_OP_DIRECT,
)
from core.scaler_core import (
    ScalerCalls, ScalerTableCalls, ScalerStandardFit, scaler_standard_fit_core,
    SCALER_SEEN_INT, SCALER_SEEN_FLOAT, SCALER_SEEN_INT_ROW, SCALER_SEEN_FLOAT_ROW,
)
from x_prep.transport_abi import xp_vendor_word
from bindings.prep_entry_cache_unpack import prep_entry_word, prep_entry_text
from bindings.classical_pandas_door import classical_admit_value, native_pandas_exact, native_pandas_f32
from bindings.sequential_admission_door import _ordinary_torch, _one_of
from bindings.native_torch_admission_door import native_torch_numpy
from bindings.prep_entry_cache_unpack import prep_mode_checked

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


def _ptr(value: PythonObject) raises -> F32P:
    return F32P(unsafe_from_address=Int(py=classical_address(value)))


def _input(source: PythonObject, ndim: Int, strict: Bool, context: PythonObject) raises -> PythonObject:
    classical_admit_value(source, context)
    var b = native_loaded_module("builtins")
    var prepared = source
    if native_pandas_exact(source):
        prepared = native_pandas_f32(source, context, strict, True)
    elif _ordinary_torch(py_type(source), context):
        prepared = native_torch_numpy(source)
    var value = classical_input(prepared, context)
    var rank = Int(py=value.ndim)
    var sequence = py_type(source) is b.list or py_type(source) is b.tuple or py_type(source) is b.range
    if strict and not sequence and not classical_is_f32(value, context):
        # A nonnative float32 buffer normalizes byte order, as _buffer did.
        if classical_is_array(value, context) or String(value.dtype.kind) != "f" or Int(py=value.itemsize) != 4:
            raise Error("TypeError: Scaler input must have dtype float32")
    if rank != ndim:
        raise Error("Scaler requires a nonempty two-dimensional input" if strict else "sample_weight must be one-dimensional")
    var rows = 1 if rank == 0 else Int(py=value.shape[0])
    var cols = Int(py=value.shape[1]) if rank == 2 else 1
    if rows < 1 or cols < 1:
        raise Error("Scaler requires a nonempty two-dimensional input")
    if rows > 2147483647 // cols:
        raise Error("Scaler exceeds the native Int32 indexing bound")
    var native = classical_is_array(value, context) or Bool(py=value.dtype.isnative)
    if classical_is_f32(value, context) and classical_c_order(value, context) and native:
        return value
    var output = classical_typed(Python.tuple(rows, cols) if rank == 2 else Python.tuple(rows), "<f4", context["output_context"])
    var dest = _ptr(output)
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "input", context)
    var addr = Int(py=classical_address(value))
    var s0 = Int(py=value.strides[0]) if rank else 0
    var s1 = Int(py=value.strides[1]) if rank == 2 else 0
    scaler_cast(addr, s0, s1, rows, cols, item, kind, native, dest)
    _ = value^
    _ = prepared^
    return output


def _copy_row(source: F32P, count: Int, context: PythonObject) raises -> PythonObject:
    var out = classical_typed(Python.tuple(count), "<f4", context["output_context"])
    unsafe_memcpy(dest=_ptr(out), src=source, count=count)
    return out


# ---------------------------------------------------------------- the typed core's calls (UNPACK)

struct ScalerDoorCalls(ScalerCalls, Movable):
    """The scaler doors' `ScalerCalls` (lane scaler-typed-core): the scaler
    kernel entry and the x_prep transport entry of the connected images,
    resolved once per operation from the prep entry table, each call with the
    GIL released as the Python entries released it. The one exception is the
    explicit host transform (HostStandardScaler / HostMinMaxScaler): its
    kernel is served by the shipped `_mojolearn_estimators_host`, which
    publishes no scaler table, so its two transform entries are still called
    as compiled Python entries (`table_kernels` False). Its x_prep programs
    run through the table like every other door."""

    var table: ScalerTableCalls
    var has_table: Bool
    var entries: PythonObject

    def __init__(out self, entries: PythonObject, table_kernels: Bool) raises:
        var mode = prep_entry_word(entries, "x_prep_numeric_mode", -1)
        var vendor = xp_vendor_word(prep_entry_text(entries, "x_prep_vendor"))
        if mode < 0 or mode > 1 or vendor < 0:
            raise Error("ImportError: mojolearn: the x_prep binding publishes no native entry table")
        var key = PrepEntryKey(mode, vendor)
        var scaler = entry_ref_unresolved()
        var direct = False
        if table_kernels:
            scaler = scaler_ref(key)
            var a: Array[Int, 0] = []
            var p: Array[Int, 0] = []
            var f: Array[Float64, 0] = []
            direct = (String(getenv("MOJOLEARN_SCALER_DIRECT", "1").strip()) != "0"
                      and entry_ref_call(scaler, SCALER_OP_DIRECT, a, p, f) == 1)
        self.table = ScalerTableCalls(scaler, xp_transport_ref(key), direct)
        self.has_table = table_kernels
        self.entries = entries

    def direct(self) -> Bool:
        return self.table.use_direct

    def standard_fit(self, r: ScalerFitRequest) raises -> Int:
        var got = 0
        with GILReleased(Python()):
            got = self.table.standard_fit(r)
        return got

    def standard_transform(self, r: ScalerTransformRequest) raises -> Int:
        if self.has_table:
            var got = 0
            with GILReleased(Python()):
                got = self.table.standard_transform(r)
            return got
        _ = py_call_typed(py_get(self.entries, "standard_transform"), r.source, r.mean, r.scale, r.dst,
                          Python.list(r.rows, r.features, Int(r.inverse), Int(r.with_mean), Int(r.with_std)))
        return 1

    def minmax_fit(self, r: MinMaxFitRequest) raises -> Int:
        var got = 0
        with GILReleased(Python()):
            got = self.table.minmax_fit(r)
        return got

    def minmax_transform(self, r: MinMaxTransformRequest) raises -> Int:
        if self.has_table:
            var got = 0
            with GILReleased(Python()):
                got = self.table.minmax_transform(r)
            return got
        _ = py_call_typed(py_get(self.entries, "minmax_transform"), r.source, r.scale, r.offset, r.dst,
                          Python.list(r.rows, r.features, Int(r.inverse), Int(r.clip), Float64(r.lower), Float64(r.upper)))
        return 1

    def run(self, arena: Int, words: Int, program: Int, stages: Int) raises:
        with GILReleased(Python()):
            self.table.run(arena, words, program, stages)


def scaler_door_calls(entries: PythonObject) raises -> ScalerDoorCalls:
    """UNPACK: the calls of a door whose kernels are all table entries."""
    return ScalerDoorCalls(entries, True)


def scaler_transform_calls(entries: PythonObject, context: PythonObject) raises -> ScalerDoorCalls:
    """UNPACK: a transform's calls; the explicit host context keeps its
    estimators-host transform entry (ScalerDoorCalls)."""
    return ScalerDoorCalls(entries, not Bool(py=py_get(context, "scaler_explicit_host", False)))


def scaler_standard_publish(fields: PythonObject, fit: ScalerStandardFit, d: Int, with_mean: Bool, with_std: Bool,
                            mode: PythonObject, context: PythonObject) raises:
    """PACK: a StandardScaler fit's fitted fields, one Array per row."""
    var stats = F32P(unsafe_from_address=Int(fit.stats.unsafe_ptr()))
    var seen = PythonObject(None)
    if fit.seen_kind == SCALER_SEEN_INT:
        seen = PythonObject(Int(fit.seen_scalar))
    elif fit.seen_kind == SCALER_SEEN_FLOAT:
        seen = PythonObject(fit.seen_scalar)
    elif fit.seen_kind == SCALER_SEEN_FLOAT_ROW:
        seen = _copy_row(F32P(unsafe_from_address=Int(fit.seen.unsafe_ptr())), d, context)
    else:
        seen = classical_typed(Python.tuple(d), "<i8", context["output_context"])
        var counts = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(seen)))
        scaler_counts(F32P(unsafe_from_address=Int(fit.seen.unsafe_ptr())), counts, d)
        _ = seen  # keep the Python object alive through the native call above
    py_setitem(fields, "mean_", _copy_row(stats, d, context) if with_mean or with_std else PythonObject(None))
    py_setitem(fields, "var_", _copy_row(stats + d, d, context) if with_std else PythonObject(None))
    py_setitem(fields, "scale_", _copy_row(stats + 2*d, d, context) if with_std else PythonObject(None))
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "n_samples_seen_", seen)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "with_mean_", with_mean)
    py_setitem(fields, "with_std_", with_std)


def _bool(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.bool:
        var numpy = py_get(context, "bool_types")
        if numpy is None or py_type(value) is not numpy[1]:
            raise Error("copy, with_mean and with_std must be bools")
    return Bool(py=value)


def scaler_fit(model: PythonObject, source: PythonObject, weight_source: PythonObject, context: PythonObject) raises -> PythonObject:
    return scaler_fit(model, source, weight_source, context, PythonObject(None))


def scaler_fit(model: PythonObject, source: PythonObject, weight_source: PythonObject, context: PythonObject,
               input_cache: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    _ = _bool(fields["copy"], context)
    var with_mean = _bool(fields["with_mean"], context)
    var with_std = _bool(fields["with_std"], context)
    var mode = fields["numeric_mode"]
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "fast" and String(mode) != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    if Bool(py=py_get(context,"scaler_explicit_host",False)) and String(mode) != "identical":
        raise Error("HostStandardScaler runs identical only on the host")
    # Preserve refit's invalidation semantics, including failure after a valid
    # configuration. Use only the native dict operations, never model hooks.
    var names = py_call(b.list, fields)
    for i in range(len(names)):  # driver-loop: over the fitted attribute names being invalidated
        if String(names[i]).endswith("_"):
            _ = py_call(fields.pop, names[i])
    var entries = context["scaler_bindings"][mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    if py_get(entries,"standard_fit") is None:
        raise Error("ImportError: HostStandardScaler training requires _mojolearn_preprocessing_host: " + String(py_get(entries,"scaler_training_error","standard_fit is unavailable")))
    var expected = 1 if String(mode) == "identical" else 0
    # The two mode probes run once per binding (prep_entry_cache).
    if not prep_mode_checked(entries, expected):
        raise Error("RuntimeError: Scaler native numeric mode disagrees with requested mode")
    var snapshot = scaler_operation_input(source, context, input_cache)
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    var x = captured[].values
    var n = captured[].metadata.source.rows
    var d = captured[].metadata.source.cols
    var calls = scaler_door_calls(entries)
    var weighted = weight_source is not None
    var weight = PythonObject(None)
    if weighted:
        classical_admit_value(weight_source, context)
        var raw = classical_input(weight_source, context, "sample_weight")
        var scalar_rank = Int(py=raw.ndim) == 0
        var weight_type = py_type(weight_source)
        if weight_type is b.bool:
            raise Error("sample_weight must be one-dimensional")
        var scalar = weight_type is b.int or weight_type is b.float
        var scalar_types = py_get(context, "scalar_types")
        var bool_types = py_get(context, "bool_types")
        if scalar_types is not None and bool_types is not None:
            scalar = scalar or (_one_of(weight_type, scalar_types) and weight_type is not bool_types[1])
        # Real numeric scalars broadcast. A zero-dimensional buffer is
        # still a one-element weight vector, matching the existing API.
        weight = _input(weight_source, 0 if scalar_rank else 1, False, context)
        if scalar:
            # The raw pointer does not retain its Python buffer owner.
            # Keep the scalar buffer alive while allocating and filling
            # its broadcast replacement.
            var scalar_weight = weight
            var first = _ptr(scalar_weight)[0]
            weight = classical_typed(Python.tuple(n), "<f4", context["output_context"])
            scaler_fill(_ptr(weight), n, first)
            _ = scalar_weight^
        elif Int(py=weight.size) != n:
            raise Error("sample_weight length differs from X")
    # RUN: the typed core (core/scaler_core.mojo) on the borrowed buffers.
    var fit = scaler_standard_fit_core(calls, _ptr(x), n, d, with_mean, with_std,
                                       Int(py=classical_address(weight)) if weighted else 0)
    _ = weight^
    _ = x^
    # PACK
    scaler_standard_publish(fields, fit, d, with_mean, with_std, mode, context)
    return model


def scaler_fit_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 5:
            raise Error("TypeError: StandardScaler.fit requires model, X, y, sample_weight and context")
        classical_check_layout(args[0], args[4])
        var context = args[4]
        var hosts = py_get(context, "scaler_host_contexts")
        var host = py_get(hosts, py_type(args[0])) if hosts is not None else PythonObject(None)
        if host is not None:
            context = host
        return scaler_fit(args[0], args[1], args[3], context).steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
