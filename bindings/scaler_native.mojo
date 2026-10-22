# SPDX-License-Identifier: Apache-2.0
# StandardScaler.fit owns admission, buffers, route selection, launches and
# fitted-state publication here. The existing statistics kernels are unchanged.
from std.memory import memcpy, bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.os import getenv
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_arrays import classical_input, classical_address, classical_typed, classical_is_array, classical_is_f32, classical_cast_kind, classical_c_order
from core.scaler_fit_impl import scaler_cast, scaler_all_finite, scaler_check_stats, scaler_fill, scaler_check_missing, scaler_counts
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, native_pandas_f32
from bindings.sequential_admission import _ordinary_torch, _one_of
from bindings.native_torch_admission import native_torch_numpy

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


def _ptr(value: PythonObject) raises -> F32P:
    return F32P(unsafe_from_address=Int(py=classical_address(value)))


def _input(source: PythonObject, ndim: Int, strict: Bool, context: PythonObject) raises -> PythonObject:
    classical_admit_value(source, context)
    var b = Python.import_module("builtins")
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


def _stage(mut program: List[Int32], op: Int, total: Int, *params: Int):
    program.append(Int32(op))
    program.append(Int32(total))
    for i in range(len(params)):  # small-loop(params: at most fourteen native stage arguments)
        program.append(Int32(params[i]))
    for i in range(14 - len(params)):  # small-loop(params: pad the fixed fourteen-word stage ABI)
        program.append(0)


def _copy_row(source: F32P, count: Int, context: PythonObject) raises -> PythonObject:
    var out = classical_typed(Python.tuple(count), "<f4", context["output_context"])
    memcpy(dest=_ptr(out), src=source, count=count)
    return out


def _bool(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    if py_type(value) is not b.bool:
        var numpy = py_get(context, "bool_types")
        if numpy is None or py_type(value) is not numpy[1]:
            raise Error("copy, with_mean and with_std must be bools")
    return Bool(py=value)


def scaler_fit(model: PythonObject, source: PythonObject, weight_source: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
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
    if Int(py=py_call_typed(entries["preprocessing_numeric_mode"])) != expected or Int(py=py_call_typed(entries["x_prep_numeric_mode"])) != expected:
        raise Error("RuntimeError: Scaler native numeric mode disagrees with requested mode")
    var x = _input(source, 2, True, context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var xp = _ptr(x)
    var output = classical_typed(Python.tuple(3, d), "<f4", context["output_context"])
    var stats = _ptr(output)
    var finite = True
    var weighted = weight_source is not None
    var direct = py_get(entries, "standard_fit_direct")
    var direct_switch = getenv("MOJOLEARN_SCALER_DIRECT", "1")
    if not weighted and direct is not None and String(direct_switch.strip()) != "0":
        finite = Int(py=py_call(direct, classical_address(x), classical_address(output), Python.list(n, d, Int(with_mean), Int(with_std)))) != 0
    else:
        finite = scaler_all_finite(xp, n*d)
        if finite and not weighted:
            _ = py_call_typed(entries["standard_fit"], classical_address(x), classical_address(output), Python.list(n, d, Int(with_mean), Int(with_std)))
    var seen = PythonObject(n)
    var mean = PythonObject(None)
    var variance = PythonObject(None)
    var scale = PythonObject(None)
    if finite and not weighted:
        scaler_check_stats(stats, stats+d, stats+2*d, d)
        mean = _copy_row(stats, d, context)
        variance = _copy_row(stats + d, d, context)
        scale = _copy_row(stats + 2*d, d, context)
    else:
        # Identical opcodes, stage order and arena words to _standard_stats.
        # These stages call the existing CPU/GPU kernels, never a host math
        # replacement. The input and all buffers stay alive through the call.
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
        var wo = n*d if weighted else -1
        var st = n*d + (n if weighted else 0)
        var out = st + 6*d
        var sc = out + 3*d
        var ch = max(4096, (n + 4095) // 4096)
        var nch = (n + ch - 1) // ch
        var wc = sc + d
        var size = wc + (nch + 1 if weighted else 0)
        if size > 2147483647:
            raise Error("x_prep: the program exceeds the native Int32 indexing bound")
        var arena = List[Float32](length=size, fill=0)
        var arena_ptr = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
        memcpy(dest=arena.unsafe_ptr(), src=xp, count=n*d)
        if weighted:
            memcpy(dest=arena.unsafe_ptr() + wo, src=_ptr(weight), count=n)
        var program = List[Int32]()
        _stage(program, 1, d, 0, n, d, st)
        _stage(program, 101, d, 0, n, d, wo, out)
        _stage(program, 102, d, out + 2*d, sc)
        if weighted:
            _stage(program, 270, nch, wo, n, ch, wc)
            _stage(program, 248, 1, wc, nch, wc+nch)
        _ = py_call_typed(entries["x_prep_run"], Int(arena.unsafe_ptr()), size, Int(program.unsafe_ptr()), 5 if weighted else 3)
        # Integer addresses do not carry ownership across the C ABI. Keep
        # the instruction storage alive until the synchronous executor returns.
        _ = program^
        if weighted and bitcast[DType.int32](arena[wc+nch]) != 0:
            raise Error("sample_weight must be finite")
        var all_same = scaler_check_missing(arena_ptr, st, out, sc, d)
        mean = _copy_row(arena_ptr+out+d, d, context)
        variance = _copy_row(arena_ptr+out+2*d, d, context)
        scale = _copy_row(arena_ptr+sc, d, context)
        if all_same:
            seen = PythonObject(Float64(arena[out])) if weighted else PythonObject(Int(arena[out]))
        elif weighted:
            seen = _copy_row(arena_ptr+out, d, context)
        else:
            seen = classical_typed(Python.tuple(d), "<i8", context["output_context"])
            var counts = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(seen)))
            scaler_counts(arena_ptr+out, counts, d)
        _ = arena^
        _ = weight^
    py_setitem(fields, "mean_", mean if with_mean or with_std else PythonObject(None))
    py_setitem(fields, "var_", variance if with_std else PythonObject(None))
    py_setitem(fields, "scale_", scale if with_std else PythonObject(None))
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "n_samples_seen_", seen)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "with_mean_", with_mean)
    py_setitem(fields, "with_std_", with_std)
    _ = x^
    _ = output^
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
