# SPDX-License-Identifier: Apache-2.0
# Operation-local Prophet input snapshots and native storage conversion.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_warnings import classical_warning
from bindings.sequential_admission import _one_of
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.classical_pandas import classical_admit_value, native_pandas_exact
from providers.python.resample_pandas_pyapi import resample_pandas_array
from bindings.sequential_arrays import sequential_array
from bindings.xseq_native_common import xseq_input, xseq_addr
from providers.python.array_metadata_pyapi import array_shape_once, array_metadata_once
from providers.python.prophet_lifecycle_pyapi import prophet_scalar
from core.cast_in import cast_in_kind
from core.prophet_input_marshal import prophet_numeric_copy, prophet_datetime_copy


def _admit(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: ProphetForecaster input nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        for i in range(len(value)):  # driver-loop: exact sequence leaves before inference or conversion
            _admit(value[i], context, depth + 1)
    elif cls is context["xseq_prophet_datetime_type"] or cls is context["xseq_prophet_timedelta_type"] or cls is b.bytes or cls is b.complex or _one_of(cls, context["xseq_prophet_extra_scalars"]):
        pass  # exact native scalar storage, never a subclass protocol
    else:
        classical_admit_value(value, context["xseq_lifecycle_context"])


def _leaves(value: PythonObject, depth: Int, mut dims: List[Int], mut leaf_depth: Int,
            leaves: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        var count = len(value)
        if depth >= 32:
            raise Error("ValueError: ProphetForecaster input has too many dimensions")
        if depth == len(dims):
            dims.append(count)
        elif dims[depth] != count:
            raise Error("ValueError: ProphetForecaster input sequences must be rectangular")
        for i in range(count):  # driver-loop: native nested input shape and passive leaves
            _leaves(value[i], depth + 1, dims, leaf_depth, leaves)
    else:
        if leaf_depth >= 0 and leaf_depth != depth:
            raise Error("ValueError: ProphetForecaster input sequences must be rectangular")
        leaf_depth = depth
        _ = py_call(leaves.append, value)


def _object_source(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var leaves = Python.list()
    var dims = List[Int]()
    var leaf_depth = -1
    _leaves(value, 0, dims, leaf_depth, leaves)
    var shape = Python.list()
    for size in dims:  # driver-loop: captured input shape into native ndarray constructor metadata
        _ = py_call(shape.append, size)
    var all_dates = len(leaves) > 0
    var all_deltas = len(leaves) > 0
    var any_complex = False
    var all_numeric = True
    var b = native_loaded_module("builtins")
    for i in range(len(leaves)):  # driver-loop: infer a datetime64 sequence without NumPy asarray
        var cls = py_type(leaves[i])
        all_dates = all_dates and cls is context["xseq_prophet_datetime_type"]
        all_deltas = all_deltas and cls is context["xseq_prophet_timedelta_type"]
        any_complex = any_complex or cls is b.complex or _one_of(cls, context["xseq_prophet_complex_scalars"])
        all_numeric = all_numeric and (cls is b.complex or cls is b.float or cls is b.int or cls is b.bool
            or _one_of(cls, context["xseq_lifecycle_context"]["scalar_types"])
            or _one_of(cls, context["xseq_prophet_extended_scalars"]))
    var shape_tuple = py_call(b.tuple, shape)
    if all_dates or all_deltas:
        var dtype = leaves[0].dtype
        for i in range(1, len(leaves)):  # driver-loop: common native temporal dtype metadata, no array conversion
            dtype = py_call(context["xseq_prophet_promote_types"], dtype, leaves[i].dtype)
        var out = sequential_array(shape_tuple, context["int64"], context)
        var destination = Int(py=xseq_addr(out))
        for i in range(len(leaves)):  # driver-loop: native temporal scalar storage converted to its common input unit
            var source = py_call(leaves[i].reshape, Python.tuple())
            prophet_temporal_into(source, dtype, destination + 8 * i, True, context)
        return py_call(out.view, dtype)
    if any_complex and all_numeric:
        var out = sequential_array(shape_tuple, context["xseq_prophet_complex_dtype"], context)
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=xseq_addr(out)))
        for i in range(len(leaves)):  # driver-loop: native complex scalar storage, copied without NumPy conversion hooks
            var value = leaves[i]
            var cls = py_type(value)
            if cls is b.complex:
                words[2 * i] = Float64(py=value.real)
                words[2 * i + 1] = Float64(py=value.imag)
            elif _one_of(cls, context["xseq_prophet_extended_scalars"]):
                var scalar = native_scalar_buffer(value)
                words[2 * i] = Float64(py=scalar.to_python_number(False))
                if scalar.complex:
                    var imag = scalar
                    imag.real = scalar.imag
                    words[2 * i + 1] = Float64(py=imag.to_python_number(False))
            else:
                words[2 * i] = Float64(py=py_call(b.float, value))
        return out
    var out = py_call(context["ndarray"], shape_tuple, dtype=context["xseq_prophet_object_dtype"])
    var flat = out.flat
    for i in range(len(leaves)):  # driver-loop: native object storage receives already admitted scalar leaves
        flat[i] = leaves[i]
    return out


def prophet_source(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var lifecycle = context["xseq_lifecycle_context"]
    _admit(value, context)
    var source = value
    if native_pandas_exact(value):
        source = resample_pandas_array(value, lifecycle)
    var array = PythonObject(None)
    try:
        array = xseq_input(source, context)
    except error:
        var b = native_loaded_module("builtins")
        var cls = py_type(source)
        if cls is not b.list and cls is not b.tuple and cls is not b.str and cls is not b.bytes and cls is not b.complex and cls is not context["xseq_prophet_datetime_type"] and cls is not context["xseq_prophet_timedelta_type"] and not _one_of(cls, context["xseq_prophet_extra_scalars"]):
            raise error
        array = _object_source(source, context)
    # Exact ndarray's C view captures shape/dtype/strides and retains its
    # original exporter independently of later caller metadata mutation.
    return py_call(array.view)


def _numbers(value: PythonObject, result: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: ProphetForecaster object payload nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        for i in range(len(value)):  # driver-loop: passive object/string-array leaves before native scalar conversion
            _numbers(value[i], result, context, depth + 1)
    else:
        var cls = py_type(value)
        if _one_of(cls, context["xseq_prophet_extended_scalars"]):
            var scalar = native_scalar_buffer(value)
            if scalar.complex:
                classical_warning("Casting complex values to real discards the imaginary part",
                    context["xseq_prophet_complex_warning"], context["xseq_lifecycle_context"], PythonObject(None))
            _ = py_call(result.append, scalar.to_python_number(False))
        else:
            if cls is not b.bytes and cls is not context["xseq_prophet_datetime_type"] and cls is not context["xseq_prophet_timedelta_type"] and not _one_of(cls, context["xseq_prophet_extra_scalars"]):
                prophet_scalar(value, context)
            _ = py_call(result.append, py_call(b.float, value))


def prophet_contiguous(source: PythonObject, wide: Bool, context: PythonObject) raises -> PythonObject:
    var lifecycle = context["xseq_lifecycle_context"]
    var dtype = source.dtype
    var letter = String(dtype.kind)
    if letter == "M" or letter == "m":
        # Numeric casts of temporal arrays consume their signed tick words;
        # only the time-axis entry interprets datetime units as nanoseconds.
        var words = py_call(source.view, context["xseq_prophet_big_i64"] if String(dtype.byteorder) == ">" else context["int64"])
        return prophet_contiguous(words, wide, context)
    var shape = array_shape_once(source, lifecycle)
    var info = array_metadata_once(source, shape^, "ProphetForecaster input", lifecycle,
                                  allow_non_native=True, allow_unsupported=True)
    var out = sequential_array(source.shape, context["float64"] if wide else context["float32"], context)
    if info.count == 0:
        return out
    if letter == "O" or letter == "U" or letter == "S":
        var values = Python.list()
        # ndarray.tolist is a native storage reader. Its object payloads are
        # admitted before conversion, so no user __float__ can be invoked.
        _numbers(py_call(source.tolist), values, context)
        var numbers = xseq_input(values, context)
        var numbers_info = array_metadata_once(numbers, array_shape_once(numbers, lifecycle), "ProphetForecaster values", lifecycle)
        prophet_numeric_copy(numbers_info.source.addr, numbers_info.shape, numbers_info.strides,
            numbers_info.source.itemsize, numbers_info.source.kind, numbers_info.native_byte_order,
            Int(py=xseq_addr(out)), wide)
        _ = numbers_info
    else:
        var kind = -1 if info.dtype_number == 13 else (-2 if info.dtype_number == 14 else (-3 if info.dtype_number == 15 else -4))
        if info.dtype_number < 13 or info.dtype_number > 16:
            kind = cast_in_kind(info.dtype_number, info.source.itemsize, "ProphetForecaster input")
        if info.dtype_number >= 14 and info.dtype_number <= 16:
            classical_warning("Casting complex values to real discards the imaginary part",
                context["xseq_prophet_complex_warning"], lifecycle, PythonObject(None))
        prophet_numeric_copy(info.source.addr, info.shape, info.strides, info.source.itemsize,
            kind, info.native_byte_order, Int(py=xseq_addr(out)), wide)
    _ = info
    return out


def _time_unit(dtype: PythonObject) raises -> PythonObject:
    var text = String(dtype.str)
    var parts = text.split("[")
    var unit = String("generic")
    var multiplier = 1
    if len(parts) == 2:
        var tail = String(parts[1])
        var stop = tail.byte_length() - 1
        var start = 0
        while start < stop:  # driver-loop: native datetime dtype unit metadata
            var digit = String(tail[byte=start:start + 1])
            if digit < "0" or digit > "9":
                break
            start += 1
        if start > 0:
            multiplier = Int(String(tail[byte=0:start]))
        unit = String(tail[byte=start:stop])
    return Python.tuple(unit, multiplier)


def prophet_temporal_into(source: PythonObject, target_dtype: PythonObject, destination: Int,
                           common: Bool, context: PythonObject) raises:
    var dtype = source.dtype
    var little = String(dtype.byteorder) != ">"
    var unit = _time_unit(dtype)
    var target = _time_unit(target_dtype)
    # A dtype view reinterprets bytes only. Unit conversion arithmetic runs
    # in Mojo, with common-unit promotion kept separate from ns time axes.
    var words = py_call(source.view, context["int64"])
    var lifecycle = context["xseq_lifecycle_context"]
    var info = array_metadata_once(words, array_shape_once(words, lifecycle), "ProphetForecaster times", lifecycle, allow_unsupported=True)
    prophet_datetime_copy(info.source.addr, info.shape, info.strides, little, String(unit[0]), Int(py=unit[1]),
                          destination, String(target[0]), Int(py=target[1]), common)
    _ = info


def prophet_nanoseconds(source: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = sequential_array(Python.tuple(Int(py=source.size)), context["int64"], context)
    prophet_temporal_into(source, context["xseq_prophet_datetime_ns"], Int(py=xseq_addr(out)), False, context)
    return out
