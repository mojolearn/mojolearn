# SPDX-License-Identifier: Apache-2.0
# Fit guard state is product execution: native admission, NumPy-versioned
# promotion, arithmetic, warnings and publication all stay in this owner.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_numpy_metadata import native_numpy_dtype, native_numpy_type, native_numpy_module_fields
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.optimizer_public_words import stash_python_error
from bindings.sequence_array_warnings import sequence_array_errstate, sequence_array_report
from core.sequence_array_cast import SEQ_CAST_OVERFLOW, SEQ_CAST_INVALID
from core.native_scalar_decode import NativeRealBits
from core.native_depth_state import NativeDepthFormat, native_depth_float_cast, native_depth_add_one


def _depth_dtype(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True)):
        return PythonObject(None)
    if py_get(context,"ndarray") is None:
        return PythonObject(None)
    var array = cls is context["ndarray"]
    if array and Int(py=value.size) != 1:
        raise Error("TypeError: scalar model state requires a scalar or zero-dimensional numeric array")
    var constructor = native_numpy_dtype(context)
    for code in ["?","b","B","h","H","i","I","l","L","q","Q","e","f","d","g","F","D","G"]:  # driver-loop: exact primitive native scalar identities, including distinct aliases
        var dtype = py_call(constructor,code)
        if array and py_type(value.dtype) is py_type(dtype):
            return value.dtype
        if cls is dtype.type:
            return dtype
    return PythonObject(None)


def _depth_builtin_admit(value: PythonObject, context: PythonObject) raises:
    var types = context["host_depth_primitive_types"]
    for i in range(len(types)):  # driver-loop: exact retained builtin primitives have no application number slots
        if py_type(value) is types[i]:
            return
    raise Error("TypeError: fit depth requires native primitive values without Python arithmetic hooks")


def host_iforest_depth_admit(value: PythonObject, context: PythonObject) raises:
    if _depth_dtype(value,context) is None:
        _depth_builtin_admit(value,context)


def host_iforest_depth_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    if _depth_dtype(value,context) is not None:
        return native_scalar_buffer(value).truth()
    _depth_builtin_admit(value,context)
    return Bool(py=value)


def _depth_release(context: PythonObject) raises -> Int:
    var module = py_get(context["modules"],"numpy")
    if module is None or py_type(module) is not context["host_iforest_module_type"]:
        raise Error("TypeError: fit depth requires native NumPy version metadata")
    var version = py_get(native_numpy_module_fields(module),"__version__")
    if py_type(version) is not py_type(PythonObject("")):
        raise Error("TypeError: fit depth requires native NumPy version text")
    return Int(String(String(version).split(".")[0]))


def _depth_warning_guard(context: PythonObject) raises:
    # The shared native warning owner still reads these builtin module
    # fields. Refuse replacement hooks before reaching any of its calls.
    # Fixed identities are retained at package connection, never inferred
    # from a callable's spelling or native-looking type.
    var module = context["host_depth_builtins"]
    if py_get(context["modules"],"builtins") is not module:
        raise Error("TypeError: fit-depth warnings require the connected native builtin module")
    var fields = native_numpy_module_fields(module)
    var expected = context["host_depth_builtin_types"]
    for name in ["dict","int","list","object","str","tuple","type"]:  # driver-loop: fixed transitive warning implementation dependencies
        if py_get(fields,PythonObject(name)) is not expected[PythonObject(name)]:
            raise Error("TypeError: fit-depth warnings cannot invoke replaced builtin hooks")


def _depth_report(status: Int, operation: String, scalar: Bool, context: PythonObject,
                  reported_operation: String = "") raises:
    if status == 0:
        return
    _depth_warning_guard(context)
    var local = context["host_depth_warning_contexts"][PythonObject(operation)]
    var settings = sequence_array_errstate(local)
    var name = reported_operation if reported_operation.byte_length() > 0 else (String("scalar ") if scalar else String(""))+operation
    sequence_array_report(status,settings,local,name)


def _depth_box(data: MutPointer[UInt8, MutUntrackedOrigin], dtype: PythonObject,
               shape: PythonObject, scalar: Bool, context: PythonObject) raises -> PythonObject:
    # Allocate native numeric storage, then copy already-computed bytes.
    # Neither a NumPy cast nor a scalar numeric constructor participates.
    var array = py_call(native_numpy_type("ndarray",context),shape,dtype=dtype)
    var width = Int(py=dtype.itemsize)
    var view = List[Int](length=10,fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ref cpy = Python().cpython()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(array._obj_ptr,pointer,Int32(0x119))
    if status != 0:
        var failure = stash_python_error()
        raise failure^
    try:
        if view[2] != width or width < 1 or width > 32:
            raise Error("TypeError: fit depth publication requires one native primitive element")
        var destination = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=view[0])
        for i in range(width):  # driver-loop: publish at most one complex long-double element
            destination[i] = data[i]
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    _ = view^
    if not scalar:
        return array
    var index = Python.tuple()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(array._obj_ptr,index._obj_ptr)
    _ = index^
    _ = array^
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def host_iforest_depth_step(value: PythonObject, subtract: Bool, context: PythonObject) raises -> PythonObject:
    var dtype = _depth_dtype(value,context)
    if dtype is None:
        # Exact builtin scalar admission precedes this native CPython
        # metadata operation. No NumPy slot or application protocol enters.
        _depth_builtin_admit(value,context)
        ref cpy = Python().cpython()
        var one = PythonObject(1)
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Subtract" if subtract else "PyNumber_Add")(value._obj_ptr,one._obj_ptr)
        _ = one^
        if not pointer:
            var failure = stash_python_error()
            raise failure^
        return PythonObject(from_owned=pointer)
    var ndarray = py_type(value) is context["ndarray"]
    var scalar_result = not ndarray or Int(py=value.ndim) == 0
    var shape = Python.tuple() if scalar_result else value.shape
    var code = String(dtype.char)
    var kind = String(dtype.kind)
    var width = Int(py=dtype.itemsize)
    var source = native_scalar_buffer(value)
    var release = _depth_release(context)
    if kind == "b":
        code = "l"
        kind = "i"
        width = 8
    elif release < 2 and scalar_result:
        if (kind == "f" and code != "g") or (kind == "u" and width == 8):
            code = "d"
            kind = "f"
            width = 8
        elif kind == "c" and code != "G":
            code = "D"
            width = 16
        elif (kind == "i" or kind == "u") and width < 8:
            code = "l"
            kind = "i"
            width = 8
    var target = py_call(native_numpy_dtype(context),PythonObject(code))
    var bytes = List[UInt8](length=32,fill=UInt8(0))
    var data = bytes.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var status = 0
    if kind == "i" or kind == "u":
        var bits = UInt64(0)-source.real.low if source.real.negative else source.real.low
        var mask = ~UInt64(0) if width == 8 else (UInt64(1) << UInt64(width*8))-1
        bits &= mask
        var signed = kind == "i"
        var overflow = bits == ((mask >> 1)+1 if signed else UInt64(0)) if subtract else bits == (mask >> 1 if signed else mask)
        bits = (bits-UInt64(1) if subtract else bits+UInt64(1)) & mask
        for i in range(width):  # driver-loop: integer ufunc arithmetic wraps within the promoted primitive width
            data[i] = UInt8((bits >> UInt64(i*8)) & 255)
        if overflow and not ndarray:
            status = SEQ_CAST_OVERFLOW
    else:
        var complex = kind == "c"
        var real_code = String("f") if code == "F" else String("d") if code == "D" else String("g") if code == "G" else code
        var format = NativeDepthFormat(real_code,width//2 if complex else width)
        var cast_invalid = source.real.precision != format.precision and source.real.special == 2 and source.real.bit(source.real.precision-2) == 0
        if complex:
            cast_invalid = cast_invalid or (source.imag.precision != format.precision and source.imag.special == 2 and source.imag.bit(source.imag.precision-2) == 0)
        if cast_invalid:
            # NumPy1 promotes small scalar/0-D floats before +1. A signaling
            # NaN fails or warns during that conversion, with "cast" text,
            # then becomes quiet for the following arithmetic operation.
            _depth_report(SEQ_CAST_INVALID,String("subtract") if subtract else String("add"),False,context,"cast")
        var real = native_depth_float_cast(source.real,format.precision,format.minimum_power)
        if real.special == 2 and real.bit(format.precision-2) == 0:
            status = SEQ_CAST_INVALID
        real = native_depth_add_one(real,subtract,format.precision,format.minimum_power)
        format.write(data,0,real)
        if complex:
            var imag = native_depth_float_cast(source.imag,format.precision,format.minimum_power)
            if imag.special == 2:
                if imag.bit(format.precision-2) == 0:
                    status = SEQ_CAST_INVALID
                # Complex imaginary +/-0 quiets a NaN like the real part.
                imag = native_depth_add_one(imag,subtract,format.precision,format.minimum_power)
            elif not imag.truth() and not subtract:
                imag.negative = False
            format.write(data,format.width,imag)
    # Warning-as-error occurs before the completed operation is published.
    # A 0-D first operation is a ufunc, whose result becomes a scalar for
    # the second operation; this distinction controls integer overflow.
    _depth_report(status,String("subtract") if subtract else String("add"),not ndarray,context)
    var result = _depth_box(data,target,shape,scalar_result,context)
    _ = bytes^
    return result


def host_iforest_depth_positive(value: PythonObject, context: PythonObject) raises -> Bool:
    if _depth_dtype(value,context) is None:
        _depth_builtin_admit(value,context)
        ref cpy = Python().cpython()
        var zero = PythonObject(0)
        var result = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(value._obj_ptr,zero._obj_ptr,Int32(4))
        _ = zero^
        if result < 0:
            var failure = stash_python_error()
            raise failure^
        return result != 0
    var source = native_scalar_buffer(value)
    var array = py_type(value) is context["ndarray"]
    if source.real.special == 2:
        if source.complex and array and _depth_release(context) >= 2:
            _depth_report(SEQ_CAST_INVALID,"greater",False,context)
        return False
    if source.complex and source.imag.special == 2:
        # Native scalar ordering short-circuits on a nonzero real part.
        # The array ufunc rejects a NaN imaginary part even then. NumPy2
        # emits invalid only if it actually compares that NaN component.
        if array:
            if not source.real.truth() and _depth_release(context) >= 2:
                _depth_report(SEQ_CAST_INVALID,"greater",False,context)
            return False
        return source.real.truth() and not source.real.negative
    if source.real.truth():
        return not source.real.negative
    return source.complex and source.imag.truth() and not source.imag.negative
