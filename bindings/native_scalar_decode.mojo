# SPDX-License-Identifier: Apache-2.0
# Shared numeric buffer decoding. Callers admit exact primitive scalar/array
# owners first. This is not a general Python buffer/protocol admission helper.
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from core.native_scalar_decode import NativeRealBits, native_scalar_real


struct NativeScalarBuffer(ImplicitlyCopyable, Movable):
    var real: NativeRealBits
    var imag: NativeRealBits
    var complex: Bool
    var longdouble: Bool

    def __init__(out self, real: NativeRealBits, imag: NativeRealBits,
                 complex: Bool, longdouble: Bool):
        self.real = real
        self.imag = imag
        self.complex = complex
        self.longdouble = longdouble

    def truth(self) -> Bool:
        return self.real.truth() or self.imag.truth()

    def equals_integer(self, target: Int) -> Bool:
        return not self.imag.truth() and self.real.equals_integer(target)

    def to_python_number(self, integer: Bool) raises -> PythonObject:
        # Complex scalar warnings and the different ndarray complex error
        # contract belong to the public caller. This converts the real part
        # only after the caller has selected its existing conversion policy.
        if not integer:
            return PythonObject(bitcast[DType.float64](self.real.float64_bits()))
        var name = String("longdouble") if self.longdouble else String("float")
        if self.real.special == 1:
            raise Error("OverflowError: cannot convert " + name + " infinity to integer")
        if self.real.special == 2:
            raise Error("ValueError: cannot convert " + name + " NaN to integer")
        var text = self.real.integer_text()
        # CPython publishes the already-computed digits as an ordinary
        # arbitrary-precision integer. No caller object or NumPy number slot
        # reaches this fixed native builtin parser.
        return py_call(py_type(PythonObject(0)), PythonObject(text), 2)


def native_scalar_buffer(value: PythonObject) raises -> NativeScalarBuffer:
    # Callers have admitted the exact native ndarray/scalar and primitive
    # dtype. Its native descriptor supplies width/code/byte order without
    # NumPy computation. Requesting PEP3118 format would reject byte-swapped
    # long double even though its raw buffer bytes are supported.
    var dtype = value.dtype
    var width = Int(py=dtype.itemsize)
    var code = String(dtype.char)
    var order = String(dtype.byteorder)
    var little = order != ">"  # all supported host ABIs are little-endian
    if width < 1 or width > 32:
        raise Error("TypeError: numeric scalar decoding requires one primitive buffer element")
    var complex = code == "F" or code == "D" or code == "G"
    var component_width = width
    if complex:
        if width % 2 != 0:
            raise Error("TypeError: complex scalar storage requires two equal components")
        component_width //= 2
        code = String("f") if code == "F" else String("d") if code == "D" else String("g")
    # Py_buffer is a public CPython structure, ten pointer-sized words on
    # our supported 64-bit CPython targets; word 2 is len, word 3 itemsize.
    # PyBUF_FULL_RO without PyBUF_FORMAT (0x118) asks for native shape and
    # strides but never requests the unsupported PEP3118 spelling.
    # Allocate both owners before acquisition. Every successful export is
    # released on all error paths; fixed private bytes outlive both decodes.
    var view = List[Int](length=10, fill=0)
    var bytes = List[UInt8](length=32, fill=UInt8(0))
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var data = bytes.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ref cpy = Python().cpython()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr,pointer,Int32(0x118))
    if status != 0:
        raise cpy.unsafe_get_error()
    try:
        if view[2] != width or view[3] != width:
            raise Error("TypeError: numeric scalar decoding requires one primitive buffer element")
        status = cpy.lib.get_function[Int32]("PyBuffer_ToContiguous")(data,pointer,width,Int8(67))
        if status != 0:
            raise cpy.unsafe_get_error()
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    width = component_width
    var real = native_scalar_real(data, 0, code, width, little)
    var imag = NativeRealBits()
    if complex:
        imag = native_scalar_real(data, width, code, width, little)
    _ = bytes^
    _ = view^
    return NativeScalarBuffer(real, imag, complex, code == "g")
