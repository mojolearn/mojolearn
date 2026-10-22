# SPDX-License-Identifier: Apache-2.0
# Shared numeric buffer decoding. Callers admit exact primitive scalar/array
# owners first. This is not a general Python buffer/protocol admission helper.
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_buffer_owner import native_buffer_view
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
    # PyMemoryView_FromObject is a documented buffer acquisition; for an
    # admitted exact numeric NumPy scalar/ndarray it never performs a cast.
    # memoryview.tobytes copies storage, including nonnative endian data.
    var view = native_buffer_view(value)
    var width = Int(py=view.itemsize)
    if Int(py=view.nbytes) != width or width < 1 or width > 32:
        raise Error("TypeError: numeric scalar decoding requires one primitive buffer element")
    var code = String(view.format)
    var little = True  # supported host targets are little-endian
    if code.byte_length() > 0:
        var first = String(code[byte=0:1])
        if first == "@" or first == "=" or first == "<" or first == ">" or first == "!":
            little = first != ">" and first != "!"
            var rest = String(code[byte=1:])
            code = rest^
    var complex = code.startswith("Z")
    if complex:
        if width % 2 != 0:
            raise Error("TypeError: complex scalar storage requires two equal components")
        var rest = String(code[byte=1:])
        code = rest^
        width //= 2
        if code != "e" and code != "f" and code != "d" and code != "g":
            raise Error("TypeError: complex scalar storage requires floating components")
    var raw = py_call(view.tobytes)
    ref cpy = Python().cpython()
    var data = cpy.lib.get_function[MutPointer[UInt8, MutUntrackedOrigin]]("PyBytes_AsString")(raw._obj_ptr)
    if not data:
        raise cpy.unsafe_get_error()
    var real = native_scalar_real(data, 0, code, width, little)
    var imag = NativeRealBits()
    if complex:
        imag = native_scalar_real(data, width, code, width, little)
    _ = raw^
    _ = view^
    return NativeScalarBuffer(real, imag, complex, code == "g")
