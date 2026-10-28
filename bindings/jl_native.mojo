# SPDX-License-Identifier: Apache-2.0
# The native owner of `johnson_lindenstrauss_min_dim`: the public function
# is one call into jl_c, which admits and captures scalar inputs, checks
# eps and n_samples natively, and computes scikit-learn's bound
# 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3) with the same binary64
# operations: eps^2 and eps^3 correctly rounded (`_portable_math.powi`,
# DEVIATION 6900: an exact integer power rounded once to nearest-even) and
# the pinned log (`_portable_math.log`, packaging/portable_math/portable_math.c
# `mojolearn_log`, x_metrics/epilogue.mojo `portable_log_c`). So the same
# bits as the shell; CPython publishes the completed truncated integer.
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_type
from bindings.classical_attributes import classical_connect_numpy
from providers.python.jl_scalar_pyapi import jl_scalar
from x_metrics.epilogue import portable_log_c

comptime _LIMB = UInt64(0xFFFFFFFF)


def _limbs_mul(a: List[UInt64], b: List[UInt64]) -> List[UInt64]:
    """The exact product of two little-endian base-2^32 integers."""
    var out = List[UInt64](length=len(a) + len(b), fill=UInt64(0))
    for i in range(len(a)):  # driver-loop: over the limbs of a 53-bit power (at most five)
        var carry = UInt64(0)
        for j in range(len(b)):  # driver-loop: over the limbs of a 53-bit mantissa (two)
            var t = a[i] * b[j] + out[i + j] + carry
            out[i + j] = t & _LIMB
            carry = t >> 32
        var k = i + len(b)
        while carry != 0:  # driver-loop: over the carry's limbs
            var t = out[k] + carry
            out[k] = t & _LIMB
            carry = t >> 32
            k += 1
    return out^


def _bit(a: List[UInt64], i: Int) -> Bool:
    return ((a[i // 32] >> UInt64(i % 32)) & UInt64(1)) != 0


def _bit_length(a: List[UInt64]) -> Int:
    var k = len(a) - 1
    while k >= 0 and a[k] == 0:  # driver-loop: over the limbs of an exact power
        k -= 1
    if k < 0:
        return 0
    var top = a[k]
    var bits = 0
    while top != 0:  # driver-loop: over the bits of one limb
        top >>= 1
        bits += 1
    return k * 32 + bits


def _cube_rounded(x: Float64) -> Float64:
    """x^3 correctly rounded to nearest-even for a finite positive x whose
    cube is normal (`_portable_math.powi(x, 3)`); x = m 2^e, the exact m^3
    rounded once to 53 bits."""
    var bits = bitcast[DType.uint64](x)
    var field = Int((bits >> 52) & UInt64(0x7FF))
    if field == 0 or field < 1023 - 300:
        # A subnormal or tiny eps: eps^3 / 3 is below half an ulp of
        # eps^2 / 2 by far more than any rounding of the cube, so the
        # bound's denominator is the same for every rounding of it.
        return x * x * x
    var m = (bits & UInt64(0xFFFFFFFFFFFFF)) | UInt64(0x10000000000000)
    var e = field - 1075
    var mm: List[UInt64] = [m & _LIMB, m >> 32]
    var power = _limbs_mul(_limbs_mul(mm, mm), mm)
    var length = _bit_length(power)
    var shift = length - 53
    var q = UInt64(0)
    for i in range(53):  # driver-loop: over the 53 kept bits of the exact cube
        if _bit(power, shift + i):
            q |= UInt64(1) << UInt64(i)
    var half = _bit(power, shift - 1)
    var sticky = False
    for i in range(shift - 1):  # driver-loop: over the discarded bits of the exact cube
        if _bit(power, i):
            sticky = True
            break
    if half and (sticky or (q & UInt64(1)) != 0):
        q += 1
        if q == (UInt64(1) << 53):
            q >>= 1
            shift += 1
    # q in [2^52, 2^53): the normal binary64 q 2^(shift + 3e), built from its fields
    var biased = shift + 3 * e + 52 + 1023
    return bitcast[DType.float64]((UInt64(biased) << 52) | (q & UInt64(0xFFFFFFFFFFFFF)))


def _pinned_log(x: Float64) -> Float64:
    # `_portable_math.log` of a positive x: the special values as
    # `special_log`, finite values through the pinned polynomial.
    if x - x != 0.0:
        return x  # NaN, +inf
    return portable_log_c(x)


def jl_min_dim(n_samples: PythonObject, eps: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve epsilon-before-count validation and the original arithmetic.
    var e = jl_scalar(eps, True, context)
    var n = jl_scalar(n_samples, False, context)
    var denominator = (e * e) / 2.0 - _cube_rounded(e) / 3.0
    var bound = 4.0 * _pinned_log(n) / denominator
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyLong_FromDouble")(bound)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def jl_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: jl_native_call requires model, operation, arguments and connected context")
        if py_type(call[1]) is not py_type(PythonObject("")) or py_type(call[2]) is not py_type(Python.tuple()) or py_type(call[3]) is not py_type(Python.dict()):
            raise Error("TypeError: native JL requires an operation, argument tuple and connected context")
        var operation = String(call[1])
        var args = call[2]
        if operation != "min_dim" or len(args) != 2:
            raise Error("unknown private johnson_lindenstrauss_min_dim operation")
        var context = call[3]
        classical_connect_numpy(context)
        return jl_min_dim(args[0], args[1], context).steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("OverflowError:"):
            return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
