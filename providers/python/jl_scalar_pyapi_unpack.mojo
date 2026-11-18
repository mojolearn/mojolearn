# SPDX-License-Identifier: Apache-2.0
# Scalar admission and decoding for the Johnson-Lindenstrauss public bound.
# Only exact native scalar owners can reach a conversion or buffer descriptor.
#
# Perf (lane fb12-rows, 2026-11-17; algos/jl-min-dim 8.27 -> 15.7 ms on the
# same box vs 0.8.55, one public call per grid point, ~10,000 calls): the
# exact builtin int/bool/float inputs are decided on borrowed pointers
# against class words read once per process (`jl_type_words`), with the
# import-resolved CPython table. Before, every call built fresh int, float
# and bool sample objects and two error Strings per scalar, a one-word List,
# and resolved the Python interface (a named-global probe) once per type
# query and per temporary's release: about 1 us per call of fixed cost.
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table, CPythonTable
from bindings.py_balanced import py_call, py_get, py_type
from bindings.py_typed_call import typed_python_error
from bindings.cpython_borrowed_unpack import borrowed_type_word
from bindings.sequential_admission_door import _one_of
from bindings.native_scalar_decode_unpack import native_scalar_buffer
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from core.shared_slot import SharedSlot


comptime JL_POSITIVE_ERROR = "The JL bound is defined for n_samples greater than zero"
comptime JL_EPSILON_ERROR = "The JL bound is defined for eps in ]0, 1["


struct JlTypeWords(Defaultable, Movable):
    # Addresses of CPython's exact int, bool, float, str, tuple and dict
    # classes (static, immortal types), 0 until first read.
    var int_word: Int
    var bool_word: Int
    var float_word: Int
    var str_word: Int
    var tuple_word: Int
    var dict_word: Int

    def __init__(out self):
        self.int_word = 0
        self.bool_word = 0
        self.float_word = 0
        self.str_word = 0
        self.tuple_word = 0
        self.dict_word = 0


comptime _JL_TYPE_WORDS = SharedSlot[JlTypeWords, "MojolearnJlTypeWordsV1", 1]


@no_inline
def _jl_fill_type_words(words: UnsafePointer[JlTypeWords, MutUntrackedOrigin]) raises:
    # Once per process, under the GIL: the exact builtin classes by identity.
    var one = PythonObject(1)
    var truth = PythonObject(True)
    var real = PythonObject(1.5)
    var text = PythonObject("")
    var pair = Python.tuple()
    var mapping = Python.dict()
    words[].int_word = borrowed_type_word(one._obj_ptr)
    words[].bool_word = borrowed_type_word(truth._obj_ptr)
    words[].float_word = borrowed_type_word(real._obj_ptr)
    words[].str_word = borrowed_type_word(text._obj_ptr)
    words[].tuple_word = borrowed_type_word(pair._obj_ptr)
    # Last: a nonzero dict word marks the record complete.
    words[].dict_word = borrowed_type_word(mapping._obj_ptr)
    _ = one
    _ = truth
    _ = real
    _ = text
    _ = pair
    _ = mapping


def jl_type_words() raises -> UnsafePointer[JlTypeWords, MutUntrackedOrigin]:
    """The process-wide exact builtin class words (one slot probe per call)."""
    var words = _JL_TYPE_WORDS.get_or_create_ptr()
    if words[].dict_word == 0:
        _jl_fill_type_words(words)
    return words


@always_inline
def jl_type_word(raw: PyObjectPtr) -> Int:
    """The class word of a live borrowed object: PyObject.ob_type at +8, the
    stable-ABI word the standard library's `Py_TYPE` reads (no reference, no
    interface lookup; bindings/cpython_layout.mojo)."""
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(raw) + 8)[0]


@always_inline
def jl_builtin_scalar(words: UnsafePointer[JlTypeWords, MutUntrackedOrigin], cls: Int) -> Bool:
    """An exact int, bool or float: decided by `jl_scalar_word` without the
    connected NumPy metadata."""
    return cls == words[].float_word or cls == words[].int_word or cls == words[].bool_word


def jl_scalar_word(value: PyObjectPtr, cls: Int, epsilon: Bool, context: PyObjectPtr,
                   table: UnsafePointer[CPythonTable, MutAnyOrigin],
                   words: UnsafePointer[JlTypeWords, MutUntrackedOrigin]) raises -> Float64:
    """One borrowed scalar of the bound as binary64 (`cls` is its class word).
    Exact floats and int64-range exact ints/bools are read directly; every
    other admitted kind takes `jl_scalar`, the general admission."""
    if cls == words[].float_word:
        # An exact float cannot fail its conversion.
        var number = table[].PyFloat_AsDouble(value)
        if epsilon:
            if not (number > 0.0 and number < 1.0):
                raise Error(JL_EPSILON_ERROR)
        elif number <= 0.0:
            raise Error(JL_POSITIVE_ERROR)
        return number
    if cls == words[].int_word or cls == words[].bool_word:
        # No integer lies strictly between zero and one.
        if epsilon:
            raise Error(JL_EPSILON_ERROR)
        # An int64-range count, widened to binary64 with round-to-nearest-even
        # (PyLong_AsDouble's own rounding), so the same value. A count outside
        # int64 keeps the exact CPython comparison and conversion in jl_scalar
        # (arbitrary size: 1 << 2000 raises PyLong_AsDouble's OverflowError,
        # -(1 << 2000) the n_samples refusal).
        # Fix (fb12-rows, 2026-11-18; smoke of c989eb9222,
        # test_jl_native_boundary.py::test_jl_preserves_nonfinite_integer_publication_errors):
        # the overflow flag was an `int*` out-parameter into a stack Int32
        # passed as an address word, and its write was not seen (1 << 2000
        # read as -1 with no overflow and refused as non-positive). No
        # out-parameter now: PyLong_AsLongLong returns -1 with an error set
        # only when an exact int does not fit (its one possible error for an
        # exact int), so the fast path is the same one compare on the result
        # (`word == -1`), and the error check runs only for -1.
        var word = table[].PyLong_AsLongLong(value)
        if word == -1 and table[].PyErr_Occurred():
            # Out of int64: drop the overflow error, take the exact path.
            table[].PyErr_Clear()
            return jl_scalar(PythonObject(from_borrowed=value), cls, epsilon, PythonObject(from_borrowed=context), words)
        if word <= 0:
            raise Error(JL_POSITIVE_ERROR)
        return Float64(word)
    return jl_scalar(PythonObject(from_borrowed=value), cls, epsilon, PythonObject(from_borrowed=context), words)


def jl_scalar(value: PythonObject, cls_word: Int, epsilon: Bool, context: PythonObject,
              words: UnsafePointer[JlTypeWords, MutUntrackedOrigin]) raises -> Float64:
    """The general admission: an exact int beyond int64, a native NumPy
    scalar or a zero-dimensional ndarray (exact floats and int64-range ints
    are read by `jl_scalar_word` first)."""
    if cls_word == words[].int_word or cls_word == words[].bool_word:
        # The direct CPython integer predicate handles arbitrary-size n
        # without narrowing first; exact type admission excludes every user
        # comparison/conversion slot.
        if epsilon:
            raise Error(JL_EPSILON_ERROR)
        ref cpy = Python().cpython()
        var zero = PythonObject(0)
        var nonpositive = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(
            value._obj_ptr, zero._obj_ptr, Int32(1))
        _ = zero^
        if nonpositive < 0:
            raise typed_python_error()
        if nonpositive != 0:
            raise Error(JL_POSITIVE_ERROR)
        var number = cpy.lib.get_function[Float64]("PyLong_AsDouble")(value._obj_ptr)
        if cpython_table()[].PyErr_Occurred():
            raise typed_python_error()
        return number
    var cls = py_type(value)
    var array = cls is py_get(context, "ndarray")
    var scalar = _one_of(cls, context["scalar_types"])
    # The generic shared list includes f16/f32/f64. Preserve long-double
    # scalar admission too; construct its exact native type without invoking
    # a mutable NumPy module export or a user scalar subclass.
    if not scalar and not array and py_get(context, "ndarray") is not None:
        scalar = cls is py_call(native_numpy_dtype(context), "g").type
    if not scalar and not array:
        raise Error("TypeError: johnson_lindenstrauss_min_dim requires native real scalars without Python hooks")
    if array and Int(py=value.ndim) != 0:
        raise Error("TypeError: johnson_lindenstrauss_min_dim requires scalar or zero-dimensional numeric inputs")
    # One capture of dtype/buffer metadata and bytes, retaining the original
    # owner throughout. Object dtype is rejected by the decoder before any
    # contained object's comparison, truth or numeric conversion can execute.
    var decoded = native_scalar_buffer(value)
    if decoded.complex:
        raise Error("TypeError: johnson_lindenstrauss_min_dim requires native real scalars")
    var real = decoded.real
    if epsilon:
        # Check the original precision before the historical binary64
        # conversion. A long-double value just below one may round to one.
        if real.special != 0 or real.negative or not real.truth() or real.highest() + real.scale >= 0:
            raise Error(JL_EPSILON_ERROR)
    elif real.special != 2 and (real.negative or not real.truth()):
        raise Error(JL_POSITIVE_ERROR)
    return bitcast[DType.float64](real.float64_bits())
