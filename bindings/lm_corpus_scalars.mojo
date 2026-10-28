# SPDX-License-Identifier: Apache-2.0
"""Native scalar preparation for the original int(step * batch * length).

NumPy scalar arithmetic is reproduced with native typed words. Calling even an
exact NumPy scalar's multiplication slot can reach np.seterrcall on overflow.
Admission and native refusal therefore precede arithmetic; no ufunc or warning
callback runs here. Native integer overflow is refused before a wrapped offset.

Promotion follows NumPy NEP 50 for Python scalar operands:
https://numpy.org/neps/nep-0050-scalar-promotion.html
The temporary promotion-state getter exists through NumPy 2.1, and 2.2 removed
it in favor of weak promotion:
https://numpy.org/doc/2.3/release/2.2.0-notes.html#nep-50-promotion-state-option-removed
"""
from std.python import Python, PythonObject
from std.math import isfinite
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.native_numpy_metadata import native_numpy_module_fields
from bindings.neural_block_admission import neural_native_scalar


def _numpy_weak_promotion(context: PythonObject) raises -> Bool:
    var bi = native_loaded_module("builtins")
    var admission = context["admission_context"]
    var numpy = py_get(admission["modules"], "numpy")
    var fields = native_numpy_module_fields(numpy)
    var version = py_get(fields, "__version__")
    if py_type(version) is not bi.str:
        raise Error("TypeError: corpus requires native NumPy version metadata")
    var parts = py_call(version.split, ".", 2)
    if len(parts) < 2:
        raise Error("TypeError: corpus requires native NumPy version metadata")
    var major = Int(py=py_call(bi.int, parts[0]))
    var minor = Int(py=py_call(bi.int, parts[1]))
    if major > 2 or (major == 2 and minor >= 2):
        return True
    var owner = py_get(admission["modules"], "numpy._core._multiarray_umath")
    if owner is None:
        owner = py_get(admission["modules"], "numpy.core._multiarray_umath")
    var native_fields = native_numpy_module_fields(owner)
    var getter = py_get(native_fields, "_get_promotion_state")
    if py_type(getter) is not py_type(bi.len):
        raise Error("TypeError: corpus NumPy promotion requires its native state getter")
    if getter.__self__ is not owner or String(getter.__name__) != "_get_promotion_state":
        raise Error("TypeError: corpus NumPy promotion getter belongs to a different native entry")
    var retained = py_get(admission, "corpus_promotion_getter")
    if retained is not None and retained is not getter:
        raise Error("TypeError: corpus NumPy promotion getter was replaced")
    py_setitem(admission, "corpus_promotion_getter", getter)
    # The documented native getter only reads the enum and creates a string.
    # Capture it once per public ids operation, including weak_and_warn state;
    # the completed arithmetic does not invoke Python warning handlers.
    var mode = py_call(getter)
    if py_type(mode) is not bi.str:
        raise Error("TypeError: corpus NumPy promotion state requires a native string")
    if String(mode) == "legacy":
        return False
    if String(mode) == "weak" or String(mode) == "weak_and_warn":
        return True
    raise Error("ValueError: corpus NumPy promotion state is unsupported")


def _integer_base(step: PythonObject, batch: Int, length: Int,
                  bits: Int, unsigned: Bool, weak: Bool) raises -> Int:
    var bi = native_loaded_module("builtins")
    var limit = py_call(bi.pow, 2, bits if unsigned else bits - 1)
    var minimum = PythonObject(0) if unsigned else -limit
    var maximum = limit - PythonObject(1)
    if weak and (Bool(py=PythonObject(batch) > maximum) or Bool(py=PythonObject(length) > maximum)):
        raise Error("ValueError: corpus batch dimensions exceed the NumPy scalar promotion width")
    # Exact builtin integer words avoid NumPy overflow/warning callbacks.
    # Reject overflow at each original multiplication, before int conversion.
    var first = py_call(bi.int, step) * PythonObject(batch)
    if Bool(py=first < minimum) or Bool(py=first > maximum):
        raise Error("ValueError: corpus step multiplication overflows its native scalar width")
    var product = first * PythonObject(length)
    if Bool(py=product < minimum) or Bool(py=product > maximum):
        raise Error("ValueError: corpus step multiplication overflows its native scalar width")
    return Int(py=product)


def _floating_base(step: PythonObject, batch: Int, length: Int, width: Int) raises -> Int:
    var bi = native_loaded_module("builtins")
    var value = Float64(py=py_call(bi.float, step))
    if not isfinite(value):
        raise Error("ValueError: corpus step multiplication requires a finite native scalar")
    var result: Float64
    if width == 2:
        # NumPy half scalar multiplication rounds each Float32 product back
        # to Float16 before using that scalar in the following multiplication.
        var left = Float16(value)
        var row_count = Float16(batch)
        var row_length = Float16(length)
        if not isfinite(Float32(row_count)) or not isfinite(Float32(row_length)):
            raise Error("ValueError: corpus dimensions overflow NumPy float16 promotion")
        var first = Float16(Float32(left) * Float32(row_count))
        var product = Float16(Float32(first) * Float32(row_length))
        if not isfinite(Float32(first)) or not isfinite(Float32(product)):
            raise Error("ValueError: corpus step multiplication overflows NumPy float16")
        result = Float64(product)
    elif width == 4:
        var first = Float32(value) * Float32(batch)
        var product = first * Float32(length)
        if not isfinite(first) or not isfinite(product):
            raise Error("ValueError: corpus step multiplication overflows NumPy float32")
        result = Float64(product)
    else:
        var first = value * Float64(batch)
        var product = first * Float64(length)
        if not isfinite(first) or not isfinite(product):
            raise Error("ValueError: corpus step multiplication overflows NumPy float64")
        result = product
    # Conversion through an exact builtin float retains truncation toward zero
    # and CPython's checked conversion into the native signed indexing word.
    return Int(py=py_call(bi.int, PythonObject(result)))


def corpus_numeric_base(step: PythonObject, batch: Int, length: Int, context: PythonObject) raises -> Int:
    var bi = native_loaded_module("builtins")
    var cls = py_type(step)
    if cls is bi.int or cls is bi.float or cls is bi.bool:
        # Preserve the original multiply-before-int order for ordinary scalar
        # values. These exact CPython number slots cannot bind caller hooks.
        var product = step * PythonObject(batch) * PythonObject(length)
        return Int(py=py_call(bi.int, product))
    if not neural_native_scalar(step, context):
        raise Error("TypeError: corpus step_index requires a native numeric scalar without multiplication hooks")
    var admission = context["admission_context"]
    var classes = admission["class_dtypes"]
    var kind = -1
    var width = 0
    for k in range(4):  # small-loop(four native NumPy scalar kinds)
        for size in [1, 2, 4, 8]:  # small-loop(four supported native NumPy scalar widths)
            var dtype = py_get(classes, Python.tuple(k, size))
            if dtype is not None and dtype.type is cls:
                kind = k
                width = size
    if kind < 0:
        raise Error("TypeError: corpus step_index requires an admitted NumPy numeric scalar")
    var weak = _numpy_weak_promotion(context)
    if kind == 2:
        return _floating_base(step, batch, length, width if weak else 8)
    if kind == 0 and width == 8 and not weak:
        # Legacy promotion of uint64 with the Python signed int operand is
        # float64; retaining the uint64 route here would change token offsets.
        return _floating_base(step, batch, length, 8)
    if kind == 3 or not weak:
        return _integer_base(step, batch, length, 64, False, False)
    return _integer_base(step, batch, length, 8 * width, kind == 0, True)
