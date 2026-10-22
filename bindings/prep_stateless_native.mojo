# SPDX-License-Identifier: Apache-2.0
# The native owner of the stateless x_prep transformers: Binarizer, Normalizer
# and PolynomialFeatures. Each public fit, transform and fit_transform is one
# call into prep_stateless_c, which checks the parameters, records the fitted
# metadata, builds the x_prep program (the same stages, operands and order the
# Python `_Prog` built) and runs it through the shared transport
# (bindings/x_prep_native_owner.mojo `xprep_run`). fit_transform converts X once
# and runs the one transform program.
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed, classical_wrap_array
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.scaler_native import _stage, _ptr
from bindings.scaler_lifecycle_native import _entries
from bindings.x_prep_native_owner import xprep_storage, xprep_take, xprep_values, xprep_run

alias F32P = MutPointer[Float32, MutUntrackedOrigin]
alias I64P = MutPointer[Int64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_BINARIZE = 12
comptime OP_NORMALIZE = 47
comptime OP_POLY = 48
comptime OP_TRANSPOSE = 209
# `_Prog.output`: below this many words the output is plain arena words.
comptime OUT_MIN_WORDS = 134217728


def _kind(model: PythonObject, context: PythonObject) raises -> String:
    # The admitted class this model is (or subclasses), by its MRO.
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var kinds = context["prep_stateless_kinds"]
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var kind = py_get(kinds, mro[k])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: the native stateless prep owner requires a Binarizer, Normalizer or PolynomialFeatures")


def _name(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def _repr(value: PythonObject) raises -> String:
    return String(py_call(Python.import_module("builtins").repr, value))


def _int_like(value: PythonObject, context: PythonObject) raises -> Bool:
    # numbers.Integral without its Python __instancecheck__: Python ints
    # (bool included) and NumPy integer scalars when NumPy is connected.
    var b = Python.import_module("builtins")
    if Bool(py=py_call(b.isinstance, value, b.int)):
        return True
    var integer = py_get(context, "integer")
    return integer is not None and Bool(py=py_call(b.isinstance, value, integer))


def _norm_code(fields: PythonObject) raises -> Int:
    var b = Python.import_module("builtins")
    var norm = fields["norm"]
    if py_type(norm) is b.str:
        var text = String(norm)
        if text == "l1":
            return 0
        if text == "l2":
            return 1
        if text == "max":
            return 2
    raise Error("mojolearn: invalid norm " + _repr(norm))


def _degrees(fields: PythonObject, context: PythonObject) raises -> Tuple[Int, Int]:
    var b = Python.import_module("builtins")
    var degree = fields["degree"]
    var lo = 0
    var hi = 0
    if _int_like(degree, context):
        hi = Int(py=py_call(b.int, degree))
    else:
        var pair = py_call(b.tuple, degree)
        if len(pair) != 2:
            raise Error("mojolearn: invalid degree " + _repr(degree))
        lo = Int(py=py_call(b.int, pair[0]))
        hi = Int(py=py_call(b.int, pair[1]))
    if hi < 0 or lo < 0 or lo > hi:
        raise Error("mojolearn: invalid degree " + _repr(degree))
    return (lo, hi)


def _emit(mut index: List[Int], mut start: List[Int], terms: List[Int]):
    for i in range(len(terms)):  # driver-loop: over one polynomial term's feature indices
        index.append(terms[i])
    start.append(len(index))


def _combinations(d: Int, k: Int, replacement: Bool, mut index: List[Int], mut start: List[Int]):
    # itertools.combinations(_with_replacement)(range(d), k), in its order.
    if k == 0:
        _emit(index, start, List[Int]())
        return
    if not replacement and k > d:
        return
    var c = List[Int](length=k, fill=0)
    if not replacement:
        for i in range(k):  # driver-loop: over one polynomial term's positions
            c[i] = i
    while True:  # driver-loop: over polynomial terms of feature indices
        _emit(index, start, c)
        var i = k - 1
        while i >= 0 and c[i] >= (d - 1 if replacement else d - k + i):  # driver-loop: over one term's positions
            i -= 1
        if i < 0:
            return
        c[i] += 1
        for j in range(i + 1, k):  # driver-loop: over one term's positions
            c[j] = c[i] if replacement else c[j - 1] + 1


def _words(values: List[Int], context: PythonObject) raises -> PythonObject:
    # Index words as the float32 values `_Prog.put_list` packed (exact).
    var count = max(len(values), 1)
    var out = classical_typed(Python.tuple(count), "<f4", context["output_context"])
    var dest = _ptr(out)
    for i in range(len(values)):  # driver-loop: over polynomial term table words
        dest[i] = Float32(values[i])
    return out


def _fit(model: PythonObject, kind: String, values: PythonObject, context: PythonObject) raises:
    var fields = py_fields(model)
    var d = Int(py=values[4])
    if kind == "normalizer":
        _ = _norm_code(fields)
    elif kind == "polynomial":
        var order = fields["order"]
        if py_type(order) is not Python.import_module("builtins").str or (String(order) != "C" and String(order) != "F"):
            raise Error("mojolearn: PolynomialFeatures order must be 'C' or 'F'")
        var degrees = _degrees(fields, context)
        var replacement = not Bool(py=fields["interaction_only"])
        var index = List[Int]()
        var start = List[Int]()
        start.append(0)
        if Bool(py=fields["include_bias"]):
            _combinations(d, 0, replacement, index, start)
        for k in range(max(1, degrees[0]), degrees[1] + 1):  # driver-loop: over polynomial degrees
            _combinations(d, k, replacement, index, start)
        var nout = len(start) - 1
        var powers = classical_typed(Python.tuple(max(nout, 1), d), "<i8", context["output_context"])
        var table = I64P(unsafe_from_address=Int(py=classical_address(powers)))
        for t in range(nout):  # driver-loop: over polynomial terms
            for i in range(start[t], start[t + 1]):  # driver-loop: over one term's feature indices
                table[t * d + index[i]] += 1
        py_setitem(fields, "_poly_index", _words(index, context))
        py_setitem(fields, "_poly_start", _words(start, context))
        py_setitem(fields, "n_output_features_", nout)
        py_setitem(fields, "powers_", powers)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "numeric_mode_", context["backend_fields"]["_DEFAULT_MODE"])


def _take_f(storage: PythonObject, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # The (rows, cols) column-major block p2m_transpose wrote, as an F-order Array.
    var b = Python.import_module("builtins")
    var part = py_call(b.memoryview, storage)[py_call(b.slice, offset, offset + rows * cols)]
    var store = py_call(context["array_type"], "f")
    _ = py_call(store.frombytes, py_call(part.cast, "B"))
    return classical_wrap_array(store, Python.tuple(rows, cols), "<f4", "F", context["output_context"])


def _transform(model: PythonObject, kind: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var fitted = fields["n_features_in_"]
    if py_type(fitted) is not Python.import_module("builtins").int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + _name(model) + " was fitted with " + String(py_call(Python.import_module("builtins").str, fitted)))
    var entries = _entries(fields["numeric_mode_"], context)
    var inputs = Python.list(Python.tuple(0, values))
    var program = List[Int32]()
    if kind == "binarizer":
        # X, the threshold word, then the output as arena words.
        var threshold = classical_typed(Python.tuple(1), "<f4", context["output_context"])
        _ptr(threshold)[0] = Float32(Float64(py=py_call(Python.import_module("builtins").float, fields["threshold"])))
        _ = py_call(inputs.append, Python.tuple(n * d, Python.tuple(0, threshold, PythonObject(None), 1, 1)))
        var out = n * d + 1
        var host = out + n * d
        if host > 2147483647:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        _stage(program, OP_BINARIZE, n * d, 0, n * d, n * d, out)
        var arena = xprep_storage(host, context)
        xprep_run(arena, host, 0, PythonObject(None), 0, program^, inputs, entries, context)
        var result = xprep_take(arena[0], out, n, d, context["output_context"])
        _ = threshold^
        _ = arena^
        return result
    if kind == "normalizer":
        var code = _norm_code(fields)
        var words = n * d
        var dev_out = words >= OUT_MIN_WORDS and py_get(entries, "x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT", "1")) != "0"
        var host = words + (0 if dev_out else words)
        if host + (words if dev_out else 0) > 2147483647:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        var arena = xprep_storage(host, context)
        var output_store = xprep_storage(words, context) if dev_out else PythonObject(None)
        _stage(program, OP_NORMALIZE, n, 0, n, d, code, host if dev_out else words)
        xprep_run(arena, host, 0, output_store, words if dev_out else 0, program^, inputs, entries, context, words if dev_out else 0)
        var result = xprep_take(output_store[0] if dev_out else arena[0], 0 if dev_out else words, n, d, context["output_context"])
        _ = arena^
        return result
    # PolynomialFeatures: X, the term indices, the term starts, the products.
    var index = fields["_poly_index"]
    var start = fields["_poly_start"]
    var nout = Int(py=fields["n_output_features_"])
    if Int(py=start.size) != nout + 1:
        raise Error("RuntimeError: PolynomialFeatures term table does not match n_output_features_")
    var io = n * d
    var so = io + Int(py=index.size)
    var out = so + Int(py=start.size)
    var words = n * nout
    var fortran = String(fields["order"]) == "F"
    var transpose = fortran and words > 0
    var dev_out = transpose and words >= OUT_MIN_WORDS and py_get(entries, "x_prep_run_out") is not None and String(getenv("MOJOLEARN_XPREP_OUT", "1")) != "0"
    var host = out + words + (words if transpose and not dev_out else 0)
    if host + (words if dev_out else 0) > 2147483647:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    _ = py_call(inputs.append, Python.tuple(io, Python.tuple(0, index, PythonObject(None), Int(py=index.size), 1)))
    _ = py_call(inputs.append, Python.tuple(so, Python.tuple(0, start, PythonObject(None), Int(py=start.size), 1)))
    _stage(program, OP_POLY, words, 0, n, d, io, so, nout, out)
    var fo = out + words
    if transpose:
        _stage(program, OP_TRANSPOSE, words, out, n, nout, host if dev_out else fo)
    var arena = xprep_storage(host, context)
    var output_store = xprep_storage(words, context) if dev_out else PythonObject(None)
    xprep_run(arena, host, 0, output_store, words if dev_out else 0, program^, inputs, entries, context, words if dev_out else 0)
    var result = PythonObject(None)
    if transpose:
        result = _take_f(output_store[0] if dev_out else arena[0], 0 if dev_out else fo, n, nout, context)
    elif fortran:
        # an empty block
        result = classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(n, nout), "<f4", "F", context["output_context"])
    elif words == 0:
        result = classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(n, nout), "<f4", "C", context["output_context"])
    else:
        result = xprep_take(arena[0], out, n, nout, context["output_context"])
    _ = arena^
    return result


def prep_stateless_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var kind = _kind(model, context)
        var result = PythonObject(None)
        if operation == "fit" or operation == "fit_transform" or operation == "transform":
            var mode = context["backend_fields"]["_DEFAULT_MODE"]
            if operation == "transform":
                var fields = py_fields(model)
                if not py_contains(fields, "n_features_in_"):
                    raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")
                mode = fields["numeric_mode_"]
            var values = xprep_values(args[0], _entries(mode, context), context)
            if operation == "transform":
                result = _transform(model, kind, values, context)
            else:
                _fit(model, kind, values, context)
                result = model
                if operation == "fit_transform":
                    # The X converted for the fit is the transform's input.
                    result = _transform(model, kind, values, context)
            _ = values^
        else:
            raise Error("unknown native stateless prep operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
