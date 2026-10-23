# SPDX-License-Identifier: Apache-2.0
"""`byte_lm_host_trainer_call(trainer, "train_step", (ids,), context)`: the
one native call of `LanguageModelHostTrainer.train_step`
(python/mojolearn/_byte_lm_host.py).

The ids are admitted and copied into C-ordered int32 words (the
`as_i32_c(ids, ndim=2)` refusals in its words), checked against the trainer
shape and the vocabulary (the first offending id named by row and
position), the four outputs are made, the compiled
`byte_lm_host_train_step` runs, and the trainer's state is replaced only
after it returned. The same entry and argument words as the Python body,
so the step keeps its bits.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind, classical_wrap_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from core.cast_in import CAST_KIND_FLOAT
from core.xseq_marshal import neural_cast_i32_2d, neural_first_bad_id

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _IGNORE_INDEX = -100


def _text(value: PythonObject) raises -> String:
    var b = Python.import_module("builtins")
    return String(py=py_call(b.str, value))


def _ids_i32(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_i32_c(source, ndim=2, name="ids")[0]`: a C-ordered int32
    (rows, cols) Array."""
    classical_admit_value(source, context)
    var value = classical_input(source, context, "ids")
    var rank = Int(py=value.ndim)
    if rank != 2:
        var hint = String("")
        if rank < 2:
            hint = (". Reshape your data either using array.reshape(-1, 1) if your data "
                    + "has a single feature or array.reshape(1, -1) if it contains a single sample.")
        raise Error("mojolearn: ids must be 2-D, got " + String(rank) + "-D shape " + _text(value.shape) + hint)
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    if rows * cols == 0:
        var what = String("feature(s)") if rows != 0 else String("sample(s)")
        raise Error("mojolearn: ids is empty, shape " + _text(value.shape) + ": Found array with 0 " + what
                    + " (shape=" + _text(value.shape) + ") while a minimum of 1 is required.")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "ids", context)
    if kind == CAST_KIND_FLOAT:
        raise Error("TypeError: ids must be an integer array")
    var store = py_call(context["array_type"], PythonObject("i"), py_call(context["native_bytes_type"], 4 * rows * cols))
    var tokens = classical_wrap_array(store, Python.tuple(rows, cols), "<i4", "C", context)
    neural_cast_i32_2d(Int(py=classical_address(value)), Int(py=value.strides[0]), Int(py=value.strides[1]),
                       rows, cols, item, kind, _I32P(unsafe_from_address=Int(py=classical_address(tokens))))
    _ = value  # keep the Python object alive through the native call above
    _ = tokens  # keep the Python object alive through the native call above
    _ = value^
    return tokens


def _f32_array(n: Int, context: PythonObject) raises -> PythonObject:
    """`_buffer.empty((n,), "<f4")`: a zero-filled mojolearn Array, on every
    install (the trainer's state is Arrays, NumPy or not)."""
    var store = py_call(context["array_type"], PythonObject("f"), py_call(context["native_bytes_type"], 4 * n))
    return classical_wrap_array(store, Python.tuple(n), "<f4", "C", context)


def byte_lm_host_train_step(trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = py_fields(trainer)
    # (batch, length + 1, vocab_size, n_total), retained at construction
    var words = fields["_step_words"]
    var batch = Int(py=words[0])
    var width = Int(py=words[1])
    var vocab = Int(py=words[2])
    var n = Int(py=words[3])
    var tokens = _ids_i32(args[0], context)
    if Int(py=tokens.shape[0]) != batch or Int(py=tokens.shape[1]) != width:
        raise Error("ids must be [" + String(batch) + ", " + String(width) + "]")
    var bad = neural_first_bad_id(_I32P(unsafe_from_address=Int(py=classical_address(tokens))), batch, width, vocab,
                                  width - 1, _IGNORE_INDEX)
    _ = tokens  # keep the Python object alive through the native call above
    if bad >= 0:
        var v = Int(_I32P(unsafe_from_address=Int(py=classical_address(tokens)))[bad])
        _ = tokens  # keep the Python object alive through the native call above
        raise Error("ids must be byte values in [0, " + String(vocab) + "); got " + String(v) + " at row "
                    + String(bad // width) + ", position " + String(bad % width))
        _ = tokens  # keep the Python object alive through the native call above
    _ = tokens  # keep the Python object alive through the native calls above
    var grad = _f32_array(n, context)
    var post_p = _f32_array(n, context)
    var post_m = _f32_array(n, context)
    var post_v = _f32_array(n, context)
    py_setitem(fields, "_gradient", PythonObject(None))
    # the trainer's binding entry, retained at construction
    var entry = fields["_train_entry"]
    var bits = py_call_typed(entry, Python.list(classical_address(fields["_parameters"]), classical_address(fields["_m"]),
                             classical_address(fields["_v"]), classical_address(tokens), classical_address(grad),
                             classical_address(post_p), classical_address(post_m), classical_address(post_v)),
                             fields["_native"], fields["_scalars"], fields["_completed"])
    _ = grad  # keep the Python object alive through the native call above
    _ = tokens  # keep the Python object alive through the native call above
    _ = post_v  # keep the Python object alive through the native call above
    _ = post_m  # keep the Python object alive through the native call above
    _ = post_p  # keep the Python object alive through the native call above
    _ = tokens^
    var result = Int(py=bits)
    _ = post_v  # keep the Python object alive through the native call above
    _ = post_m  # keep the Python object alive through the native call above
    _ = post_p  # keep the Python object alive through the native call above
    _ = grad  # keep the Python object alive through the native call above
    py_setitem(fields, "_gradient", grad)
    py_setitem(fields, "_parameters", post_p)
    py_setitem(fields, "_m", post_m)
    py_setitem(fields, "_v", post_v)
    py_setitem(fields, "_completed", Int(py=fields["_completed"]) + 1)
    return PythonObject(result)


def byte_lm_host_trainer_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(py=call[1])
        if operation != "train_step":
            raise Error("RuntimeError: unknown byte LM host trainer operation " + operation)
        return byte_lm_host_train_step(call[0], call[2], call[3]).steal_data()
    except error:
        var text = String(error)
        if text.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if text.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if text.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
