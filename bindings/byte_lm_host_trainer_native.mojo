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
from std.memory import bitcast
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry, neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_info
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
    var shape = value.shape
    var strides = value.strides
    var rank = len(shape)
    if rank != 2:
        var hint = String("")
        if rank < 2:
            hint = (". Reshape your data either using array.reshape(-1, 1) if your data "
                    + "has a single feature or array.reshape(1, -1) if it contains a single sample.")
        raise Error("mojolearn: ids must be 2-D, got " + String(rank) + "-D shape " + _text(shape) + hint)
    var rows = Int(py=shape[0])
    var cols = Int(py=shape[1])
    if rows * cols == 0:
        var what = String("feature(s)") if rows != 0 else String("sample(s)")
        raise Error("mojolearn: ids is empty, shape " + _text(shape) + ": Found array with 0 " + what
                    + " (shape=" + _text(shape) + ") while a minimum of 1 is required.")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "ids", context)
    if kind == CAST_KIND_FLOAT:
        raise Error("TypeError: ids must be an integer array")
    var store = py_call(context["array_type"], PythonObject("i"), py_call(context["native_bytes_type"], 4 * rows * cols))
    var tokens = classical_wrap_array(store, Python.tuple(rows, cols), "<i4", "C", context)
    neural_cast_i32_2d(Int(py=classical_address(value)), Int(py=strides[0]), Int(py=strides[1]),
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


def _host_provider(fields: PythonObject) raises -> PythonObject:
    var factory = neural_descriptor_entry(fields["_binding"], "byte_lm_host_api_v1")
    var provider = neural_descriptor_reuse(py_get(fields, "_native_provider"), factory,
        "mojolearn.training.byte_lm_host.v1", 4)
    py_setitem(fields, "_native_provider", provider)
    return provider


def _host_words(fields: PythonObject, threaded: Int = 0, threads: Int = 0,
                batch: Int = 0, length: Int = 0, completed: Int = 0) raises -> List[Int]:
    var native = fields["_native"]
    var b = native_loaded_module("builtins")
    if py_type(native) is not b.list and py_type(native) is not b.tuple:
        raise Error("TypeError: host model shape requires native list or tuple storage")
    if len(native) != 7 and len(native) != 9 and len(native) != 10:
        raise Error("byte LM host: expected 7, 9 or 10 shape integers")
    var words = List[Int]()
    for i in range(len(native)):  # driver-loop: over shape metadata once per public operation
        if py_type(native[i]) is not b.int:
            raise Error("TypeError: host model dimensions require exact native integers")
        words.append(Int(py=native[i]))
    if len(words) == 7:
        words.append(2)
        words.append(256)
    if len(words) == 9:
        words.append(0)
    words.append(threaded)
    words.append(threads)
    words.append(batch)
    words.append(length)
    words.append(completed)
    return words^


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
    var provider = _host_provider(fields)
    var descriptors: List[Int] = [Int(py=classical_address(fields["_parameters"])),
        Int(py=classical_address(fields["_m"])), Int(py=classical_address(fields["_v"])),
        Int(py=classical_address(tokens)), Int(py=classical_address(grad)),
        Int(py=classical_address(post_p)), Int(py=classical_address(post_m)), Int(py=classical_address(post_v))]
    var optimizer = fields["_scalars"]
    classical_admit_value(optimizer, context)
    if len(optimizer) != 5:
        raise Error("byte LM host: train step expects 5 optimizer scalars")
    var scalars = List[Float64]()
    for i in range(5):  # driver-loop: over the five admitted optimizer metadata fields
        scalars.append(Float64(py=optimizer[i]))
    var completed = fields["_completed"]
    if py_type(completed) is not native_loaded_module("builtins").int:
        raise Error("TypeError: completed_steps requires a native integer")
    var native = _host_words(fields, completed=Int(py=completed))
    # Preserve the old GIL-held host operation and transactional output publish.
    var bits = neural_descriptor_execute(provider, 3, descriptors, native, scalars, False)
    _ = grad  # keep the Python object alive through the native call above
    _ = tokens  # keep the Python object alive through the native call above
    _ = post_v  # keep the Python object alive through the native call above
    _ = post_m  # keep the Python object alive through the native call above
    _ = post_p  # keep the Python object alive through the native call above
    _ = tokens^
    var result = bits
    _ = post_v  # keep the Python object alive through the native call above
    _ = post_m  # keep the Python object alive through the native call above
    _ = post_p  # keep the Python object alive through the native call above
    _ = grad  # keep the Python object alive through the native call above
    py_setitem(fields, "_gradient", grad)
    py_setitem(fields, "_parameters", post_p)
    py_setitem(fields, "_m", post_m)
    py_setitem(fields, "_v", post_v)
    py_setitem(fields, "_completed", Int(py=completed) + 1)
    return PythonObject(result)


def byte_lm_host_trainer_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(py=call[1])
        if operation == "native_provider_info":
            return neural_descriptor_info(py_fields(call[0])["_native_provider"]).steal_data()
        if operation != "train_step" and operation != "loss":
            raise Error("RuntimeError: unknown byte LM host trainer operation " + operation)
        var result = byte_lm_host_train_step(call[0], call[2], call[3])
        if operation == "loss":
            return PythonObject(Float64(bitcast[DType.float32](UInt32(py=result)))).steal_data()
        return result.steal_data()
    except error:
        var text = String(error)
        if text.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if text.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if text.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
