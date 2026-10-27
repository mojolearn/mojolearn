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
from bindings.native_numpy_metadata import native_numpy_type
from bindings.native_buffer_owner import native_storage_allocate
from providers.python.byte_lm_metadata_pyapi import _Shape, _shape_of
from providers.python.training_buffer_pyapi import training_f32_cast_owned
from core.training_state_validation import training_byte_shape_validate
from core.sha256 import sha256_hex_addr
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
        "mojolearn.training.byte_lm_host.v1", 5)
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


def _host_shape(config: PythonObject, context: PythonObject) raises -> Tuple[_Shape, PythonObject]:
    var bi = native_loaded_module("builtins")
    var state = bi.dict()
    var object = config
    if config is not None:
        if py_type(config) is not context["byte_lm_config_class"]:
            raise Error("TypeError: shape must be a ByteLanguageModelConfig")
        var fields = py_fields(config)
        classical_admit_value(fields, context)
        py_setitem(state, "model_shape", fields)
    var shape = _shape_of(state)
    training_byte_shape_validate(shape.d, shape.chunked)
    if config is None:
        object = native_storage_allocate(context["byte_lm_config_class"])
        var fields = py_fields(object)
        var names = [String("batch"), String("length"), String("d_model"), String("n_heads"), String("n_kv"),
                     String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size")]
        for i in range(9):  # driver-loop: over immutable default model dimensions
            py_setitem(fields, names[i], shape.d[i])
        py_setitem(fields, "chunked_lm_head_v2", shape.chunked)
    return (shape^, object)


def _host_profile(fields: PythonObject) raises -> PythonObject:
    var provider = _host_provider(fields)
    var result = List[String](length=1, fill=String(""))
    _ = neural_descriptor_execute(provider, 4, [Int(result.unsafe_ptr())], _host_words(fields), List[Float64](), False)
    return PythonObject(result[0])


def byte_lm_host_construct(trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(trainer) is not context["byte_lm_host_trainer_class"]:
        raise Error("TypeError: host training requires the connected native trainer class")
    classical_connect_numpy(context)
    var profile = py_get(context["byte_lm_profile_defaults"], "training", PythonObject("fp32_v1"))
    if py_type(profile) is not bi.str or not Bool(py=context["byte_lm_profiles"][profile]["training"]):
        raise Error("NotImplementedError: the selected numeric profile is unavailable for Byte-LM training")
    var captured = _host_shape(args[3], context)
    var shape = captured[0]
    var binding = context["byte_lm_host_binding"]
    if py_type(binding) is bi.str:
        raise Error(String(binding))
    var entry = neural_descriptor_entry(binding, "byte_lm_host_train_step")
    var n = shape.n_total()
    var parameters = training_f32_cast_owned(args[0], [n], "parameters", context)
    var moment = _f32_array(n, context) if args[1] is None else training_f32_cast_owned(args[1], [n], "m", context)
    var variance = _f32_array(n, context) if args[2] is None else training_f32_cast_owned(args[2], [n], "v", context)
    if py_type(args[4]) is not bi.int:
        raise Error("completed_steps must be an int")
    var completed = Int(py=args[4])
    if completed < 0:
        raise Error("completed_steps must not be negative")
    var betas = args[6]
    classical_admit_value(betas, context)
    var beta_cls = py_type(betas)
    var native_beta = beta_cls is bi.list or beta_cls is bi.tuple or beta_cls is bi.range or beta_cls is bi.memoryview or beta_cls is context["array_type"]
    var numpy = py_get(context["modules"], "numpy")
    if numpy is not None:
        native_beta = native_beta or beta_cls is native_numpy_type("ndarray", context)
    if not native_beta or len(betas) != 2:
        raise Error("betas must contain two real numbers")
    var values = Python.list(args[5], betas[0], betas[1], args[7], args[8])
    classical_admit_value(values, context)
    var names = [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]
    var scalars = List[Float64]()
    var scalar_objects = Python.list()
    for i in range(5):  # driver-loop: over optimizer metadata captured once
        var cls = py_type(values[i])
        var floating = cls is bi.float
        if numpy is not None:
            floating = floating or cls is native_numpy_type("float64", context)
        if cls is not bi.int and not floating:
            raise Error(names[i] + " must be a real number")
        var value = Float64(py=values[i])
        scalars.append(value)
        _ = py_call(scalar_objects.append, value)
    if not (scalars[0] > 0 and scalars[3] > 0):
        raise Error("lr and eps must be positive")
    if not (scalars[1] >= 0 and scalars[1] < 1 and scalars[2] >= 0 and scalars[2] < 1):
        raise Error("betas must be in [0, 1)")
    if scalars[4] < 0:
        raise Error("weight_decay must not be negative")
    var fields = py_fields(trainer)
    py_setitem(fields, "_shape", captured[1])
    py_setitem(fields, "_native", shape.native_shape())
    py_setitem(fields, "_binding", binding)
    py_setitem(fields, "_train_entry", entry)
    py_setitem(fields, "_parameters", parameters)
    py_setitem(fields, "_m", moment)
    py_setitem(fields, "_v", variance)
    py_setitem(fields, "_completed", completed)
    py_setitem(fields, "_scalars", scalar_objects)
    py_setitem(fields, "_step_words", Python.tuple(shape.batch(), shape.length() + 1, shape.vocab(), n))
    py_setitem(fields, "_gradient", PythonObject(None))
    _ = _host_provider(fields)
    return PythonObject(None)


def byte_lm_host_from_state(cls: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if cls is not context["byte_lm_host_trainer_class"]:
        raise Error("TypeError: host restoration requires the connected native trainer class")
    var bi = native_loaded_module("builtins")
    var options = args[4]
    classical_admit_value(options, context)
    if py_type(options) is not bi.dict:
        raise Error("TypeError: host trainer options require native dictionary storage")
    var keys = py_call(bi.list, options)
    for i in range(len(keys)):  # driver-loop: over admitted constructor keyword metadata
        var key = String(keys[i])
        if key != "shape" and key != "lr" and key != "betas" and key != "eps" and key != "weight_decay":
            raise Error("TypeError: host trainer received an unexpected keyword argument '" + key + "'")
    var constructor = Python.tuple(args[0], args[1], args[2], py_get(options, "shape"), args[3],
        py_get(options, "lr", PythonObject(.001)), py_get(options, "betas", Python.tuple(.9, .999)),
        py_get(options, "eps", PythonObject(.00000001)), py_get(options, "weight_decay", PythonObject(.01)))
    var result = native_storage_allocate(cls)
    _ = byte_lm_host_construct(result, constructor, context)
    return result


def byte_lm_host_metadata(trainer: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    if py_type(trainer) is not context["byte_lm_host_trainer_class"]:
        raise Error("TypeError: host metadata requires the connected native trainer class")
    var fields = py_fields(trainer)
    if operation == "profile":
        return _host_profile(fields)
    if operation == "parameters_sha256":
        var value = fields["_parameters"]
        classical_admit_value(value, context)
        return PythonObject(sha256_hex_addr(Int(py=classical_address(value)), Int(py=value.size) * 4))
    var name = String("_" + operation)
    if operation == "completed_steps":
        name = "_completed"
    return fields[name]


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
        if operation == "construct":
            return byte_lm_host_construct(call[0], call[2], call[3]).steal_data()
        if operation == "from_state":
            return byte_lm_host_from_state(call[0], call[2], call[3]).steal_data()
        if operation == "shape" or operation == "profile" or operation == "completed_steps" or operation == "parameters" or operation == "m" or operation == "v" or operation == "gradient" or operation == "parameters_sha256":
            return byte_lm_host_metadata(call[0], operation, call[3]).steal_data()
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
        if text.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if text.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
