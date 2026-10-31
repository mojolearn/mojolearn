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
from std.memory import bitcast, memcpy
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.byte_lm_trainer_native import _validated_state
from bindings.byte_checkpoint_io import ByteCheckpointIOError
from providers.python.byte_lm_json_checkpoint_pyapi import byte_json_payload, byte_json_arrays, byte_json_read
from bindings.linalg_ops_native import linalg_ops_operation
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_numpy_metadata import native_numpy_type
from bindings.native_buffer_owner import native_storage_allocate
from providers.python.byte_lm_metadata_pyapi import _Shape, _shape_of, byte_lm_lengths, byte_lm_config_operation
from providers.python.training_buffer_pyapi import training_f32_cast_owned
from core.training_state_validation import training_byte_shape_validate
from core.sha256 import sha256_hex_addr
from bindings.tsa_hw_native import tsa_array
from core.neural_block_public_impl import neural_copy_row_prefixes
from core.host_helpers_impl import argmax_rows_f32
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


def _host_words(mut mods: NativeModules, fields: PythonObject, threaded: Int = 0, threads: Int = 0,
                batch: Int = 0, length: Int = 0, completed: Int = 0) raises -> List[Int]:
    var native = fields["_native"]
    var b = mods.module("builtins")
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


def _host_shape(mut mods: NativeModules, config: PythonObject, context: PythonObject) raises -> Tuple[_Shape, PythonObject]:
    var bi = mods.module("builtins")
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


def _host_profile_pair(mut mods: NativeModules, fields: PythonObject) raises -> List[String]:
    var provider = _host_provider(fields)
    mods.released()
    var result = List[String](length=2, fill=String(""))
    _ = neural_descriptor_execute(provider, 4, [Int(result.unsafe_ptr())], _host_words(mods, fields), List[Float64](), False)
    mods.released()
    return result^


def _host_profile(mut mods: NativeModules, fields: PythonObject) raises -> PythonObject:
    var result = _host_profile_pair(mods, fields)
    return PythonObject(result[0])


def _inference_parameters(mut mods: NativeModules, source: PythonObject, shape: _Shape,
                          context: PythonObject) raises -> Tuple[PythonObject, String]:
    var bi = mods.module("builtins")
    if py_type(source) is not bi.dict:
        return (training_f32_cast_owned(source, [shape.n_total()], "parameters", context), String("float32"))
    # Inspect native dictionary keys without invoking mapping/protocol hooks.
    # Packed classes are storage only; their buffers pass the same admission as
    # a direct linalg call, and decoding uses that existing provider/kernel.
    var names = shape.names()
    var shapes = shape.shapes()
    var keys = py_call(bi.list, source)
    var missing = Python.list()
    var extra = Python.list()
    for i in range(len(keys)):  # driver-loop: over native key types before any dictionary lookup
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: parameter dictionary names must be exact strings")
    for i in range(len(names)):  # driver-loop: over immutable parameter registry metadata
        if not py_contains(source, names[i]):
            _ = py_call(missing.append, names[i])
    for i in range(len(keys)):  # driver-loop: over supplied native parameter dictionary keys
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: parameter dictionary names must be exact strings")
        var known = False
        for j in range(len(names)):  # driver-loop: over immutable parameter registry names
            known = known or String(py=keys[i]) == names[j]
        if not known:
            _ = py_call(extra.append, keys[i])
    if len(missing) != 0 or len(extra) != 0:
        raise Error("mojolearn.LanguageModelInference: parameter dict mismatch; missing "
            + String(py=py_call(bi.repr, missing)) + ", unknown " + String(py=py_call(bi.repr, extra)))
    var format = String("float32")
    var owners = List[PythonObject]()
    var formats = List[Int]()
    for i in range(len(names)):  # driver-loop: over packed storage tags before allocating model state
        var item = source[names[i]]
        var selected = String("float32")
        var tag = 0
        if py_type(item) is context["byte_lm_bf16_class"]:
            selected = "bfloat16"
            tag = 1
            item = Python.tuple(PythonObject(None), py_fields(item)["bits"])
        elif py_type(item) is context["byte_lm_int8_class"]:
            selected = "int8"
            tag = 2
            var fields = py_fields(item)
            item = Python.tuple(PythonObject(None), fields["codes"], fields["exponents"])
        owners.append(item)
        formats.append(tag)
        if selected != "float32":
            if format != "float32" and format != selected:
                raise Error("mojolearn.lowbit: weights mix formats ['bfloat16', 'int8']; store a model one way")
            format = selected
    var result = _f32_array(shape.n_total(), context)
    var target = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result)))
    var offset = 0
    for i in range(len(names)):  # driver-loop: over admitted named tensors, not model arithmetic
        var item = owners[i]
        if formats[i] == 1:
            item = linalg_ops_operation(PythonObject(None), "family_lin_from_bf16", item, context["byte_lm_linalg_context"])
            mods.released()
        elif formats[i] == 2:
            item = linalg_ops_operation(PythonObject(None), "family_lin_dequantize_int8", item, context["byte_lm_linalg_context"])
            mods.released()
        var owned = training_f32_cast_owned(item, shapes[i], names[i], context)
        var count = 1
        for j in range(len(shapes[i])):  # driver-loop: over admitted tensor shape metadata
            count *= shapes[i][j]
        memcpy(dest=target + offset,
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(owned))), count=count)
        offset += count
        _ = owned
    return (result, format^)


def byte_lm_host_inference_construct(mut mods: NativeModules, model: PythonObject, args: PythonObject,
                                     context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(model) is not context["byte_lm_host_inference_class"]:
        raise Error("TypeError: host inference requires the connected native inference class")
    classical_connect_numpy(context)
    var captured = _host_shape(mods, args[1], context)
    var shape = captured[0].copy()
    if py_type(args[2]) is not bi.bool:
        raise Error("TypeError: threaded must be a bool")
    var pending = bi.dict()
    py_setitem(pending, "_threaded", args[2])
    py_setitem(pending, "_threads", args[3])
    _ = _host_threads(mods, pending, PythonObject(None), PythonObject(None))
    var packed = _inference_parameters(mods, args[0], shape, context)
    var binding = context["byte_lm_host_binding"]
    if py_type(binding) is bi.str:
        raise Error(String(py=binding))
    py_setitem(pending, "_binding", binding)
    py_setitem(pending, "_native", shape.native_shape())
    var profiles = _host_profile_pair(mods, pending)
    var expected = shape.profile(String(py=context["byte_lm_default_profile"])) + profiles[1]
    if profiles[0] != expected:
        raise Error("RuntimeError: byte LM host profile mismatch: " + profiles[0] + " != " + expected)
    py_setitem(pending, "_shape", captured[1])
    py_setitem(pending, "_parameters", packed[0])
    py_setitem(pending, "weight_format", packed[1])
    # Publish only after admission, allocation, decoding and profile agreement.
    _ = py_call(py_fields(model).update, pending)
    return PythonObject(None)


def byte_lm_host_inference_checkpoint(mut mods: NativeModules, cls: PythonObject, args: PythonObject,
        context: PythonObject, mut io_error: ByteCheckpointIOError) raises -> PythonObject:
    if cls is not context["byte_lm_host_inference_class"]:
        raise Error("TypeError: host checkpoint restoration requires the connected native inference class")
    classical_connect_numpy(context)
    var source = byte_json_payload(byte_json_read(args[0], io_error))
    mods.released()
    var described = _shape_of(source)
    training_byte_shape_validate(described.d, described.chunked)
    byte_json_arrays(source, described.n_total(), described.n_tensors())
    mods.released()
    # This is the same complete state validator used by the GPU trainer:
    # integrity, mode/profile, tensor registry, optimizer, cursor and schedule.
    var state_context = context["byte_lm_checkpoint_context"]
    classical_connect_numpy(state_context)
    var state = _validated_state(mods, source, state_context)
    var shape = _host_shape_from_state(state, context)
    mods.released()
    var result = native_storage_allocate(cls)
    _ = byte_lm_host_inference_construct(mods, result,
        Python.tuple(state["parameters"], shape, args[1], args[2]), context)
    return result


def _host_shape_from_state(state: PythonObject, context: PythonObject) raises -> PythonObject:
    # Reuse the config owner instead of invoking its Python constructor.
    return byte_lm_config_operation(state, "config_state", Python.tuple(), context)


def byte_lm_host_construct(mut mods: NativeModules, trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(trainer) is not context["byte_lm_host_trainer_class"]:
        raise Error("TypeError: host training requires the connected native trainer class")
    classical_connect_numpy(context)
    var profile = py_get(context["byte_lm_profile_defaults"], "training", PythonObject("fp32_v1"))
    if py_type(profile) is not bi.str or not Bool(py=context["byte_lm_profiles"][profile]["training"]):
        raise Error("NotImplementedError: the selected numeric profile is unavailable for Byte-LM training")
    var captured = _host_shape(mods, args[3], context)
    var shape = captured[0].copy()
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
    mods.released()
    return PythonObject(None)


def byte_lm_host_from_state(mut mods: NativeModules, cls: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if cls is not context["byte_lm_host_trainer_class"]:
        raise Error("TypeError: host restoration requires the connected native trainer class")
    var bi = mods.module("builtins")
    var options = args[4]
    if py_type(options) is not bi.dict:
        raise Error("TypeError: host trainer options require native dictionary storage")
    var keys = py_call(bi.list, options)
    for i in range(len(keys)):  # driver-loop: over admitted constructor keyword metadata
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: host trainer options require string keyword names")
        var key = String(keys[i])
        if key != "shape" and key != "lr" and key != "betas" and key != "eps" and key != "weight_decay":
            raise Error("TypeError: host trainer received an unexpected keyword argument '" + key + "'")
        # shape is a ByteLanguageModelConfig owner: _host_shape admits its
        # exact class and native fields in construct. Admitting the owner
        # itself as a generic value refuses every valid restoration.
        if key != "shape":
            classical_admit_value(options[keys[i]], context)
    var constructor = Python.tuple(args[0], args[1], args[2], py_get(options, "shape"), args[3],
        py_get(options, "lr", PythonObject(.001)), py_get(options, "betas", Python.tuple(.9, .999)),
        py_get(options, "eps", PythonObject(.00000001)), py_get(options, "weight_decay", PythonObject(.01)))
    var result = native_storage_allocate(cls)
    _ = byte_lm_host_construct(mods, result, constructor, context)
    return result


def byte_lm_host_metadata(mut mods: NativeModules, trainer: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    if py_type(trainer) is not context["byte_lm_host_trainer_class"] and py_type(trainer) is not context["byte_lm_host_inference_class"]:
        raise Error("TypeError: host metadata requires the connected native trainer class")
    var fields = py_fields(trainer)
    if operation == "profile":
        return _host_profile(mods, fields)
    if operation == "parameters_sha256":
        var value = fields["_parameters"]
        classical_admit_value(value, context)
        return PythonObject(sha256_hex_addr(Int(py=classical_address(value)), Int(py=value.size) * 4))
    var name = String("_" + operation)
    if operation == "completed_steps":
        name = "_completed"
    return fields[name]


def _host_threads(mut mods: NativeModules, fields: PythonObject, threaded: PythonObject, threads: PythonObject) raises -> Tuple[Int, Int]:
    var bi = mods.module("builtins")
    var flag = fields["_threaded"] if threaded is None else threaded
    if py_type(flag) is not bi.bool:
        raise Error("TypeError: threaded must be a bool or None")
    var selected = fields["_threads"] if threads is None else threads
    var count = 0
    if selected is not None:
        if py_type(selected) is not bi.int:
            raise Error("TypeError: threads must be an int or None")
        count = Int(py=selected)
        if count < 1 or count > 1024:
            raise Error("threads must be in [1, 1024]")
    return (1 if Bool(py=flag) else 0, count)


def byte_lm_host_infer(mut mods: NativeModules, model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if py_type(model) is not context["byte_lm_host_inference_class"]:
        raise Error("TypeError: host inference requires the connected native model class")
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var threaded = _host_threads(mods, fields, args[2], args[3])
    var words = _host_words(mods, fields, threaded[0], threaded[1])
    var provider = _host_provider(fields)
    mods.released()
    var is_loss = operation == "loss_bits" or operation == "inference_loss"
    classical_admit_value(args[0], context)
    classical_admit_value(args[1], context)
    var value = classical_input(args[0], context, "ids")
    var dims = value.shape
    var strides = value.strides
    if len(dims) != 2:
        raise Error("ids must be a two-dimensional array")
    var batch = Int(py=dims[0])
    var length = Int(py=dims[1])
    var vocab = words[8]
    if is_loss:
        if batch != words[0] or length != words[1] + 1:
            raise Error("ids must be [" + String(words[0]) + ", " + String(words[1] + 1) + "]")
    elif batch < 1 or batch > 1048576 or length < 1 or length > words[1]:
        raise Error("ids must be [batch, 1.." + String(words[1]) + "]")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "ids", context)
    var tokens = tsa_array("i", "<i4", Python.tuple(batch, length), batch * length, 4, context)
    var token_address = Int(py=classical_address(tokens))
    neural_cast_i32_2d(Int(py=classical_address(value)), Int(py=strides[0]), Int(py=strides[1]),
        batch, length, item, kind, _I32P(unsafe_from_address=token_address))
    _ = value^
    var sizes = List[Int](length=batch, fill=length)
    if args[1] is not None:
        sizes = byte_lm_lengths(args[1], batch, length, context)
        mods.released()
        var padded = tsa_array("i", "<i4", Python.tuple(batch, length), batch * length, 4, context)
        neural_copy_row_prefixes(Int(py=classical_address(padded)), token_address, sizes, batch, length, 1)
        # token_address only borrows the old Array's storage. Keep its owner
        # alive through the prefix copy before replacing it with padded.
        # Installed CPU verifier failures and both threaded/unthreaded public
        # regressions are retained in byte-lm-ragged-20261030. The output copy
        # below already keeps its corresponding source owner alive this way.
        _ = tokens
        tokens = padded
        token_address = Int(py=classical_address(tokens))
    var bad = neural_first_bad_id(_I32P(unsafe_from_address=token_address), batch, length, vocab,
        length - 1 if is_loss else -1, _IGNORE_INDEX)
    if bad >= 0:
        var found = Int(_I32P(unsafe_from_address=token_address)[bad])
        raise Error("ids must be byte values in [0, " + String(vocab) + "); got " + String(found)
            + " at row " + String(bad // length) + ", position " + String(bad % length))
    var parameters = fields["_parameters"]
    var addresses: List[Int] = [Int(py=classical_address(parameters)), token_address]
    words[12] = batch
    words[13] = length
    if is_loss:
        var bits = neural_descriptor_execute(provider, 2, addresses, words, List[Float64](), False)
        mods.released()
        _ = tokens
        _ = parameters
        if operation == "inference_loss":
            return PythonObject(Float64(bitcast[DType.float32](UInt32(bits))))
        return PythonObject(bits)
    if operation == "next_bytes" and args[1] is None and threaded[0] == 1:
        var next = tsa_array("i", "<i4", Python.tuple(batch), batch, 4, context)
        var next_address = Int(py=classical_address(next))
        addresses.append(next_address)
        var written = neural_descriptor_execute(provider, 1, addresses, words, List[Float64](), False)
        mods.released()
        if written != batch:
            raise Error("RuntimeError: byte LM host wrote an unexpected number of next bytes")
        var output = Python.list()
        var ids = _I32P(unsafe_from_address=next_address)
        for row in range(batch):  # driver-loop: over completed next-token results, publishing one per batch row
            _ = py_call(output.append, Int(ids[row]))
        _ = next
        _ = tokens
        _ = parameters
        return output
    if batch * length > 268435456 // vocab:
        raise Error("byte LM host: logits exceed the admitted span")
    var out = tsa_array("f", "<f4", Python.tuple(batch, length, vocab), batch * length * vocab, 4, context)
    var out_address = Int(py=classical_address(out))
    addresses.append(out_address)
    var written = neural_descriptor_execute(provider, 0, addresses, words, List[Float64](), False)
    mods.released()
    if written != batch * length * vocab:
        raise Error("RuntimeError: byte LM host wrote an unexpected number of logits")
    _ = tokens
    _ = parameters
    if operation == "next_bytes":
        var indices = List[Int64](length=batch, fill=0)
        var source = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_address)
        for row in range(batch):  # driver-loop: over admitted sequences, selecting each native result row
            argmax_rows_f32(source + (row * length + sizes[row] - 1) * vocab, 1, vocab,
                MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(indices.unsafe_ptr()) + row * 8))
        var output = Python.list()
        for row in range(batch):  # driver-loop: over completed next-token results, publishing one per batch row
            _ = py_call(output.append, Int(indices[row]))
        _ = out
        return output
    if args[1] is not None:
        var padded = tsa_array("f", "<f4", Python.tuple(batch, length, vocab), batch * length * vocab, 4, context)
        neural_copy_row_prefixes(Int(py=classical_address(padded)), out_address, sizes, batch, length, vocab)
        _ = out
        return padded
    return out


def byte_lm_host_train_step(mut mods: NativeModules, trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
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
    mods.released()
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
    if py_type(completed) is not mods.module("builtins").int:
        raise Error("TypeError: completed_steps requires a native integer")
    var native = _host_words(mods, fields, completed=Int(py=completed))
    # Preserve the old GIL-held host operation and transactional output publish.
    var bits = neural_descriptor_execute(provider, 3, descriptors, native, scalars, False)
    mods.released()
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
    var mods = NativeModules()
    var io_error = ByteCheckpointIOError()
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(py=call[1])
        if operation == "logits" or operation == "next_bytes" or operation == "loss_bits" or operation == "inference_loss":
            return byte_lm_host_infer(mods, call[0], operation, call[2], call[3]).steal_data()
        if operation == "inference_from_checkpoint":
            return byte_lm_host_inference_checkpoint(mods, call[0], call[2], call[3], io_error).steal_data()
        if operation == "inference_construct":
            return byte_lm_host_inference_construct(mods, call[0], call[2], call[3]).steal_data()
        if operation == "construct":
            return byte_lm_host_construct(mods, call[0], call[2], call[3]).steal_data()
        if operation == "from_state":
            return byte_lm_host_from_state(mods, call[0], call[2], call[3]).steal_data()
        if operation == "shape" or operation == "profile" or operation == "completed_steps" or operation == "parameters" or operation == "m" or operation == "v" or operation == "gradient" or operation == "parameters_sha256":
            return byte_lm_host_metadata(mods, call[0], operation, call[3]).steal_data()
        if operation == "native_provider_info":
            return neural_descriptor_info(py_fields(call[0])["_native_provider"]).steal_data()
        if operation != "train_step" and operation != "loss":
            raise Error("RuntimeError: unknown byte LM host trainer operation " + operation)
        var result = byte_lm_host_train_step(mods, call[0], call[2], call[3])
        if operation == "loss":
            return PythonObject(Float64(bitcast[DType.float32](UInt32(py=result)))).steal_data()
        return result.steal_data()
    except error:
        if io_error.pending:
            return io_error^.restore()
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
