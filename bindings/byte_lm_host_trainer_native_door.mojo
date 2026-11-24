# SPDX-License-Identifier: Apache-2.0
"""`byte_lm_host_trainer_call(model, operation, args, context)`: the native
calls of `LanguageModelInference` (python/mojolearn/_byte_lm_host.py), the
CPU forward over the host oracles. CPU training is
`LanguageModelTrainer(device="cpu")` (bindings/byte_lm_trainer_native_door.mojo).
"""
from bindings.host_class_device import host_class_device
from std.python import Python, PythonObject
from std.memory import bitcast, unsafe_memcpy
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.native_array_storage_door import native_array_zeros
from bindings.byte_lm_trainer_native_door import byte_state_unpack, _tier_process
from bindings.byte_checkpoint_io_unpack import ByteCheckpointIOError
from providers.python.byte_lm_json_checkpoint_pyapi_door import byte_json_payload, byte_json_arrays, byte_json_read
from bindings.linalg_ops_native_unpack import linalg_ops_operation
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_buffer_owner_pack import native_storage_allocate
from providers.python.byte_lm_metadata_pyapi_door import _Shape, _shape_of, byte_lm_lengths, byte_lm_config_operation
from bindings.admitted_sequence_door import admit_nd_owned_f32
from core.training_state_validation import training_byte_shape_validate
from core.sha256 import sha256_hex_addr
from bindings.tsa_hw_native_door import tsa_array
from core.neural_block_public_impl import neural_copy_row_prefixes
from core.host_helpers_impl import argmax_rows_f32
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry, neural_descriptor_reuse, neural_descriptor_execute
from bindings.classical_arrays_door import classical_input, classical_address, classical_cast_kind, classical_wrap_array
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED
from core.xseq_marshal import neural_first_bad_id
from core.admitted_ids import admitted_ids_i32
from bindings.byte_perf_helpers_pack import byte_int_list_from_i32, byte_int_list_from_i64

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _IGNORE_INDEX = -100


def _text(value: PythonObject) raises -> String:
    var b = native_loaded_module("builtins")
    return String(py=py_call(b.str, value))


def _f32_array(n: Int, context: PythonObject) raises -> PythonObject:
    """`_buffer.empty((n,), "<f4")`: a zero-filled mojolearn Array, on every
    install (the trainer's state is Arrays, NumPy or not)."""
    var store = _zero_store("f", n, context)
    return classical_wrap_array(store, Python.tuple(n), "<f4", "C", context)


def _zero_store(code: String, n: Int, context: PythonObject) raises -> PythonObject:
    # One zero-filled native store (sequence repeat of a zero seed), instead
    # of a zero bytes object copied into a second store. Same storage type;
    # an unexpected storage type keeps the original two-step allocation.
    var store = native_array_zeros(code, n)
    if py_type(store) is context["array_type"]:
        return store
    return py_call(context["array_type"], PythonObject(code), py_call(context["native_bytes_type"], 4 * n))


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
        return (admit_nd_owned_f32(source, [shape.n_total()], "parameters", context), String("float32"))
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
        var owned = admit_nd_owned_f32(item, shapes[i], names[i], context)
        var count = 1
        for j in range(len(shapes[i])):  # driver-loop: over admitted tensor shape metadata
            count *= shapes[i][j]
        unsafe_memcpy(dest=target + offset,
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(owned))), count=count)
        offset += count
        _ = owned
    return (result, format^)


def byte_lm_host_inference_construct(mut mods: NativeModules, model: PythonObject, args: PythonObject,
                                     context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(model) is not context["byte_lm_host_inference_class"]:
        raise Error("TypeError: host inference requires the connected native inference class")
    # `device=` (lane vdf-parallel, 2026-11-24; bindings/host_class_device.mojo):
    # the last item, read once before any work. Host only: None and "cpu" are
    # today's route, "gpu" is refused naming the class.
    _ = host_class_device(args[4], "LanguageModelInference", bi)
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
    # `device=`, as `byte_lm_host_inference_construct` (lane vdf-parallel).
    _ = host_class_device(args[3], "LanguageModelInference", mods.module("builtins"))
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
    var tier = _tier_process(mods, state_context)
    # UNPACK into the trainer's typed state (bindings/byte_lm_trainer_native_run.mojo
    # `ByteLmStateData`): all four arrays are admitted, the moments and flags
    # only to be validated. The model keeps the parameters alone.
    var data = byte_state_unpack(mods, source, state_context, tier)
    var shape = _host_shape_from_state(source, context)
    mods.released()
    var count = data.n_total
    var parameters = tsa_array("f", "<f4", Python.tuple(count), count, 4, state_context)
    unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(parameters))),
                  src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(data.parameters.unsafe_ptr())),
                  count=count)
    _ = data^  # the typed state lives through the copy above
    var result = native_storage_allocate(cls)
    _ = byte_lm_host_inference_construct(mods, result,
        Python.tuple(parameters, shape, args[1], args[2], PythonObject(None)), context)  # device admitted above
    return result


def _host_shape_from_state(state: PythonObject, context: PythonObject) raises -> PythonObject:
    # Reuse the config owner instead of invoking its Python constructor.
    return byte_lm_config_operation(state, "config_state", Python.tuple(), context)


def byte_lm_host_metadata(mut mods: NativeModules, trainer: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    if py_type(trainer) is not context["byte_lm_host_inference_class"]:
        raise Error("TypeError: host metadata requires the connected native inference class")
    var fields = py_fields(trainer)
    if operation == "profile":
        return _host_profile(mods, fields)
    if operation == "parameters_sha256":
        var value = fields["_parameters"]
        classical_admit_value(value, context)
        var digest = PythonObject(sha256_hex_addr(Int(py=classical_address(value)), Int(py=value.size) * 4))
        _ = value^  # read by address inside the call above: named again so it outlives it (lane lastuse-addr-arg)
        return digest
    return fields[String("_" + operation)]


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
    _ = admitted_ids_i32(Int(py=classical_address(value)), Int(py=strides[0]), Int(py=strides[1]),
        batch, length, item, kind == CAST_KIND_SIGNED, False, _I32P(unsafe_from_address=token_address))
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
        var output = byte_int_list_from_i32(next_address, batch)
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
        var output = byte_int_list_from_i64(Int(indices.unsafe_ptr()), batch)
        _ = indices^
        _ = out
        return output
    if args[1] is not None:
        var padded = tsa_array("f", "<f4", Python.tuple(batch, length, vocab), batch * length * vocab, 4, context)
        neural_copy_row_prefixes(Int(py=classical_address(padded)), out_address, sizes, batch, length, vocab)
        _ = out
        return padded
    return out


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
        if operation == "shape" or operation == "profile" or operation == "parameters_sha256":
            return byte_lm_host_metadata(mods, call[0], operation, call[3]).steal_data()
        # TOMBSTONE: LanguageModelHostTrainer (construct, from_state, train_step, loss, m, v, gradient,
        # completed_steps) deleted 2026-11-22 by lane neural-consolidate: the same composed host step
        # (training/byte_lm_host_execution.mojo byte_host_train_native) is LanguageModelTrainer(device="cpu"),
        # bit for bit; code recoverable at the lane's parent commit.
        raise Error("RuntimeError: unknown byte LM host inference operation " + operation)
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
