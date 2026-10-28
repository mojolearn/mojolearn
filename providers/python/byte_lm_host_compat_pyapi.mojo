# SPDX-License-Identifier: Apache-2.0
# Thin compatibility exports for CPU-only LanguageModelTrainer. All state and
# orchestration live in ByteHostSession; all computation uses existing host
# execution. The same common provider table serves normal product operations.
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.py_balanced import py_type, py_call
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from training.byte_lm_host_session import ByteHostSession, host_optimizer
from training.byte_lm_host_trainer_dispatch import BYTE_HOST_TRAINER_TABLE, host_trainer_operation
from training.byte_lm_config import ByteConfig, byte_lm_arithmetic_suffix
from bindings.sha256_door import sha256_hexdigest
from core.sha256_file import sha256_file
from bindings.native_json import native_path


def _int(value: PythonObject) raises -> Int:
    if py_type(value) is not native_loaded_module("builtins").int:
        raise Error("byte LM: metadata requires native integers without conversion hooks")
    return Int(py=value)


def _words(value: PythonObject) raises -> List[Int]:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.list and py_type(value) is not b.tuple:
        raise Error("byte LM: descriptors require native lists or tuples")
    var result = List[Int]()
    for i in range(len(value)):  # driver-loop: over public ABI descriptor metadata
        result.append(_int(value[i]))
    return result^


def _shape(value: PythonObject) raises -> List[Int]:
    var result = _words(value)
    if len(result) != 7 and len(result) != 9 and len(result) != 10:
        raise Error("byte LM: expected 7, 9 or 10 shape integers")
    if len(result) == 7:
        result.append(2)
        result.append(256)
    if len(result) == 9:
        result.append(0)
    return result^


def _config(p: List[Int]) raises -> ByteConfig:
    if p[9] != 0 and p[9] != 1:
        raise Error("byte LM: chunked head selector must be 0 or 1")
    var cfg = ByteConfig(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9] == 1)
    cfg.validate()
    return cfg^


def _plan(shape: PythonObject, params: PythonObject) raises -> Tuple[List[Int], List[Float64]]:
    var b = native_loaded_module("builtins")
    if (py_type(params) is not b.list and py_type(params) is not b.tuple) or len(params) != 12:
        raise Error("byte LM: expected 12 scalar parameters")
    var words = _shape(shape)
    words.append(_int(params[0]))
    words.append(_int(params[1]))
    words.append(_int(params[2]))
    words.append(_int(params[10]))
    words.append(0)
    words.append(0)
    words.append(0)
    var scalars = List[Float64]()
    for i in [3, 4, 5, 6, 7, 8, 9, 11]:  # driver-loop: over fixed optimizer ABI scalar positions
        if py_type(params[i]) is not b.int and py_type(params[i]) is not b.float:
            raise Error("byte LM: optimizer requires native scalar metadata")
        scalars.append(Float64(py=params[i]))
    _ = _config(words)
    _ = host_optimizer(words, scalars)
    return (words^, scalars^)


def _defaults(shape: PythonObject, completed: Int) raises -> Tuple[List[Int], List[Float64]]:
    return _plan(shape, Python.list(0, completed, 2, 0.001, 0.9, 0.999, 1e-8, 0.01, 0, 0, 0, 0))


def _execute(op: Int, session: PythonObject, addresses: PythonObject, params: PythonObject,
             shape: PythonObject, dims: PythonObject, named: PythonObject, release: Int = 0) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteHostSession]()
    var plan = _plan(shape, params)
    var words = plan[0].copy()
    words[16] = release
    if dims is not None:
        var dimensions = _words(dims)
        if len(dimensions) != 2:
            raise Error("byte LM logits: expected 2 dimensions")
        words[14] = dimensions[0]
        words[15] = dimensions[1]
    var values = _words(addresses)
    var descriptors: List[Int] = [Int(owner)]
    for i in range(len(values)):  # driver-loop: over admitted operation buffer descriptors
        descriptors.append(values[i])
    if named is not None:
        var extras = _words(named)
        for i in range(len(extras)):  # driver-loop: over optional named gradient buffers
            descriptors.append(extras[i])
    return PythonObject(host_trainer_operation(op, descriptors, words, plan[1]))


def host_compat_api() raises -> PythonObject:
    var table = BYTE_HOST_TRAINER_TABLE.get_or_create_ptr()
    return neural_descriptor_capsule(table[].name, table[].entries)


def host_compat_handle(session: PythonObject) raises -> PythonObject:
    return PythonObject(Int(session.downcast_value_ptr[ByteHostSession]()))


def host_compat_create() raises -> PythonObject:
    return PythonObject(alloc=ByteHostSession())


def host_compat_close(session: PythonObject) raises -> PythonObject:
    session.downcast_value_ptr[ByteHostSession]()[].close()
    return PythonObject(0)


def host_compat_info(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteHostSession]()
    return Python.list(owner[].completed if owner[].opened else -1, owner[].grad_step if owner[].opened else -1,
                       Int(owner[].usable), Int(owner[].opened))


def host_compat_mode() -> PythonObject:
    return PythonObject(1)


def host_compat_vendor() raises -> PythonObject:
    return PythonObject("cpu")


def host_compat_profile() raises -> PythonObject:
    # The former CPU proxy used the public config default (chunked=False),
    # independent of a binary experiment switch's optional default.
    return PythonObject(ByteConfig(chunked_lm_head_v2=False).profile())


def host_compat_suffix() raises -> PythonObject:
    return PythonObject(byte_lm_arithmetic_suffix())


def host_compat_shape_profile(shape: PythonObject) raises -> PythonObject:
    return PythonObject(_config(_shape(shape)).profile())


def host_compat_sha(data: PythonObject) raises -> PythonObject:
    return sha256_hexdigest(data)


def host_compat_file_sha(path: PythonObject) raises -> PythonObject:
    return PythonObject(sha256_file(String(py=native_path(path))))


def host_compat_run(addresses: PythonObject, named: PythonObject, params: PythonObject, shape: PythonObject) raises -> PythonObject:
    var session = host_compat_create()
    return _execute(8, session, addresses, params, shape, PythonObject(None), named)


def host_compat_run_configured(addresses: PythonObject, params: PythonObject, shape: PythonObject) raises -> PythonObject:
    return host_compat_run(addresses, Python.list(), params, shape)


def host_compat_run_default(addresses: PythonObject, params: PythonObject) raises -> PythonObject:
    return host_compat_run_configured(addresses, params, Python.list(2, 32, 32, 4, 2, 8, 64, 2, 256))


def host_compat_trainer_step(session: PythonObject, addresses: PythonObject, params: PythonObject, shape: PythonObject) raises -> PythonObject:
    return _execute(1, session, addresses, params, shape, PythonObject(None), PythonObject(None))


def host_compat_trainer_logits(session: PythonObject, addresses: PythonObject, dims: PythonObject,
                               params: PythonObject, shape: PythonObject) raises -> PythonObject:
    return _execute(2, session, addresses, params, shape, dims, PythonObject(None))


def host_compat_trainer_next(session: PythonObject, addresses: PythonObject, dims: PythonObject,
                            params: PythonObject, shape: PythonObject) raises -> PythonObject:
    return _execute(6, session, addresses, params, shape, dims, PythonObject(None))


def host_compat_trainer_export(session: PythonObject, addresses: PythonObject, shape: PythonObject,
                              completed: PythonObject, release: PythonObject) raises -> PythonObject:
    var plan = _defaults(shape, _int(completed))
    var owner = session.downcast_value_ptr[ByteHostSession]()
    var values = _words(addresses)
    var a: List[Int] = [Int(owner)]
    for i in range(len(values)):  # driver-loop: over state output descriptors
        a.append(values[i])
    var words = plan[0].copy()
    if py_type(release) is not native_loaded_module("builtins").bool:
        raise Error("byte LM: release must be a bool")
    words[16] = Int(Bool(py=release))
    return PythonObject(host_trainer_operation(3, a, words, plan[1]))


def host_compat_trainer_grad(session: PythonObject, addresses: PythonObject, shape: PythonObject,
                            completed: PythonObject) raises -> PythonObject:
    var plan = _defaults(shape, _int(completed))
    var values = _words(addresses)
    var a: List[Int] = [Int(session.downcast_value_ptr[ByteHostSession]())]
    for i in range(len(values)):  # driver-loop: over gradient output descriptors
        a.append(values[i])
    return PythonObject(host_trainer_operation(10, a, plan[0], plan[1]))


def host_compat_logits(addresses: PythonObject, dims: PythonObject, shape: PythonObject) raises -> PythonObject:
    var plan = _defaults(shape, 0)
    var words = plan[0].copy()
    var dimensions = _words(dims)
    if len(dimensions) != 2:
        raise Error("byte LM logits: expected 2 dimensions")
    words[14] = dimensions[0]
    words[15] = dimensions[1]
    return PythonObject(host_trainer_operation(7, _words(addresses), words, plan[1]))


def host_compat_open(session: PythonObject, addresses: PythonObject, params: PythonObject, shape: PythonObject) raises -> PythonObject:
    var plan = _plan(shape, params)
    var opt = host_optimizer(plan[0], plan[1])
    return PythonObject(session.downcast_value_ptr[ByteHostSession]()[].open(
        _words(addresses), _config(plan[0]), opt, plan[0][11]))


def host_compat_step(session: PythonObject, addresses: PythonObject, params: PythonObject, shape: PythonObject) raises -> PythonObject:
    var plan = _plan(shape, params)
    if plan[0][10] != 1:
        raise Error("byte LM: session step requires action 1")
    return PythonObject(session.downcast_value_ptr[ByteHostSession]()[].step(
        _words(addresses), _config(plan[0]), host_optimizer(plan[0], plan[1]), plan[0][11]))


def host_compat_eval(session: PythonObject, addresses: PythonObject, params: PythonObject, shape: PythonObject) raises -> PythonObject:
    var plan = _plan(shape, params)
    if plan[0][10] != 0:
        raise Error("byte LM: session eval requires action 0")
    return PythonObject(session.downcast_value_ptr[ByteHostSession]()[].evaluate(
        _words(addresses), _config(plan[0]), host_optimizer(plan[0], plan[1]), plan[0][11]))


def host_compat_rollback(session: PythonObject) raises -> PythonObject:
    return PythonObject(session.downcast_value_ptr[ByteHostSession]()[].rollback())


def host_compat_export(session: PythonObject, addresses: PythonObject, shape: PythonObject) raises -> PythonObject:
    return PythonObject(session.downcast_value_ptr[ByteHostSession]()[].export_state(_words(addresses), _config(_shape(shape))))


def host_compat_grad(session: PythonObject, addresses: PythonObject, shape: PythonObject) raises -> PythonObject:
    var values = _words(addresses)
    if len(values) != 1:
        raise Error("byte LM: byte_lm_session_export_gradients expects 1 address")
    return PythonObject(session.downcast_value_ptr[ByteHostSession]()[].export_gradients(values[0], _config(_shape(shape))))


def _resident_logits(session: PythonObject, addresses: PythonObject, dims: PythonObject,
                     shape: PythonObject, completed: PythonObject, next_bytes: Bool) raises -> PythonObject:
    var dimensions = _words(dims)
    var cfg = _config(_shape(shape))
    if len(dimensions) != 2 or dimensions[0] < 1 or dimensions[0] > 1024 or dimensions[1] < 1 or dimensions[1] > cfg.length:
        raise Error("byte LM logits: batch or length outside admitted bounds")
    if dimensions[0] * dimensions[1] > 268435456 // cfg.vocab_size:
        raise Error("byte LM logits: batch * length * vocab exceeds the admitted span")
    return PythonObject(session.downcast_value_ptr[ByteHostSession]()[].logits(
        _words(addresses), dimensions[0], dimensions[1], cfg, _int(completed), next_bytes))


def host_compat_resident_logits(session: PythonObject, addresses: PythonObject, dims: PythonObject,
                               shape: PythonObject, completed: PythonObject) raises -> PythonObject:
    return _resident_logits(session, addresses, dims, shape, completed, False)


def host_compat_resident_next(session: PythonObject, addresses: PythonObject, dims: PythonObject,
                             shape: PythonObject, completed: PythonObject) raises -> PythonObject:
    return _resident_logits(session, addresses, dims, shape, completed, True)


def register(mut module: PythonModuleBuilder) raises:
    _ = module.add_type[ByteHostSession]("_ByteHostSession")
    module.def_function[host_compat_api]("byte_lm_trainer_api_v1")
    module.def_function[host_compat_handle]("byte_lm_native_session_handle_v1")
    module.def_function[host_compat_create]("byte_lm_session_create")
    module.def_function[host_compat_close]("byte_lm_session_close")
    module.def_function[host_compat_info]("byte_lm_session_info")
    module.def_function[host_compat_mode]("byte_lm_numeric_mode")
    module.def_function[host_compat_vendor]("byte_lm_vendor")
    module.def_function[host_compat_profile]("byte_lm_profile")
    module.def_function[host_compat_suffix]("byte_lm_arithmetic_suffix")
    module.def_function[host_compat_shape_profile]("byte_lm_config_profile")
    module.def_function[host_compat_sha]("sha256_hex")
    module.def_function[host_compat_file_sha]("byte_lm_file_sha256")
    module.def_function[host_compat_run_default]("byte_lm_run")
    module.def_function[host_compat_run_configured]("byte_lm_run_configured")
    module.def_function[host_compat_run]("byte_lm_trainer_run")
    module.def_function[host_compat_logits]("byte_lm_logits")
    module.def_function[host_compat_trainer_step]("byte_lm_trainer_step")
    module.def_function[host_compat_trainer_logits]("byte_lm_trainer_logits")
    module.def_function[host_compat_trainer_next]("byte_lm_trainer_next_bytes")
    module.def_function[host_compat_trainer_export]("byte_lm_trainer_export_state")
    module.def_function[host_compat_trainer_grad]("byte_lm_trainer_export_gradients")
    module.def_function[host_compat_open]("byte_lm_session_open")
    module.def_function[host_compat_step]("byte_lm_session_step")
    module.def_function[host_compat_eval]("byte_lm_session_eval")
    module.def_function[host_compat_rollback]("byte_lm_session_rollback")
    module.def_function[host_compat_export]("byte_lm_session_export_state")
    module.def_function[host_compat_grad]("byte_lm_session_export_gradients")
    module.def_function[host_compat_resident_logits]("byte_lm_session_logits")
    module.def_function[host_compat_resident_next]("byte_lm_session_next_bytes")
