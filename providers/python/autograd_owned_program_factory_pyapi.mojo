# SPDX-License-Identifier: Apache-2.0
# Native creation/load preparation and finished owner conversion. Executor
# allocation is supplied by the selected HostExec/DeviceExec binding image.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type
from core.native_descriptor_call import native_descriptor_words
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session_registry_token
from providers.python.neural_descriptor_pyapi import neural_descriptor_resource_handle
from providers.python.autograd_owned_program_public_pyapi import NativeOwnedProgramCreateRequest
from providers.python.autograd_distributed_pyapi import distributed_devices, distributed_open, distributed_load, distributed_object
from providers.python.autograd_partition_pyapi import partition_open, partition_load
from providers.python.autograd_program_pyapi import _integer
from providers.python.autograd_program_arrays_pyapi import program_native_input
from bindings.native_json import native_path


def owned_program_create_envelope[KIND: Int](
    aa: Int, na: Int, pp: Int, np: Int, nf: Int, result: Int,
) raises -> MutPointer[NativeOwnedProgramCreateRequest, MutUntrackedOrigin]:
    var addresses = native_descriptor_words(aa, na)
    var scope = native_descriptor_words(pp, np)
    if na != 1 or np != 2 or nf != 0 or result == 0:
        raise Error("native captured program: invalid creation envelope")
    if addresses[0] == 0 or scope[1] != KIND or (scope[0] != 0 and scope[0] != 1):
        raise Error("native captured program: creation belongs to another executor")
    var request = MutPointer[NativeOwnedProgramCreateRequest, MutUntrackedOrigin](unsafe_from_address=addresses[0])
    var owner_scope: List[Int] = [ops_session_registry_token(), KIND]
    if neural_descriptor_resource_handle(request[].lifetime, owner_scope) != request[].sid or request[].sid <= 0:
        raise Error("TypeError: captured program creation belongs to another Session registry")
    var partitioned = request[].operation == "partition" or request[].operation == "load_partitioned"
    if scope[0] != Int(partitioned):
        raise Error("TypeError: captured program creation has a different owner kind")
    return request


def owned_program_create_devices(request: NativeOwnedProgramCreateRequest) raises -> List[Int]:
    var args = request.args
    if py_type(args) is not py_type(Python.tuple()) or len(args) != 2:
        raise Error("TypeError: captured program creation requires two native arguments")
    var op = request.operation
    if op == "load_distributed" or op == "load_partitioned":
        _ = native_path(args[0])
        return distributed_devices(args[1])
    if op != "distribute" and op != "partition":
        raise Error("ValueError: unsupported captured program creation operation")
    var options = args[1]
    var count = 6 if op == "partition" else 4
    if py_type(options) is not py_type(Python.tuple()) or len(options) != count:
        raise Error("TypeError: captured program creation requires native option storage")
    # Validate all options before allocating any device worker. Native integer
    # values are immutable; input-buffer descriptors are captured only at fit/run.
    for i in range(count - 2):  # small-loop(count: topology words and optimizer handle)
        _ = _integer(options[i], "captured program option", 1)
    _ = _integer(options[count - 1], "seed", 0)
    var parameters = program_native_input(options[count - 2])
    for i in range(len(parameters)):  # driver-loop: over explicit trainable parameter handles
        _ = _integer(parameters[i], "parameter", 1)
    return distributed_devices(args[0])


def owned_program_create_execute[E: Exec & Movable & Deinitable, KIND: Int](
    mut request: NativeOwnedProgramCreateRequest, ids: List[Int], var executors: List[E],
) raises:
    var op = request.operation
    var args = request.args
    var handle = PythonObject(None)
    if op == "load_partitioned":
        handle = partition_load[E, KIND](args[0], ids, executors^)
    elif op == "load_distributed":
        handle = distributed_load[E, KIND](args[0], ids, executors^)
    elif op == "partition":
        var settings = args[1]
        handle = partition_open[E, KIND](PythonObject(request.sid), PythonObject(request.pid), ids,
            settings[0], settings[1], settings[2], settings[3], settings[4], settings[5], executors^)
    else:
        var settings = args[1]
        handle = distributed_open[E, KIND](PythonObject(request.sid), PythonObject(request.pid), ids,
            settings[0], settings[1], settings[2], settings[3], executors^)
    request.result = distributed_object(request.cls, request.binding, handle)
