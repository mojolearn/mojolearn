# SPDX-License-Identifier: Apache-2.0
"""Native local process lifecycle and ordered request waves for model owners.

Workers import their installed ABI connection once, then enter a compiled
dispatch loop. The coordinator owns device visibility, framing, authentication,
process cleanup and request ordering. No pickle or Python worker loop is used.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_transport import (
    TRANSPORT_BOUNDARY, transport_initialize, transport_listen, transport_accept,
    transport_connect, transport_send, transport_receive, transport_close,
    transport_spawn, transport_wait, transport_terminate, transport_clock,
    transport_sleep, _transport_json,
)
from bindings.native_codec import codec_pack, codec_unpack
from bindings.native_json import native_json_parse
from bindings.py_balanced import py_call, py_get, py_setitem, py_type


comptime PROCESS_MAX_BYTES = (1 << 63) - 1


struct ProcessBoundary(Defaultable, Movable):
    var bi: PythonObject
    var environ: PythonObject
    var random: PythonObject
    var pid: PythonObject
    var executable: String
    var vendor: String

    def __init__(out self):
        self.bi = PythonObject(None)
        self.environ = PythonObject(None)
        self.random = PythonObject(None)
        self.pid = PythonObject(None)
        self.executable = String()
        self.vendor = String()


comptime PROCESS_BOUNDARY = _Global[
    StorageType=ProcessBoundary, name="MojoNativeProcessPool",
    init_fn=ProcessBoundary.__init__,
]


def process_initialize(executable: String, environ: PythonObject, vendor: String) raises:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    p[].bi = Python.import_module("builtins")
    if py_type(environ) is not p[].bi.dict or executable == "":
        raise Error("process: initialization requires an executable and native environment mapping")
    # Cache the underlying bytes dictionary, not os._Environ.get/__iter__.
    # os.environ mutation updates this same dictionary, preserving selection
    # semantics when a caller changes its mask before creating the pool.
    p[].environ = environ
    p[].executable = executable
    p[].vendor = vendor
    var posix = Python.import_module("posix")
    p[].random = posix.urandom
    p[].pid = posix.getpid
    if py_type(p[].random) is not py_type(p[].bi.len) or py_type(p[].pid) is not py_type(p[].bi.len):
        raise Error("process: process metadata entries must be native C builtins")
    transport_initialize()


def _process_environment(devices: PythonObject, group: PythonObject) raises -> List[String]:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    var bi = p[].bi
    var env = bi.dict()
    var keys = py_call(bi.list, p[].environ)
    for i in range(len(keys)):  # driver-loop: over process environment records
        py_setitem(env, py_call(keys[i].decode, "utf-8", "surrogateescape"), py_call(p[].environ[keys[i]].decode, "utf-8", "surrogateescape"))
    py_setitem(env, "MOJOLEARN_NUMERIC_MODE", "identical")
    # The parent selected its installed route at connection time. Later
    # visibility-mask edits still apply, but a changed process environment
    # must not silently move an existing GPU owner onto CPU workers.
    py_setitem(env, "MOJOLEARN_VENDOR", p[].vendor)
    var mask = String()
    if p[].vendor == "cuda":
        mask = "CUDA_VISIBLE_DEVICES"
    elif p[].vendor == "hip":
        mask = "ROCR_VISIBLE_DEVICES" if "ROCR_VISIBLE_DEVICES" in env else "HIP_VISIBLE_DEVICES"
        _ = py_call(env.pop, "HIP_VISIBLE_DEVICES" if mask == "ROCR_VISIBLE_DEVICES" else "ROCR_VISIBLE_DEVICES", PythonObject(None))
    elif p[].vendor == "metal":
        if len(group) != 1 or Int(py=group[0]) != 0:
            raise Error("process: Metal supports only the selected device zero")
    elif p[].vendor != "cpu":
        raise Error("process: unsupported installed vendor")
    if mask != "":
        var visible = py_get(env, mask)
        var selected = bi.list()
        var used = bi.set()
        var tokens = py_call(visible.split, ",") if visible is not None else bi.list()
        for i in range(len(devices)):  # driver-loop: over all selected device identities before launching a worker
            var device = Int(py=devices[i])
            if visible is not None and device >= len(tokens):
                raise Error("process: device index outside visibility mask")
            var identity = tokens[device].strip() if visible is not None else PythonObject(String(device))
            if len(identity) == 0 or identity in used:
                raise Error("process: selected devices repeat or omit a visibility identifier")
            _ = py_call(used.add, identity)
        for i in range(len(group)):  # driver-loop: over worker device visibility records
            var device = Int(py=group[i])
            _ = py_call(selected.append, tokens[device].strip() if visible is not None else PythonObject(String(device)))
        py_setitem(env, mask, py_call(PythonObject(",").join, selected))
    # small-loop(the fixed family device-count variable names)
    for family in [String("KMEANS"), String("GBDT"), String("GRAM"), String("QR"),
                   String("OPTIMIZER"), String("GLM"), String("SOLVER"), String("IFOREST"),
                   String("FOREST"), String("SVM"), String("GP"), String("GMM"),
                   String("RESAMPLE"), String("DBSCAN"), String("NEIGHBORS"), String("HIERARCHY")]:
        py_setitem(env, "MOJOLEARN_" + family + "_DEVICE_COUNT", String(len(group)))
    var result = List[String]()
    keys = py_call(bi.list, env)
    for i in range(len(keys)):  # driver-loop: over native process environment records
        result.append(String(keys[i]) + "=" + String(env[keys[i]]))
    return result^


def process_pool_create(devices: PythonObject, cooperative: Bool, child_module: String) raises -> PythonObject:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    var bi = p[].bi
    var selected = py_call(bi.tuple, devices)
    var seen = bi.set()
    if len(selected) == 0 or child_module == "":
        raise Error("process: nonempty devices and child module are required")
    for i in range(len(selected)):  # driver-loop: over admitted physical device indices
        if py_type(selected[i]) is not bi.int or Int(py=selected[i]) < 0 or selected[i] in seen:
            raise Error("process: devices must be distinct nonnegative integer indices")
        _ = py_call(seen.add, selected[i])
    var result = bi.dict()
    py_setitem(result, "devices", selected)
    py_setitem(result, "cooperative", cooperative)
    py_setitem(result, "module", child_module)
    py_setitem(result, "workers", bi.list())
    py_setitem(result, "closed", False)
    py_setitem(result, "sequence", 0)
    return result


def process_send(socket: PythonObject, kind: String, sequence: Int, value: PythonObject) raises:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    var packed = codec_pack(value, PROCESS_MAX_BYTES)
    var description = py_call(PythonObject(_transport_json(packed[0])).encode, "ascii")
    var header = p[].bi.dict()
    py_setitem(header, "kind", kind)
    py_setitem(header, "sequence", sequence)
    py_setitem(header, "description_bytes", len(description))
    # Descriptors travel in the binary frame so tensor registries do not
    # inherit the transport's small control-header size limit.
    var body = py_call(p[].bi.bytearray, description)
    _ = py_call(body.extend, packed[1])
    transport_send(socket, header, body)


def process_receive(socket: PythonObject) raises -> PythonObject:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    var frame = transport_receive(socket, PROCESS_MAX_BYTES)
    var header = frame[0]
    var body = frame[1]
    var raw = py_get(header, "description_bytes")
    if py_type(raw) is not p[].bi.int:
        raise Error("process: missing structural descriptor length")
    var length = Int(py=raw)
    if length <= 0 or length > len(body):
        raise Error("process: structural descriptor outside frame")
    var descriptor = native_json_parse(body[py_call(p[].bi.slice, 0, length)])
    var decoded = codec_unpack(descriptor, body[py_call(p[].bi.slice, length, len(body))], PROCESS_MAX_BYTES)
    var result = p[].bi.list()
    _ = py_call(result.append, header)
    _ = py_call(result.append, decoded)
    return py_call(p[].bi.tuple, result)


def _process_reap(pid: Int) raises:
    var status = transport_wait(pid, True)
    if status != -1:
        return
    # Completed native workers return before forced shutdown. Verification
    # children also flush their complete-operation audit after that return.
    var deadline = transport_clock() + 1.0
    while transport_clock() < deadline:  # driver-loop: over graceful native child completion status
        if transport_wait(pid, True) != -1:
            return
        transport_sleep(0.01)
    try:
        transport_terminate(pid)
    except:
        if transport_wait(pid, True) != -1:
            return
        raise
    deadline = transport_clock() + 4.0
    while transport_clock() < deadline:  # driver-loop: over termination status within the existing total five-second grace interval
        if transport_wait(pid, True) != -1:
            return
        transport_sleep(0.01)
    try:
        transport_terminate(pid, 9)
    except:
        if transport_wait(pid, True) != -1:
            return
        raise
    _ = transport_wait(pid)


def process_pool_close(pool: PythonObject) raises:
    var workers = pool["workers"]
    var bi = PROCESS_BOUNDARY.get_or_create_ptr()[].bi
    py_setitem(pool, "workers", bi.list())
    py_setitem(pool, "closed", True)
    var failure = String()
    for i in range(len(workers)):  # driver-loop: over owned worker sockets and process IDs
        var worker = workers[i]
        if worker["socket"] is not None:
            try:
                process_send(worker["socket"], "close", -1, PythonObject(None))
            except:
                # Failed workers can already have closed their socket. Their
                # process is still reaped below; no owner is abandoned.
                pass
        try:
            transport_close(worker["socket"])
        except error:
            if failure == "":
                failure = String(error)
        try:
            _process_reap(Int(py=worker["pid"]))
        except error:
            # A failed reap remains owned and can be retried by close().
            _ = py_call(pool["workers"].append, worker)
            if failure == "":
                failure = String(error)
    if failure != "":
        raise Error(failure)


def process_pool_start(pool: PythonObject) raises:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    var bi = p[].bi
    if Bool(py=pool["closed"]):
        raise Error("process: pool is closed")
    if len(pool["workers"]) != 0:
        return
    var devices = pool["devices"]
    var cooperative = Bool(py=pool["cooperative"])
    if p[].vendor == "cpu" and cooperative and len(devices) != 1:
        raise Error("NotImplementedError: cooperative neural updates across multiple CPU devices have no host implementation")
    var server = transport_listen("127.0.0.1", 0, 120.0)
    var socket_type = TRANSPORT_BOUNDARY.get_or_create_ptr()[].socket
    var port = Int(py=py_call(socket_type.getsockname, server)[1])
    var groups = 1 if cooperative else len(devices)
    try:
        for i in range(groups):  # driver-loop: over native worker process groups
            var group = bi.list()
            if cooperative:
                group = py_call(bi.list, devices)
            else:
                _ = py_call(group.append, devices[i])
            var env = _process_environment(devices, group)
            var token = String(py_call(p[].random, 32).hex())
            var argv = List[String]()
            argv.append(String(p[].executable))
            argv.append(String("-m"))
            argv.append(String(pool["module"]))
            argv.append(String(port))
            argv.append(String(token))
            var pid = transport_spawn(p[].executable, argv, env)
            var worker = bi.dict()
            py_setitem(worker, "pid", pid)
            py_setitem(worker, "socket", PythonObject(None))
            py_setitem(worker, "devices", py_call(bi.tuple, group))
            _ = py_call(pool["workers"].append, worker)
            var connection = transport_accept(server, 120.0)
            py_setitem(worker, "socket", connection)
            var hello = transport_receive(connection, 0)[0]
            if String(py_get(hello, "kind", "")) != "hello" or String(py_get(hello, "token", "")) != token or Int(py=py_get(hello, "pid", -1)) != pid:
                raise Error("process: child authentication or process identity mismatch")
            if String(py_get(hello, "vendor", "")) != p[].vendor:
                raise Error("process: child vendor differs from the parent native route")
            # Existing local model workers have no training-operation timeout.
            # Keep only the connection handshake bounded; large supported
            # model steps must not acquire a new two-minute runtime limit.
            _ = py_call(socket_type.settimeout, connection, PythonObject(None))
            py_setitem(worker, "identity", hello)
    except error:
        transport_close(server)
        try:
            process_pool_close(pool)
        except cleanup_error:
            raise Error("RuntimeError: native process startup failed: " + String(error) + "; cleanup also failed: " + String(cleanup_error))
        py_setitem(pool, "closed", False)
        raise error
    transport_close(server)


def process_pool_map(pool: PythonObject, requests: PythonObject) raises -> PythonObject:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    var bi = p[].bi
    var inputs = py_call(bi.list, requests)
    process_pool_start(pool)
    var workers = pool["workers"]
    var result = bi.list()
    var sequence = Int(py=pool["sequence"])
    var offset = 0
    try:
        while offset < len(inputs):  # driver-loop: over logical request waves across physical worker groups
            var width = min(len(workers), len(inputs) - offset)
            for i in range(width):  # driver-loop: over independent requests launched before consuming wave results
                process_send(workers[i]["socket"], "request", sequence + i, inputs[offset + i])
            var failure = String()
            for i in range(width):  # driver-loop: over worker replies in fixed logical request order
                try:
                    var reply = process_receive(workers[i]["socket"])
                    if Int(py=py_get(reply[0], "sequence", -1)) != sequence + i:
                        raise Error("process: response sequence mismatch")
                    if String(py_get(reply[0], "kind", "")) == "error":
                        raise Error("RuntimeError: native model worker failed: " + String(reply[1]))
                    if String(py_get(reply[0], "kind", "")) != "result":
                        raise Error("process: unexpected worker response")
                    _ = py_call(result.append, reply[1])
                except error:
                    if failure == "":
                        failure = String(error)
            if failure != "":
                raise Error(failure)
            offset += width
            sequence += width
            py_setitem(pool, "sequence", sequence)
    except error:
        try:
            process_pool_close(pool)
        except cleanup_error:
            raise Error("RuntimeError: native process request failed: " + String(error) + "; cleanup also failed: " + String(cleanup_error))
        py_setitem(pool, "closed", False)
        raise error
    return result


def process_pool_call(pool: PythonObject, worker_index: Int, request: PythonObject) raises -> PythonObject:
    # Persistent layer owners require an addressed request. A reported model
    # error preserves the living worker; a transport/protocol failure tears
    # down the pool because remote state ownership can no longer be trusted.
    var count = 1 if Bool(py=pool["cooperative"]) else len(pool["devices"])
    if worker_index < 0 or worker_index >= count:
        raise Error("process: addressed worker index outside the selected group")
    process_pool_start(pool)
    var sequence = Int(py=pool["sequence"])
    var reply = PythonObject(None)
    try:
        var worker = pool["workers"][worker_index]
        process_send(worker["socket"], "request", sequence, request)
        reply = process_receive(worker["socket"])
        if Int(py=py_get(reply[0], "sequence", -1)) != sequence:
            raise Error("process: response sequence mismatch")
        var kind = String(py_get(reply[0], "kind", ""))
        if kind != "result" and kind != "error":
            raise Error("process: unexpected addressed worker response")
        py_setitem(pool, "sequence", sequence + 1)
    except error:
        try:
            process_pool_close(pool)
        except cleanup_error:
            raise Error("RuntimeError: native process request failed: " + String(error) + "; cleanup also failed: " + String(cleanup_error))
        py_setitem(pool, "closed", False)
        raise error
    if String(reply[0]["kind"]) == "error":
        raise Error("RuntimeError: native model worker failed: " + String(reply[1]))
    return reply[1]


def process_worker_connect(argv: PythonObject) raises -> PythonObject:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    if len(argv) != 3:
        raise Error("process: worker requires the private port and authentication token")
    var port = Int(py=py_call(p[].bi.int, argv[1]))
    var token = String(argv[2])
    if port <= 0 or port > 65535 or token.byte_length() != 64:
        raise Error("process: invalid worker connection envelope")
    var socket = transport_connect("127.0.0.1", port, 120.0)
    var hello = p[].bi.dict()
    py_setitem(hello, "kind", "hello")
    py_setitem(hello, "token", token)
    py_setitem(hello, "pid", p[].pid())
    py_setitem(hello, "vendor", p[].vendor)
    transport_send(socket, hello, PythonObject(None))
    _ = py_call(TRANSPORT_BOUNDARY.get_or_create_ptr()[].socket.settimeout, socket, PythonObject(None))
    return socket


def process_echo_worker(argv: PythonObject) raises:
    # Native protocol probe used by the installed verifier. Model workers use
    # the same framing helpers with their compiled model dispatch table.
    var socket = process_worker_connect(argv)
    try:
        while True:  # driver-loop: over authenticated local protocol-verifier requests
            var request = process_worker_next(socket)
            if request is None:
                break
            var sequence = Int(py=py_get(request[0], "sequence", -1))
            process_send(socket, "result", sequence, request[1])
    except:
        transport_close(socket)
        raise
    transport_close(socket)


def process_worker_next(socket: PythonObject) raises -> PythonObject:
    var request = process_receive(socket)
    var kind = String(py_get(request[0], "kind", ""))
    if kind == "close":
        return PythonObject(None)
    if kind != "request":
        raise Error("process: worker received an invalid request kind")
    return request


def process_call(operation: String, args: PythonObject) raises -> PythonObject:
    if operation == "initialize":
        process_initialize(String(args[0]), args[1], String(args[2]))
    elif operation == "create":
        return process_pool_create(args[0], Bool(py=args[1]), String(args[2]))
    elif operation == "map":
        return process_pool_map(args[0], args[1])
    elif operation == "call":
        return process_pool_call(args[0], Int(py=args[1]), args[2])
    elif operation == "close":
        process_pool_close(args[0])
    elif operation == "worker_echo":
        process_echo_worker(args)
    elif operation == "environment":
        var env = _process_environment(args[0], args[1])
        var out = PROCESS_BOUNDARY.get_or_create_ptr()[].bi.list()
        for i in range(len(env)):  # driver-loop: over verifier-visible process environment entries
            _ = py_call(out.append, env[i])
        return out
    else:
        raise Error("process: unknown operation " + operation)
    return PythonObject(None)


def process_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = process_call(String(args[0]), args[1])
        return result.steal_data()
    except error:
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
