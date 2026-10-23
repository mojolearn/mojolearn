# SPDX-License-Identifier: Apache-2.0
# Native framed transport. POSIX owns fd/process operations; CPython is used
# only for exact builtin containers and an opaque fd-owning capsule. Calling
# CPython's socket/posix wrappers would invoke Python audit hooks.
from std.ffi import _Global, c_int, c_uint, c_size_t, external_call
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.sys.info import CompilationTarget
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.native_error_class import error_without_class
from std.math import isfinite
from bindings.native_json import native_json_parse
from bindings.native_idna import idna_initialize, idna_encode
from bindings.native_buffer_address import native_buffer_address
from bindings.native_numpy_metadata import native_numpy_type, native_numpy_module_fields
from bindings.native_posix_transport import (
    PosixBytes, PosixWords, posix_clock, posix_sleep, posix_open, posix_accept,
    posix_close, posix_transfer, posix_spawn, posix_wait, posix_terminate,
    posix_errno, posix_failure,
)


struct TransportBoundary(Defaultable, Movable):
    var bi: PythonObject
    var json_string: PythonObject
    var capsule_name: String
    var array_type: PythonObject
    var numpy_context: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.json_string = PythonObject(None)
        self.array_type = PythonObject(None)
        self.numpy_context = PythonObject(None)
        # PyCapsule retains this name pointer, not a copy. The DSO-global
        # owner is never reassigned and outlives every socket capsule.
        self.capsule_name = String("mojolearn.native_transport.fd.v1")


comptime TRANSPORT_BOUNDARY = _Global[
    StorageType=TransportBoundary, name="MojoNativeTransportBoundary",
    init_fn=TransportBoundary.__init__,
]


def transport_initialize() raises:
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    t[].bi = Python.import_module("builtins")
    var entry = Python.import_module("_json").encode_basestring_ascii
    if py_type(entry) is not py_type(t[].bi.len):
        raise Error("transport: JSON quoting requires its native C entry")
    t[].json_string = entry
    var array_module = Python.import_module("array")
    if py_type(array_module) is not py_type(Python.import_module("sys")):
        raise Error("transport: array buffers require their native module")
    var array_type = py_get(native_numpy_module_fields(array_module), "array")
    if py_type(array_type) is not t[].bi.type:
        raise Error("transport: array buffers require their native type")
    # array.array is a module-associated C heap type on CPython >=3.10.
    # The public C API refuses Python-created classes before buffer hooks.
    ref cpy = Python().cpython()
    var owner = cpy.lib.get_function[PyObjectPtr]("PyType_GetModule")(array_type._obj_ptr)
    if not owner:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        raise Error("transport: array buffers require their native type")
    if owner != array_module._obj_ptr:
        raise Error("transport: array buffer type has a different extension owner")
    t[].array_type = array_type
    t[].numpy_context = t[].bi.dict()
    py_setitem(t[].numpy_context, "modules", Python.import_module("sys").modules)
    idna_initialize()


def _transport_destroy(capsule: PyObjectPtr) abi("C") -> NoneType:
    # This native destructor cannot invoke Python callbacks. The capsule
    # owns exactly two words: fd and Float64 timeout; explicit close marks
    # fd=-1 so destruction remains idempotent.
    try:
        ref cpy = Python().cpython()
        var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
        var raw = cpy.lib.get_function[Optional[PosixBytes]]("PyCapsule_GetPointer")(capsule, t[].capsule_name.as_c_string_slice().unsafe_ptr())
        if raw:
            var state = PosixWords(unsafe_from_address=Int(raw.value()))
            posix_close(state[0])
            external_call["free", NoneType](raw.value())
        else:
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
    except:
        pass


def _transport_own(fd: Int, timeout: Float64) raises -> PythonObject:
    var raw = external_call["malloc", Optional[PosixBytes]](c_size_t(16))
    if not raw:
        posix_close(fd)
        raise Error("transport: socket owner allocation failed")
    var state = PosixWords(unsafe_from_address=Int(raw.value()))
    state[0] = fd
    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(raw.value()))[1] = timeout
    ref cpy = Python().cpython()
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    var result = cpy.lib.get_function[PyObjectPtr]("PyCapsule_New")(raw.value(), t[].capsule_name.as_c_string_slice().unsafe_ptr(), _transport_destroy)
    if not result:
        posix_close(fd)
        external_call["free", NoneType](raw.value())
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def _transport_state(socket: PythonObject, require_open: Bool = True) raises -> PosixWords:
    ref cpy = Python().cpython()
    var name = TRANSPORT_BOUNDARY.get_or_create_ptr()[].capsule_name.as_c_string_slice()
    if cpy.lib.get_function[c_int]("PyCapsule_IsValid")(socket._obj_ptr, name.unsafe_ptr()) == 0:
        raise Error("transport: expected an owned native socket")
    var raw = cpy.lib.get_function[Optional[PosixBytes]]("PyCapsule_GetPointer")(socket._obj_ptr, name.unsafe_ptr())
    if not raw:
        raise cpy.unsafe_get_error()
    var state = PosixWords(unsafe_from_address=Int(raw.value()))
    if require_open and state[0] < 0:
        raise Error("transport: socket is closed")
    return state


def transport_timeout(socket: PythonObject, seconds: Float64) raises:
    # -1 is the existing unbounded post-handshake training policy.
    if seconds != -1 and (seconds <= 0 or not isfinite(seconds)):
        raise Error("transport: timeout must be finite and positive, or -1")
    var state = _transport_state(socket)
    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(state))[1] = seconds
    _ = socket


def transport_clock() raises -> Float64:
    return posix_clock()


def transport_sleep(seconds: Float64) raises:
    posix_sleep(seconds)


def _transport_timeout(seconds: Float64) raises:
    if seconds <= 0 or not isfinite(seconds):
        raise Error("transport: timeout must be finite and positive")


def _transport_host(host: String) raises -> String:
    if host == "":
        return String()
    # The native IDNA helper returns ASCII bytes. Exact bytes.decode is a
    # built-in operation; it does not enter the Python codec search path.
    var encoded = idna_encode(PythonObject(host))
    return String(py_call(encoded.decode, "ascii"))


def transport_listen(host: String, port: Int, timeout: Float64) raises -> PythonObject:
    _transport_timeout(timeout)
    if port < 0 or port > 65535:
        raise Error("transport: port outside [0, 65535]")
    return _transport_own(posix_open(_transport_host(host), port, timeout, True), timeout)


def transport_accept(server: PythonObject, timeout: Float64) raises -> PythonObject:
    _transport_timeout(timeout)
    var state = _transport_state(server)
    var fd = posix_accept(state[0], timeout)
    _ = server
    return _transport_own(fd, timeout)


def transport_connect(host: String, port: Int, timeout: Float64) raises -> PythonObject:
    _transport_timeout(timeout)
    if port < 1 or port > 65535:
        raise Error("transport: remote port outside [1, 65535]")
    return _transport_own(posix_open(_transport_host(host), port, timeout, False), timeout)


def transport_address(socket: PythonObject, peer: Bool = False) raises -> PythonObject:
    var state = _transport_state(socket)
    var storage = Array[Int64, 16](uninitialized=True)  # sockaddr_storage: 128 bytes, aligned
    var length = Array[c_uint, 1](c_uint(128))
    var status = c_int(0)
    if peer:
        status = external_call["getpeername", c_int](c_int(state[0]), storage.unsafe_ptr(), length.unsafe_ptr())
    else:
        status = external_call["getsockname", c_int](c_int(state[0]), storage.unsafe_ptr(), length.unsafe_ptr())
    _ = socket
    if status != 0:
        posix_failure("socket address", posix_errno())
    var bytes = PosixBytes(unsafe_from_address=Int(storage.unsafe_ptr()))
    var family = Int(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=Int(bytes))[])
    var ipv6 = 10
    comptime if CompilationTarget.is_macos():
        family = Int(bytes[1])
        ipv6 = 30
    if family != 2 and family != ipv6:
        raise Error("transport: socket address is not IPv4 or IPv6")
    var text = Array[Int8, 64](uninitialized=True)
    var raw = bytes + (4 if family == 2 else 8)
    var found = external_call["inet_ntop", Optional[PosixBytes]](c_int(family), raw, text.unsafe_ptr(), c_uint(64))
    if not found:
        posix_failure("inet_ntop", posix_errno())
    var result = Python.list()
    _ = py_call(result.append, String(unsafe_from_utf8_ptr=text.unsafe_ptr()))
    _ = py_call(result.append, (Int(bytes[2]) << 8) | Int(bytes[3]))
    if family == ipv6:
        var flow = (Int(bytes[4]) << 24) | (Int(bytes[5]) << 16) | (Int(bytes[6]) << 8) | Int(bytes[7])
        var scope = Int(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(bytes) + 24)[])
        _ = py_call(result.append, flow)
        _ = py_call(result.append, scope)
    _ = storage^
    return py_call(TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi.tuple, result)


def transport_hostname() raises -> String:
    var buffer = Array[Int8, 256](uninitialized=True)
    if external_call["gethostname", c_int](buffer.unsafe_ptr(), c_size_t(256)) != 0:
        posix_failure("gethostname", posix_errno())
    buffer[255] = 0
    return String(unsafe_from_utf8_ptr=buffer.unsafe_ptr())


def transport_port(socket: PythonObject) raises -> Int:
    return Int(py=transport_address(socket)[1])


def transport_close(socket: PythonObject) raises:
    if socket is not None:
        var state = _transport_state(socket, False)
        var fd = state[0]
        state[0] = -1
        posix_close(fd)
        _ = socket


def _transport_json(value: PythonObject, depth: Int = 0) raises -> String:
    if depth > 64:
        raise Error("transport: JSON header nesting exceeds 64")
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    var cls = py_type(value)
    if value is None:
        return "null"
    if cls is t[].bi.str:
        return String(py_call(t[].json_string, value))
    if cls is t[].bi.bool:
        return "true" if Bool(py=value) else "false"
    if cls is t[].bi.int:
        return String(value)
    if cls is t[].bi.float:
        if not isfinite(Float64(py=value)):
            raise Error("transport: JSON numbers must be finite")
        return String(value)
    var result = String()
    if cls is t[].bi.list or cls is t[].bi.tuple:
        result += "["
        for i in range(len(value)):  # driver-loop: over header array fields
            if i != 0:
                result += ","
            result += _transport_json(value[i], depth + 1)
        return result + "]"
    if cls is t[].bi.dict:
        var keys = py_call(t[].bi.list, value)
        for i in range(len(keys)):  # driver-loop: over header object keys
            if py_type(keys[i]) is not t[].bi.str:
                raise Error("transport: JSON header keys must be strings")
        keys = py_call(t[].bi.sorted, keys)
        result += "{"
        for i in range(len(keys)):  # driver-loop: over header object fields
            if i != 0:
                result += ","
            result += String(py_call(t[].json_string, keys[i])) + ":" + _transport_json(value[keys[i]], depth + 1)
        return result + "}"
    raise Error("transport: JSON header contains an unsupported object")


def _transport_send_bytes(socket: PythonObject, value: PythonObject) raises:
    var state = _transport_state(socket)
    var timeout = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(state))[1]
    # This helper receives only exact bytes/bytearray, or an already-created
    # native memoryview. Hold the export through every partial write.
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[c_int]("PyObject_GetBuffer")(value._obj_ptr, pointer, c_int(0)) != 0:
        raise cpy.unsafe_get_error()
    try:
        posix_transfer(state[0], PosixBytes(unsafe_from_address=view[0]), view[2], timeout, True)
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    _ = view^
    _ = value
    _ = socket


def transport_send(socket: PythonObject, header: PythonObject, payload: PythonObject) raises:
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    _ = _transport_state(socket)
    if py_type(header) is not t[].bi.dict:
        raise Error("transport: send requires a native socket and dict header")
    var cls = py_type(payload)
    if payload is not None and cls is not t[].bi.bytes and cls is not t[].bi.bytearray and cls is not t[].bi.memoryview and cls is not t[].array_type:
        var numpy = py_get(t[].numpy_context["modules"], "numpy")
        if numpy is None or cls is not native_numpy_type("ndarray", t[].numpy_context):
            raise Error("transport: payload requires a native contiguous buffer; Python buffer hooks are not accepted")
    var header_keys = py_call(t[].bi.list, header)
    for i in range(len(header_keys)):
        if py_type(header_keys[i]) is not t[].bi.str:
            raise Error("transport: JSON header keys must be strings")
    var bytes = t[].bi.bytes() if payload is None else py_call(py_call(t[].bi.memoryview, payload).cast, "B")
    var envelope = py_call(t[].bi.dict, header)
    py_setitem(envelope, "nbytes", PythonObject(len(bytes)))
    var raw = py_call(PythonObject(_transport_json(envelope)).encode, "ascii")
    var n = len(raw)
    if n > (1 << 20):
        raise Error("transport: oversized header")
    var prefix = py_call(t[].bi.bytearray, 4)
    for i in range(4):  # small-loop(the 4-byte frame length word)
        py_setitem(prefix, i, PythonObject((n >> (24 - i * 8)) & 255))
    _transport_send_bytes(socket, prefix)
    _transport_send_bytes(socket, raw)
    _transport_send_bytes(socket, bytes)
    _ = payload


def _transport_exact(socket: PythonObject, count: Int) raises -> PythonObject:
    if count < 0:
        raise Error("transport: negative receive size")
    var state = _transport_state(socket)
    var timeout = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(state))[1]
    var owner = py_call(TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi.bytearray, count)
    var address = native_buffer_address(owner)
    posix_transfer(state[0], PosixBytes(unsafe_from_address=address), count, timeout, False)
    _ = socket
    return py_call(TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi.bytes, owner)


def transport_receive(socket: PythonObject, max_payload: Int) raises -> PythonObject:
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    _ = _transport_state(socket)
    if max_payload < 0:
        raise Error("transport: receive requires a native socket and nonnegative bound")
    var prefix = _transport_exact(socket, 4)
    var count = 0
    for i in range(4):  # small-loop(the 4-byte frame length word)
        count = (count << 8) | Int(py=prefix[i])
    if count > (1 << 20):
        raise Error("transport: oversized header")
    var header = native_json_parse(_transport_exact(socket, count))
    if py_type(header) is not t[].bi.dict:
        raise Error("transport: framed header must be an object")
    var raw_count = py_get(header, "nbytes", PythonObject(0))
    if py_type(raw_count) is not t[].bi.int:
        raise Error("transport: payload size must be an integer")
    var n = Int(py=raw_count)
    if n < 0 or n > max_payload:
        raise Error("transport: payload exceeds caller bound")
    var result = t[].bi.list()
    _ = py_call(result.append, header)
    _ = py_call(result.append, _transport_exact(socket, n))
    return py_call(t[].bi.tuple, result)


def transport_spawn(executable: String, argv: List[String], environment: List[String]) raises -> Int:
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    if executable == "" or len(argv) == 0:
        raise Error("transport: executable and argv must be nonempty")
    var env = t[].bi.dict()
    for i in range(len(environment)):  # driver-loop: last duplicate NAME wins, as before
        var pair = py_call(PythonObject(environment[i]).partition, "=")
        if String(pair[1]) != "=" or String(pair[0]) == "":
            raise Error("transport: environment entry must be NAME=value")
        py_setitem(env, pair[0], pair[2])
    var keys = py_call(t[].bi.list, env)
    var records = List[String]()
    for i in range(len(keys)):  # driver-loop: exact builtin string environment records
        records.append(String(keys[i]) + "=" + String(env[keys[i]]))
    return posix_spawn(executable, argv, records)


def transport_wait(pid: Int, nohang: Bool = False) raises -> Int:
    if pid <= 0 or pid > 2147483647:
        raise Error("transport: invalid child process ID")
    return posix_wait(pid, nohang)


def transport_terminate(pid: Int, signal: Int = 15) raises:
    if pid <= 0 or pid > 2147483647 or signal <= 0 or signal > 2147483647:
        raise Error("transport: invalid process ID or signal")
    posix_terminate(pid, signal)


def transport_call(operation: String, args: PythonObject) raises -> PythonObject:
    """Single native operation entry for admission and installed protocol tests."""
    if operation == "initialize":
        transport_initialize()
    elif operation == "idna":
        return idna_encode(args[0])
    elif operation == "listen":
        return transport_listen(String(args[0]), Int(py=args[1]), Float64(py=args[2]))
    elif operation == "accept":
        return transport_accept(args[0], Float64(py=args[1]))
    elif operation == "connect":
        return transport_connect(String(args[0]), Int(py=args[1]), Float64(py=args[2]))
    elif operation == "address":
        return transport_address(args[0], Bool(py=args[1]))
    elif operation == "port":
        return PythonObject(transport_port(args[0]))
    elif operation == "timeout":
        transport_timeout(args[0], Float64(py=args[1]))
    elif operation == "send":
        transport_send(args[0], args[1], args[2])
    elif operation == "receive":
        return transport_receive(args[0], Int(py=args[1]))
    elif operation == "close":
        transport_close(args[0])
    elif operation == "spawn":
        var argv = List[String]()
        var environment = List[String]()
        for i in range(len(args[1])):  # small-loop(process argument words)
            argv.append(String(args[1][i]))
        for i in range(len(args[2])):  # small-loop(process environment words)
            environment.append(String(args[2][i]))
        return PythonObject(transport_spawn(String(args[0]), argv, environment))
    elif operation == "wait":
        return PythonObject(transport_wait(Int(py=args[0]), Bool(py=args[1])))
    elif operation == "terminate":
        transport_terminate(Int(py=args[0]), Int(py=args[1]))
    else:
        raise Error("transport: unknown operation " + operation)
    return PythonObject(None)


def transport_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = transport_call(String(args[0]), args[1])
        return result.steal_data()
    except error:
        return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
