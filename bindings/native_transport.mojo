# SPDX-License-Identifier: Apache-2.0
"""Native process and socket transport shared by distributed model owners.

Mojo owns framing, validation, allocation and operation order. CPython's C
socket/POSIX implementations supply platform ABI details; socket.py,
subprocess.py, multiprocessing, pickle and Python callbacks never execute.
Only exact C builtin functions/types are cached at ABI setup. No numerical
work runs here and no architecture identity is inferred from transport.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.math import isfinite
from bindings.native_json import native_json_parse


struct TransportBoundary(Defaultable, Movable):
    var bi: PythonObject
    var socket: PythonObject
    var getaddrinfo: PythonObject
    var spawn: PythonObject
    var wait: PythonObject
    var kill: PythonObject
    var monotonic: PythonObject
    var sleep: PythonObject
    var json_string: PythonObject
    var sol_socket: Int
    var reuse_address: Int
    var passive: Int
    var nohang: Int

    def __init__(out self):
        self.bi = PythonObject(None)
        self.socket = PythonObject(None)
        self.getaddrinfo = PythonObject(None)
        self.spawn = PythonObject(None)
        self.wait = PythonObject(None)
        self.kill = PythonObject(None)
        self.monotonic = PythonObject(None)
        self.sleep = PythonObject(None)
        self.json_string = PythonObject(None)
        self.sol_socket = 0
        self.reuse_address = 0
        self.passive = 0
        self.nohang = 0


comptime TRANSPORT_BOUNDARY = _Global[
    StorageType=TransportBoundary, name="MojoNativeTransportBoundary",
    init_fn=TransportBoundary.__init__,
]


def _transport_builtin(entry: PythonObject) raises -> PythonObject:
    var bi = TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi
    if bi.type(entry) is not bi.type(bi.len):
        raise Error("transport: operation must be a compiled C builtin")
    return entry


def transport_initialize() raises:
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    t[].bi = Python.import_module("builtins")
    var sockets = Python.import_module("_socket")
    var processes = Python.import_module("posix")
    var time = Python.import_module("time")
    t[].socket = sockets.socket
    if t[].bi.type(t[].socket) is not t[].bi.type or String(t[].socket.__module__) != "_socket":
        raise Error("transport: socket owner must be the native socket type")
    t[].getaddrinfo = _transport_builtin(sockets.getaddrinfo)
    t[].spawn = _transport_builtin(processes.posix_spawnp)
    t[].wait = _transport_builtin(processes.waitpid)
    t[].kill = _transport_builtin(processes.kill)
    t[].monotonic = _transport_builtin(time.monotonic)
    t[].sleep = _transport_builtin(time.sleep)
    t[].json_string = _transport_builtin(Python.import_module("_json").encode_basestring_ascii)
    t[].sol_socket = Int(py=sockets.SOL_SOCKET)
    t[].reuse_address = Int(py=sockets.SO_REUSEADDR)
    t[].passive = Int(py=sockets.AI_PASSIVE)
    t[].nohang = Int(py=processes.WNOHANG)


def transport_clock() raises -> Float64:
    return Float64(TRANSPORT_BOUNDARY.get_or_create_ptr()[].monotonic())


def transport_sleep(seconds: Float64) raises:
    if seconds < 0 or not isfinite(seconds):
        raise Error("transport: invalid sleep interval")
    _ = TRANSPORT_BOUNDARY.get_or_create_ptr()[].sleep(seconds)


def _transport_timeout(seconds: Float64) raises:
    if seconds <= 0 or not isfinite(seconds):
        raise Error("transport: timeout must be finite and positive")


def transport_listen(host: String, port: Int, timeout: Float64) raises -> PythonObject:
    _transport_timeout(timeout)
    if port < 0 or port > 65535:
        raise Error("transport: port outside [0, 65535]")
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    var addresses = t[].getaddrinfo(PythonObject(None) if host == "" else PythonObject(host),
                                  port, 0, 1, 0, t[].passive)
    for i in range(len(addresses)):  # driver-loop: over resolved network endpoints
        var row = addresses[i]
        var socket = t[].socket(row[0], row[1], row[2])
        try:
            socket.setsockopt(t[].sol_socket, t[].reuse_address, 1)
            socket.settimeout(timeout)
            socket.bind(row[4])
            socket.listen()
            return socket
        except:
            socket.close()
            if i + 1 == len(addresses):
                raise Error("transport: could not bind listening endpoint")
    raise Error("transport: no resolved listening endpoint")


def transport_accept(server: PythonObject, timeout: Float64) raises -> PythonObject:
    _transport_timeout(timeout)
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    if t[].bi.type(server) is not t[].socket:
        raise Error("transport: listener is not a native socket")
    var accepted = server._accept()
    # _socket.socket is the C owner. The fourth positional argument adopts
    # the accepted fd, with no socket.py object construction or accept wrapper.
    var socket = t[].socket(server.family, server.type, server.proto, accepted[0])
    socket.settimeout(timeout)
    return socket


def transport_connect(host: String, port: Int, timeout: Float64) raises -> PythonObject:
    _transport_timeout(timeout)
    if port < 1 or port > 65535:
        raise Error("transport: remote port outside [1, 65535]")
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    var addresses = t[].getaddrinfo(host, port, 0, 1)
    for i in range(len(addresses)):  # driver-loop: over resolved network endpoints
        var row = addresses[i]
        var socket = t[].socket(row[0], row[1], row[2])
        try:
            socket.settimeout(timeout)
            socket.connect(row[4])
            return socket
        except:
            socket.close()
            if i + 1 == len(addresses):
                raise Error("transport: connection failed")
    raise Error("transport: no resolved remote endpoint")


def transport_close(socket: PythonObject) raises:
    if socket is not None:
        TRANSPORT_BOUNDARY.get_or_create_ptr()[].socket.close(socket)


def _transport_json(value: PythonObject, depth: Int = 0) raises -> String:
    if depth > 64:
        raise Error("transport: JSON header nesting exceeds 64")
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    var cls = t[].bi.type(value)
    if value is None:
        return "null"
    if cls is t[].bi.str:
        return String(t[].json_string(value))
    if cls is t[].bi.bool:
        return "true" if Bool(py=value) else "false"
    if cls is t[].bi.int:
        return String(value)
    if cls is t[].bi.float:
        if not isfinite(Float64(value)):
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
        var keys = t[].bi.list(value)
        for i in range(len(keys)):
            if t[].bi.type(keys[i]) is not t[].bi.str:
                raise Error("transport: JSON header keys must be strings")
        keys = t[].bi.sorted(keys)
        result += "{"
        for i in range(len(keys)):  # driver-loop: over header object fields
            if i != 0:
                result += ","
            result += String(t[].json_string(keys[i])) + ":" + _transport_json(value[keys[i]], depth + 1)
        return result + "}"
    raise Error("transport: JSON header contains an unsupported object")


def transport_send(socket: PythonObject, header: PythonObject, payload: PythonObject = PythonObject(None)) raises:
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    if not Bool(py=t[].bi.isinstance(socket, t[].socket)) or t[].bi.type(header) is not t[].bi.dict:
        raise Error("transport: send requires a native socket and dict header")
    var bytes = t[].bi.bytes() if payload is None else t[].bi.memoryview(payload).cast("B")
    var envelope = t[].bi.dict(header)
    envelope["nbytes"] = PythonObject(len(bytes))
    var raw = PythonObject(_transport_json(envelope)).encode("ascii")
    var n = len(raw)
    if n > (1 << 20):
        raise Error("transport: oversized header")
    var prefix = t[].bi.bytearray(4)
    for i in range(4):
        prefix[i] = PythonObject((n >> (24 - i * 8)) & 255)
    t[].socket.sendall(socket, prefix)
    t[].socket.sendall(socket, raw)
    if len(bytes) != 0:
        t[].socket.sendall(socket, bytes)


def _transport_exact(socket: PythonObject, count: Int) raises -> PythonObject:
    if count < 0:
        raise Error("transport: negative receive size")
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    var owner = t[].bi.bytearray(count)
    var view = t[].bi.memoryview(owner)
    var received = 0
    while received < count:  # driver-loop: over bytes received for one native frame
        var got = Int(py=t[].socket.recv_into(socket, view[t[].bi.slice(received, count)], count - received))
        if got <= 0:
            raise Error("transport: peer closed the connection")
        received += got
    return t[].bi.bytes(owner)


def transport_receive(socket: PythonObject, max_payload: Int) raises -> PythonObject:
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    if not Bool(py=t[].bi.isinstance(socket, t[].socket)) or max_payload < 0:
        raise Error("transport: receive requires a native socket and nonnegative bound")
    var prefix = _transport_exact(socket, 4)
    var count = 0
    for i in range(4):
        count = (count << 8) | Int(py=prefix[i])
    if count > (1 << 20):
        raise Error("transport: oversized header")
    var header = native_json_parse(_transport_exact(socket, count))
    if t[].bi.type(header) is not t[].bi.dict:
        raise Error("transport: framed header must be an object")
    var raw_count = header.get("nbytes", PythonObject(0))
    if t[].bi.type(raw_count) is not t[].bi.int:
        raise Error("transport: payload size must be an integer")
    var n = Int(py=raw_count)
    if n < 0 or n > max_payload:
        raise Error("transport: payload exceeds caller bound")
    var result = t[].bi.list()
    result.append(header)
    result.append(_transport_exact(socket, n))
    return t[].bi.tuple(result)


def transport_spawn(executable: String, argv: List[String], environment: List[String]) raises -> Int:
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    if executable == "" or len(argv) == 0:
        raise Error("transport: executable and argv must be nonempty")
    var args = t[].bi.list()
    var env = t[].bi.dict()
    for i in range(len(argv)):  # driver-loop: over native process argument records
        args.append(PythonObject(argv[i]))
    for i in range(len(environment)):  # driver-loop: over native process environment records
        var pair = PythonObject(environment[i]).partition("=")
        if String(pair[1]) != "=" or String(pair[0]) == "":
            raise Error("transport: environment entry must be NAME=value")
        env[pair[0]] = pair[2]
    # posix.posix_spawnp is the native C entry exposed by os.posix_spawnp,
    # checked during initialization; it directly performs supported POSIX
    # spawn, with no shell, Python child callback or interpreter worker loop.
    return Int(py=t[].spawn(PythonObject(executable), args, env))


def transport_wait(pid: Int, nohang: Bool = False) raises -> Int:
    if pid <= 0:
        raise Error("transport: invalid child process ID")
    var t = TRANSPORT_BOUNDARY.get_or_create_ptr()
    var result = t[].wait(pid, t[].nohang if nohang else 0)
    return -1 if Int(py=result[0]) == 0 else Int(py=result[1])


def transport_terminate(pid: Int, signal: Int = 15) raises:
    if pid <= 0 or signal <= 0:
        raise Error("transport: invalid process ID or signal")
    _ = TRANSPORT_BOUNDARY.get_or_create_ptr()[].kill(pid, signal)


def transport_call(operation: String, args: PythonObject) raises -> PythonObject:
    """Single native operation entry for admission and installed protocol tests."""
    if operation == "initialize":
        transport_initialize()
    elif operation == "listen":
        return transport_listen(String(args[0]), Int(py=args[1]), Float64(args[2]))
    elif operation == "accept":
        return transport_accept(args[0], Float64(args[1]))
    elif operation == "connect":
        return transport_connect(String(args[0]), Int(py=args[1]), Float64(args[2]))
    elif operation == "send":
        transport_send(args[0], args[1], args[2])
    elif operation == "receive":
        return transport_receive(args[0], Int(py=args[1]))
    elif operation == "close":
        transport_close(args[0])
    elif operation == "spawn":
        var argv = List[String]()
        var environment = List[String]()
        for i in range(len(args[1])):
            argv.append(String(args[1][i]))
        for i in range(len(args[2])):
            environment.append(String(args[2][i]))
        return PythonObject(transport_spawn(String(args[0]), argv, environment))
    elif operation == "wait":
        return PythonObject(transport_wait(Int(py=args[0]), Bool(py=args[1])))
    elif operation == "terminate":
        transport_terminate(Int(py=args[0]), Int(py=args[1]))
    else:
        raise Error("transport: unknown operation " + operation)
    return PythonObject(None)
