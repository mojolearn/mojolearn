# SPDX-License-Identifier: Apache-2.0
"""Native local process lifecycle and ordered request waves for model owners.

Workers import their installed ABI connection once, then enter a compiled
dispatch loop. The coordinator owns device visibility, framing, authentication,
process cleanup and request ordering. No pickle or Python worker loop is used.
"""
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_transport import (
    transport_initialize, transport_listen, transport_accept, transport_port, transport_timeout,
    transport_connect, transport_send, transport_send_parts, transport_receive, transport_receive_owned, transport_close,
    transport_spawn, transport_wait, transport_terminate, transport_clock,
    transport_sleep, _transport_json,
)
from bindings.native_codec import codec_pack_owned, codec_unpack
from bindings.native_json import native_json_parse
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from core.shared_slot import SharedSlot


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


comptime PROCESS_BOUNDARY = SharedSlot[ProcessBoundary, "MojolearnNativeProcessPoolV1"]


def process_initialize(executable: String, environ: PythonObject, vendor: String) raises:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    p[].bi = native_loaded_module("builtins")
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
    return _process_environment(devices, group, PythonObject(None))


def _process_environment(devices: PythonObject, group: PythonObject, overrides: PythonObject) raises -> List[String]:
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
    if overrides is not None:
        if py_type(overrides) is not bi.dict:
            raise Error("process: worker environment overrides require an exact dict")
        var override_keys = py_call(bi.list, overrides)
        for i in range(len(override_keys)):  # driver-loop: admit before dict hashing or string protocols
            if py_type(override_keys[i]) is not bi.str:
                raise Error("process: worker environment keys require exact strings")
        for i in range(len(override_keys)):  # driver-loop: copy admitted worker setup metadata
            var key = override_keys[i]
            var value = overrides[key]
            if py_type(value) is not bi.str:
                raise Error("process: worker environment values require exact strings")
            # Setup metadata cannot override routing, visibility, numeric mode
            # or inherited library search paths. This is the one fixed native
            # CV dependency flag, consumed only during worker import/setup.
            if String(key) != "MOJOLEARN_NATIVE_CV_NUMPY" or (String(value) != "0" and String(value) != "1"):
                raise Error("process: unsupported worker environment override")
            py_setitem(env, key, value)
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
    var packed = codec_pack_owned(value, PROCESS_MAX_BYTES)
    var description = py_call(PythonObject(_transport_json(packed[0])).encode, "ascii")
    var header = p[].bi.dict()
    py_setitem(header, "kind", kind)
    py_setitem(header, "sequence", sequence)
    py_setitem(header, "description_bytes", len(description))
    # Descriptors travel in the binary frame so tensor registries do not
    # inherit the transport's small control-header size limit. The frame
    # payload is the descriptor then the codec bytes, written from the two
    # buffers (no joined copy of the payload).
    transport_send_parts(socket, header, description, packed[1])


def process_receive(socket: PythonObject) raises -> PythonObject:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    var frame = transport_receive_owned(socket, PROCESS_MAX_BYTES)
    var header = frame[0]
    var body = frame[1]
    var raw = py_get(header, "description_bytes")
    if py_type(raw) is not p[].bi.int:
        raise Error("process: missing structural descriptor length")
    var length = Int(py=raw)
    if length <= 0 or length > len(body):
        raise Error("process: structural descriptor outside frame")
    var descriptor = native_json_parse(body[py_call(p[].bi.slice, 0, length)])
    # A zero-copy window onto the received bytearray past the descriptor.
    var window = py_call(p[].bi.memoryview, body)[py_call(p[].bi.slice, length, len(body))]
    var decoded = codec_unpack(descriptor, window, PROCESS_MAX_BYTES)
    _ = window
    var result = p[].bi.list()
    _ = py_call(result.append, header)
    _ = py_call(result.append, decoded)
    return py_call(p[].bi.tuple, result)


struct ReapOutcome(Movable):
    """Per worker, in pool order: the reap's error ("" when reaped), the
    seconds from the start of the reap to it, and the signal it needed."""
    var errors: List[String]
    var seconds: List[Float64]
    var signal: List[Int]

    def __init__(out self, n: Int):
        self.errors = List[String](length=n, fill=String())
        self.seconds = List[Float64](length=n, fill=0.0)
        self.signal = List[Int](length=n, fill=0)


def _reap_poll(pids: List[Int], mut done: List[Bool], mut outcome: ReapOutcome, start: Float64, window: Float64, level: Int) raises:
    """Nonblocking reaps of every open worker until all are reaped, failed,
    or `window` seconds pass (one shared window for the whole group)."""
    var deadline = transport_clock() + window
    while True:  # driver-loop: over graceful native child completion status
        var waiting = False
        for i in range(len(pids)):  # driver-loop: over owned worker process IDs
            if done[i]:
                continue
            try:
                if transport_wait(pids[i], True) != -1:
                    done[i] = True
                    outcome.seconds[i] = transport_clock() - start
                    outcome.signal[i] = level
                else:
                    waiting = True
            except error:
                done[i] = True
                outcome.errors[i] = String(error)
        if not waiting or transport_clock() >= deadline:
            return
        transport_sleep(0.01)


def _reap_signal(pids: List[Int], mut done: List[Bool], mut outcome: ReapOutcome, start: Float64, signal: Int) raises:
    for i in range(len(pids)):  # driver-loop: signal every still-running owned worker
        if done[i]:
            continue
        try:
            transport_terminate(pids[i], signal)
        except error:
            # An already-exited child cannot be signalled; reap it instead.
            try:
                if transport_wait(pids[i], True) != -1:
                    done[i] = True
                    outcome.seconds[i] = transport_clock() - start
                    outcome.signal[i] = 0
                else:
                    done[i] = True
                    outcome.errors[i] = String(error)
            except wait_error:
                done[i] = True
                outcome.errors[i] = String(wait_error)


def _process_reap_group(pids: List[Int]) raises -> ReapOutcome:
    """Reap every worker of a closing pool with the existing escalation per
    worker (one second to exit on its own, SIGTERM, four more seconds,
    SIGKILL, then a blocking wait), but with each window shared by the whole
    group instead of spent once per worker in turn."""
    var out = ReapOutcome(len(pids))
    var done = List[Bool](length=len(pids), fill=False)
    var start = transport_clock()
    # Completed native workers return before forced shutdown. Verification
    # children also flush their complete-operation audit after that return.
    _reap_poll(pids, done, out, start, 1.0, 0)
    _reap_signal(pids, done, out, start, 15)
    # Termination status within the existing total five-second grace interval.
    _reap_poll(pids, done, out, start, 4.0, 15)
    _reap_signal(pids, done, out, start, 9)
    for i in range(len(pids)):  # driver-loop: blocking reap of every killed owned worker
        if done[i]:
            continue
        try:
            _ = transport_wait(pids[i])
            out.seconds[i] = transport_clock() - start
            out.signal[i] = 9
        except error:
            out.errors[i] = String(error)
        done[i] = True
    return out^


def process_pool_close(pool: PythonObject) raises:
    _process_pool_close(pool, False)


def _process_pool_close(pool: PythonObject, abort: Bool) raises:
    var workers = pool["workers"]
    var bi = PROCESS_BOUNDARY.get_or_create_ptr()[].bi
    py_setitem(pool, "workers", bi.list())
    py_setitem(pool, "closed", True)
    var failure = String()
    if abort:
        # Cancel every owned child before serial reaping; do not spend one
        # full grace interval per still-running peer or drain its reply.
        for i in range(len(workers)):  # driver-loop: signal the authenticated owned worker set
            try:
                transport_terminate(Int(py=workers[i]["pid"]))
            except:
                pass  # Reap below distinguishes an already-exited child.
    # Every worker is told to close and its socket closed before any reap, and
    # the reap's grace windows are shared by the whole group: the workers'
    # exits overlap, so a k-worker close waits about one worker's exit, not k
    # (each worker was closed and reaped in turn, up to five seconds each,
    # before 2026-11-05). The per-worker escalation is unchanged.
    var pids = List[Int]()
    for i in range(len(workers)):  # driver-loop: over owned worker sockets and process IDs
        var worker = workers[i]
        if worker["socket"] is not None and not abort:
            try:
                # Cleanup must not inherit the unbounded training timeout.
                # A nonreading worker can fill its socket before close starts.
                transport_timeout(worker["socket"], 1.0)
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
        pids.append(Int(py=worker["pid"]))
    var reaped = _process_reap_group(pids)
    var close_seconds = bi.list()
    var close_signal = bi.list()
    for i in range(len(workers)):  # driver-loop: over owned worker reap outcomes
        _ = py_call(close_seconds.append, PythonObject(reaped.seconds[i]))
        _ = py_call(close_signal.append, PythonObject(reaped.signal[i]))
        if reaped.errors[i] != "":
            # A failed reap remains owned and can be retried by close().
            _ = py_call(pool["workers"].append, workers[i])
            if failure == "":
                failure = reaped.errors[i]
    # Seconds from the close request to each worker's reap, and the signal
    # its reap needed (0: none, 15: SIGTERM, 9: SIGKILL).
    var timing = py_get(pool, "timing", PythonObject(None))
    if timing is not None and len(workers) > 0:
        py_setitem(timing, "close_seconds", close_seconds)
        py_setitem(timing, "close_signal", close_signal)
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
    var port = transport_port(server)
    var groups = 1 if cooperative else len(devices)
    # Every worker is spawned before the first handshake, so the workers'
    # interpreter start, package import and binding connection overlap: a
    # k-device pool starts in about one worker's start-up, not k of them
    # (they were spawned and authenticated one after another before
    # 2026-11-05). Each worker keeps its own single-use token; a hello is
    # matched to its worker by that token and must carry that worker's pid,
    # so authentication is as strict as the serial order was. `workers`
    # stays in spawn (device) order, whatever order the workers connect in.
    var pending = bi.dict()
    var timing = bi.dict()
    var started = bi.list()
    py_setitem(timing, "start_seconds", started)
    py_setitem(pool, "timing", timing)
    try:
        for i in range(groups):  # driver-loop: over native worker process groups (spawn all first)
            var group = bi.list()
            if cooperative:
                group = py_call(bi.list, devices)
            else:
                _ = py_call(group.append, devices[i])
            var env = _process_environment(devices, group, py_get(pool, "environment"))
            var token = String(py_call(p[].random, 32).hex())
            var argv = List[String]()
            argv.append(String(p[].executable))
            argv.append(String("-m"))
            argv.append(String(pool["module"]))
            argv.append(String(port))
            argv.append(String(token))
            var spawned = transport_clock()
            var pid = transport_spawn(p[].executable, argv, env)
            var worker = bi.dict()
            py_setitem(worker, "pid", pid)
            py_setitem(worker, "socket", PythonObject(None))
            py_setitem(worker, "devices", py_call(bi.tuple, group))
            _ = py_call(pool["workers"].append, worker)
            _ = py_call(started.append, PythonObject(None))
            var entry = bi.list()
            _ = py_call(entry.append, worker)
            _ = py_call(entry.append, PythonObject(i))
            _ = py_call(entry.append, PythonObject(spawned))
            py_setitem(pending, PythonObject(token), entry)
        for _ in range(groups):  # driver-loop: authenticate each spawned worker as it connects
            var connection = transport_accept(server, 120.0)
            var attached = False
            try:
                var hello = transport_receive(connection, 0)[0]
                var claimed = PythonObject(String(py_get(hello, "token", "")))
                var entry = py_get(pending, claimed, PythonObject(None))
                if String(py_get(hello, "kind", "")) != "hello" or entry is None or Int(py=py_get(hello, "pid", -1)) != Int(py=entry[0]["pid"]):
                    raise Error("process: child authentication or process identity mismatch")
                _ = py_call(pending.pop, claimed)  # a token authenticates one connection only
                var worker = entry[0]
                py_setitem(worker, "socket", connection)
                attached = True
                if String(py_get(hello, "vendor", "")) != p[].vendor:
                    raise Error("process: child vendor differs from the parent native route")
                # Existing local model workers have no training-operation timeout.
                # Keep only the connection handshake bounded; large supported
                # model steps must not acquire a new two-minute runtime limit.
                transport_timeout(connection, -1.0)
                py_setitem(worker, "identity", hello)
                # Seconds from this worker's spawn to its authenticated hello
                # (interpreter start, package import, binding connection).
                py_setitem(started, entry[1], PythonObject(transport_clock() - Float64(py=entry[2])))
            except error:
                if not attached:
                    try:
                        transport_close(connection)
                    except:
                        pass  # The handshake failure below is the reported error.
                raise error^
    except error:
        transport_close(server)
        try:
            _process_pool_close(pool, String(error).startswith("InterruptedError:"))
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
                    if String(error).startswith("InterruptedError:"):
                        raise error  # Cancel instead of draining unbounded peer replies.
                    if failure == "":
                        failure = String(error)
            if failure != "":
                raise Error(failure)
            offset += width
            sequence += width
            py_setitem(pool, "sequence", sequence)
    except error:
        try:
            _process_pool_close(pool, String(error).startswith("InterruptedError:"))
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
            _process_pool_close(pool, String(error).startswith("InterruptedError:"))
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
    transport_timeout(socket, -1.0)
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
        var env = _process_environment(args[0], args[1], args[2] if len(args) > 2 else PythonObject(None))
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
        if String(error).startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
