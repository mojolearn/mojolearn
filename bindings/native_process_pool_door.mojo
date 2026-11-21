# SPDX-License-Identifier: Apache-2.0
"""Native local process lifecycle and ordered request waves for model owners.

Workers import their installed ABI connection once, then enter a compiled
dispatch loop. The coordinator owns device visibility, framing, authentication,
process cleanup and request ordering. No pickle or Python worker loop is used.

State (lane pools-frame, 2026-11-21). A pool's state is a typed record
(`ProcessPoolRecord`, bindings/native_process_pool_run.mojo: devices, worker
process ids, socket descriptors, sequence, closed) held by one native owner,
`ProcessPoolOwner`, stored in the pool dictionary under `record`. Every
operation reads the record. The dictionary's other entries are of two kinds:
`module` and `environment` are a caller's configuration, UNPACKed when the
workers are spawned; `devices`, `cooperative`, `workers`, `closed`, `sequence`
and `timing` are a mirror this door publishes for readers and never reads back.

Wire. Every request and reply is one typed frame
(bindings/native_process_wire_door.mojo, core/parallel_worker_frame.mojo).
REMOVED 2026-11-21: the codec frame (JSON header dict, JSON structural
descriptor, joined payload; `process_send`, `process_receive`) and the `typed`
flag that chose between the two. Only the hello of a starting worker is still
a transport frame with a JSON header.
"""
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_transport_door import (
    transport_initialize,
    transport_listen,
    transport_accept,
    transport_port,
    transport_timeout,
    transport_connect,
    transport_send,
    transport_receive,
    transport_close,
    transport_spawn,
    transport_wait,
    transport_sleep,
    _transport_state,
)
from bindings.native_transport_run import transport_terminate, transport_clock
from bindings.native_process_wire_door import (
    WireOutgoing, wire_send, wire_receive, wire_send_fd, wire_receive_fd, wire_unpack, wire_text,
)
from core.parallel_worker_frame import WIRE_REQUEST, WIRE_RESULT, WIRE_ERROR, WIRE_CLOSE
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from core.shared_slot import SharedSlot
from bindings.native_process_pool_run import (
    ReapOutcome, ProcessPoolRecord, pool_index_admit, pool_wave_width, pool_reply_failed,
)
from bindings.native_keys_unpack import native_keys
from bindings.native_keys_unpack import (
    NATIVE_PROCESS_OPN_INITIALIZE, NATIVE_PROCESS_OPN_CREATE, NATIVE_PROCESS_OPN_MAP, NATIVE_PROCESS_OPN_CLOSE, NATIVE_PROCESS_OPN_WORKER_ECHO, NATIVE_PROCESS_OPN_ENVIRONMENT,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item


struct ProcessPoolOwner(Movable, Writable):
    """One pool's native owner, stored once in the pool dictionary as
    `record`. `native` is the typed state every operation works on. The Python
    objects here are boundary pins only, one per started worker in spawn
    order: the socket capsule that owns the worker's descriptor (dropping the
    pool drops the capsule, which closes the descriptor, so an abandoned
    pool's workers still see their connection close) and the worker's
    admitted hello, kept for the published mirror."""
    var native: ProcessPoolRecord
    var sockets: List[PythonObject]
    var identities: List[PythonObject]

    def __init__(out self, var native: ProcessPoolRecord):
        self.native = native^
        self.sockets = List[PythonObject]()
        self.identities = List[PythonObject]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ProcessPoolOwner(workers=", len(self.native.pids), ", closed=", self.native.closed, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


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
    var nk = native_keys()
    py_setitem(env, nk[].k_MOJOLEARN_NUMERIC_MODE, "identical")
    # The parent selected its installed route at connection time. Later
    # visibility-mask edits still apply, but a changed process environment
    # must not silently move an existing GPU owner onto CPU workers.
    py_setitem(env, nk[].k_MOJOLEARN_VENDOR, p[].vendor)
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
    var words = List[Int]()
    for i in range(len(selected)):  # driver-loop: over admitted physical device indices
        if py_type(selected[i]) is not bi.int or Int(py=selected[i]) < 0 or selected[i] in seen:
            raise Error("process: devices must be distinct nonnegative integer indices")
        _ = py_call(seen.add, selected[i])
        words.append(Int(py=selected[i]))
    var result = bi.dict()
    var nk = native_keys()
    var native = ProcessPoolRecord(words^, cooperative)
    # The record's Python type is registered by every binding that builds this
    # door (bindings/native_process_pool_type_door.mojo).
    py_setitem(result, nk[].k_record, PythonObject(alloc=ProcessPoolOwner(native^)))
    # Configuration a caller may still set before the first request.
    py_setitem(result, nk[].k_module, child_module)
    # The published mirror.
    py_setitem(result, nk[].k_devices, selected)
    py_setitem(result, nk[].k_cooperative, cooperative)
    py_setitem(result, nk[].k_workers, bi.list())
    py_setitem(result, nk[].k_closed, False)
    py_setitem(result, nk[].k_sequence, 0)
    return result


def _pool_publish(pool: PythonObject) raises:
    """PACK: the mirror of the record (`workers`, `closed`, `sequence`)."""
    var bi = PROCESS_BOUNDARY.get_or_create_ptr()[].bi
    var nk = native_keys()
    var record = py_item(pool, nk[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    var workers = bi.list()
    for i in range(len(rec[].native.pids)):  # driver-loop: over the pool's workers, one mirror entry each
        var group = bi.list()
        if rec[].native.cooperative:
            for d in range(len(rec[].native.devices)):  # small-loop(the devices one cooperative worker owns)
                _ = py_call(group.append, PythonObject(rec[].native.devices[d]))
        else:
            _ = py_call(group.append, PythonObject(rec[].native.devices[i]))
        var worker = bi.dict()
        py_setitem(worker, nk[].k_pid, PythonObject(rec[].native.pids[i]))
        py_setitem(worker, nk[].k_devices, py_call(bi.tuple, group))
        if rec[].identities[i] is not None:
            py_setitem(worker, nk[].k_identity, rec[].identities[i])
        _ = py_call(workers.append, worker)
    py_setitem(pool, nk[].k_workers, workers)
    py_setitem(pool, nk[].k_closed, PythonObject(rec[].native.closed))
    py_setitem(pool, nk[].k_sequence, PythonObject(rec[].native.sequence))
    _ = record


def _pool_abandon(pool: PythonObject, message: String, what: String) raises:
    """After a failed start or a transport failure: every worker is closed
    and reaped (remote state can no longer be trusted) and the pool is left
    reusable. The caller raises its own error afterwards."""
    try:
        _process_pool_close(pool, message.startswith("InterruptedError:"))
    except cleanup_error:
        raise Error("RuntimeError: native process " + what + " failed: " + message + "; cleanup also failed: " + String(cleanup_error))
    var nk = native_keys()
    var record = py_item(pool, nk[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    rec[].native.closed = False
    py_setitem(pool, nk[].k_closed, False)
    _ = record


def process_pool_closed(pool: PythonObject) raises -> Bool:
    """Whether `pool` was closed, from its record."""
    var record = py_item(pool, native_keys()[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    var closed = rec[].native.closed
    _ = record
    return closed


def process_pool_devices(pool: PythonObject) raises -> List[Int]:
    """The pool's selected physical devices, in the order given at creation."""
    var record = py_item(pool, native_keys()[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    var devices = rec[].native.devices.copy()
    _ = record
    return devices^


def process_pool_pids(pool: PythonObject) raises -> List[Int]:
    """The process ids of the pool's started workers, in worker order."""
    var record = py_item(pool, native_keys()[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    var pids = rec[].native.pids.copy()
    _ = record
    return pids^


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
    var nk = native_keys()
    var bi = PROCESS_BOUNDARY.get_or_create_ptr()[].bi
    var record = py_item(pool, nk[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    # The worker set leaves the record before any socket or process call, so
    # a failure below cannot leave a half-closed worker addressable.
    var pids = rec[].native.pids.copy()
    var fds = rec[].native.fds.copy()
    var sockets = rec[].sockets.copy()
    var identities = rec[].identities.copy()
    rec[].native.pids.clear()
    rec[].native.fds.clear()
    rec[].sockets.clear()
    rec[].identities.clear()
    rec[].native.closed = True
    var failure = String()
    if abort:
        # Cancel every owned child before serial reaping; do not spend one
        # full grace interval per still-running peer or drain its reply.
        for i in range(len(pids)):  # driver-loop: signal the authenticated owned worker set
            try:
                transport_terminate(pids[i])
            except:
                pass  # Reap below distinguishes an already-exited child.
    # Every worker is told to close and its socket closed before any reap, and
    # the reap's grace windows are shared by the whole group: the workers'
    # exits overlap, so a k-worker close waits about one worker's exit, not k
    # (each worker was closed and reaped in turn, up to five seconds each,
    # before 2026-11-05). The per-worker escalation is unchanged.
    var no_request = List[Int64]()
    for i in range(len(pids)):  # driver-loop: over owned worker sockets and process IDs
        if fds[i] >= 0 and not abort:
            try:
                # Cleanup must not inherit the unbounded training timeout.
                # A nonreading worker can fill its socket before close starts.
                wire_send_fd(fds[i], 1.0, WIRE_CLOSE, -1, no_request, WireOutgoing())
            except:
                # Failed workers can already have closed their socket. Their
                # process is still reaped below; no owner is abandoned.
                pass
        try:
            transport_close(sockets[i])
        except error:
            if failure == "":
                failure = String(error)
    var reaped = _process_reap_group(pids)
    var close_seconds = bi.list()
    var close_signal = bi.list()
    for i in range(len(pids)):  # driver-loop: over owned worker reap outcomes
        _ = py_call(close_seconds.append, PythonObject(reaped.seconds[i]))
        _ = py_call(close_signal.append, PythonObject(reaped.signal[i]))
        if reaped.errors[i] != "":
            # A failed reap remains owned and can be retried by close(); its
            # socket is closed, so it is never addressed again.
            rec[].native.pids.append(pids[i])
            rec[].native.fds.append(-1)
            rec[].sockets.append(PythonObject(None))
            rec[].identities.append(identities[i])
            if failure == "":
                failure = reaped.errors[i]
    _pool_publish(pool)
    # Seconds from the close request to each worker's reap, and the signal
    # its reap needed (0: none, 15: SIGTERM, 9: SIGKILL).
    var timing = py_get(pool, nk[].k_timing, PythonObject(None))
    if timing is not None and py_type(timing) is bi.dict and len(pids) > 0:
        py_setitem(timing, nk[].k_close_seconds, close_seconds)
        py_setitem(timing, nk[].k_close_signal, close_signal)
    _ = record
    if failure != "":
        raise Error(failure)


def _pool_timing(pool: PythonObject, seconds: List[Float64]) raises:
    """PACK: seconds from each worker's spawn to its authenticated hello
    (interpreter start, package import, binding connection); None for a
    worker that did not authenticate."""
    var bi = PROCESS_BOUNDARY.get_or_create_ptr()[].bi
    var nk = native_keys()
    var started = bi.list()
    for i in range(len(seconds)):  # small-loop(one start time per worker)
        if seconds[i] >= 0:
            _ = py_call(started.append, PythonObject(seconds[i]))
        else:
            _ = py_call(started.append, PythonObject(None))
    var timing = bi.dict()
    py_setitem(timing, nk[].k_start_seconds, started)
    py_setitem(pool, nk[].k_timing, timing)


def process_pool_start(pool: PythonObject) raises:
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    var bi = p[].bi
    var nk = native_keys()
    var record = py_item(pool, nk[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    if rec[].native.closed:
        raise Error("process: pool is closed")
    if rec[].native.started():
        return
    var cooperative = rec[].native.cooperative
    var groups = rec[].native.width()
    if p[].vendor == "cpu" and cooperative and len(rec[].native.devices) != 1:
        raise Error("NotImplementedError: cooperative neural updates across multiple CPU devices have no host implementation")
    # UNPACK of the caller's configuration, once per start.
    var child_module = String(py_item(pool, nk[].k_module))
    var overrides = py_get(pool, nk[].k_environment)
    var devices = bi.list()
    for i in range(len(rec[].native.devices)):  # small-loop(the pool's selected devices)
        _ = py_call(devices.append, PythonObject(rec[].native.devices[i]))
    var server = transport_listen("127.0.0.1", 0, 120.0)
    var port = transport_port(server)
    # Every worker is spawned before the first handshake, so the workers'
    # interpreter start, package import and binding connection overlap: a
    # k-device pool starts in about one worker's start-up, not k of them
    # (they were spawned and authenticated one after another before
    # 2026-11-05). Each worker keeps its own single-use token; a hello is
    # matched to its worker by that token and must carry that worker's pid,
    # so authentication is as strict as the serial order was. The record
    # stays in spawn (device) order, whatever order the workers connect in.
    var tokens = List[String]()
    var spawned = List[Float64]()
    var seconds = List[Float64](length=groups, fill=-1.0)
    try:
        for i in range(groups):  # driver-loop: over native worker process groups (spawn all first)
            var group = bi.list()
            if cooperative:
                group = py_call(bi.list, devices)
            else:
                _ = py_call(group.append, devices[i])
            var env = _process_environment(devices, group, overrides)
            var token = String(py_call(p[].random, 32).hex())
            var argv = List[String]()
            argv.append(String(p[].executable))
            argv.append(String("-m"))
            argv.append(child_module.copy())
            argv.append(String(port))
            argv.append(token.copy())
            spawned.append(transport_clock())
            var pid = transport_spawn(p[].executable, argv, env)
            rec[].native.pids.append(pid)
            rec[].native.fds.append(-1)
            rec[].sockets.append(PythonObject(None))
            rec[].identities.append(PythonObject(None))
            tokens.append(token.copy())
        for _ in range(groups):  # driver-loop: authenticate each spawned worker as it connects
            var connection = transport_accept(server, 120.0)
            var attached = False
            try:
                var hello = transport_receive(connection, 0)[0]
                var claimed = String(py_get(hello, nk[].k_token, ""))
                var index = -1
                for t in range(len(tokens)):  # small-loop(one pending token per spawned worker)
                    if tokens[t] != "" and tokens[t] == claimed:
                        index = t
                if String(py_get(hello, nk[].k_kind, "")) != "hello" or index < 0 or Int(py=py_get(hello, nk[].k_pid, -1)) != rec[].native.pids[index]:
                    raise Error("process: child authentication or process identity mismatch")
                tokens[index] = String("")  # a token authenticates one connection only
                rec[].sockets[index] = connection
                rec[].native.fds[index] = _transport_state(connection)[0]
                attached = True
                if String(py_get(hello, nk[].k_vendor, "")) != p[].vendor:
                    raise Error("process: child vendor differs from the parent native route")
                # Existing local model workers have no training-operation timeout.
                # Keep only the connection handshake bounded; large supported
                # model steps must not acquire a new two-minute runtime limit.
                # (The waves pass no timeout with the record's descriptor; the
                # capsule's own word is kept in step for its other readers.)
                transport_timeout(connection, -1.0)
                rec[].identities[index] = hello
                seconds[index] = transport_clock() - spawned[index]
            except error:
                if not attached:
                    try:
                        transport_close(connection)
                    except:
                        pass  # The handshake failure below is the reported error.
                raise error^
    except error:
        transport_close(server)
        _pool_timing(pool, seconds)
        _pool_abandon(pool, String(error), "startup")
        raise error
    transport_close(server)
    _pool_timing(pool, seconds)
    _pool_publish(pool)
    _ = record


def _process_pool_wave(pool: PythonObject, inputs: PythonObject, words: List[List[Int64]]) raises -> PythonObject:
    """Ordered request waves over the pool's workers: request `i` of a wave
    goes to worker `i` as one typed frame (its words, when `words` has them,
    and the record UNPACKed from `inputs[i]`), then the replies are read in
    request order. Every worker of a wave has its request before the first
    reply is awaited, so the workers compute at the same time."""
    var bi = PROCESS_BOUNDARY.get_or_create_ptr()[].bi
    var total = len(inputs)
    if len(words) != 0 and len(words) != total:
        raise Error("process: typed request words differ from their payloads")
    process_pool_start(pool)
    var nk = native_keys()
    var record = py_item(pool, nk[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    # The descriptors are copied out once per map: no reference into the
    # record is held across a socket wait.
    var fds = rec[].native.fds.copy()
    var sequence = rec[].native.sequence
    var no_request = List[Int64]()
    var result = bi.list()
    var offset = 0
    try:
        while offset < total:  # driver-loop: over logical request waves across physical worker groups
            var width = pool_wave_width(len(fds), total - offset)
            for i in range(width):  # driver-loop: over independent requests launched before consuming wave results
                var outgoing = wire_unpack(inputs[offset + i])
                if len(words) != 0:
                    wire_send_fd(fds[i], -1.0, WIRE_REQUEST, sequence + i, words[offset + i], outgoing)
                else:
                    wire_send_fd(fds[i], -1.0, WIRE_REQUEST, sequence + i, no_request, outgoing)
            var failure = String()
            for i in range(width):  # driver-loop: over worker replies in fixed logical request order
                try:
                    var reply = wire_receive_fd(fds[i], -1.0)
                    if pool_reply_failed(reply.kind, reply.sequence, sequence + i, False):
                        raise Error("RuntimeError: native model worker failed: " + String(reply.value))
                    _ = py_call(result.append, reply.value)
                except error:
                    if String(error).startswith("InterruptedError:"):
                        raise error  # Cancel instead of draining unbounded peer replies.
                    if failure == "":
                        failure = String(error)
            if failure != "":
                raise Error(failure)
            offset += width
            sequence += width
            rec[].native.sequence = sequence
    except error:
        _pool_abandon(pool, String(error), "request")
        raise error
    # The mirror's sequence, once per map (the record has it per wave).
    py_setitem(pool, nk[].k_sequence, PythonObject(sequence))
    _ = record
    return result


def process_pool_map(pool: PythonObject, requests: PythonObject) raises -> PythonObject:
    """One reply per request, in request order; each request is a value the
    typed frame carries (its record), with no request words."""
    var inputs = py_call(PROCESS_BOUNDARY.get_or_create_ptr()[].bi.list, requests)
    var no_words = List[List[Int64]]()
    return _process_pool_wave(pool, inputs, no_words)


def process_pool_map_typed(pool: PythonObject, words: List[List[Int64]], payloads: PythonObject) raises -> PythonObject:
    """`process_pool_map` for typed requests: `words[i]` is one encoded
    `WorkerRequest` (core/parallel_worker_request.mojo) and `payloads[i]` the
    value that rides with it (None for none). `payloads` is an exact list."""
    if py_type(payloads) is not PROCESS_BOUNDARY.get_or_create_ptr()[].bi.list:
        raise Error("process: typed request payloads require an exact list")
    return _process_pool_wave(pool, payloads, words)


def process_pool_request(pool: PythonObject, worker_index: Int, request: List[Int64], outgoing: WireOutgoing) raises -> PythonObject:
    # One addressed typed request (lane parallel-payload-transport): the
    # request's words and its record travel as one typed frame, the reply
    # comes back as one (bindings/native_process_wire_door.mojo). Persistent
    # layer owners require an addressed request. A reported model error
    # preserves the living worker; a transport/protocol failure tears down the
    # pool because remote state ownership can no longer be trusted.
    var nk = native_keys()
    var record = py_item(pool, nk[].k_record)
    var rec = record.downcast_value_ptr[ProcessPoolOwner]()
    pool_index_admit(rec[].native.width(), worker_index)
    process_pool_start(pool)
    if worker_index >= len(rec[].native.fds):
        # Only workers whose reap failed are left (a close that could not finish).
        raise Error("transport: socket is closed")
    var sequence = rec[].native.sequence
    var fd = rec[].native.fds[worker_index]
    var value = PythonObject(None)
    var failed = False
    try:
        wire_send_fd(fd, -1.0, WIRE_REQUEST, sequence, request, outgoing)
        var reply = wire_receive_fd(fd, -1.0)
        failed = pool_reply_failed(reply.kind, reply.sequence, sequence, True)
        value = reply.value
        rec[].native.sequence = sequence + 1
        py_setitem(pool, nk[].k_sequence, PythonObject(sequence + 1))
    except error:
        _pool_abandon(pool, String(error), "request")
        raise error
    _ = record
    if failed:
        raise Error("RuntimeError: native model worker failed: " + String(value))
    return value


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
    var nk = native_keys()
    py_setitem(hello, nk[].k_kind, "hello")
    py_setitem(hello, nk[].k_token, token)
    py_setitem(hello, nk[].k_pid, p[].pid())
    py_setitem(hello, nk[].k_vendor, p[].vendor)
    transport_send(socket, hello, PythonObject(None))
    transport_timeout(socket, -1.0)
    return socket


struct WorkerTurn(Movable):
    """One request a worker received: `open` is False for the coordinator's
    close (the worker then leaves its loop), `sequence` is echoed in the
    reply and `value` is the request's PACKed record."""
    var open: Bool
    var sequence: Int
    var value: PythonObject

    def __init__(out self, open: Bool, sequence: Int, value: PythonObject):
        self.open = open
        self.sequence = sequence
        self.value = value


def process_worker_next(socket: PythonObject) raises -> WorkerTurn:
    var message = wire_receive(socket)
    if message.kind == WIRE_CLOSE:
        return WorkerTurn(False, -1, PythonObject(None))
    if message.kind != WIRE_REQUEST:
        raise Error("process: worker received an invalid request kind")
    return WorkerTurn(True, message.sequence, message.value)


def process_worker_reply(socket: PythonObject, sequence: Int, value: PythonObject) raises:
    """A worker's result: `value` UNPACKed whole before the first byte is
    written, so a value the frame cannot carry leaves the socket in step and
    the caller can still send its refusal."""
    var outgoing = wire_unpack(value)
    var no_request = List[Int64]()
    wire_send(socket, WIRE_RESULT, sequence, no_request, outgoing)


def process_worker_refuse(socket: PythonObject, sequence: Int, message: String) raises:
    """A worker's model error: the reply is the error text; the worker lives on."""
    var no_request = List[Int64]()
    wire_send(socket, WIRE_ERROR, sequence, no_request, wire_text(message))


def process_echo_worker(argv: PythonObject) raises:
    # Native protocol probe used by the installed verifier. Model workers use
    # the same framing helpers with their compiled model dispatch table.
    var socket = process_worker_connect(argv)
    try:
        while True:  # driver-loop: over authenticated local protocol-verifier requests
            var turn = process_worker_next(socket)
            if not turn.open:
                break
            process_worker_reply(socket, turn.sequence, turn.value)
    except:
        transport_close(socket)
        raise
    transport_close(socket)


def process_call(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    # `operation`: a NATIVE_PROCESS_OPN_* code; `name` is the caller's str, read
    # only for the unknown-operation text.
    if operation == NATIVE_PROCESS_OPN_INITIALIZE:
        process_initialize(String(args[0]), args[1], String(args[2]))
    elif operation == NATIVE_PROCESS_OPN_CREATE:
        return process_pool_create(args[0], Bool(py=args[1]), String(args[2]))
    elif operation == NATIVE_PROCESS_OPN_MAP:
        return process_pool_map(args[0], args[1])
    elif operation == NATIVE_PROCESS_OPN_CLOSE:
        process_pool_close(args[0])
    elif operation == NATIVE_PROCESS_OPN_WORKER_ECHO:
        process_echo_worker(args)
    elif operation == NATIVE_PROCESS_OPN_ENVIRONMENT:
        var env = _process_environment(args[0], args[1], args[2] if len(args) > 2 else PythonObject(None))
        var out = PROCESS_BOUNDARY.get_or_create_ptr()[].bi.list()
        for i in range(len(env)):  # driver-loop: over verifier-visible process environment entries
            _ = py_call(out.append, env[i])
        return out
    else:
        raise Error("process: unknown operation " + String(name))
    return PythonObject(None)


def process_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var nk = native_keys()
        var result = process_call(py_op_code(nk[].ops_process, args[0]), args[0], args[1])
        return result.steal_data()
    except error:
        if String(error).startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
