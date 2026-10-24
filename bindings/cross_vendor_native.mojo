# SPDX-License-Identifier: Apache-2.0
"""Native byte admission, identity folds and state hashes for live training."""
from std.ffi import _Global
from std.math import isfinite
from std.memory import memcpy, bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_transport import (
    transport_initialize, transport_send, transport_receive, _transport_exact,
    transport_listen, transport_accept, transport_close, transport_clock, transport_sleep, transport_connect,
    transport_port, transport_address, transport_timeout, transport_hostname,
    _transport_json,
)
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.parallel_byte_native import parallel_byte_call, parallel_byte_init, _parallel_array, _parallel_int, _parallel_float
from bindings.generation_buffers import generation_array
from bindings.cross_vendor_admission import (
    cv_types_initialize, cv_int, cv_float, cv_bool, cv_text, cv_sequence, cv_buffer, cv_path, cv_json, cv_state,
)
from bindings.byte_checkpoint_native import byte_checkpoint_initialize, byte_checkpoint_save, byte_checkpoint_load
from bindings.byte_checkpoint_io import ByteCheckpointIOError
from bindings.native_json import native_json_parse
from bindings.sha256_door import sha256_update, sha256_hexdigest, sha256_sliced8
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from core.sha256 import Sha256


struct CrossVendorBoundary(Defaultable, Movable):
    var bi: PythonObject
    var arrays: PythonObject
    var array_cls: PythonObject
    var mismatch_cls: PythonObject
    var fold_pair: PythonObject
    var parallel_cls: PythonObject
    var feed_cls: PythonObject
    var rate_cls: PythonObject
    var vendor: PythonObject
    var worker_cls: PythonObject
    var coordinator_cls: PythonObject
    var ready_cls: PythonObject
    var argv: PythonObject
    var console_write: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.arrays = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.mismatch_cls = PythonObject(None)
        self.fold_pair = PythonObject(None)
        self.parallel_cls = PythonObject(None)
        self.feed_cls = PythonObject(None)
        self.rate_cls = PythonObject(None)
        self.vendor = PythonObject(None)
        self.worker_cls = PythonObject(None)
        self.coordinator_cls = PythonObject(None)
        self.ready_cls = PythonObject(None)
        self.argv = PythonObject(None)
        self.console_write = PythonObject(None)


comptime CROSS_VENDOR_BOUNDARY = _Global[
    StorageType=CrossVendorBoundary, name="MojoCrossVendorBoundary",
    init_fn=CrossVendorBoundary.__init__,
]


def cross_vendor_initialize(array_cls: PythonObject, mismatch_cls: PythonObject, fold_pair: PythonObject) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    c[].bi = Python.import_module("builtins")
    c[].arrays = Python.import_module("array").array
    c[].array_cls = array_cls
    c[].mismatch_cls = mismatch_cls
    c[].fold_pair = fold_pair
    cv_types_initialize(array_cls)
    byte_checkpoint_initialize(array_cls)
    c[].argv = Python.import_module("sys").argv
    c[].console_write = Python.import_module("posix").write
    if py_type(c[].fold_pair) is not py_type(c[].bi.len):
        raise Error("cross_vendor: fold must be a native extension entry")
    transport_initialize()


def cross_vendor_bytes(value: PythonObject) raises -> PythonObject:
    return py_call(cv_buffer(value).cast, "B")


def cross_vendor_f32(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var raw = cross_vendor_bytes(value)
    if len(raw) % 4 != 0:
        raise Error("cross_vendor: gradient bytes must be a multiple of four")
    return py_call(raw.cast, "f")


def cross_vendor_fold(gradients: PythonObject, prefix: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var parts = cv_sequence(gradients, "gradients")
    var first = 0
    var source = prefix
    if source is None:
        if len(parts) == 0:
            raise Error("ordered_fold needs at least one gradient")
        source = parts[0]
        first = 1
    var store = py_call(c[].arrays, "f")
    _ = py_call(store.frombytes, py_call(cross_vendor_f32(source).cast, "B"))
    var n = len(store)
    var total = store.buffer_info()[0]
    for k in range(first, len(parts)):  # driver-loop: over logical shard gradient buffers in immutable order
        var shard = py_call(c[].arrays, "f")
        _ = py_call(shard.frombytes, py_call(cross_vendor_f32(parts[k]).cast, "B"))
        if len(shard) != n:
            raise Error("ordered_fold: gradients differ in length")
        # Retain the selected operation's existing GPU/CPU execution route.
        # Mojo owns the immutable fold; the cached compiled entry owns its
        # arithmetic, including FTZ and the one float32 rounding per add.
        _ = py_call(c[].fold_pair, total, shard.buffer_info()[0], n)
        # Explicitly retain the C array owning the address across every read;
        # converting to an integer pointer erases the compiler's origin edge.
        _ = shard^
    var result = store.tobytes()
    _ = store
    return result


def cross_vendor_state_hash(state: PythonObject) raises -> PythonObject:
    cv_state(state)
    var hasher = Sha256()
    for key in [String("parameters"), String("m"), String("v"), String("flags")]:  # small-loop(four state keys): one digest update per named state field
        sha256_update(hasher, cross_vendor_bytes(state[key]))
    return PythonObject(hasher.hexdigest())


def cross_vendor_call(operation: String, args: PythonObject, mut io_error: ByteCheckpointIOError) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if operation == "byte_checkpoint_initialize":
        byte_checkpoint_initialize(args[0])
    elif operation == "byte_checkpoint_save":
        return byte_checkpoint_save(args[0], args[1], io_error)
    elif operation == "byte_checkpoint_load":
        return byte_checkpoint_load(args[0], io_error)
    elif operation == "main":
        return cross_vendor_main(args[0])
    elif operation == "initialize":
        cross_vendor_initialize(args[0], args[1], args[2])
    elif operation == "worker_initialize":
        c[].parallel_cls = args[0]
        c[].feed_cls = args[1]
        c[].rate_cls = args[2]
        c[].vendor = args[3]
        c[].worker_cls = args[4]
        c[].coordinator_cls = args[5]
        c[].ready_cls = args[6]
    elif operation == "worker_init":
        return cross_vendor_worker_init(args)
    elif operation == "worker_connect":
        return cross_vendor_worker_connect(args[0])
    elif operation == "worker_run":
        return cross_vendor_worker_run(args[0], io_error)
    elif operation == "trainer_hash":
        return cross_vendor_state_hash(_cross_vendor_trainer(args[0], "export_raw", c[].bi.list()))
    elif operation == "feed_init":
        return cross_vendor_feed_init(args)
    elif operation == "feed_batch":
        return cross_vendor_feed_batch(args[0], cv_int(args[1], "step"), cv_int(args[2], "shard"))
    elif operation == "rates_init":
        return cross_vendor_rates_init(args)
    elif operation == "rates_value":
        return cross_vendor_rate(args[0], cv_int(args[1], "step"))
    elif operation == "drain_events":
        if py_type(args[0]) is not c[].worker_cls and py_type(args[0]) is not c[].coordinator_cls:
            raise Error("cross_vendor: event owner must be an exact Worker or Coordinator")
        var events = args[0]._events
        py_setattr(args[0], "_events", c[].bi.list())
        return events
    elif operation == "coordinator_init":
        return cross_vendor_coordinator_init(args)
    elif operation == "coordinator_run":
        return cross_vendor_coordinator_run(args[0])
    elif operation == "coordinator_drive":
        return cross_vendor_drive(args[0])
    elif operation == "coordinator_refuse":
        cross_vendor_refuse(args[0], cv_text(args[1], "refusal reason"))
    elif operation.startswith("event_"):
        return cross_vendor_event(operation, args)
    elif operation == "f32":
        return cross_vendor_f32(args[0])
    elif operation == "fold":
        return cross_vendor_fold(args[0], args[1])
    elif operation == "pair":
        var parts = c[].bi.list()
        _ = py_call(parts.append, args[0])
        _ = py_call(parts.append, args[1])
        return cross_vendor_fold(parts, PythonObject(None))
    elif operation == "state_hash":
        return cross_vendor_state_hash(args[0])
    elif operation == "send":
        transport_send(args[0], args[1], args[2])
    elif operation == "receive":
        return transport_receive(args[0], Int(py=args[1]))
    elif operation == "receive_exact":
        return _transport_exact(args[0], Int(py=args[1]))
    else:
        raise Error("cross_vendor: unknown native operation " + operation)
    return PythonObject(None)


def cross_vendor_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    var operation = String()
    var io_error = ByteCheckpointIOError()
    try:
        operation = String(args[0])
        var result = cross_vendor_call(operation, args[1], io_error)
        return result.steal_data()
    except error:
        if io_error.pending:
            return io_error^.restore()
        if String(error).startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if String(error).startswith("CrossVendorMismatch:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 2)
                return causal_set_exception(error_without_class(error, "CrossVendorMismatch"), causal_borrowed_item(context, "mismatch"))
            except nested:
                return raise_python_exception(nested, ExceptionType("PyExc_RuntimeError"))
        if operation == "send" or operation == "receive" or operation == "receive_exact":
            return raise_python_exception(error, ExceptionType("PyExc_ConnectionError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


def cross_vendor_coordinator_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var owner = args[0]
    if py_type(owner) is not c[].coordinator_cls:
        raise Error("cross_vendor: Coordinator subclasses and Python attribute hooks are not accepted")
    if args[6] is not None:
        raise Error("cross_vendor: on_step callbacks are not accepted; use drain_events or log_path")
    var workers = cv_int(args[2], "workers")
    var shards = cv_int(args[3], "logical_shards")
    if workers < 1 or shards < workers:
        raise Error("need 1 <= workers <= logical_shards")
    var port = cv_int(args[1], "port")
    if port < 0 or port > 65535 or cv_int(args[4], "steps") < 0:
        raise Error("coordinator requires port in [0,65535] and nonnegative steps")
    _ = cv_text(args[5], "host")
    if cv_float(args[7], "timeout") <= 0 or cv_float(args[8], "accept_timeout") <= 0 or cv_int(args[9], "max_gradient_bytes") < 0:
        raise Error("coordinator timeouts must be positive and byte limit nonnegative")
    _ = cv_bool(args[10], "chained")
    py_setattr(owner, "port", args[1])
    py_setattr(owner, "n_workers", PythonObject(workers))
    py_setattr(owner, "K", PythonObject(shards))
    py_setattr(owner, "steps", args[4])
    py_setattr(owner, "host", args[5])
    py_setattr(owner, "on_step", args[6])
    py_setattr(owner, "timeout", args[7])
    py_setattr(owner, "accept_timeout", args[8])
    py_setattr(owner, "max_bytes", args[9])
    py_setattr(owner, "chained", py_call(c[].bi.bool, args[10]))
    py_setattr(owner, "peers", c[].bi.list())
    py_setattr(owner, "_events", c[].bi.list())
    py_setattr(owner, "log_path", PythonObject(None) if args[12] is None else cv_path(args[12]))
    py_setattr(owner, "bound_port", PythonObject(None))
    var ready = py_call(c[].bi.object.__new__, args[11])
    py_setattr(ready, "_set", PythonObject(False))
    py_setattr(owner, "ready", ready)
    return PythonObject(None)


def cross_vendor_event(operation: String, args: PythonObject) raises -> PythonObject:
    var event = args[0]
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(event) is not c[].ready_cls:
        raise Error("cross_vendor: readiness requires the native event owner")
    if operation == "event_init" or operation == "event_clear":
        py_setattr(event, "_set", PythonObject(False))
    elif operation == "event_set":
        py_setattr(event, "_set", PythonObject(True))
    elif operation == "event_is_set":
        return event._set
    elif operation == "event_wait":
        var unlimited = args[1] is None
        var timeout = Float64(0) if unlimited else cv_float(args[1], "readiness timeout")
        if not unlimited and not isfinite(timeout):
            raise Error("readiness timeout must be finite or None")
        var start = transport_clock()
        while not cv_bool(event._set, "readiness state"):  # driver-loop: over native readiness wait notifications
            if not unlimited and transport_clock() - start >= timeout:
                return PythonObject(False)
            transport_sleep(0.001)
        return PythonObject(True)
    return PythonObject(None)


def cross_vendor_refuse(owner: PythonObject, reason: String) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var header = c[].bi.dict()
    py_setitem(header, "cmd", PythonObject("refuse"))
    py_setitem(header, "reason", PythonObject(reason))
    for i in range(len(owner.peers)):  # driver-loop: over connected worker refusal notices
        try:
            transport_send(owner.peers[i]["sock"], header, PythonObject(None))
        except:
            pass
    raise Error("CrossVendorMismatch: " + reason)


def _cross_vendor_elapsed(start: Float64) raises -> PythonObject:
    return py_call(CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi.round, PythonObject(transport_clock() - start), 3)


def cross_vendor_drive(owner: PythonObject) raises -> PythonObject:
    cross_vendor_coordinator_admit(owner)
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var peers = owner.peers
    var count = len(peers)
    var shards = Int(py=owner.K)
    var chained = Bool(py=owner.chained)
    if count != Int(py=owner.n_workers):
        cross_vendor_refuse(owner, "worker count differs from the admitted group")
    var names = bi.set()
    var owned = bi.list()
    for i in range(count):  # driver-loop: over worker protocol and shard ownership declarations
        var peer = peers[i]
        var name = py_get(peer, "name")
        if py_type(name) is not bi.str or py_contains(names, name):
            cross_vendor_refuse(owner, "workers require distinct string names")
        _ = py_call(names.add, name)
        if String(py_get(peer, "protocol")) != "mojolearn.cross-vendor.v1":
            cross_vendor_refuse(owner, "protocol mismatch: " + String(name))
        var ids = py_get(peer, "shards")
        if py_type(ids) is not bi.list or len(ids) == 0:
            cross_vendor_refuse(owner, "workers require nonempty shard lists")
        for k in range(len(ids)):  # driver-loop: over one worker's declared shard IDs
            if py_type(ids[k]) is not bi.int:
                cross_vendor_refuse(owner, "shard IDs must be integers")
            _ = py_call(owned.append, ids[k])
        for key in [String("state"), String("completed"), String("n_total")]:  # small-loop(three start fields): state, completed, n_total
            if not Bool(py=peer[key] == peers[0][key]):
                cross_vendor_refuse(owner, "workers start from different " + key)
    owned = py_call(bi.sorted, owned)
    if len(owned) != shards:
        cross_vendor_refuse(owner, "workers must own every logical shard exactly once")
    for k in range(shards):  # driver-loop: over logical shards (ownership check)
        if Int(py=owned[k]) != k:
            cross_vendor_refuse(owner, "workers must own shards 0.." + String(shards - 1) + " exactly once")
    var total_elements = Int(py=peers[0]["n_total"])
    var maximum = Int(py=owner.max_bytes)
    if total_elements < 0 or maximum < 0 or total_elements > maximum // 4:
        cross_vendor_refuse(owner, "gradient exceeds the coordinator byte bound")
    var total_bytes = total_elements * 4
    if chained:
        # Stable insertion by the first integer shard; no Python sort key.
        for i in range(1, count):  # driver-loop: over worker blocks in immutable shard order
            var peer = peers[i]
            var j = i
            while j > 0 and Int(py=peers[j - 1]["shards"][0]) > Int(py=peer["shards"][0]):  # driver-loop: over worker blocks in immutable shard order
                py_setitem(peers, j, peers[j - 1])
                j -= 1
            py_setitem(peers, j, peer)
        for i in range(count):  # driver-loop: over workers (contiguous block check)
            var ids = peers[i]["shards"]
            for k in range(len(ids)):  # driver-loop: over one worker's declared shard IDs
                if Int(py=ids[k]) != Int(py=ids[0]) + k:
                    cross_vendor_refuse(owner, "chained fold: worker must own a contiguous block")
            if not Bool(py=py_get(peers[i], "chained", PythonObject(False))):
                cross_vendor_refuse(owner, "chained fold: every worker must start with chained=True")
    var rows = bi.list()
    var completed = Int(py=peers[0]["completed"])
    for step in range(completed, Int(py=owner.steps)):  # driver-loop: over requested lockstep training rounds
        var started = transport_clock()
        var timing = bi.dict()
        for i in range(count):  # driver-loop: over worker step requests
            var head = bi.dict()
            py_setitem(head, "cmd", PythonObject("step"))
            py_setitem(head, "step", PythonObject(step))
            py_setitem(head, "chained", PythonObject(chained))
            py_setitem(head, "first", PythonObject(chained and i == 0))
            transport_send(peers[i]["sock"], head, PythonObject(None))
        var gradients = bi.dict()
        var losses = bi.dict()
        for i in range(count):  # driver-loop: over worker gradient replies
            var peer = peers[i]
            var n = len(peer["shards"])
            if not chained and total_bytes != 0 and n > maximum // total_bytes:
                cross_vendor_refuse(owner, "worker gradient payload exceeds coordinator byte bound")
            var response = transport_receive(peer["sock"], 0 if chained else n * total_bytes)
            var head = response[0]
            var payload = response[1]
            if not Bool(py=py_get(head, "step") == PythonObject(step)) or not Bool(py=py_get(head, "shards") == peer["shards"]):
                cross_vendor_refuse(owner, String(peer["name"]) + " answered the wrong step or shards")
            if len(head["losses"]) != n or (not chained and len(payload) != n * total_bytes):
                cross_vendor_refuse(owner, String(peer["name"]) + " sent a wrong gradient or loss count")
            for k in range(n):  # driver-loop: over one worker's logical shard replies
                var id = peer["shards"][k]
                py_setitem(losses, id, head["losses"][k])
                if not chained:
                    py_setitem(gradients, id, payload[py_call(bi.slice, k * total_bytes, (k + 1) * total_bytes)])
            if chained:
                py_setitem(timing, "gradients_" + String(peer["name"]), _cross_vendor_elapsed(started))
        var total = bi.bytes()
        if chained:
            for i in range(count):  # driver-loop: over chained prefix handoffs in fixed shard order
                var head = bi.dict()
                py_setitem(head, "cmd", PythonObject("fold"))
                py_setitem(head, "step", PythonObject(step))
                var folded_at = transport_clock()
                transport_send(peers[i]["sock"], head, total)
                var response = transport_receive(peers[i]["sock"], total_bytes)
                total = response[1]
                if not Bool(py=py_get(response[0], "step") == PythonObject(step)) or len(total) != total_bytes:
                    cross_vendor_refuse(owner, String(peers[i]["name"]) + " returned a wrong fold")
                py_setitem(timing, "fold_" + String(peers[i]["name"]), _cross_vendor_elapsed(folded_at))
        else:
            var parts = bi.list()
            for k in range(shards):  # driver-loop: over logical shard gradients in fixed order
                _ = py_call(parts.append, gradients[k])
            total = cross_vendor_fold(parts, PythonObject(None))
        var all_losses = bi.list()
        for k in range(shards):  # driver-loop: over logical shard losses in fixed order
            _ = py_call(all_losses.append, losses[k])
        var total_hash = sha256_hexdigest(total)
        var apply = bi.dict()
        py_setitem(apply, "cmd", PythonObject("apply"))
        py_setitem(apply, "step", PythonObject(step))
        py_setitem(apply, "total_sha256", total_hash)
        py_setitem(apply, "losses", all_losses)
        for i in range(count):  # driver-loop: over identical optimizer commit requests
            transport_send(peers[i]["sock"], apply, bi.bytes() if chained and i == count - 1 else total)
        py_setitem(timing, "apply_sent", _cross_vendor_elapsed(started))
        var hashes = bi.dict()
        var agreed = PythonObject(None)
        var vendors = bi.dict()
        for i in range(count):  # driver-loop: over committed worker state witnesses
            var peer = peers[i]
            var response = transport_receive(peer["sock"], 0)
            var head = response[0]
            if not Bool(py=py_get(head, "completed") == PythonObject(step + 1)):
                cross_vendor_refuse(owner, String(peer["name"]) + " did not commit step " + String(step + 1))
            py_setitem(hashes, peer["name"], head["state"])
            py_setitem(vendors, peer["name"], py_get(peer, "vendor"))
            if i == 0:
                agreed = head["state"]
            elif not Bool(py=head["state"] == agreed):
                cross_vendor_refuse(owner, "replicas disagree after step " + String(step + 1))
            py_setitem(timing, "applied_" + String(peer["name"]), _cross_vendor_elapsed(started))
        py_setitem(timing, "step_seconds", _cross_vendor_elapsed(started))
        var row = bi.dict()
        py_setitem(row, "step", PythonObject(step + 1))
        py_setitem(row, "losses", all_losses)
        py_setitem(row, "state", agreed)
        py_setitem(row, "workers", hashes)
        py_setitem(row, "vendors", vendors)
        py_setitem(row, "total_sha256", total_hash)
        py_setitem(row, "protocol", PythonObject("chained" if chained else "gathered"))
        py_setitem(row, "timing", timing)
        _ = py_call(rows.append, row)
        _ = py_call(owner._events.append, row)
        cross_vendor_log(owner.log_path, row)
    var done = bi.dict()
    py_setitem(done, "cmd", PythonObject("done"))
    for i in range(count):  # driver-loop: over group completion notices
        transport_send(peers[i]["sock"], done, PythonObject(None))
    return rows


def _cross_vendor_close_peers(owner: PythonObject) raises:
    for i in range(len(owner.peers)):  # driver-loop: over native socket cleanup
        try:
            transport_close(owner.peers[i]["sock"])
        except:
            pass


def cross_vendor_coordinator_run(owner: PythonObject) raises -> PythonObject:
    cross_vendor_coordinator_admit(owner)
    var bi = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi
    var server = transport_listen(String(owner.host), Int(py=owner.port), Float64(py=owner.accept_timeout))
    py_setattr(owner, "bound_port", transport_port(server))
    py_setattr(owner.ready, "_set", PythonObject(True))
    var rows = PythonObject(None)
    try:
        while len(owner.peers) < Int(py=owner.n_workers):  # driver-loop: over admitted worker connections
            var socket = transport_accept(server, Float64(py=owner.timeout))
            try:
                var response = transport_receive(socket, 0)
                var peer = py_call(bi.dict, response[0])
                py_setitem(peer, "sock", socket)
                py_setitem(peer, "where", transport_address(socket, True))
                _ = py_call(owner.peers.append, peer)
            except error:
                transport_close(socket)
                raise error
        rows = cross_vendor_drive(owner)
    except error:
        _cross_vendor_close_peers(owner)
        transport_close(server)
        raise error
    _cross_vendor_close_peers(owner)
    transport_close(server)
    return rows


def cross_vendor_feed_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var owner = args[0]
    if py_type(owner) is not c[].feed_cls:
        raise Error("BatchFeed subclasses and Python attribute hooks are not accepted")
    var value = args[1]
    var view = cv_buffer(value)
    var shape = value.shape if py_type(value) is c[].array_cls else view.shape
    if len(shape) != 4:
        raise Error("BatchFeed requires int32[steps,logical_shards,batch,length+1]")
    var dims = List[Int]()
    for i in range(4):  # small-loop(four batch-feed dimensions): steps, shards, batch, length + 1
        dims.append(_parallel_int(shape[i], 0 if i == 0 else 1, 2147483647, "batch feed dimension", False))
    _ = cv_int(args[2], "start_step")
    py_setattr(owner, "_start", PythonObject(_parallel_int(args[2], 0, 999999, "start_step", False)))
    py_setattr(owner, "_data", _parallel_array(value, dims, True))
    return PythonObject(None)


def cross_vendor_feed_batch(owner: PythonObject, step: Int, shard: Int) raises -> PythonObject:
    if py_type(owner) is not CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].feed_cls:
        raise Error("cross_vendor: batch owner must be an exact BatchFeed")
    var data = owner._data
    var local = step - Int(py=owner._start)
    var k = Int(py=data.shape[1])
    if local < 0 or local >= Int(py=data.shape[0]) or shard < 0 or shard >= k:
        raise Error("BatchFeed step or shard falls outside its retained data")
    var batch = Int(py=data.shape[2])
    var width = Int(py=data.shape[3])
    var n = batch * width
    var output = generation_array([batch, width], "i")
    var source = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(py=data._addr))
    var target = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
    memcpy(dest=target, src=source + (local * k + shard) * n, count=n)
    _ = data^
    return output


def cross_vendor_rates_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(args[0]) is not c[].rate_cls:
        raise Error("LearningRateTable subclasses and Python attribute hooks are not accepted")
    var rates = cv_sequence(args[1], "rates")
    if len(rates) == 0:
        raise Error("LearningRateTable requires at least one rate")
    var values = c[].bi.list()
    for i in range(len(rates)):  # driver-loop: over the declared learning-rate schedule
        _ = cv_float(rates[i], "learning rate")
        var value = _parallel_float(rates[i], "learning rate")
        if value <= 0:
            raise Error("learning rates must be positive")
        _ = py_call(values.append, PythonObject(Float64(value)))
    py_setattr(args[0], "_values", values)
    _ = cv_int(args[2], "start_step")
    py_setattr(args[0], "_start", PythonObject(_parallel_int(args[2], 0, 999999, "start_step", False)))
    return PythonObject(None)


def cross_vendor_rate(table: PythonObject, step: Int) raises -> PythonObject:
    if py_type(table) is not CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].rate_cls:
        raise Error("cross_vendor: rate owner must be an exact LearningRateTable")
    var at = step - Int(py=table._start)
    if at < 0 or at >= len(table._values):
        raise Error("LearningRateTable step falls outside its retained schedule")
    return table._values[at]


def _cross_vendor_trainer(trainer: PythonObject, operation: String, params: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(trainer) is c[].parallel_cls:
        var args = c[].bi.list()
        _ = py_call(args.append, trainer)
        if operation == "export_raw":
            _ = py_call(args.append, 0)
            _ = py_call(args.append, PythonObject(None))
        else:
            for i in range(len(params)):  # driver-loop: over the native trainer operation envelope
                _ = py_call(args.append, params[i])
        return parallel_byte_call(operation, args)
    raise Error("cross_vendor: trainer must be an exact ParallelByteLanguageModelTrainer; Python trainer protocols are not accepted")


def _cross_vendor_one(value: PythonObject) raises -> PythonObject:
    var result = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi.list()
    _ = py_call(result.append, value)
    return result


def cross_vendor_log(path: PythonObject, row: PythonObject) raises:
    if path is None:
        return
    var bi = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi
    var stream = py_call(bi.open, cv_path(path), "ab")
    try:
        var encoded = py_call(PythonObject(_transport_json(row) + "\n").encode, "ascii")
        if Int(py=py_call(stream.write, encoded)) != len(encoded):
            raise Error("cross_vendor: short event log write")
        stream.close()
    except error:
        stream.close()
        raise error


def cross_vendor_worker_arguments(trainer: PythonObject, batches: PythonObject, rates: PythonObject, callback: PythonObject) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if trainer is not None and py_type(trainer) is not c[].parallel_cls:
        raise Error("cross_vendor: trainer must be an exact ParallelByteLanguageModelTrainer; Python trainer protocols are not accepted")
    if py_type(batches) is not c[].feed_cls:
        raise Error("cross_vendor: batches must be an exact BatchFeed; Python batch callbacks are not accepted")
    if rates is not None and py_type(rates) is not c[].rate_cls:
        raise Error("cross_vendor: lr_for_step must be an exact LearningRateTable; Python schedules are not accepted")
    if callback is not None:
        raise Error("cross_vendor: on_commit callbacks are not accepted; use drain_events or native checkpoint configuration")


def cross_vendor_commit_configuration(owner: PythonObject, directory: PythonObject, steps: PythonObject, expected: PythonObject, retain: PythonObject, scheme: PythonObject) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var ids = cv_sequence(steps, "checkpoint_steps")
    var chosen = c[].bi.set()
    for i in range(len(ids)):  # driver-loop: over explicit native checkpoint steps
        var step = cv_int(ids[i], "checkpoint step")
        if step < 1:
            raise Error("cross_vendor: checkpoint steps must be positive")
        _ = py_call(chosen.add, step)
    if len(chosen) != 0 and directory is None:
        raise Error("cross_vendor: checkpoint_steps requires checkpoint_dir")
    py_setattr(owner, "checkpoint_dir", PythonObject(None) if directory is None else cv_path(directory))
    py_setattr(owner, "_checkpoint_steps", chosen)
    py_setattr(owner, "retain_gradients", PythonObject(cv_bool(retain, "retain_gradients")))
    var hash_scheme = cv_text(scheme, "hash_scheme")
    if hash_scheme != "sha256.v1" and hash_scheme != "sliced-sha256-8.v2":
        raise Error("cross_vendor: unsupported native hash_scheme")
    py_setattr(owner, "hash_scheme", scheme)
    var rows = cv_sequence(expected, "expected_commits")
    var registry = c[].bi.dict()
    for i in range(len(rows)):  # driver-loop: over explicit expected committed hashes
        var row = rows[i]
        cv_json(row)
        if py_type(row) is not c[].bi.dict or not py_contains(row, "step"):
            raise Error("cross_vendor: expected commits require step and state or total_sha256")
        var step = cv_int(row["step"], "expected step")
        if step < 1 or py_contains(registry, step):
            raise Error("cross_vendor: expected commit steps must be unique and positive")
        var keys = py_call(c[].bi.list, row)
        if len(keys) < 2:
            raise Error("cross_vendor: expected commits require state or total_sha256")
        for j in range(len(keys)):  # small-loop(expected commit fields): the named keys of one configured row
            var key = String(keys[j])
            if key == "step":
                continue
            if key == "losses_f32_hex":
                var losses = cv_sequence(row[keys[j]], "expected loss words")
                for k in range(len(losses)):  # driver-loop: over logical shard loss words of one configured row
                    if len(PythonObject(cv_text(losses[k], "expected loss word"))) != 8:
                        raise Error("cross_vendor: expected loss words must have eight hexadecimal characters")
                continue
            if key == "lr_f32_hex":
                if len(PythonObject(cv_text(row[keys[j]], "expected learning-rate word"))) != 8:
                    raise Error("cross_vendor: expected learning-rate word must have eight hexadecimal characters")
                continue
            if key == "hash_scheme":
                _ = cv_text(row[keys[j]], "expected hash scheme")
                continue
            if key != "state" and key != "total_sha256" and key != "state_sha256" and key != "gradient_sha256":
                raise Error("cross_vendor: unknown expected commit field")
            _ = cv_text(row[keys[j]], "expected hash")
            if len(row[keys[j]]) != 64:
                raise Error("cross_vendor: expected hashes must have 64 hexadecimal characters")
            var raw = py_call(row[keys[j]].encode, "ascii")
            for k in range(64):  # small-loop(64 hex characters): one configured digest string
                var ch = Int(py=raw[k])
                if not (ch >= 48 and ch <= 57) and not (ch >= 97 and ch <= 102):
                    raise Error("cross_vendor: expected hashes must have 64 hexadecimal characters")
        py_setitem(registry, step, native_json_parse(PythonObject(_transport_json(row))))
    py_setattr(owner, "_expected_commits", registry)


def cross_vendor_sliced_hash(value: PythonObject) raises -> PythonObject:
    # The eight-slice v2 digest format is computed in core/sha256.mojo
    # (sha256_sliced8_hex) over the same bytes.
    return sha256_sliced8(cross_vendor_bytes(value))


def cross_vendor_float_word(value: PythonObject) raises -> PythonObject:
    var word = bitcast[DType.uint32](Float32(cv_float(value, "recorded float32")))
    return py_call(CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi.format, Int(word), "08x")


def cross_vendor_commit(owner: PythonObject, trainer: PythonObject, row: PythonObject, snapshot: PythonObject, started: Float64, mut io_error: ByteCheckpointIOError) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var event = py_call(c[].bi.dict, row)
    var hash_start = transport_clock()
    py_setitem(event, "hash_scheme", owner.hash_scheme)
    if String(owner.hash_scheme) == "sha256.v1":
        py_setitem(event, "state_sha256", row["state"])
        py_setitem(event, "gradient_sha256", row["total_sha256"])
    else:
        var digest = Sha256()
        for key in [String("parameters"), String("m"), String("v"), String("flags")]:  # small-loop(four state keys): one digest update per named state field
            digest.update_text(String(cross_vendor_sliced_hash(snapshot[key])))
        py_setitem(event, "state_sha256", PythonObject(digest.hexdigest()))
        py_setitem(event, "gradient_sha256", cross_vendor_sliced_hash(row["total_bytes"]))
    py_setitem(event, "lr_f32_hex", cross_vendor_float_word(trainer._state["config"]["lr"]))
    var losses = c[].bi.list()
    var values = cv_sequence(row["losses"], "committed losses")
    for i in range(len(values)):  # driver-loop: over logical shard loss metadata
        _ = py_call(losses.append, cross_vendor_float_word(values[i]))
    py_setitem(event, "losses_f32_hex", losses)
    py_setitem(event, "hash_seconds", PythonObject(transport_clock() - hash_start))
    if not cv_bool(owner.retain_gradients, "retain_gradients"):
        _ = py_call(event.pop, "total_bytes")
    var step = Int(py=row["step"])
    var expected = py_get(owner._expected_commits, step)
    if expected is not None:
        var keys = py_call(c[].bi.list, expected)
        for i in range(len(keys)):  # driver-loop: over configured committed-state checks
            var key = String(keys[i])
            if key != "step" and not Bool(py=event[key] == expected[key]):
                py_setitem(event, "expected_mismatch", PythonObject(key))
                _ = py_call(owner._events.append, event)
                cross_vendor_log(owner.log_path, row_without_gradient(event))
                raise Error("CrossVendorMismatch: committed " + key + " differs from the configured expectation at step " + String(step))
    if py_contains(owner._checkpoint_steps, step):
        var checkpoint_start = transport_clock()
        var state = py_call(c[].bi.dict, trainer._state)
        _ = py_call(state.update, snapshot)
        var filename = PythonObject("ckpt_") + py_call(PythonObject(String(step)).zfill, 8) + PythonObject(".blm")
        var path = py_call(owner.checkpoint_dir.rstrip, "/") + PythonObject("/") + filename
        var digest = byte_checkpoint_save(path, state, io_error)
        py_setitem(event, "checkpoint_path", path)
        py_setitem(event, "checkpoint_sha256", digest)
        py_setitem(event, "checkpoint_seconds", PythonObject(transport_clock() - checkpoint_start))
    py_setitem(event, "seconds", PythonObject(transport_clock() - started))
    _ = py_call(owner._events.append, event)
    cross_vendor_log(owner.log_path, row_without_gradient(event))
    return event


def row_without_gradient(row: PythonObject) raises -> PythonObject:
    var result = py_call(CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi.dict, row)
    _ = py_call(result.pop, "total_bytes", PythonObject(None))
    return result


def cross_vendor_worker_admit(owner: PythonObject) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(owner) is not c[].worker_cls:
        raise Error("cross_vendor: worker owner must be an exact Worker")
    cross_vendor_worker_arguments(owner.trainer, owner.batches, owner.lr_for_step, owner.on_commit)
    var shards = cv_sequence(owner.shards, "shards")
    for i in range(len(shards)):  # driver-loop: over admitted native shard metadata
        _ = cv_int(shards[i], "shard")
    var address = cv_sequence(owner.address, "address")
    if len(address) != 2:
        raise Error("cross_vendor: worker address requires two entries")
    _ = cv_text(address[0], "host")
    _ = cv_int(address[1], "port")
    _ = cv_text(owner.name, "name")
    _ = cv_float(owner.connect_timeout, "connect_timeout")
    _ = cv_float(owner.timeout, "timeout")
    _ = cv_bool(owner.chained, "chained")
    _ = cv_bool(owner.retain_gradients, "retain_gradients")
    var scheme = cv_text(owner.hash_scheme, "hash_scheme")
    if scheme != "sha256.v1" and scheme != "sliced-sha256-8.v2":
        raise Error("cross_vendor: unsupported native hash_scheme")
    if owner.checkpoint_dir is not None:
        _ = cv_path(owner.checkpoint_dir)
    if py_type(owner._checkpoint_steps) is not c[].bi.set or py_type(owner._expected_commits) is not c[].bi.dict or py_type(owner._events) is not c[].bi.list:
        raise Error("cross_vendor: policy and event storage must be native containers")
    var checkpoint_steps = py_call(c[].bi.list, owner._checkpoint_steps)
    for i in range(len(checkpoint_steps)):  # driver-loop: over explicit native checkpoint steps
        _ = cv_int(checkpoint_steps[i], "checkpoint step")
    var expected_steps = py_call(c[].bi.list, owner._expected_commits)
    for i in range(len(expected_steps)):  # driver-loop: over explicit expected committed hashes
        _ = cv_int(expected_steps[i], "expected step")
        cv_json(owner._expected_commits[expected_steps[i]])
    if owner.log_path is not None:
        _ = cv_path(owner.log_path)


def cross_vendor_coordinator_admit(owner: PythonObject) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(owner) is not c[].coordinator_cls:
        raise Error("cross_vendor: coordinator owner must be an exact Coordinator")
    if owner.on_step is not None:
        raise Error("cross_vendor: on_step callbacks are not accepted; use drain_events or log_path")
    _ = cv_int(owner.port, "port")
    _ = cv_int(owner.n_workers, "workers")
    _ = cv_int(owner.K, "logical_shards")
    _ = cv_int(owner.steps, "steps")
    _ = cv_int(owner.max_bytes, "max_gradient_bytes")
    _ = cv_text(owner.host, "host")
    _ = cv_float(owner.timeout, "timeout")
    _ = cv_float(owner.accept_timeout, "accept_timeout")
    _ = cv_bool(owner.chained, "chained")
    if py_type(owner.peers) is not c[].bi.list or py_type(owner._events) is not c[].bi.list:
        raise Error("cross_vendor: peer and event storage must be native lists")
    if py_type(owner.ready) is not c[].ready_cls:
        raise Error("cross_vendor: readiness requires the native event owner")
    if owner.log_path is not None:
        _ = cv_path(owner.log_path)


def cross_vendor_worker_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var owner = args[0]
    if py_type(owner) is not c[].worker_cls:
        raise Error("cross_vendor: Worker subclasses and Python attribute hooks are not accepted")
    var trainer = args[7]
    cross_vendor_worker_arguments(trainer, args[3], args[11], args[12])
    var shards = cv_sequence(args[2], "shards")
    var ids = c[].bi.list()
    for i in range(len(shards)):  # driver-loop: over the worker's logical shard ids
        var shard = cv_int(shards[i], "shard")
        if shard < 0:
            raise Error("cross_vendor: shards must be nonnegative")
        _ = py_call(ids.append, shard)
    var address = cv_sequence(args[4], "address")
    if len(address) != 2:
        raise Error("worker address requires host and port in [1,65535]")
    _ = cv_text(address[0], "worker host")
    var port = cv_int(address[1], "worker port")
    if port < 1 or port > 65535:
        raise Error("worker address requires host and port in [1,65535]")
    if args[5] is not None:
        _ = cv_text(args[5], "worker name")
    if cv_float(args[8], "connect_timeout") <= 0 or cv_float(args[9], "timeout") <= 0:
        raise Error("worker timeouts must be positive")
    _ = cv_bool(args[10], "chained")
    _ = cv_int(args[6], "device")
    # Admit every optional policy before retaining/copying model state. Invalid
    # callback replacements must not construct even a lazy trainer owner.
    py_setattr(owner, "log_path", PythonObject(None) if args[13] is None else cv_path(args[13]))
    cross_vendor_commit_configuration(owner, args[14], args[15], args[16], args[17], args[18])
    if trainer is None:
        cv_state(args[1])
        trainer = py_call(c[].bi.object.__new__, c[].parallel_cls)
        var device = _cross_vendor_one(args[6])
        _ = parallel_byte_init(trainer, args[1], device, PythonObject(1), PythonObject(False))
    py_setattr(owner, "trainer", trainer)
    py_setattr(owner, "shards", py_call(c[].bi.sorted, ids))
    py_setattr(owner, "batches", args[3])
    py_setattr(owner, "address", py_call(c[].bi.tuple, address))
    py_setattr(owner, "name", args[5] if Bool(py=args[5]) else PythonObject(transport_hostname()))
    py_setattr(owner, "connect_timeout", args[8])
    py_setattr(owner, "timeout", args[9])
    py_setattr(owner, "chained", py_call(c[].bi.bool, args[10]))
    py_setattr(owner, "lr_for_step", args[11])
    py_setattr(owner, "on_commit", args[12])
    py_setattr(owner, "_events", c[].bi.list())
    return PythonObject(None)


def cross_vendor_worker_connect(owner: PythonObject) raises -> PythonObject:
    cross_vendor_worker_admit(owner)
    var deadline = transport_clock() + Float64(py=owner.connect_timeout)
    while True:  # driver-loop: over native connection retries within the configured deadline
        try:
            var socket = transport_connect(String(owner.address[0]), Int(py=owner.address[1]), 30.0)
            transport_timeout(socket, Float64(py=owner.timeout))
            return socket
        except error:
            if String(error).startswith("InterruptedError:"):
                raise error
            if transport_clock() > deadline:
                raise error
            transport_sleep(2.0)


def _cross_vendor_batch(owner: PythonObject, step: Int, shard: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(owner.batches) is c[].feed_cls:
        return cross_vendor_feed_batch(owner.batches, step, Int(py=shard))
    raise Error("cross_vendor: batches must be an exact BatchFeed; Python batch callbacks are not accepted")


def cross_vendor_worker_run(owner: PythonObject, mut io_error: ByteCheckpointIOError) raises -> PythonObject:
    cross_vendor_worker_admit(owner)
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var tr = owner.trainer
    var empty_args = bi.list()
    var state = _cross_vendor_trainer(tr, "export_raw", empty_args)
    var n = len(cross_vendor_bytes(state["parameters"])) // 4
    var hello = bi.dict()
    py_setitem(hello, "protocol", PythonObject("mojolearn.cross-vendor.v1"))
    py_setitem(hello, "name", owner.name)
    py_setitem(hello, "vendor", c[].vendor)
    py_setitem(hello, "shards", owner.shards)
    py_setitem(hello, "completed", _cross_vendor_trainer(tr, "step", empty_args))
    py_setitem(hello, "state", cross_vendor_state_hash(state))
    py_setitem(hello, "n_total", PythonObject(n))
    py_setitem(hello, "chained", owner.chained)
    var deadline = transport_clock() + Float64(py=owner.connect_timeout)
    var socket = PythonObject(None)
    var response = PythonObject(None)
    while True:  # driver-loop: over initial tunnel handshakes within the retry deadline
        socket = cross_vendor_worker_connect(owner)
        try:
            transport_send(socket, hello, PythonObject(None))
            response = transport_receive(socket, n * 4)
            break
        except error:
            transport_close(socket)
            if String(error).startswith("InterruptedError:"):
                raise error
            if transport_clock() > deadline:
                raise error
            transport_sleep(5.0)
    var committed = 0
    var held = bi.list()
    var total = PythonObject(None)
    var on_device = False
    var first = True
    var chained = Bool(py=owner.chained)
    var step_started = transport_clock()
    try:
        while True:  # driver-loop: over native coordinator commands until completion
            if first:
                first = False
            else:
                response = transport_receive(socket, n * 4)
            var head = response[0]
            var payload = response[1]
            var command = String(py_get(head, "cmd"))
            if command == "step":
                step_started = transport_clock()
                var step = Int(py=head["step"])
                if Int(py=_cross_vendor_trainer(tr, "step", empty_args)) != step:
                    raise Error("CrossVendorMismatch: worker completed step differs from coordinator")
                if Bool(py=py_get(head, "chained", PythonObject(False))) != chained:
                    raise Error("CrossVendorMismatch: coordinator and worker disagree on the protocol")
                if owner.lr_for_step is not None:
                    var lr = cross_vendor_rate(owner.lr_for_step, step)
                    _ = _cross_vendor_trainer(tr, "set_lr", _cross_vendor_one(lr))
                var losses = bi.list()
                var parts = bi.list()
                var device_fold = chained and Bool(py=_cross_vendor_trainer(tr, "has_device_fold", empty_args))
                if device_fold and Bool(py=py_get(head, "first", PythonObject(False))):
                    _ = _cross_vendor_trainer(tr, "fold_reset", _cross_vendor_one(PythonObject(None)))
                    for k in range(len(owner.shards)):  # driver-loop: over this worker's immutable logical shards
                        var batch = _cross_vendor_batch(owner, step, owner.shards[k])
                        _ = py_call(losses.append, _cross_vendor_trainer(tr, "shard_gradient_fold", _cross_vendor_one(batch)))
                    held = bi.list()
                    total = PythonObject(None)
                    on_device = True
                else:
                    for k in range(len(owner.shards)):  # driver-loop: over this worker's immutable logical shards
                        var batch = _cross_vendor_batch(owner, step, owner.shards[k])
                        var result = _cross_vendor_trainer(tr, "shard_gradient", _cross_vendor_one(batch))
                        _ = py_call(losses.append, result[0])
                        _ = py_call(parts.append, cross_vendor_bytes(result[1]).tobytes())
                    on_device = False
                var reply = bi.dict()
                py_setitem(reply, "step", PythonObject(step))
                py_setitem(reply, "shards", owner.shards)
                py_setitem(reply, "losses", losses)
                if chained:
                    if not on_device:
                        held = parts
                        total = PythonObject(None)
                    transport_send(socket, reply, PythonObject(None))
                else:
                    transport_send(socket, reply, py_call(bi.bytes().join, parts))
            elif command == "fold":
                if not chained or (len(held) == 0 and not on_device):
                    raise Error("CrossVendorMismatch: worker was asked to fold with nothing held")
                if on_device:
                    total = _cross_vendor_trainer(tr, "fold_export", _cross_vendor_one(PythonObject(None)))
                elif Bool(py=_cross_vendor_trainer(tr, "has_device_fold", empty_args)):
                    _ = _cross_vendor_trainer(tr, "fold_reset", _cross_vendor_one(payload if len(payload) != 0 else PythonObject(None)))
                    for k in range(len(held)):  # driver-loop: over held logical gradients continuing the prefix
                        _ = _cross_vendor_trainer(tr, "fold_add", _cross_vendor_one(held[k]))
                    total = _cross_vendor_trainer(tr, "fold_export", _cross_vendor_one(PythonObject(None)))
                else:
                    total = cross_vendor_fold(held, payload if len(payload) != 0 else PythonObject(None))
                held = bi.list()
                on_device = False
                var reply = bi.dict()
                py_setitem(reply, "step", head["step"])
                transport_send(socket, reply, total)
            elif command == "apply":
                if len(payload) != 0:
                    total = payload
                if total is None or len(total) != n * 4:
                    raise Error("CrossVendorMismatch: summed gradient has the wrong size")
                var expected_hash = py_get(head, "total_sha256")
                if expected_hash is not None and not Bool(py=sha256_hexdigest(total) == expected_hash):
                    raise Error("CrossVendorMismatch: total hash differs from the coordinator")
                var gradient = py_call(c[].arrays, "f")
                _ = py_call(gradient.frombytes, total)
                _ = _cross_vendor_trainer(tr, "apply_gradient", _cross_vendor_one(gradient))
                var applied = total
                total = PythonObject(None)
                committed += 1
                var done = _cross_vendor_trainer(tr, "step", empty_args)
                var snapshot = _cross_vendor_trainer(tr, "export_raw", empty_args)
                var digest = cross_vendor_state_hash(snapshot)
                var row = bi.dict()
                py_setitem(row, "step", done)
                py_setitem(row, "state", digest)
                py_setitem(row, "total_sha256", expected_hash)
                py_setitem(row, "losses", py_get(head, "losses"))
                py_setitem(row, "total_bytes", applied)
                _ = cross_vendor_commit(owner, tr, row, snapshot, step_started, io_error)
                var reply = bi.dict()
                py_setitem(reply, "completed", done)
                py_setitem(reply, "state", digest)
                transport_send(socket, reply, PythonObject(None))
            elif command == "done":
                transport_close(socket)
                return PythonObject(committed)
            elif command == "refuse":
                raise Error("CrossVendorMismatch: coordinator refused: " + String(py_get(head, "reason", PythonObject(""))))
            else:
                raise Error("cross_vendor: unknown coordinator command " + command)
    except error:
        transport_close(socket)
        raise error


def cross_vendor_console(text: String) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var raw = py_call(PythonObject(text + "\n").encode, "utf-8")
    var sent = 0
    while sent < len(raw):  # driver-loop: over partial native console writes
        var n = Int(py=py_call(c[].console_write, 1, raw[py_call(c[].bi.slice, sent, len(raw))]))
        if n < 1:
            raise Error("cross_vendor: short console write")
        sent += n


def cross_vendor_main(argv: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var values = cv_sequence(c[].argv[py_call(c[].bi.slice, 1, None)] if argv is None else argv, "arguments")
    var usage = "usage: python -m mojolearn.cross_vendor coordinator --workers N --shards K --steps N [--host HOST] [--port PORT] [--log PATH] [--chained]"
    if len(values) == 0:
        raise Error(usage)
    if cv_text(values[0], "command") == "--help" or String(values[0]) == "-h":
        cross_vendor_console(usage)
        return PythonObject(0)
    if String(values[0]) != "coordinator":
        raise Error(usage)
    var host = PythonObject("0.0.0.0")
    var port = PythonObject(7777)
    var workers = PythonObject(None)
    var shards = PythonObject(None)
    var steps = PythonObject(None)
    var path = PythonObject(None)
    var chained = PythonObject(False)
    var at = 1
    while at < len(values):  # driver-loop: over native command-line options
        var option = cv_text(values[at], "option")
        at += 1
        if option == "--help" or option == "-h":
            cross_vendor_console(usage)
            return PythonObject(0)
        if option == "--chained":
            chained = PythonObject(True)
            continue
        if at == len(values):
            raise Error("missing value for " + option)
        var value = values[at]
        _ = cv_text(value, "option value")
        at += 1
        if option == "--host":
            host = value
        elif option == "--log":
            path = value
        elif option == "--port":
            port = py_call(c[].bi.int, value)
        elif option == "--workers":
            workers = py_call(c[].bi.int, value)
        elif option == "--shards":
            shards = py_call(c[].bi.int, value)
        elif option == "--steps":
            steps = py_call(c[].bi.int, value)
        else:
            raise Error("unknown coordinator option " + option)
    if workers is None or shards is None or steps is None:
        raise Error("coordinator requires --workers, --shards and --steps")
    var owner = py_call(c[].bi.object.__new__, c[].coordinator_cls)
    var args = c[].bi.list()
    _ = py_call(args.append, owner)
    _ = py_call(args.append, port)
    _ = py_call(args.append, workers)
    _ = py_call(args.append, shards)
    _ = py_call(args.append, steps)
    _ = py_call(args.append, host)
    _ = py_call(args.append, PythonObject(None))
    _ = py_call(args.append, 3600.0)
    _ = py_call(args.append, 3600.0)
    _ = py_call(args.append, 2147483648)
    _ = py_call(args.append, chained)
    _ = py_call(args.append, c[].ready_cls)
    _ = py_call(args.append, path)
    _ = cross_vendor_coordinator_init(args)
    var rows = cross_vendor_coordinator_run(owner)
    cross_vendor_console("AGREED on " + String(len(rows)) + " steps across " + String(workers) + " workers")
    return PythonObject(0)
