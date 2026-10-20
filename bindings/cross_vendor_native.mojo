# SPDX-License-Identifier: Apache-2.0
"""Native byte admission, identity folds and state hashes for live training."""
from std.ffi import _Global
from std.math import isfinite
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_transport import (
    transport_initialize, transport_send, transport_receive, _transport_exact,
    transport_listen, transport_accept, transport_close, transport_clock, transport_sleep,
)
from bindings.causal_exception import causal_borrowed_item, causal_set_exception


struct CrossVendorBoundary(Defaultable, Movable):
    var bi: PythonObject
    var arrays: PythonObject
    var array_cls: PythonObject
    var hash_type: PythonObject
    var mismatch_cls: PythonObject
    var fold_pair: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.arrays = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.hash_type = PythonObject(None)
        self.mismatch_cls = PythonObject(None)
        self.fold_pair = PythonObject(None)


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
    c[].hash_type = Python.import_module("hashlib").sha256
    c[].fold_pair = fold_pair
    if c[].bi.type(c[].hash_type) is not c[].bi.type(c[].bi.len):
        raise Error("cross_vendor: SHA256 must be a native extension entry")
    if c[].bi.type(c[].fold_pair) is not c[].bi.type(c[].bi.len):
        raise Error("cross_vendor: fold must be a native extension entry")
    transport_initialize()


def cross_vendor_bytes(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if c[].bi.type(value) is c[].array_cls:
        return value._mv.cast("B")
    return c[].bi.memoryview(value).cast("B")


def cross_vendor_f32(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var raw = cross_vendor_bytes(value)
    if len(raw) % 4 != 0:
        raise Error("cross_vendor: gradient bytes must be a multiple of four")
    return raw.cast("f")


def cross_vendor_fold(gradients: PythonObject, prefix: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var parts = c[].bi.list(gradients)
    var first = 0
    var source = prefix
    if source is None:
        if len(parts) == 0:
            raise Error("ordered_fold needs at least one gradient")
        source = parts[0]
        first = 1
    var store = c[].arrays("f")
    store.frombytes(cross_vendor_f32(source).cast("B"))
    var n = len(store)
    var total = store.buffer_info()[0]
    for k in range(first, len(parts)):  # driver-loop: over logical shard gradient buffers in immutable order
        var shard = c[].arrays("f")
        shard.frombytes(cross_vendor_f32(parts[k]).cast("B"))
        if len(shard) != n:
            raise Error("ordered_fold: gradients differ in length")
        # Retain the selected operation's existing GPU/CPU execution route.
        # Mojo owns the immutable fold; the cached compiled entry owns its
        # arithmetic, including FTZ and the one float32 rounding per add.
        _ = c[].fold_pair(total, shard.buffer_info()[0], n)
        # Explicitly retain the C array owning the address across every read;
        # converting to an integer pointer erases the compiler's origin edge.
        _ = shard^
    var result = store.tobytes()
    _ = store
    return result


def cross_vendor_state_hash(state: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if c[].bi.type(state) is not c[].bi.dict:
        raise Error("cross_vendor: state must be a dict")
    var hasher = c[].hash_type()
    for key in [String("parameters"), String("m"), String("v"), String("flags")]:
        hasher.update(cross_vendor_bytes(state[key]))
    return hasher.hexdigest()


def cross_vendor_call(operation: String, args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if operation == "initialize":
        cross_vendor_initialize(args[0], args[1], args[2])
    elif operation == "coordinator_init":
        return cross_vendor_coordinator_init(args)
    elif operation == "coordinator_run":
        return cross_vendor_coordinator_run(args[0])
    elif operation == "coordinator_drive":
        return cross_vendor_drive(args[0])
    elif operation == "coordinator_refuse":
        cross_vendor_refuse(args[0], String(args[1]))
    elif operation.startswith("event_"):
        return cross_vendor_event(operation, args)
    elif operation == "f32":
        return cross_vendor_f32(args[0])
    elif operation == "fold":
        return cross_vendor_fold(args[0], args[1])
    elif operation == "pair":
        var parts = c[].bi.list()
        parts.append(args[0])
        parts.append(args[1])
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
    try:
        operation = String(args[0])
        var result = cross_vendor_call(operation, args[1])
        return result.steal_data()
    except error:
        if String(error).startswith("CrossVendorMismatch:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 2)
                return causal_set_exception(error, causal_borrowed_item(context, "mismatch"))
            except nested:
                return raise_python_exception(nested, ExceptionType("PyExc_RuntimeError"))
        if operation == "send" or operation == "receive" or operation == "receive_exact":
            return raise_python_exception(error, ExceptionType("PyExc_ConnectionError"))
        return raise_python_exception(error, ExceptionType.ValueError)


def cross_vendor_coordinator_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var owner = args[0]
    var workers = Int(py=args[2])
    var shards = Int(py=args[3])
    if workers < 1 or shards < workers:
        raise Error("need 1 <= workers <= logical_shards")
    owner.port = args[1]
    owner.n_workers = PythonObject(workers)
    owner.K = PythonObject(shards)
    owner.steps = args[4]
    owner.host = args[5]
    owner.on_step = args[6]
    owner.timeout = args[7]
    owner.accept_timeout = args[8]
    owner.max_bytes = args[9]
    owner.chained = c[].bi.bool(args[10])
    owner.peers = c[].bi.list()
    owner.bound_port = PythonObject(None)
    var ready = c[].bi.object.__new__(args[11])
    ready._set = PythonObject(False)
    owner.ready = ready
    return PythonObject(None)


def cross_vendor_event(operation: String, args: PythonObject) raises -> PythonObject:
    var event = args[0]
    if operation == "event_init" or operation == "event_clear":
        event._set = PythonObject(False)
    elif operation == "event_set":
        event._set = PythonObject(True)
    elif operation == "event_is_set":
        return event._set
    elif operation == "event_wait":
        var unlimited = args[1] is None
        var timeout = Float64(0) if unlimited else Float64(py=args[1])
        if not unlimited and not isfinite(timeout):
            raise Error("readiness timeout must be finite or None")
        var start = transport_clock()
        while not Bool(py=event._set):  # driver-loop: over native readiness wait notifications
            if not unlimited and transport_clock() - start >= timeout:
                return PythonObject(False)
            transport_sleep(0.001)
        return PythonObject(True)
    return PythonObject(None)


def cross_vendor_refuse(owner: PythonObject, reason: String) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var header = c[].bi.dict()
    header["cmd"] = PythonObject("refuse")
    header["reason"] = PythonObject(reason)
    for i in range(len(owner.peers)):  # driver-loop: over connected worker refusal notices
        try:
            transport_send(owner.peers[i]["sock"], header)
        except:
            pass
    raise Error("CrossVendorMismatch: " + reason)


def _cross_vendor_elapsed(start: Float64) raises -> PythonObject:
    return CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi.round(PythonObject(transport_clock() - start), 3)


def cross_vendor_drive(owner: PythonObject) raises -> PythonObject:
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
        var name = peer.get("name")
        if bi.type(name) is not bi.str or Bool(py=names.__contains__(name)):
            cross_vendor_refuse(owner, "workers require distinct string names")
        names.add(name)
        if String(peer.get("protocol")) != "mojolearn.cross-vendor.v1":
            cross_vendor_refuse(owner, "protocol mismatch: " + String(name))
        var ids = peer.get("shards")
        if bi.type(ids) is not bi.list or len(ids) == 0:
            cross_vendor_refuse(owner, "workers require nonempty shard lists")
        for k in range(len(ids)):
            if bi.type(ids[k]) is not bi.int:
                cross_vendor_refuse(owner, "shard IDs must be integers")
            owned.append(ids[k])
        for key in [String("state"), String("completed"), String("n_total")]:
            if not Bool(py=peer[key] == peers[0][key]):
                cross_vendor_refuse(owner, "workers start from different " + key)
    owned = bi.sorted(owned)
    if len(owned) != shards:
        cross_vendor_refuse(owner, "workers must own every logical shard exactly once")
    for k in range(shards):
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
            while j > 0 and Int(py=peers[j - 1]["shards"][0]) > Int(py=peer["shards"][0]):
                peers[j] = peers[j - 1]
                j -= 1
            peers[j] = peer
        for i in range(count):
            var ids = peers[i]["shards"]
            for k in range(len(ids)):
                if Int(py=ids[k]) != Int(py=ids[0]) + k:
                    cross_vendor_refuse(owner, "chained fold: worker must own a contiguous block")
            if not Bool(py=peers[i].get("chained", PythonObject(False))):
                cross_vendor_refuse(owner, "chained fold: every worker must start with chained=True")
    var rows = bi.list()
    var completed = Int(py=peers[0]["completed"])
    for step in range(completed, Int(py=owner.steps)):  # driver-loop: over requested lockstep training rounds
        var started = transport_clock()
        var timing = bi.dict()
        for i in range(count):  # driver-loop: over worker step requests
            var head = bi.dict()
            head["cmd"] = PythonObject("step")
            head["step"] = PythonObject(step)
            head["chained"] = PythonObject(chained)
            head["first"] = PythonObject(chained and i == 0)
            transport_send(peers[i]["sock"], head)
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
            if not Bool(py=head.get("step") == PythonObject(step)) or not Bool(py=head.get("shards") == peer["shards"]):
                cross_vendor_refuse(owner, String(peer["name"]) + " answered the wrong step or shards")
            if len(head["losses"]) != n or (not chained and len(payload) != n * total_bytes):
                cross_vendor_refuse(owner, String(peer["name"]) + " sent a wrong gradient or loss count")
            for k in range(n):
                var id = peer["shards"][k]
                losses[id] = head["losses"][k]
                if not chained:
                    gradients[id] = payload[bi.slice(k * total_bytes, (k + 1) * total_bytes)]
            if chained:
                timing["gradients_" + String(peer["name"])] = _cross_vendor_elapsed(started)
        var total = bi.bytes()
        if chained:
            for i in range(count):  # driver-loop: over chained prefix handoffs in fixed shard order
                var head = bi.dict()
                head["cmd"] = PythonObject("fold")
                head["step"] = PythonObject(step)
                var folded_at = transport_clock()
                transport_send(peers[i]["sock"], head, total)
                var response = transport_receive(peers[i]["sock"], total_bytes)
                total = response[1]
                if not Bool(py=response[0].get("step") == PythonObject(step)) or len(total) != total_bytes:
                    cross_vendor_refuse(owner, String(peers[i]["name"]) + " returned a wrong fold")
                timing["fold_" + String(peers[i]["name"])] = _cross_vendor_elapsed(folded_at)
        else:
            var parts = bi.list()
            for k in range(shards):
                parts.append(gradients[k])
            total = cross_vendor_fold(parts, PythonObject(None))
        var all_losses = bi.list()
        for k in range(shards):
            all_losses.append(losses[k])
        var total_hash = c[].hash_type(total).hexdigest()
        var apply = bi.dict()
        apply["cmd"] = PythonObject("apply")
        apply["step"] = PythonObject(step)
        apply["total_sha256"] = total_hash
        apply["losses"] = all_losses
        for i in range(count):  # driver-loop: over identical optimizer commit requests
            transport_send(peers[i]["sock"], apply, bi.bytes() if chained and i == count - 1 else total)
        timing["apply_sent"] = _cross_vendor_elapsed(started)
        var hashes = bi.dict()
        var agreed = PythonObject(None)
        var vendors = bi.dict()
        for i in range(count):  # driver-loop: over committed worker state witnesses
            var peer = peers[i]
            var response = transport_receive(peer["sock"], 0)
            var head = response[0]
            if not Bool(py=head.get("completed") == PythonObject(step + 1)):
                cross_vendor_refuse(owner, String(peer["name"]) + " did not commit step " + String(step + 1))
            hashes[peer["name"]] = head["state"]
            vendors[peer["name"]] = peer.get("vendor")
            if i == 0:
                agreed = head["state"]
            elif not Bool(py=head["state"] == agreed):
                cross_vendor_refuse(owner, "replicas disagree after step " + String(step + 1))
            timing["applied_" + String(peer["name"])] = _cross_vendor_elapsed(started)
        timing["step_seconds"] = _cross_vendor_elapsed(started)
        var row = bi.dict()
        row["step"] = PythonObject(step + 1)
        row["losses"] = all_losses
        row["state"] = agreed
        row["workers"] = hashes
        row["vendors"] = vendors
        row["total_sha256"] = total_hash
        row["protocol"] = PythonObject("chained" if chained else "gathered")
        row["timing"] = timing
        rows.append(row)
        # Existing user callback contract is preserved pending the owner's
        # explicit choice between native events and caller callback exceptions.
        # A Python callback here is retained runtime debt and must fail the
        # whole-operation native audit; moving its invocation does not close it.
        if Bool(py=owner.on_step):
            _ = owner.on_step(row)
    var done = bi.dict()
    done["cmd"] = PythonObject("done")
    for i in range(count):  # driver-loop: over group completion notices
        transport_send(peers[i]["sock"], done)
    return rows


def _cross_vendor_close_peers(owner: PythonObject) raises:
    for i in range(len(owner.peers)):  # driver-loop: over native socket cleanup
        try:
            transport_close(owner.peers[i]["sock"])
        except:
            pass


def cross_vendor_coordinator_run(owner: PythonObject) raises -> PythonObject:
    var bi = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi
    var server = transport_listen(String(owner.host), Int(py=owner.port), Float64(py=owner.accept_timeout))
    owner.bound_port = server.getsockname()[1]
    owner.ready._set = PythonObject(True)
    var rows = PythonObject(None)
    try:
        while len(owner.peers) < Int(py=owner.n_workers):  # driver-loop: over admitted worker connections
            var socket = transport_accept(server, Float64(py=owner.timeout))
            try:
                var response = transport_receive(socket, 0)
                var peer = bi.dict(response[0])
                peer["sock"] = socket
                peer["where"] = socket.getpeername()
                owner.peers.append(peer)
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
