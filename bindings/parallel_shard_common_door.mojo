# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shared parts of the native Python-sharded multi-GPU drivers
(lane/native-parallel-driver-20261024,
docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md, "kind B").

These drivers cut their work into logical shards (`*_per_shard` rows,
columns, series, trees or classes), run each shard's plain native owner in a
worker process per device and place every shard's block at its own offset.
The old Python driver (`_parallel_pool.DevicePool` plus pickle) did the same
in Python; here the partition, the request records, the native local
process pool (bindings/native_process_pool_door.mojo: one process per device, the
visibility mask narrowed to it, every device count 1) and the join are
native. The wave order is the old pool's: `process_pool_map_typed` sends waves of
`len(devices)` requests and hands wave position j to worker j, so shard i
runs on rank i % len(devices), as `DevicePool.map` did.

The driver negative control is the old `_parallel_pool.driver_read_shift`,
word for word: with MOJOLEARN_PAR_DRIVER_SABOTAGE=1 and
MOJOLEARN_HOST_ALLOW_SABOTAGE=1 in the process environment (read once per
process) or after `_native_parallel_shard('sabotage_arm', (True,))`, every shard on an owner above rank 0 whose
true start is above 0 READS one position early while its block is still
PLACED at the true offset. Inert at one device.

No floating-point value is computed here: copies only.

UNPACK and PACK only: the partition, the device-index admission, the idle
pool rule, the batch checks, the transport negative control's decision and
the block copies of the join are typed, in
bindings/parallel_shard_common_run.mojo. The pool, its requests and the
trace records are typed (the pool record of the process pool door,
`ShardTraceRecord`); requests and replies are typed frames, and a request
is an operation word (`ShardWave`, the codes of
bindings/parallel_shard_common_run.mojo) beside its payload record: no
operation name is sent or looked up.
"""
from std.ffi import external_call
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_attr import py_attr
from bindings.classical_arrays_door import classical_address
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_process_pool_door import (
    process_pool_create, process_pool_map_typed, process_pool_map_replies, process_pool_close, process_pool_closed, process_pool_pids, PROCESS_BOUNDARY,
    process_pool_trace, process_pool_trace_set,
)
from bindings.native_process_wire_door import wire_reply_pack, wire_record_pack
from core.parallel_worker_frame import WireRecord
from bindings.native_worker_inventory_door import native_worker_inventory, native_require_worker_inventory
from bindings.tsa_hw_native_door import tsa_array
from core.shared_slot import SharedSlot
from bindings.parallel_shard_common_run import (
    shard_read_shift,
    shard_faults_take,
    SHARD_FAULT_NONE,
    SHARD_FAULT_DROP_RESULT,
    shard_store_token,
    shard_item,
    SHARD_DEVICE_NOT_AN_INDEX, SHARD_TRACE_RECORDS, SHARD_IDLE_FORGET, SHARD_IDLE_REUSE,
    ShardTraceRecord, shard_trace_find,
    shard_width_admit, shard_bounds, shard_indices_admit, shard_stored_index_admit, shard_idle_action,
    shard_batch_admit, shard_stores_admit, shard_fault_action, shard_rows_admit, shard_block_copy,
    shard_join_block, shard_join_complete,
    shard_worker_op_name, shard_worker_op_numerical, shard_request_words,
    PARALLEL_SHARD_WORKER_OPN_IDENTITY, PARALLEL_SHARD_WORKER_OPN_SHARD_STORE, PARALLEL_SHARD_WORKER_OPN_SHARD_RELEASE,
)
from bindings.parallel_keys_unpack import parallel_keys
from bindings.py_balanced import py_item


def shard_b() raises -> PythonObject:
    return native_loaded_module("builtins")


struct ShardWave(Movable):
    """The requests of one `shard_run` / `shard_map` call, typed: request i
    is the operation code `ops[i]`, its wire words `words[i]` (one encoded
    `WorkerRequest`) and its payload record `payloads[i]`, the tuple
    `(state, args)` the worker reads (None for a request without operands).
    The drivers build one with `shard_wave()` and `add`; the operation never
    travels or is compared as a name."""
    var ops: List[Int]
    var words: List[List[Int64]]
    var payloads: PythonObject

    def __init__(out self, payloads: PythonObject):
        self.ops = List[Int]()
        self.words = List[List[Int64]]()
        self.payloads = payloads

    def add(mut self, op: Int, state: PythonObject, args: PythonObject) raises:
        """One request: `op` (a worker operation code) on `(state, args)`."""
        self.ops.append(op)
        self.words.append(shard_request_words(op))
        _ = py_call(self.payloads.append, Python.tuple(state, args))

    def add_plain(mut self, op: Int) raises:
        """One request without operands (the identity round)."""
        self.ops.append(op)
        self.words.append(shard_request_words(op))
        _ = py_call(self.payloads.append, PythonObject(None))

    def count(self) -> Int:
        return len(self.ops)


def shard_wave() raises -> ShardWave:
    """An empty request wave (its payload list is an exact list)."""
    return ShardWave(shard_b().list())


def shard_ranges(count: Int, width: PythonObject, name: String, empty_message: String) raises -> List[Int]:
    """The shard bounds [lo0, hi0, lo1, hi1, ...] of `range(0, count, width)`
    with `min(start + width, count)` ends; `width` must be a positive int
    (`name` names it in the refusal)."""
    var exact = py_type(width) is shard_b().int
    shard_width_admit(exact, Int(py=width) if exact else 0, name)
    return shard_bounds(count, Int(py=width), empty_message)


def shard_devices(devices: PythonObject) raises -> Int:
    """`DevicePool`'s admission of the device list, in its words; the number
    of devices."""
    var b = shard_b()
    if py_type(devices) is not b.tuple and py_type(devices) is not b.list and py_type(devices) is not b.range:
        raise Error("TypeError: devices requires a native tuple, list or range; Python iteration hooks are unsupported")
    var selected = py_call(b.tuple, devices)
    var indices = List[Int](capacity=len(selected))
    for i in range(len(selected)):  # driver-loop: over the admitted device indices
        indices.append(Int(py=selected[i]) if py_type(selected[i]) is b.int else SHARD_DEVICE_NOT_AN_INDEX)
    shard_indices_admit(indices)
    return len(selected)

struct ShardTrace(Defaultable, Movable):
    # The fault counter is typed RUN state (bindings/parallel_shard_common_run.mojo).
    # The records are typed too (lane pools-frame, 2026-11-21: they were a
    # Python list of dictionaries, one new dictionary per served request).
    # `inventories[i]` holds the identity replies of record i's workers as
    # the typed records their frames carried (it was a pinned Python list of
    # the PACKed replies); `shard_trace_take` PACKs them for the receipt.
    var records: List[ShardTraceRecord]
    var inventories: List[List[WireRecord]]
    var next_id: Int

    def __init__(out self):
        self.records = List[ShardTraceRecord]()
        self.inventories = List[List[WireRecord]]()
        self.next_id = 1


#: Every pool the sharded drivers started in this process since the last
#: `shard_trace_take`: its devices and its workers' own driver inventories
#: (the record the verifier's two-device witness reads,
#: python/mojolearn/_verify_par.py). Bounded: the oldest records go first.
comptime SHARD_TRACE = SharedSlot[ShardTrace, "MojolearnParallelShardTraceV5", 5]


struct ShardIdlePool(Movable):
    """One started pool between two `shard_run` calls, or the pool a call
    holds: its key (child module, device indices), the process that owns its
    workers and two boundary pins, the pool and the parent environment
    mapping snapshot it was spawned under (compared whole, by one mapping
    comparison, when the pool is taken again)."""
    var child: String
    var devices: List[Int]
    var pid: Int
    var pool: PythonObject
    var environ: PythonObject

    def __init__(out self, var child: String, var devices: List[Int], pid: Int, pool: PythonObject,
                 environ: PythonObject):
        self.child = child^
        self.devices = devices^
        self.pid = pid
        self.pool = pool
        self.environ = environ

    def keyed(self, child: String, devices: List[Int]) -> Bool:
        if self.child != child or len(self.devices) != len(devices):
            return False
        for i in range(len(devices)):  # small-loop(a pool's device indices)
            if self.devices[i] != devices[i]:
                return False
        return True


struct ShardIdle(Defaultable, Movable):
    # The idle pools, typed (lane pools-frame, 2026-11-21: it was a Python
    # dict {(child module, devices tuple): [pool, environment snapshot, pid]},
    # a key tuple built and hashed per call). At most one entry per key; a
    # process holds a few.
    var pools: List[ShardIdlePool]
    # A WORKER's stored call operands (`shard_store`), typed (lane pools-frame,
    # 2026-11-21: it was a Python dict {token: payload tuple}). One entry per
    # call in flight on this worker, so the token scan is over one entry in
    # practice. `stored_payloads[i]` is the boundary pin of the payload tuple
    # as its frame was PACKed; `stored_models[i]` is the model built from the
    # call's stored state at its first request (None until then), dropped
    # with the store at the call's release.
    var stored_tokens: List[String]
    var stored_payloads: List[PythonObject]
    var stored_models: List[PythonObject]

    def __init__(out self):
        self.pools = List[ShardIdlePool]()
        self.stored_tokens = List[String]()
        self.stored_payloads = List[PythonObject]()
        self.stored_models = List[PythonObject]()

    def take(self, child: String, devices: List[Int]) -> Int:
        """The idle entry of this key, or -1."""
        for i in range(len(self.pools)):  # small-loop(the idle pools of this process: a few)
            if self.pools[i].keyed(child, devices):
                return i
        return -1

    def find(self, token: String) -> Int:
        """The store entry of `token`, or -1."""
        for i in range(len(self.stored_tokens)):  # small-loop(the calls in flight on this worker: one)
            if self.stored_tokens[i] == token:
                return i
        return -1


def shard_store_put(token: PythonObject, payload: PythonObject) raises -> PythonObject:
    """A worker's `shard_store`: keep `payload` under `token` for the call."""
    var idle = SHARD_IDLE.get_or_create_ptr()
    if py_type(token) is not shard_b().str:
        raise Error("parallel shards: a store token is a string")
    if py_type(payload) is not shard_b().tuple:
        raise Error("parallel shards: a stored payload is a tuple")
    var name = String(token)
    var at = idle[].find(name)
    if at >= 0:
        idle[].stored_payloads[at] = payload
        idle[].stored_models[at] = PythonObject(None)
    else:
        idle[].stored_tokens.append(name^)
        idle[].stored_payloads.append(payload)
        idle[].stored_models.append(PythonObject(None))
    return PythonObject(True)


def shard_store_drop(token: PythonObject) raises -> PythonObject:
    """A worker's `shard_release`: drop the call's stored operands and the
    model built from them."""
    var idle = SHARD_IDLE.get_or_create_ptr()
    if py_type(token) is shard_b().str:
        var at = idle[].find(String(token))
        if at >= 0:
            _ = idle[].stored_tokens.pop(at)
            _ = idle[].stored_payloads.pop(at)
            _ = idle[].stored_models.pop(at)
    return PythonObject(True)


def _stored_entry(reference: PythonObject) raises -> Int:
    """The store entry a `shard_stored_ref` names; refused when this worker
    holds none."""
    var at = SHARD_IDLE.get_or_create_ptr()[].find(String(reference[1]))
    if at < 0:
        raise Error("parallel shards: no stored operands for this call on this worker")
    return at


def shard_resolve_stored(value: PythonObject) raises -> PythonObject:
    """A request's state or argument tuple with every top-level
    `shard_stored_ref` replaced by the stored operand it names; any other
    value is returned as is. One pass: each operand is tested once (it was
    tested in a first pass and again in the second), and the result tuple is
    built only from the first reference on."""
    var b = shard_b()
    if py_type(value) is not b.tuple:
        return value
    var out = PythonObject(None)
    for i in range(len(value)):  # small-loop(a request tuple's few operands)
        var item = value[i]
        if _is_stored_ref(item):
            if out is None:
                out = b.list()
                for j in range(i):  # small-loop(the operands before the first reference)
                    _ = py_call(out.append, value[j])
            var payload = SHARD_IDLE.get_or_create_ptr()[].stored_payloads[_stored_entry(item)]
            var k = Int(py=item[2])
            shard_stored_index_admit(k, len(payload))
            _ = py_call(out.append, payload[k])
        elif out is not None:
            _ = py_call(out.append, item)
    if out is None:
        return value
    return py_call(b.tuple, out)


def shard_stored_model(cls: PythonObject, state: PythonObject) raises -> PythonObject:
    """A worker's model of `cls` for one request's `state`.

    `state` is either the state record itself (a call with at most one
    request per worker sends it with that request) or a one-operand tuple
    holding a `shard_stored_ref` (a call with more requests than workers
    stores the record once per worker). For a stored record the model is
    built at the call's first request on this worker and held beside the
    store until the call's release: one model per call per worker, where
    every request used to carry the whole record and rebuild the model."""
    var b = shard_b()
    if py_type(state) is not b.tuple:
        return shard_model(cls, state)
    if len(state) != 1 or not _is_stored_ref(state[0]):
        raise Error("parallel shards: a stored model state is one stored operand")
    var reference = state[0]
    var at = _stored_entry(reference)
    var idle = SHARD_IDLE.get_or_create_ptr()
    if idle[].stored_models[at] is None:
        var payload = idle[].stored_payloads[at]
        var k = Int(py=reference[2])
        shard_stored_index_admit(k, len(payload))
        var built = shard_model(cls, payload[k])
        # The slot is read again: building a model can run a collection.
        idle = SHARD_IDLE.get_or_create_ptr()
        idle[].stored_models[at] = built
    return idle[].stored_models[at]


def _is_stored_ref(item: PythonObject) raises -> Bool:
    var b = shard_b()
    if py_type(item) is not b.tuple or len(item) != 3:
        return False
    var mark = item[0]
    return py_type(mark) is b.str and String(mark) == SHARD_STORED_MARK


#: Lane perf-trees-misc (2026-11-05): the started one-shot shard pools kept
#: between `shard_run` calls. Each call used to spawn a fresh worker per
#: device (interpreter start, package import, binding connection: about 5 s
#: per open on Linux, `~/mojolearn-evidence/svl-partA.txt`), so a
#: par-forecast fit, predict and forecast paid it three times per cell. An
#: idle pool is reused only for the same child module, the same device tuple
#: and an unchanged parent environment (the worker environment, visibility
#: mask included, is derived from it at spawn); every reuse re-admits the
#: workers by identity and starts its own trace record, as a fresh pool
#: does. A failed call closes its pool and nothing is kept. Workers held
#: here end when the parent process ends (their socket closes).
#: V3: the worker store is typed (tokens, payload pins, models).
comptime SHARD_IDLE = SharedSlot[ShardIdle, "MojolearnParallelShardIdleV4", 4]


def shard_identity() raises -> PythonObject:
    """A worker's answer to an identity request: its own process and the
    devices its visibility mask leaves it (bindings/native_worker_inventory_door.mojo)."""
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    return native_worker_inventory(p[].vendor, p[].environ)


def shard_open(devices: PythonObject, child: String) raises -> PythonObject:
    """A started pool of one worker per device whose placement is admitted:
    every worker answers an identity request first, and the replies must be
    distinct processes started here and, on CUDA and HIP, distinct physical
    GPUs, one per worker (`native_require_worker_inventory`, the CV pool's
    rule). The pool and its inventory are recorded in the trace."""
    var pool = process_pool_create(devices, False, child)
    return _shard_admit(pool, devices)


def _shard_admit(pool: PythonObject, devices: PythonObject) raises -> PythonObject:
    """`shard_open`'s admission of a pool's workers (a fresh pool, or an idle
    one `shard_run` reuses): the identity round, the inventory check and a
    new trace record. The pool is closed on a refusal."""
    var b = shard_b()
    var nk = parallel_keys()
    var asks = shard_wave()
    for i in range(len(devices)):  # driver-loop: one identity request per worker
        asks.add_plain(PARALLEL_SHARD_WORKER_OPN_IDENTITY)
    # The identity replies stay typed for the trace; their PACK here is the
    # inventory check's only.
    var replies = process_pool_map_replies(pool, asks.words, asks.payloads)
    var identities = b.list()
    var kept = List[WireRecord](capacity=len(replies))
    for i in range(len(replies)):  # driver-loop: over the workers' identity replies
        _ = py_call(identities.append, wire_reply_pack(replies[i]))
        # The record changes owner (no copy); the reply keeps an empty one.
        var record = WireRecord()
        swap(record, replies[i].record)
        kept.append(record^)
    var pids = b.list()
    var pid_words = process_pool_pids(pool)
    for i in range(len(pid_words)):  # driver-loop: over the spawned workers
        _ = py_call(pids.append, PythonObject(pid_words[i]))
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    try:
        native_require_worker_inventory(identities, p[].vendor, pids, Int(external_call["getpid", Int32]()))
    except error:
        process_pool_close(pool)
        raise error^
    var t = SHARD_TRACE.get_or_create_ptr()
    var device_words = List[Int]()
    for i in range(len(devices)):  # small-loop(the pool's admitted devices)
        device_words.append(Int(py=devices[i]))
    var id = t[].next_id
    t[].next_id = id + 1
    t[].records.append(ShardTraceRecord(id, device_words^))
    t[].inventories.append(kept^)
    process_pool_trace_set(pool, id)
    if len(t[].records) > SHARD_TRACE_RECORDS:
        _ = t[].records.pop(0)
        _ = t[].inventories.pop(0)
    return pool


def shard_trace_take() raises -> PythonObject:
    """The pools recorded since the last take, and an empty trace. PACK: one
    mapping per pool (`devices`, `inventory`, `calls` of `pid` and
    `operation`), the form the verifier's receipt reads."""
    var b = shard_b()
    var nk = parallel_keys()
    var t = SHARD_TRACE.get_or_create_ptr()
    var out = b.list()
    for r in range(len(t[].records)):  # driver-loop: over the recorded pools
        var record = b.dict()
        var devices = b.list()
        for i in range(len(t[].records[r].devices)):  # small-loop(one pool's devices)
            _ = py_call(devices.append, PythonObject(t[].records[r].devices[i]))
        var calls = b.list()
        for i in range(len(t[].records[r].pids)):  # driver-loop: over the requests one pool served
            var call = b.dict()
            py_setitem(call, nk[].k_pid, PythonObject(t[].records[r].pids[i]))
            py_setitem(call, nk[].k_operation, PythonObject(shard_worker_op_name(t[].records[r].operations[i])))
            _ = py_call(calls.append, call)
        py_setitem(record, nk[].k_devices, devices)
        var inventory = b.list()
        for i in range(len(t[].inventories[r])):  # driver-loop: over one pool's workers (PACK of their identity records)
            _ = py_call(inventory.append, wire_record_pack(t[].inventories[r][i]))
        py_setitem(record, nk[].k_inventory, inventory)
        py_setitem(record, nk[].k_calls, calls)
        _ = py_call(out.append, record)
    t[].records.clear()
    t[].inventories.clear()
    return out

def shard_map(pool: PythonObject, wave: ShardWave) raises -> PythonObject:
    """`process_pool_map_typed` of the wave's requests, with every request's
    worker and operation code added to the pool's trace record.

    THE TRANSPORT NEGATIVE CONTROL (python/mojolearn/_verify_distributed.py):
    armed by `_native_parallel_shard('faults_arm', ('drop_result',))` or
    ('reverse_results',) (`shard_faults_arm`), the first numerical batch of
    more than one result in this process is returned without its last result
    or in reverse order, once. The driver's own checks or the hash must catch
    it. Before (lane envgate-classical-deviceplan): MOJOLEARN_PAR_TRANSPORT_FAULT
    with MOJOLEARN_HOST_ALLOW_SABOTAGE=1, two getenv per numerical batch; now
    one slot read, no environment."""
    var b = shard_b()
    var results = process_pool_map_typed(pool, wave.words, wave.payloads)
    var requests = wave.count()
    var nk = parallel_keys()
    # The pool's trace word is in its typed record (it was a `trace_record`
    # entry of the pool dictionary: a dictionary read, a type test and an
    # int conversion per map).
    var traced = process_pool_trace(pool)
    var workers = process_pool_pids(pool)
    if traced != 0 and len(workers) > 0:
        var t = SHARD_TRACE.get_or_create_ptr()
        var at = shard_trace_find(t[].records, traced)
        if at >= 0:
            # Two words per request into the typed record (it was a new
            # dictionary per request, then a name String per request).
            for i in range(requests):  # driver-loop: one trace entry per request
                t[].records[at].pids.append(workers[i % len(workers)])
                t[].records[at].operations.append(wave.ops[i])
    var fault = shard_fault_action(len(results), requests,
                                   len(results) > 1 and requests > 0 and shard_worker_op_numerical(wave.ops[0]))
    if fault != SHARD_FAULT_NONE:
        var altered = py_call(b.list, results)
        if fault == SHARD_FAULT_DROP_RESULT:
            _ = py_call(altered.pop)
        else:
            _ = py_call(altered.reverse)
        return altered
    return results


def _shard_acquire(devices: PythonObject, child: String) raises -> ShardIdlePool:
    """`shard_run`'s pool: the idle pool kept for this child, devices and
    environment, re-admitted, or a fresh one. The returned entry is the
    call's own; `_shard_keep` puts it back."""
    var b = shard_b()
    var selected = py_call(b.tuple, devices)
    var indices = List[Int](capacity=len(selected))
    for i in range(len(selected)):  # small-loop(the call's admitted device indices)
        indices.append(Int(py=selected[i]))
    var environ = py_call(b.dict, PROCESS_BOUNDARY.get_or_create_ptr()[].environ)
    var pid = Int(external_call["getpid", Int32]())
    var reused = PythonObject(None)
    var idle = SHARD_IDLE.get_or_create_ptr()
    var at = idle[].take(child, indices)
    if at >= 0:
        # The entry leaves the idle list before any pool call.
        var held = idle[].pools.pop(at)
        var owned = held.pid == pid
        var action = shard_idle_action(held.pid, pid, owned and Bool(py=held.environ == environ),
                                       owned and process_pool_closed(held.pool),
                                       len(process_pool_pids(held.pool)) if owned else 0, len(indices))
        if action == SHARD_IDLE_FORGET:
            # Inherited across os.fork: the pool's workers and sockets belong
            # to the parent process. Drop the record without closing (a close
            # here would end the parent's workers on the shared sockets).
            pass
        elif action == SHARD_IDLE_REUSE:
            try:
                reused = _shard_admit(held.pool, devices)
            except:
                # a worker that died while idle: the identity round closed
                # the pool; a fresh one is opened below
                reused = PythonObject(None)
        else:
            try:
                process_pool_close(held.pool)
            except:
                pass
    if reused is None:
        reused = shard_open(devices, child)
    return ShardIdlePool(child.copy(), indices^, pid, reused, environ)


def _shard_keep(var acquired: ShardIdlePool) raises:
    """Keep a pool whose call succeeded idle for the next call (`SHARD_IDLE`)."""
    var idle = SHARD_IDLE.get_or_create_ptr()
    var at = idle[].take(acquired.child, acquired.devices)
    if at >= 0:
        var other = idle[].pools.pop(at)
        if other.pid == acquired.pid:  # another thread kept a pool under this key meanwhile
            try:
                process_pool_close(other.pool)
            except:
                pass
        idle = SHARD_IDLE.get_or_create_ptr()
    idle[].pools.append(acquired^)


def shard_run(requests: ShardWave, devices: PythonObject, child: String) raises -> PythonObject:
    """The workers' results in request order (waves of len(devices), wave
    position j on worker j). A pool whose call succeeded is kept idle for the
    next call with the same child, devices and environment (`SHARD_IDLE`);
    a failed call closes it."""
    var acquired = _shard_acquire(devices, child)
    var reused = acquired.pool
    var results = PythonObject(None)
    try:
        results = shard_map(reused, requests)
    except error:
        process_pool_close(reused)
        raise error^
    if len(results) != requests.count():
        process_pool_close(reused)
    shard_batch_admit(len(results), requests.count())
    _shard_keep(acquired^)
    return results


#: The marker of a stored operand in a shard request (`shard_stored_ref`).
comptime SHARD_STORED_MARK = "mojolearn.shard-stored.v1"


def shard_stored_ref(token: String, index: Int) raises -> PythonObject:
    """The placeholder a request carries for operand `index` of the payload
    `shard_run_stored` stored under `token` on every worker."""
    return Python.tuple(PythonObject(SHARD_STORED_MARK), PythonObject(token), PythonObject(index))

def shard_run_stored(payload: PythonObject, token: String, requests: ShardWave, devices: PythonObject,
                     child: String) raises -> PythonObject:
    """`shard_run` with the call's shared operands sent ONCE per worker
    (lane ne2-i-j, 2026-11-07, plan row N452). `payload` (a tuple: X, labels,
    ...) is stored under `token` on every worker of the one pool this call
    holds (one `shard_store` request per worker, wave position j on worker
    j); each request names its operands by `shard_stored_ref(token, i)`,
    which the worker resolves from its store (`shard_resolve_stored`); one
    `shard_release` per worker drops the store before the pool goes idle.
    The shards, their order and their workers are `shard_run`'s, so the
    results are the same words; only the bytes on the sockets change (X once
    per worker, not once per shard). A failed call closes the pool, which
    ends its workers and their stores."""
    var b = shard_b()
    var acquired = _shard_acquire(devices, child)
    var pool = acquired.pool
    var nk = parallel_keys()
    var workers = len(process_pool_pids(pool))
    var results = PythonObject(None)
    try:
        var stores = shard_wave()
        var releases = shard_wave()
        var held = PythonObject(token)
        for i in range(workers):  # driver-loop: one store and one release per worker
            stores.add(PARALLEL_SHARD_WORKER_OPN_SHARD_STORE, held, payload)
            releases.add(PARALLEL_SHARD_WORKER_OPN_SHARD_RELEASE, held, PythonObject(None))
        var stored = process_pool_map_typed(pool, stores.words, stores.payloads)
        shard_stores_admit(len(stored), workers)
        results = shard_map(pool, requests)
        shard_batch_admit(len(results), requests.count())
        _ = process_pool_map_typed(pool, releases.words, releases.payloads)
    except error:
        process_pool_close(pool)
        raise error^
    _shard_keep(acquired^)
    return results

def shard_dtype(value: PythonObject) raises -> String:
    """The typestr ("<f4", "<i8", ...) of a mojolearn Array or an ndarray.
    An Array's `dtype` is its typestr; an ndarray's `dtype` prints its name
    ("float32"), so its typestr is `dtype.str`. A driver input admitted
    in place (already C-ordered float32) is the caller's ndarray."""
    var module = py_get(native_loaded_module("sys").modules, "mojolearn._array")
    if module is not None and py_type(value) is module.Array:
        return String(py=value.dtype)
    return String(py=value.dtype.str)


def shard_array(dtype: String, shape: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    """A zero-filled C-ordered Array of `dtype` and `shape` (`count` items)."""
    var item = shard_item(dtype)
    return tsa_array(item[0], dtype, shape, count, item[1], context)


def shard_rows(source: PythonObject, lo: Int, hi: Int, context: PythonObject) raises -> PythonObject:
    """A copy of rows [lo, hi) of a C-ordered Array (any rank >= 1), with the
    source's trailing shape."""
    var b = shard_b()
    var shape = py_call(b.list, source.shape)
    var rows = Int(py=shape[0])
    shard_rows_admit(lo, hi, rows)
    var inner = 1
    for k in range(1, len(shape)):  # small-loop(the source's trailing dimensions)
        inner *= Int(py=shape[k])
    py_setitem(shape, 0, hi - lo)
    var dtype = shard_dtype(source)
    var item = shard_item(dtype)
    var out = shard_array(dtype, py_call(b.tuple, shape), (hi - lo) * inner, context)
    var bytes = (hi - lo) * inner * item[1]
    if bytes > 0:
        shard_block_copy(Int(py=classical_address(out)), Int(py=classical_address(source)) + lo * inner * item[1], bytes)
        _ = out  # keep the Python object alive through the native call above
        _ = source  # keep the Python object alive through the native call above
    return out


def shard_row_vector(source: PythonObject, row: Int, context: PythonObject) raises -> PythonObject:
    """Row `row` of a C-ordered 2-D Array, as a 1-D copy."""
    var b = shard_b()
    var width = Int(py=source.shape[1])
    var dtype = shard_dtype(source)
    var item = shard_item(dtype)
    var out = shard_array(dtype, Python.tuple(width), width, context)
    if width > 0:
        shard_block_copy(Int(py=classical_address(out)), Int(py=classical_address(source)) + row * width * item[1],
                         width * item[1])
        _ = out  # keep the Python object alive through the native call above
        _ = source  # keep the Python object alive through the native call above
    _ = b
    return out


def shard_concat(parts: PythonObject, name: String, total: Int, context: PythonObject) raises -> PythonObject:
    """The arrays `parts[k][name]` joined along their first axis in shard
    order (`shard_join_rows`)."""
    var b = shard_b()
    var arrays = b.list()
    for p in range(len(parts)):  # driver-loop: one array per device shard
        _ = py_call(arrays.append, parts[p][name])
    return shard_join_rows(arrays, total, context)


def shard_join_rows(arrays: PythonObject, total: Int, context: PythonObject) raises -> PythonObject:
    """The C-ordered arrays joined along their first axis in shard order (a
    byte copy per shard): the first array's dtype and trailing shape, `total`
    leading rows. Every array must share them."""
    var b = shard_b()
    var first = arrays[0]
    var shape = py_call(b.list, first.shape)
    var inner = 1
    for k in range(1, len(shape)):  # small-loop(the trailing dimensions)
        inner *= Int(py=shape[k])
    var dtype = shard_dtype(first)
    var item = shard_item(dtype)
    py_setitem(shape, 0, total)
    var out = shard_array(dtype, py_call(b.tuple, shape), total * inner, context)
    var cursor = 0
    for p in range(len(arrays)):  # driver-loop: one block copy per device shard
        var value = arrays[p]
        var same = shard_dtype(value) == dtype
        var rows = Int(py=value.shape[0]) if same else 0
        shard_join_block(same, cursor, rows, total)
        var bytes = rows * inner * item[1]
        if bytes > 0:
            shard_block_copy(Int(py=classical_address(out)) + cursor * inner * item[1], Int(py=classical_address(value)), bytes)
            _ = out  # keep the Python object alive through the native call above
            _ = value  # keep the Python object alive through the native call above
        cursor += rows
    shard_join_complete(cursor, total)
    return out


def shard_state(fields: PythonObject, drop: PythonObject, context: PythonObject) raises -> PythonObject:
    """A worker state record: the instance fields whose values the native
    codec carries (None, bool, int, float, str, native Arrays and tuples,
    lists and str-keyed dicts of them), without the names in `drop`.
    Device handles and other objects stay in this process."""
    var b = shard_b()
    var state = b.dict()
    var keys = py_call(b.list, fields)
    for k in range(len(keys)):  # driver-loop: over the estimator's instance fields
        var key = keys[k]
        if py_type(key) is not b.str or Bool(py=py_call(py_attr(drop, "__contains__"), key)):
            continue
        var value = fields[key]
        if _carried(value, context, 0):
            py_setitem(state, key, value)
    return state


def _carried(value: PythonObject, context: PythonObject, depth: Int) raises -> Bool:
    var b = shard_b()
    if depth > 8:
        return False
    var cls = py_type(value)
    if value is None or cls is b.bool or cls is b.int or cls is b.float or cls is b.str:
        return True
    var nk = parallel_keys()
    if cls is py_item(context, nk[].k_array_class):
        return True
    if cls is b.tuple or cls is b.list:
        for i in range(len(value)):  # driver-loop: over a structural field's entries
            if not _carried(value[i], context, depth + 1):
                return False
        return True
    if cls is b.dict:
        var keys = py_call(b.list, value)
        for i in range(len(keys)):  # driver-loop: over a structural field's entries
            if py_type(keys[i]) is not b.str or not _carried(value[keys[i]], context, depth + 1):
                return False
        return True
    return False


def shard_model(cls: PythonObject, state: PythonObject) raises -> PythonObject:
    """A model of `cls` whose instance dictionary is `state`, built without a
    Python constructor (the old worker's `cls(**params)` carried the same
    fields)."""
    var b = shard_b()
    var model = py_call(b.object.__new__, cls)
    _ = py_call(py_fields(model).update, state)
    return model
