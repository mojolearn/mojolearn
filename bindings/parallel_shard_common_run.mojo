# SPDX-License-Identifier: Apache-2.0
"""Typed parts of the parallel shard driver (RUN): the driver read shift, the numerical
operation names, the dtype item table, and the process counters (injected transport faults and
store tokens) and the trace records (`ShardTraceRecord`). The pool is a typed record of the process pool door."""
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.os import getenv
from core.shared_slot import SharedSlot
from core.parallel_worker_request import WorkerRequest, worker_request_decode, WORKER_NO_HANDLE


#: `ShardCounters.transport_fault` values: the transport negative control
#: (`shard_map`) armed by `shard_faults_arm`.
comptime SHARD_FAULT_NONE = 0
comptime SHARD_FAULT_DROP_RESULT = 1
comptime SHARD_FAULT_REVERSE_RESULTS = 2


struct ShardCounters(Defaultable, Movable):
    #: transport faults the verifier's switch injected since the last take
    var faults: Int
    #: `shard_run_stored` calls made by this process (its store tokens)
    var stored_calls: Int
    #: THE NEGATIVE CONTROLS' SWITCHES (lane envgate-classical-deviceplan,
    #: docs/plans/ENV_GATES_2026-11-15.md). Before, `shard_read_shift` read
    #: MOJOLEARN_PAR_DRIVER_SABOTAGE and MOJOLEARN_HOST_ALLOW_SABOTAGE with
    #: two getenv per shard, and `shard_map` read MOJOLEARN_HOST_ALLOW_SABOTAGE
    #: and MOJOLEARN_PAR_TRANSPORT_FAULT per numerical batch.
    #: `transport_fault`: SHARD_FAULT_*, set only by `shard_faults_arm` (the
    #: verifier's `transport_fault` and the tests call it through
    #: `_native_parallel_shard('faults_arm', (fault,))`); no environment.
    var transport_fault: Int
    #: `driver_sabotage`: -1 until first read, then 0 or 1. Seeded once per
    #: process from MOJOLEARN_PAR_DRIVER_SABOTAGE=1 and
    #: MOJOLEARN_HOST_ALLOW_SABOTAGE=1 (tools/identity_break.py sets both in
    #: the environment of the process it starts); `shard_sabotage_arm` sets
    #: it inside a running process.
    var driver_sabotage: Int

    def __init__(out self):
        self.faults = 0
        self.stored_calls = 0
        self.transport_fault = SHARD_FAULT_NONE
        self.driver_sabotage = -1


#: V2: the two switch words (lane envgate-classical-deviceplan).
comptime SHARD_COUNTERS = SharedSlot[ShardCounters, "MojolearnParallelShardCountersV2", 2]


def shard_faults_arm(fault: String) raises:
    """Arm the transport negative control of `shard_map` for this process:
    "drop_result" or "reverse_results" arms it, "" disarms it. The first
    numerical batch of more than one result after arming is altered once
    (`shard_fault_mark`); `shard_faults_take` reports it."""
    var c = SHARD_COUNTERS.get_or_create_ptr()
    if fault == "":
        c[].transport_fault = SHARD_FAULT_NONE
    elif fault == "drop_result":
        c[].transport_fault = SHARD_FAULT_DROP_RESULT
    elif fault == "reverse_results":
        c[].transport_fault = SHARD_FAULT_REVERSE_RESULTS
    else:
        raise Error("ValueError: unknown transport fault " + fault)


def shard_transport_fault() raises -> Int:
    """The armed transport fault (SHARD_FAULT_*): one slot read, no
    environment."""
    return SHARD_COUNTERS.get_or_create_ptr()[].transport_fault


def shard_sabotage_arm(on: Bool) raises:
    """Set the driver read-shift negative control for this process (the
    in-process form of MOJOLEARN_PAR_DRIVER_SABOTAGE=1 with
    MOJOLEARN_HOST_ALLOW_SABOTAGE=1)."""
    SHARD_COUNTERS.get_or_create_ptr()[].driver_sabotage = 1 if on else 0


def _driver_sabotage() -> Bool:
    """The driver read-shift switch: the environment read once per process,
    unless `shard_sabotage_arm` set it."""
    try:
        var c = SHARD_COUNTERS.get_or_create_ptr()
        if c[].driver_sabotage < 0:
            var on = (String(getenv("MOJOLEARN_PAR_DRIVER_SABOTAGE")) == "1"
                      and String(getenv("MOJOLEARN_HOST_ALLOW_SABOTAGE")) == "1")
            c[].driver_sabotage = 1 if on else 0
        return c[].driver_sabotage == 1
    except:
        return False


def shard_fault_mark() raises -> Bool:
    """Record the one injected transport fault of a take window: True when
    none was recorded yet (the caller injects it), else False."""
    var c = SHARD_COUNTERS.get_or_create_ptr()
    if c[].faults != 0:
        return False
    c[].faults = 1
    return True


def shard_read_shift(index: Int, first: Int, n_devices: Int) -> Int:
    """`_parallel_pool.driver_read_shift(index, first, devices)`: 1 for a
    shard on an owner above rank 0 (rank = index % n_devices) whose true
    start is above 0 when the driver sabotage switch is on
    (`_driver_sabotage`: the environment once per process, or
    `shard_sabotage_arm`), else 0. Per shard: one slot read, no getenv."""
    if first < 1 or n_devices < 2 or index % n_devices == 0:
        return 0
    if _driver_sabotage():
        return 1
    return 0


def shard_faults_take() raises -> Int:
    """How many transport faults the verifier's switch injected since the
    last take (`shard_map`), and zero."""
    var c = SHARD_COUNTERS.get_or_create_ptr()
    var n = c[].faults
    c[].faults = 0
    return n


def shard_store_token() raises -> String:
    """A token unique to this process and call (pid plus a counter)."""
    var c = SHARD_COUNTERS.get_or_create_ptr()
    c[].stored_calls += 1
    return String("ss-") + String(Int(external_call["getpid", Int32]())) + "-" + String(c[].stored_calls)


def shard_item(dtype: String) raises -> Tuple[String, Int]:
    """(array type code, item size) of a C-ordered native dtype string."""
    if dtype == "<f4":
        return (String("f"), 4)
    if dtype == "<f8":
        return (String("d"), 8)
    if dtype == "<i4":
        return (String("i"), 4)
    if dtype == "<i8":
        return (String("q"), 8)
    if dtype == "<u4":
        return (String("I"), 4)
    if dtype == "|u1" or dtype == "<u1":
        return (String("B"), 1)
    raise Error("ValueError: shard arrays require float32, float64, int32, int64, uint32 or uint8, got " + dtype)


# ------------------------------------------------- typed partition and join
# (lane dw-parallel-arena, 2026-11-21) The shard bounds, the device-index
# admission, the idle-pool rule, the batch checks, the transport negative
# control's decision and the block copies of the join, over typed values.
# `bindings/parallel_shard_common_door.mojo` unpacks into these and packs.

#: UNPACK hands a device index that is not an exact int as this value: it is
#: refused in the words of a negative index.
comptime SHARD_DEVICE_NOT_AN_INDEX = -1
#: The trace keeps this many pool records; the oldest go first.
comptime SHARD_TRACE_RECORDS = 256


struct ShardTraceRecord(Copyable, Movable):
    """One started pool in the trace: its id (the pool dictionary's
    `trace_record` word), its devices and, per request it served, the
    worker's process id and the operation's code (`shard_worker_op_name`
    gives its name at the take)."""
    var id: Int
    var devices: List[Int]
    var pids: List[Int]
    var operations: List[Int]

    def __init__(out self, id: Int, var devices: List[Int]):
        self.id = id
        self.devices = devices^
        self.pids = List[Int]()
        self.operations = List[Int]()


def shard_trace_find(records: List[ShardTraceRecord], id: Int) -> Int:
    """The position of the record with `id`, or -1 when the trace no longer
    holds it (taken, or pushed out by newer pools). Newest first: a pool
    in use is almost always among the last records."""
    for back in range(len(records)):  # small-loop(at most SHARD_TRACE_RECORDS pool records, once per request batch)
        var at = len(records) - 1 - back
        if records[at].id == id:
            return at
    return -1

# What `shard_run` does with the idle pool it found (`shard_idle_action`).
comptime SHARD_IDLE_FORGET = 0   # inherited across fork: drop the record, close nothing
comptime SHARD_IDLE_REUSE = 1    # re-admit its workers and use it
comptime SHARD_IDLE_CLOSE = 2    # close it; a fresh pool is opened


def shard_width_admit(exact: Bool, width: Int, name: String) raises:
    """`exact`: the width is an exact int."""
    if not exact or width < 1:
        raise Error("ValueError: " + name + " must be a positive integer")


def shard_bounds(count: Int, width: Int, empty_message: String) raises -> List[Int]:
    """The shard bounds [lo0, hi0, lo1, hi1, ...] of `range(0, count, width)`
    with `min(start + width, count)` ends, for an admitted `width`."""
    if count < 1 and empty_message != "":
        raise Error("ValueError: " + empty_message)
    var bounds = List[Int]()
    var i = 0
    while i < count:  # driver-loop: over the shard boundaries for the device pool
        bounds.append(i)
        bounds.append(min(i + width, count))
        i += width
    return bounds^


def shard_indices_admit(devices: List[Int]) raises:
    """`DevicePool`'s admission of the device list, in its words: at least
    one index, none negative, none repeated."""
    if len(devices) == 0:
        raise Error("ValueError: devices must be distinct nonnegative integer indices")
    for i in range(len(devices)):  # small-loop(admitted device indices)
        if devices[i] < 0:
            raise Error("ValueError: devices must be distinct nonnegative integer indices")
        for j in range(i):  # small-loop(earlier device indices)
            if devices[j] == devices[i]:
                raise Error("ValueError: devices must be distinct nonnegative integer indices")


def shard_stored_index_admit(index: Int, operands: Int) raises:
    if index < 0 or index >= operands:
        raise Error("parallel shards: a stored operand index is out of range")


def shard_idle_action(owner_pid: Int, pid: Int, same_environment: Bool, closed: Bool, workers: Int,
                      devices: Int) -> Int:
    """An idle pool is reused only by the process that started it, for an
    unchanged parent environment, while open and with one worker per device.
    `same_environment`, `closed` and `workers` are read only for a pool this
    process owns."""
    if owner_pid != pid:
        return SHARD_IDLE_FORGET
    if same_environment and not closed and workers == devices:
        return SHARD_IDLE_REUSE
    return SHARD_IDLE_CLOSE


def shard_batch_admit(results: Int, requests: Int) raises:
    if results != requests:
        raise Error("ValueError: shard workers returned an incomplete batch")


def shard_stores_admit(stored: Int, workers: Int) raises:
    if stored != workers:
        raise Error("ValueError: shard workers did not store the call's operands")


def shard_fault_action(results: Int, requests: Int, numerical: Bool) raises -> Int:
    """The transport fault to inject into this batch (SHARD_FAULT_*): the
    armed one, once per take window, for a numerical batch of more than one
    result. `numerical` is read only for such a batch."""
    if results <= 1 or requests <= 0 or not numerical:
        return SHARD_FAULT_NONE
    var fault = shard_transport_fault()
    if fault != SHARD_FAULT_NONE and shard_fault_mark():
        return fault
    return SHARD_FAULT_NONE


def shard_rows_admit(lo: Int, hi: Int, rows: Int) raises:
    if lo < 0 or hi > rows or hi < lo:
        raise Error("ValueError: shard rows outside the source")


def shard_block_copy(dst: Int, src: Int, bytes: Int):
    """`bytes` bytes from `src` to `dst` (one shard's block, back to back:
    one transfer, as before)."""
    if bytes > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
                      src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=src), count=bytes)


def shard_join_block(same_dtype: Bool, cursor: Int, rows: Int, total: Int) raises:
    """A shard's block of `rows` rows joins at row `cursor` of `total`."""
    if not same_dtype:
        raise Error("ValueError: shard workers returned different dtypes")
    if cursor + rows > total:
        raise Error("ValueError: shard workers returned too many rows")


def shard_join_complete(cursor: Int, total: Int) raises:
    if cursor != total:
        raise Error("ValueError: shard workers returned an incomplete result")


# ------------------------------------------------- the workers' operations
# One code space for every worker a shard pool starts (the shard worker of
# bindings/parallel_shard_native_door.mojo, with the neighbor operations of
# bindings/parallel_neighbor_worker_door.mojo, and the forecast worker of
# bindings/parallel_forecast_native_door.mojo). A request travels as one
# encoded `WorkerRequest` whose `op` is one of these codes
# (`shard_request_words`); its payload tuple `(state, args)` rides beside it.
# The workers dispatch on the decoded word: no operation name is sent, looked
# up or compared. The neighbor operations stay the last shard codes (one
# range test in the shard worker).
comptime PARALLEL_SHARD_WORKER_OPN_IDENTITY = 0
comptime PARALLEL_SHARD_WORKER_OPN_SHARD_STORE = 1
comptime PARALLEL_SHARD_WORKER_OPN_SHARD_RELEASE = 2
comptime PARALLEL_SHARD_WORKER_OPN_RBF_ROWS = 3
comptime PARALLEL_SHARD_WORKER_OPN_SCALER_FIT = 4
comptime PARALLEL_SHARD_WORKER_OPN_SCALER_TRANSFORM = 5
comptime PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_FIT = 6
comptime PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_PREDICT = 7
comptime PARALLEL_SHARD_WORKER_OPN_FOREST_FIT = 8
comptime PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD = 9
comptime PARALLEL_SHARD_WORKER_OPN_REFERENCE_STORE = 10
comptime PARALLEL_SHARD_WORKER_OPN_REFERENCE_VOTE = 11
comptime PARALLEL_SHARD_WORKER_OPN_QUERY_STATE = 12
comptime PARALLEL_SHARD_WORKER_OPN_RELEASE = 13
comptime PARALLEL_SHARD_WORKER_OPN_IVF_STORE = 14
comptime PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH = 15
comptime PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY = 16
comptime PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS = 17
# The forecast worker's operations; its identity request is the shard one.
comptime PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD = 32
comptime PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD = 33
comptime PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD = 34
comptime PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD = 35
# Bookkeeping, not numerical: a worker stores one fitted estimator's predict models (a "generation").
comptime PARALLEL_FORECAST_WORKER_OPN_GENERATION_STORE = 36


def shard_worker_op_name(op: Int) -> String:
    """The operation's name in the trace (`shard_trace_take`): the word the
    request used to carry, read by the verifier's receipt."""
    if op == PARALLEL_SHARD_WORKER_OPN_IDENTITY:
        return "identity"
    if op == PARALLEL_SHARD_WORKER_OPN_SHARD_STORE:
        return "shard_store"
    if op == PARALLEL_SHARD_WORKER_OPN_SHARD_RELEASE:
        return "shard_release"
    if op == PARALLEL_SHARD_WORKER_OPN_RBF_ROWS:
        return "rbf_rows"
    if op == PARALLEL_SHARD_WORKER_OPN_SCALER_FIT:
        return "scaler_fit"
    if op == PARALLEL_SHARD_WORKER_OPN_SCALER_TRANSFORM:
        return "scaler_transform"
    if op == PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_FIT:
        return "gpc_class_fit"
    if op == PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_PREDICT:
        return "gpc_class_predict"
    if op == PARALLEL_SHARD_WORKER_OPN_FOREST_FIT:
        return "forest_fit"
    if op == PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD:
        return "reference_shard"
    if op == PARALLEL_SHARD_WORKER_OPN_REFERENCE_STORE:
        return "reference_store"
    if op == PARALLEL_SHARD_WORKER_OPN_REFERENCE_VOTE:
        return "reference_vote"
    if op == PARALLEL_SHARD_WORKER_OPN_QUERY_STATE:
        return "query_state"
    if op == PARALLEL_SHARD_WORKER_OPN_RELEASE:
        return "release"
    if op == PARALLEL_SHARD_WORKER_OPN_IVF_STORE:
        return "ivf_store"
    if op == PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH:
        return "ivf_search"
    if op == PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY:
        return "reference_query"
    if op == PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS:
        return "query_rows"
    if op == PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD:
        return "arima_predict_shard"
    if op == PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD:
        return "hw_predict_shard"
    if op == PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD:
        return "arima_fit_shard"
    if op == PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD:
        return "hw_fit_shard"
    if op == PARALLEL_FORECAST_WORKER_OPN_GENERATION_STORE:
        return "forecast_generation_store"
    return "unknown"


def shard_worker_op_numerical(op: Int) -> Bool:
    """True for an operation whose result is numerical (the batches the
    transport negative control of `shard_map` may alter): every operation
    but the identity round and the store / release / state bookkeeping."""
    if op >= PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD:
        return op <= PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD
    if op >= PARALLEL_SHARD_WORKER_OPN_RBF_ROWS and op <= PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD:
        return True
    return (op == PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH or op == PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY
            or op == PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS)


def shard_request_words(op: Int) -> List[Int64]:
    """The wire words of one shard request: an encoded `WorkerRequest` with
    the operation code and no handle or words (the operands ride in the
    payload record)."""
    return WorkerRequest(op).encode()


def shard_request_op(address: Int, byte_count: Int) raises -> Int:
    """A worker's decode of its request words (a frame without request words
    is refused here): the operation code."""
    var request = worker_request_decode(address, byte_count)
    if request.handle != WORKER_NO_HANDLE or len(request.words) != 0:
        raise Error("parallel shards: invalid worker request envelope")
    return request.op
