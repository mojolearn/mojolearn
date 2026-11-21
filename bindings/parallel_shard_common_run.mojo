# SPDX-License-Identifier: Apache-2.0
"""Typed parts of the parallel shard driver (RUN): the driver read shift, the numerical
operation names, the dtype item table, and the process counters (injected transport faults and
store tokens) and the trace records (`ShardTraceRecord`). The pool is a typed record of the process pool door."""
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.os import getenv
from core.shared_slot import SharedSlot


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
    worker's process id and the operation's name."""
    var id: Int
    var devices: List[Int]
    var pids: List[Int]
    var operations: List[String]

    def __init__(out self, id: Int, var devices: List[Int]):
        self.id = id
        self.devices = devices^
        self.pids = List[Int]()
        self.operations = List[String]()


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
