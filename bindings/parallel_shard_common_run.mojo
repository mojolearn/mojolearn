# SPDX-License-Identifier: Apache-2.0
"""Typed parts of the parallel shard driver (RUN): the driver read shift, the numerical
operation names, the dtype item table, and the process counters (injected transport faults and
store tokens). The Python pool and trace records stay in `bindings/parallel_shard_common_door.mojo`."""
from std.ffi import external_call
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
