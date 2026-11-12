# SPDX-License-Identifier: Apache-2.0
"""Typed parts of the parallel shard driver (RUN): the driver read shift, the numerical
operation names, the dtype item table, and the process counters (injected transport faults and
store tokens). The Python pool and trace records stay in `bindings/parallel_shard_common_door.mojo`."""
from std.ffi import external_call
from std.os import getenv
from core.shared_slot import SharedSlot


struct ShardCounters(Defaultable, Movable):
    #: transport faults the verifier's switch injected since the last take
    var faults: Int
    #: `shard_run_stored` calls made by this process (its store tokens)
    var stored_calls: Int

    def __init__(out self):
        self.faults = 0
        self.stored_calls = 0


comptime SHARD_COUNTERS = SharedSlot[ShardCounters, "MojolearnParallelShardCountersV1"]


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
    start is above 0 when both switch variables are set, else 0. Read at
    call time, never cached."""
    if first < 1 or n_devices < 2 or index % n_devices == 0:
        return 0
    if (String(getenv("MOJOLEARN_PAR_DRIVER_SABOTAGE")) == "1"
            and String(getenv("MOJOLEARN_HOST_ALLOW_SABOTAGE")) == "1"):
        return 1
    return 0


def shard_faults_take() raises -> Int:
    """How many transport faults the verifier's switch injected since the
    last take (`shard_map`), and zero."""
    var c = SHARD_COUNTERS.get_or_create_ptr()
    var n = c[].faults
    c[].faults = 0
    return n


def _numerical(name: String) -> Bool:
    return (name == "arima_predict_shard" or name == "hw_predict_shard" or name == "gpc_class_fit"
            or name == "gpc_class_predict" or name == "ivf_search" or name == "rbf_rows"
            or name == "scaler_fit" or name == "scaler_transform" or name == "forest_fit"
            or name == "arima_fit_shard" or name == "hw_fit_shard" or name == "reference_shard"
            or name == "reference_query" or name == "query_rows")


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
