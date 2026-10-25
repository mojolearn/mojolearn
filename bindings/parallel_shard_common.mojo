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
process pool (bindings/native_process_pool.mojo: one process per device, the
visibility mask narrowed to it, every device count 1) and the join are
native. The wave order is the old pool's: `process_pool_map` sends waves of
`len(devices)` requests and hands wave position j to worker j, so shard i
runs on rank i % len(devices), as `DevicePool.map` did.

The driver negative control is the old `_parallel_pool.driver_read_shift`,
word for word: with MOJOLEARN_PAR_DRIVER_SABOTAGE=1 and
MOJOLEARN_HOST_ALLOW_SABOTAGE=1 every shard on an owner above rank 0 whose
true start is above 0 READS one position early while its block is still
PLACED at the true offset. Inert at one device.

No floating-point value is computed here: copies only.
"""
from std.ffi import _Global, external_call
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address
from bindings.native_process_pool import process_pool_create, process_pool_map, process_pool_close, PROCESS_BOUNDARY
from bindings.native_worker_inventory import native_worker_inventory, native_require_worker_inventory
from bindings.tsa_hw_native import tsa_array


def shard_b() raises -> PythonObject:
    return Python.import_module("builtins")


def shard_ranges(count: Int, width: PythonObject, name: String, empty_message: String) raises -> List[Int]:
    """The shard bounds [lo0, hi0, lo1, hi1, ...] of `range(0, count, width)`
    with `min(start + width, count)` ends; `width` must be a positive int
    (`name` names it in the refusal)."""
    if py_type(width) is not shard_b().int or Int(py=width) < 1:
        raise Error("ValueError: " + name + " must be a positive integer")
    if count < 1 and empty_message != "":
        raise Error("ValueError: " + empty_message)
    var w = Int(py=width)
    var out = List[Int]()
    var i = 0
    while i < count:  # driver-loop: over the shard boundaries for the device pool
        out.append(i)
        out.append(min(i + w, count))
        i += w
    return out^


def shard_devices(devices: PythonObject) raises -> Int:
    """`DevicePool`'s admission of the device list, in its words; the number
    of devices."""
    var b = shard_b()
    if py_type(devices) is not b.tuple and py_type(devices) is not b.list and py_type(devices) is not b.range:
        raise Error("TypeError: devices requires a native tuple, list or range; Python iteration hooks are unsupported")
    var selected = py_call(b.tuple, devices)
    var seen = b.set()
    if len(selected) == 0:
        raise Error("ValueError: devices must be distinct nonnegative integer indices")
    for i in range(len(selected)):  # driver-loop: over the admitted device indices
        if (py_type(selected[i]) is not b.int or Int(py=selected[i]) < 0
                or Bool(py=py_call(py_attr(seen, "__contains__"), selected[i]))):
            raise Error("ValueError: devices must be distinct nonnegative integer indices")
        _ = py_call(seen.add, selected[i])
    return len(selected)


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


struct ShardTrace(Defaultable, Movable):
    var records: PythonObject
    var faults: Int

    def __init__(out self):
        self.records = PythonObject(None)
        self.faults = 0


#: Every pool the sharded drivers started in this process since the last
#: `shard_trace_take`: its devices and its workers' own driver inventories
#: (the record the verifier's two-device witness reads,
#: python/mojolearn/_verify_par.py). Bounded: the oldest records go first.
comptime SHARD_TRACE = _Global[StorageType=ShardTrace, name="MojoParallelShardTrace", init_fn=ShardTrace.__init__]


def shard_identity() raises -> PythonObject:
    """A worker's answer to an identity request: its own process and the
    devices its visibility mask leaves it (bindings/native_worker_inventory.mojo)."""
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    return native_worker_inventory(p[].vendor, p[].environ)


def shard_open(devices: PythonObject, child: String) raises -> PythonObject:
    """A started pool of one worker per device whose placement is admitted:
    every worker answers an identity request first, and the replies must be
    distinct processes started here and, on CUDA and HIP, distinct physical
    GPUs, one per worker (`native_require_worker_inventory`, the CV pool's
    rule). The pool and its inventory are recorded in the trace."""
    var b = shard_b()
    var pool = process_pool_create(devices, False, child)
    var requests = b.list()
    for i in range(len(devices)):  # driver-loop: one identity request per worker
        _ = py_call(requests.append, Python.tuple("identity"))
    var identities = process_pool_map(pool, requests)
    var pids = b.list()
    for i in range(len(pool["workers"])):  # driver-loop: over the spawned workers
        _ = py_call(pids.append, pool["workers"][i]["pid"])
    var p = PROCESS_BOUNDARY.get_or_create_ptr()
    try:
        native_require_worker_inventory(identities, p[].vendor, pids, Int(external_call["getpid", Int32]()))
    except error:
        process_pool_close(pool)
        raise error^
    var t = SHARD_TRACE.get_or_create_ptr()
    if t[].records is None:
        t[].records = b.list()
    var record = b.dict()
    py_setitem(record, "devices", py_call(b.list, devices))
    py_setitem(record, "inventory", identities)
    py_setitem(record, "calls", b.list())
    py_setitem(pool, "trace_record", record)
    _ = py_call(t[].records.append, record)
    if len(t[].records) > 256:
        _ = py_call(t[].records.pop, 0)
    return pool


def shard_trace_take() raises -> PythonObject:
    """The pools recorded since the last take, and an empty trace."""
    var t = SHARD_TRACE.get_or_create_ptr()
    var out = t[].records
    t[].records = shard_b().list()
    return shard_b().list() if out is None else out


def shard_faults_take() raises -> Int:
    """How many transport faults the verifier's switch injected since the
    last take (`shard_map`), and zero."""
    var t = SHARD_TRACE.get_or_create_ptr()
    var n = t[].faults
    t[].faults = 0
    return n


def _operation(request: PythonObject) raises -> String:
    var name = String(request[0])
    if name == "py":
        return String(request[1])
    return name


def _numerical(name: String) -> Bool:
    return (name == "arima_predict_shard" or name == "hw_predict_shard" or name == "gpc_class_fit"
            or name == "gpc_class_predict" or name == "ivf_search" or name == "rbf_rows"
            or name == "scaler_fit" or name == "scaler_transform" or name == "forest_fit"
            or name == "arima_fit_shard" or name == "hw_fit_shard" or name == "reference_shard"
            or name == "query_rows")


def shard_map(pool: PythonObject, requests: PythonObject) raises -> PythonObject:
    """`process_pool_map(pool, requests)`, with every request's worker and
    operation added to the pool's trace record.

    THE TRANSPORT NEGATIVE CONTROL (python/mojolearn/_verify_distributed.py):
    with MOJOLEARN_PAR_TRANSPORT_FAULT=drop_result or reverse_results and
    MOJOLEARN_HOST_ALLOW_SABOTAGE=1, the first numerical batch of more than
    one result in this process is returned without its last result or in
    reverse order, once. The driver's own checks or the hash must catch it.
    No build, workflow or gate sets either variable."""
    var b = shard_b()
    var results = process_pool_map(pool, requests)
    var record = py_get(pool, PythonObject("trace_record"), PythonObject(None))
    var workers = pool["workers"]
    if record is not None and len(workers) > 0:
        for i in range(len(requests)):  # driver-loop: one trace entry per request
            var call = b.dict()
            py_setitem(call, "pid", workers[i % len(workers)]["pid"])
            py_setitem(call, "operation", _operation(requests[i]))
            _ = py_call(record["calls"].append, call)
    if len(results) > 1 and len(requests) > 0 and _numerical(_operation(requests[0])):
        var fault = String()
        if String(getenv("MOJOLEARN_HOST_ALLOW_SABOTAGE")) == "1":
            fault = String(getenv("MOJOLEARN_PAR_TRANSPORT_FAULT"))
        var t = SHARD_TRACE.get_or_create_ptr()
        if t[].faults == 0 and (fault == "drop_result" or fault == "reverse_results"):
            t[].faults = 1
            var altered = py_call(b.list, results)
            if fault == "drop_result":
                _ = py_call(altered.pop)
            else:
                _ = py_call(altered.reverse)
            return altered
    return results


def shard_run(requests: PythonObject, devices: PythonObject, child: String) raises -> PythonObject:
    """The workers' results in request order (waves of len(devices), wave
    position j on worker j); the pool always closes."""
    var pool = shard_open(devices, child)
    var results = PythonObject(None)
    try:
        results = shard_map(pool, requests)
    except error:
        process_pool_close(pool)
        raise error^
    process_pool_close(pool)
    if len(results) != len(requests):
        raise Error("ValueError: shard workers returned an incomplete batch")
    return results


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
    if lo < 0 or hi > rows or hi < lo:
        raise Error("ValueError: shard rows outside the source")
    var inner = 1
    for k in range(1, len(shape)):  # small-loop(the source's trailing dimensions)
        inner *= Int(py=shape[k])
    py_setitem(shape, 0, hi - lo)
    var dtype = String(py=source.dtype)
    var item = shard_item(dtype)
    var out = shard_array(dtype, py_call(b.tuple, shape), (hi - lo) * inner, context)
    var bytes = (hi - lo) * inner * item[1]
    if bytes > 0:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(source)))
               + lo * inner * item[1],
               count=bytes)
        _ = out  # keep the Python object alive through the native call above
        _ = source  # keep the Python object alive through the native call above
    return out


def shard_row_vector(source: PythonObject, row: Int, context: PythonObject) raises -> PythonObject:
    """Row `row` of a C-ordered 2-D Array, as a 1-D copy."""
    var b = shard_b()
    var width = Int(py=source.shape[1])
    var dtype = String(py=source.dtype)
    var item = shard_item(dtype)
    var out = shard_array(dtype, Python.tuple(width), width, context)
    if width > 0:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(source)))
               + row * width * item[1],
               count=width * item[1])
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
    var dtype = String(py=first.dtype)
    var item = shard_item(dtype)
    py_setitem(shape, 0, total)
    var out = shard_array(dtype, py_call(b.tuple, shape), total * inner, context)
    var cursor = 0
    for p in range(len(arrays)):  # driver-loop: one block copy per device shard
        var value = arrays[p]
        if String(py=value.dtype) != dtype:
            raise Error("ValueError: shard workers returned different dtypes")
        var rows = Int(py=value.shape[0])
        var bytes = rows * inner * item[1]
        if cursor + rows > total:
            raise Error("ValueError: shard workers returned too many rows")
        if bytes > 0:
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
                   + cursor * inner * item[1],
                   src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value))),
                   count=bytes)
            _ = out  # keep the Python object alive through the native call above
            _ = value  # keep the Python object alive through the native call above
        cursor += rows
    if cursor != total:
        raise Error("ValueError: shard workers returned an incomplete result")
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
    if cls is context["array_class"]:
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
