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
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address
from bindings.native_process_pool import process_pool_create, process_pool_map, process_pool_close
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
    try:
        if (String(getenv("MOJOLEARN_PAR_DRIVER_SABOTAGE")) == "1"
                and String(getenv("MOJOLEARN_HOST_ALLOW_SABOTAGE")) == "1"):
            return 1
    except:
        pass
    return 0


def shard_run(requests: PythonObject, devices: PythonObject, child: String) raises -> PythonObject:
    """The workers' results in request order (waves of len(devices), wave
    position j on worker j); the pool always closes."""
    var pool = process_pool_create(devices, False, child)
    var results = PythonObject(None)
    try:
        results = process_pool_map(pool, requests)
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
        return ("f", 4)
    if dtype == "<f8":
        return ("d", 8)
    if dtype == "<i4":
        return ("i", 4)
    if dtype == "<i8":
        return ("q", 8)
    if dtype == "|u1" or dtype == "<u1":
        return ("B", 1)
    raise Error("ValueError: shard arrays require float32, float64, int32, int64 or uint8, got " + dtype)


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


def shard_concat(parts: PythonObject, name: String, total: Int, context: PythonObject) raises -> PythonObject:
    """The arrays `parts[k][name]` joined along their first axis in shard
    order (a byte copy per shard); the first part's dtype and trailing shape,
    `total` leading rows. Every part must share them."""
    var b = shard_b()
    var first = parts[0][name]
    var shape = py_call(b.list, first.shape)
    var inner = 1
    for k in range(1, len(shape)):  # small-loop(the trailing dimensions)
        inner *= Int(py=shape[k])
    var dtype = String(py=first.dtype)
    var item = shard_item(dtype)
    py_setitem(shape, 0, total)
    var out = shard_array(dtype, py_call(b.tuple, shape), total * inner, context)
    var cursor = 0
    for p in range(len(parts)):  # driver-loop: one block copy per device shard
        var value = parts[p][name]
        if String(py=value.dtype) != dtype:
            raise Error("ValueError: shard worker returned a different dtype for " + name)
        var rows = Int(py=value.shape[0])
        var bytes = rows * inner * item[1]
        if cursor + rows > total:
            raise Error("ValueError: shard workers returned too many rows for " + name)
        if bytes > 0:
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
                   + cursor * inner * item[1],
                   src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value))),
                   count=bytes)
            _ = out  # keep the Python object alive through the native call above
            _ = value  # keep the Python object alive through the native call above
        cursor += rows
    if cursor != total:
        raise Error("ValueError: shard workers returned an incomplete " + name)
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
