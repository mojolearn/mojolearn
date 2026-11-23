# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""Concurrent use of one `DeviceStore` (core/device_store.mojo; lane linalg-owner, 2026-11-22).

A store is one process slot per binding (X_PREP_STORE, X_METRICS_STORE) and the doors release the GIL around put,
free, take_rows and the program runs that read slots, so two calls at once reallocated the slot list under each other
or replaced a slot another call was reading. The store now takes its lock word around every List step.

`STORE_CONCURRENT_CALLS` tasks at once, on one store and one context, each: put its own words, gather two rows of them
(`take_rows`), read both slots back, free both. Every word read back must equal the words that task put (alone, a
put/read is a byte copy, so the reference is the input itself); at the end no slot is live.
"""
from std.memory import bitcast
from std.testing import TestSuite
from max.gpu.host import DeviceContext
from core.device_store import DeviceStore
from core.host_parallel import host_parallelize
from checks.native_tests.testsuite_cases import require

comptime STORE_CONCURRENT_CALLS = 100
comptime _ROW = 37
comptime _ROWS = 9


def _words(unit: Int) -> List[Float32]:
    var x = List[Float32](length=_ROW * _ROWS, fill=0)
    for i in range(_ROW * _ROWS):  # small-loop(fixture: 333 words per task)
        x[i] = Float32(unit * 1000 + i) * Float32(0.5)
    return x^


def _same(got: List[Float32], want: List[Float32], lo: Int, label: String) raises:
    for i in range(len(got)):  # small-loop(at most 333 words)
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[lo + i]):
            raise Error(label + " word " + String(i) + ": " + String(got[i]) + " put " + String(want[lo + i]))


def _task(mut store: DeviceStore, ctx: DeviceContext, unit: Int) raises:
    var x = _words(unit)
    var id = store.put(ctx, Int(x.unsafe_ptr()), len(x))
    var rows: List[Int64] = [Int64(unit % _ROWS), Int64((unit + 4) % _ROWS)]
    var gathered = store.take_rows(ctx, id, _ROW, Int(rows.unsafe_ptr()), 2)
    var back = List[Float32](length=len(x), fill=0)
    store.read(ctx, id, Int(back.unsafe_ptr()), len(back))
    var two = List[Float32](length=2 * _ROW, fill=0)
    store.read(ctx, gathered, Int(two.unsafe_ptr()), len(two))
    _same(back, x, 0, "task " + String(unit) + " slot")
    var first = List[Float32](length=_ROW, fill=0)
    var second = List[Float32](length=_ROW, fill=0)
    for i in range(_ROW):  # small-loop(one row, 37 words)
        first[i] = two[i]
        second[i] = two[_ROW + i]
    _same(first, x, (unit % _ROWS) * _ROW, "task " + String(unit) + " row a")
    _same(second, x, ((unit + 4) % _ROWS) * _ROW, "task " + String(unit) + " row b")
    store.free(ctx, gathered)
    store.free(ctx, id)
    _ = x^
    _ = rows^
    _ = back^
    _ = two^


def test_store_concurrent() raises:
    var ctx = DeviceContext()
    var store = DeviceStore()
    var sp = UnsafePointer(to=store).unsafe_origin_cast[MutAnyOrigin]()
    var failed = List[Int](length=STORE_CONCURRENT_CALLS, fill=0)
    var messages = List[String](length=STORE_CONCURRENT_CALLS, fill=String(""))
    var fp = failed.unsafe_ptr()
    var mp = messages.unsafe_ptr()
    def task(unit: Int) {imm fp, imm mp, imm sp, imm ctx}:
        try:
            _task(sp[], ctx, unit)
        except problem:
            mp[unsafe_offset=unit] = String(problem)
            fp.unsafe_store(unit, 1)
    host_parallelize(task, STORE_CONCURRENT_CALLS)
    var count = 0
    var first = String("")
    for i in range(STORE_CONCURRENT_CALLS):  # small-loop(the tasks' failure words)
        if failed[i] != 0:
            if count == 0:
                first = messages[i]
            count += 1
    var live = store.live
    _ = messages^
    _ = failed^
    _ = store^
    _ = ctx^
    require(count == 0, String(count) + " of " + String(STORE_CONCURRENT_CALLS)
        + " concurrent store tasks read back other words; first: " + first)
    require(live == 0, "slots left live after every task freed its own: " + String(live))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
