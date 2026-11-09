# SPDX-License-Identifier: Apache-2.0
# Native unit threads (lane p3-msel-parallel-and-process-pool, 2026-11-09).
#
# Independent units that call compiled owners through CPython objects (a
# GridSearchCV candidate x fold fit, a cross-validation fold) run on several
# OS threads of THIS process: no child interpreter, no package import, no
# pickle. The caller runs units too; `workers - 1` more threads are started
# with `pthread_create` and joined with the GIL released.
#
# Why OS threads and not `host_parallelize` (core/host_parallel.mojo): a unit
# holds the GIL while it drives its owner and gives it up only inside the
# owners' GIL-released kernel calls. A runtime pool worker blocked on the GIL
# could be the one a nested `host_parallelize` inside another unit's fit
# waits for: a deadlock. Threads of our own never serve the runtime pool.
#
# The entry function owns the unit protocol: it takes the GIL with
# `native_unit_gil_enter` (reentrant: the caller already holds it), drains
# units in index order under the GIL, writes each result into the slot of its
# unit (the order of the results is the unit order, never the finish order)
# and gives the GIL back with `native_unit_gil_leave`. The floating-point
# environment of the caller is installed on every worker (core/host_parallel
# DEVIATION 5900), so a unit computes the same bits on any thread.
from std.ffi import c_int, c_size_t, external_call
from std.python import Python
from std.python._cpython import GILReleased
from std.sys.info import num_physical_cores

from bindings.cpython_table import cpython_table

#: The thread entry: `void *(*)(void *)` with the argument and result as
#: address words (same register class under the C ABI on every target).
comptime NativeUnitEntry = def(Int) thin abi("C") -> Int

#: Words reserved for a `pthread_attr_t` (56 bytes on glibc x86_64/aarch64,
#: 64 bytes on macOS): 128 bytes.
comptime _ATTR_WORDS = 16
#: Stack of every unit thread: owners recurse over model structure (nested
#: estimators, tree builders); a macOS secondary thread gets 512 KiB by
#: default, the main thread 8 MiB. Reserved address space, not resident.
comptime _UNIT_THREAD_STACK_BYTES = 64 << 20


def native_unit_cores() -> Int:
    """Physical cores of this host (at least 1)."""
    return max(num_physical_cores(), 1)


def native_unit_gil_enter() raises -> c_int:
    """`PyGILState_Ensure` from the one CPython table (create the table on
    the caller thread before starting workers)."""
    var table = cpython_table()
    return table[].PyGILState_Ensure()


def native_unit_gil_leave(state: c_int) raises:
    var table = cpython_table()
    table[].PyGILState_Release(state)


def native_unit_threads_run(entry: NativeUnitEntry, arg: Int, workers: Int) raises:
    """Run `entry(arg)` on the calling thread and on `workers - 1` new OS
    threads, then join them with the GIL released. The caller must hold the
    GIL. A thread that cannot be started is not an error: the units it would
    have taken are drained by the others (the entry pulls units, it is not
    handed a fixed share)."""
    # Created here, with the GIL held, so no worker creates it.
    _ = cpython_table()
    var extra = workers - 1
    var handles = List[Int](length=max(extra, 0), fill=0)
    var attr = List[Int64](length=_ATTR_WORDS, fill=Int64(0))
    var started = 0
    if extra > 0 and external_call["pthread_attr_init", c_int](Int(attr.unsafe_ptr())) == 0:
        _ = external_call["pthread_attr_setstacksize", c_int](Int(attr.unsafe_ptr()), c_size_t(_UNIT_THREAD_STACK_BYTES))
        for i in range(extra):  # driver-loop: over unit threads (at most one per core)
            var slot = Int(handles.unsafe_ptr()) + i * 8
            if external_call["pthread_create", c_int](slot, Int(attr.unsafe_ptr()), entry, arg) != 0:
                break
            started += 1
        _ = external_call["pthread_attr_destroy", c_int](Int(attr.unsafe_ptr()))
    _ = entry(arg)
    if started > 0:
        # Workers still running need the GIL to finish their units.
        with GILReleased(Python()):
            for i in range(started):  # driver-loop: over started unit threads
                _ = external_call["pthread_join", c_int](handles[i], Int(0))
    _ = handles^
    _ = attr^
