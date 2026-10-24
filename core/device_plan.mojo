# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE THREAD-SCOPED DEVICE PLAN (lane/native-parallel-driver-20261024;
docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md).

A multi-GPU driver used to run its estimator in a worker process whose
visibility mask (`CUDA_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES`) held only the selected devices and whose
`MOJOLEARN_<FAMILY>_DEVICE_COUNT` variables held their number. The owner
loops (`svm/impl/distance/kernel_matrices.mojo::_kernel_rows`,
`cholesky/multi_gpu.mojo`, `cholesky/checks/trsm.mojo`) read that count with
`getenv` and opened `DeviceContext(device_id=rank)`.

A native driver runs in the caller's process instead. It binds the ordered
list of physical device indices to the CALLING OS THREAD for the length of
one call (`device_plan_enter` / `device_plan_exit`); the owner loops read
the count through `device_plan_setting` and the device of each rank through
`device_plan_device`. With no plan bound on the thread both return exactly
what they returned before (the environment variable, the rank), so every
existing caller, check tool and Python pool worker is unchanged.

Why per thread and not per process: the bindings release the GIL around
their kernels, so two Python threads can run two drivers, or a driver and a
plain one-device call, at the same time. A process-wide setting (what the
environment variable was inside a worker) would leak one call's devices
into the other. Every read happens on the thread that entered the plan: the
owner loops read the count and the device ids before `host_parallelize`
hands the per-owner work to other threads, and those tasks call
`kernel_op(..., distribute=False)`, which reads no plan.

Storage: one process-lifetime table (`std.ffi._Global`) of fixed slots, each
owned by one thread id (`pthread_self`), claimed by compare-exchange and
released by a release store. Only the owning thread reads or writes a
slot's devices, so the slot data needs no lock. A nested plan on one thread
is refused by name. No floating-point value is read or written here.
"""
from std.atomic import Atomic, Ordering
from std.ffi import _Global, external_call
from std.os import getenv

from checks.numerics import GLOBAL_NUMERIC_MODE

comptime DEVICE_PLAN_SLOTS = 64
comptime DEVICE_PLAN_MAX_DEVICES = 64


struct _DevicePlanTable(Defaultable, Movable):
    # owners[s]: 0 when slot s is free, else the owning thread id.
    # counts[s]: the number of devices in slot s's plan.
    # ids[s * DEVICE_PLAN_MAX_DEVICES + r]: the physical device of rank r.
    # active[0]: the number of claimed slots (a fast path for the common
    # case of no plan anywhere in the process).
    # The lists are sized once here and never resized, so their storage
    # addresses are stable for the process.
    var owners: List[Int64]
    var counts: List[Int64]
    var ids: List[Int64]
    var active: List[Int64]

    def __init__(out self):
        self.owners = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.counts = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.ids = List[Int64](length=DEVICE_PLAN_SLOTS * DEVICE_PLAN_MAX_DEVICES, fill=Int64(0))
        self.active = List[Int64](length=1, fill=Int64(0))


comptime _DEVICE_PLAN_TABLE = _Global[
    StorageType=_DevicePlanTable,
    name=("MojoDevicePlanIdentical" if GLOBAL_NUMERIC_MODE == 1 else "MojoDevicePlanOther"),
    init_fn=_DevicePlanTable.__init__,
]


def _thread_id() -> Int64:
    # pthread_t is an unsigned long on Linux and a pointer on macOS: either
    # way a nonzero word that identifies the live calling thread.
    return Int64(Int(external_call["pthread_self", UInt]()))


def _words(mut list: List[Int64]) -> MutPointer[Int64, MutUntrackedOrigin]:
    return rebind[MutPointer[Int64, MutUntrackedOrigin]](list.unsafe_ptr())


def _own_slot() -> Int:
    """The calling thread's slot, or -1 when it has no plan."""
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    var active = _words(table[].active)
    if Atomic.load[ordering=Ordering.ACQUIRE](active) == 0:
        return -1
    var owners = _words(table[].owners)
    var me = _thread_id()
    for s in range(DEVICE_PLAN_SLOTS):  # driver-loop: over the fixed plan slots
        if Atomic.load[ordering=Ordering.ACQUIRE](owners + s) == me:
            return s
    return -1


def device_plan_enter(devices: List[Int]) raises:
    """Bind `devices` (rank r runs on physical device `devices[r]`) to the
    calling thread until `device_plan_exit`."""
    var count = len(devices)
    if count < 1 or count > DEVICE_PLAN_MAX_DEVICES:
        raise Error("ValueError: a device plan names 1.." + String(DEVICE_PLAN_MAX_DEVICES) + " devices, got " + String(count))
    for i in range(count):  # driver-loop: over the selected device indices
        if devices[i] < 0:
            raise Error("ValueError: devices must be distinct nonnegative integer indices")
        for j in range(i):  # driver-loop: over the earlier selected device indices
            if devices[j] == devices[i]:
                raise Error("ValueError: devices must be distinct nonnegative integer indices")
    if _own_slot() >= 0:
        raise Error("RuntimeError: a device plan is already bound to this thread; multi-GPU drivers do not nest")
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    var owners = _words(table[].owners)
    var me = _thread_id()
    var slot = -1
    for s in range(DEVICE_PLAN_SLOTS):  # driver-loop: over the fixed plan slots
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](owners + s, expected, me):
            slot = s
            break
    if slot < 0:
        raise Error("RuntimeError: more than " + String(DEVICE_PLAN_SLOTS) + " threads hold a device plan at once")
    var ids = _words(table[].ids)
    for r in range(count):  # driver-loop: over the plan's ranks
        ids[slot * DEVICE_PLAN_MAX_DEVICES + r] = Int64(devices[r])
    _words(table[].counts)[slot] = Int64(count)
    _ = Atomic[DType.int64].fetch_add(_words(table[].active), Int64(1))


def device_plan_exit() raises:
    """Release the calling thread's plan."""
    var slot = _own_slot()
    if slot < 0:
        raise Error("RuntimeError: no device plan is bound to this thread")
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    _words(table[].counts)[slot] = Int64(0)
    _ = Atomic[DType.int64].fetch_add(_words(table[].active), Int64(-1))
    Atomic.store[ordering=Ordering.RELEASE](_words(table[].owners) + slot, Int64(0))


def device_plan_size() -> Int:
    """The number of devices in the calling thread's plan, 0 without one."""
    var slot = _own_slot()
    if slot < 0:
        return 0
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    return Int(_words(table[].counts)[slot])


def device_plan_device(rank: Int) raises -> Int:
    """The physical device of owner `rank`: the plan's entry while a plan is
    bound to the calling thread, else `rank` (the worker-process meaning,
    where the visibility mask made rank r the r-th selected device)."""
    var slot = _own_slot()
    if slot < 0:
        return rank
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    var count = Int(_words(table[].counts)[slot])
    if rank < 0 or rank >= count:
        raise Error("device plan: owner rank " + String(rank) + " outside a plan of " + String(count) + " devices")
    return Int(_words(table[].ids)[slot * DEVICE_PLAN_MAX_DEVICES + rank])


def device_plan_root() -> Int:
    """The calling thread's plan's first device (its root), -1 without a plan."""
    var slot = _own_slot()
    if slot < 0:
        return -1
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    return Int(_words(table[].ids)[slot * DEVICE_PLAN_MAX_DEVICES])


def device_plan_setting(name: String) -> String:
    """What an owner loop parses as its device count: the plan's size as text
    while a plan is bound to the calling thread, else `getenv(name)`."""
    var count = device_plan_size()
    if count > 0:
        return String(count)
    return String(getenv(name))
