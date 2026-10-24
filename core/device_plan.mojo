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
comptime DEVICE_PLAN_LOG_WORDS = 4096

# THE OWNER FAMILIES A PLAN DRIVES. The cooperative worker set every
# `_parallel_pool.DEVICE_COUNT_VARIABLES` count to the device number, and
# MOJOLEARN_CHOLESKY_DEVICE_COUNT only inside the kernel-method and Cholesky
# operations (the GP and GaussianMixture workers kept their root
# factorization). A plan carries the same choice as a mask, so a driver's
# owners partition exactly where the worker's did.
comptime DEVICE_PLAN_KMEANS = 1 << 0
comptime DEVICE_PLAN_GBDT = 1 << 1
comptime DEVICE_PLAN_GRAM = 1 << 2
comptime DEVICE_PLAN_QR = 1 << 3
comptime DEVICE_PLAN_OPTIMIZER = 1 << 4
comptime DEVICE_PLAN_GLM = 1 << 5
comptime DEVICE_PLAN_SOLVER = 1 << 6
comptime DEVICE_PLAN_IFOREST = 1 << 7
comptime DEVICE_PLAN_FOREST = 1 << 8
comptime DEVICE_PLAN_SVM = 1 << 9
comptime DEVICE_PLAN_GP = 1 << 10
comptime DEVICE_PLAN_GMM = 1 << 11
comptime DEVICE_PLAN_RESAMPLE = 1 << 12
comptime DEVICE_PLAN_DBSCAN = 1 << 13
comptime DEVICE_PLAN_NEIGHBORS = 1 << 14
comptime DEVICE_PLAN_HIERARCHY = 1 << 15
comptime DEVICE_PLAN_CHOLESKY = 1 << 16
#: what the cooperative worker's environment set (`DEVICE_COUNT_VARIABLES`)
comptime DEVICE_PLAN_WORKER = (1 << 16) - 1


def _family_bit(name: String) -> Int:
    """The DEVICE_PLAN_* bit of `MOJOLEARN_<FAMILY>_DEVICE_COUNT`, 0 for any
    other name (never planned)."""
    if name == "MOJOLEARN_KMEANS_DEVICE_COUNT":
        return DEVICE_PLAN_KMEANS
    if name == "MOJOLEARN_GBDT_DEVICE_COUNT":
        return DEVICE_PLAN_GBDT
    if name == "MOJOLEARN_GRAM_DEVICE_COUNT":
        return DEVICE_PLAN_GRAM
    if name == "MOJOLEARN_QR_DEVICE_COUNT":
        return DEVICE_PLAN_QR
    if name == "MOJOLEARN_OPTIMIZER_DEVICE_COUNT":
        return DEVICE_PLAN_OPTIMIZER
    if name == "MOJOLEARN_GLM_DEVICE_COUNT":
        return DEVICE_PLAN_GLM
    if name == "MOJOLEARN_SOLVER_DEVICE_COUNT":
        return DEVICE_PLAN_SOLVER
    if name == "MOJOLEARN_IFOREST_DEVICE_COUNT":
        return DEVICE_PLAN_IFOREST
    if name == "MOJOLEARN_FOREST_DEVICE_COUNT":
        return DEVICE_PLAN_FOREST
    if name == "MOJOLEARN_SVM_DEVICE_COUNT":
        return DEVICE_PLAN_SVM
    if name == "MOJOLEARN_GP_DEVICE_COUNT":
        return DEVICE_PLAN_GP
    if name == "MOJOLEARN_GMM_DEVICE_COUNT":
        return DEVICE_PLAN_GMM
    if name == "MOJOLEARN_RESAMPLE_DEVICE_COUNT":
        return DEVICE_PLAN_RESAMPLE
    if name == "MOJOLEARN_DBSCAN_DEVICE_COUNT":
        return DEVICE_PLAN_DBSCAN
    if name == "MOJOLEARN_NEIGHBORS_DEVICE_COUNT":
        return DEVICE_PLAN_NEIGHBORS
    if name == "MOJOLEARN_HIERARCHY_DEVICE_COUNT":
        return DEVICE_PLAN_HIERARCHY
    if name == "MOJOLEARN_CHOLESKY_DEVICE_COUNT":
        return DEVICE_PLAN_CHOLESKY
    return 0


struct _DevicePlanTable(Defaultable, Movable):
    # owners[s]: 0 when slot s is free, else the owning thread id.
    # counts[s]: the number of devices in slot s's plan.
    # masks[s]: the owner families (DEVICE_PLAN_* bits) the plan drives.
    # ids[s * DEVICE_PLAN_MAX_DEVICES + r]: the physical device of rank r.
    # active[0]: the number of claimed slots (a fast path for the common
    # case of no plan anywhere in the process).
    # The lists are sized once here and never resized, so their storage
    # addresses are stable for the process.
    var owners: List[Int64]
    var counts: List[Int64]
    var masks: List[Int64]
    var ids: List[Int64]
    var active: List[Int64]
    # The placement log the verifier's two-device witness reads
    # (`device_plan_log_take`, python/mojolearn/_verify_par.py): every bound
    # plan's [count, id0, id1, ...], appended under `log_lock`, until full.
    var log: List[Int64]
    var log_used: List[Int64]
    var log_lock: List[Int64]

    def __init__(out self):
        self.owners = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.counts = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.masks = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.ids = List[Int64](length=DEVICE_PLAN_SLOTS * DEVICE_PLAN_MAX_DEVICES, fill=Int64(0))
        self.active = List[Int64](length=1, fill=Int64(0))
        self.log = List[Int64](length=DEVICE_PLAN_LOG_WORDS, fill=Int64(0))
        self.log_used = List[Int64](length=1, fill=Int64(0))
        self.log_lock = List[Int64](length=1, fill=Int64(0))


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


def device_plan_enter(devices: List[Int], families: Int = DEVICE_PLAN_WORKER) raises:
    """Bind `devices` (rank r runs on physical device `devices[r]`) to the
    calling thread until `device_plan_exit`, for the owner families in
    `families` (DEVICE_PLAN_* bits)."""
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
    _words(table[].masks)[slot] = Int64(families)
    _ = Atomic[DType.int64].fetch_add(_words(table[].active), Int64(1))
    _log_plan(devices)


def device_plan_exit() raises:
    """Release the calling thread's plan."""
    var slot = _own_slot()
    if slot < 0:
        raise Error("RuntimeError: no device plan is bound to this thread")
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    _words(table[].counts)[slot] = Int64(0)
    _words(table[].masks)[slot] = Int64(0)
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


def device_plan_setting(name: String, default: String = "") -> String:
    """What an owner loop parses as its device count: the plan's size as text
    while a plan driving `name`'s family is bound to the calling thread, else
    `getenv(name, default)`."""
    var slot = _own_slot()
    if slot >= 0:
        var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
        if (Int(_words(table[].masks)[slot]) & _family_bit(name)) != 0:
            return String(Int(_words(table[].counts)[slot]))
    return String(getenv(name, default))


def _log_lock() -> MutPointer[Int64, MutUntrackedOrigin]:
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    var lock = _words(table[].log_lock)
    while True:  # driver-loop: spin on the placement log's lock word
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](lock, expected, Int64(1)):
            return lock


def _log_plan(devices: List[Int]):
    """Append one bound plan to the placement log (dropped when full)."""
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    var lock = _log_lock()
    var used = Int(_words(table[].log_used)[0])
    if used + 1 + len(devices) <= DEVICE_PLAN_LOG_WORDS:
        var log = _words(table[].log)
        log[used] = Int64(len(devices))
        for r in range(len(devices)):  # driver-loop: over the plan's ranks
            log[used + 1 + r] = Int64(devices[r])
        _words(table[].log_used)[0] = Int64(used + 1 + len(devices))
    Atomic.store[ordering=Ordering.RELEASE](lock, Int64(0))


def device_plan_log_take() -> List[List[Int]]:
    """The plans bound since the last take, in order, and an empty log."""
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr()
    var lock = _log_lock()
    var used = Int(_words(table[].log_used)[0])
    var log = _words(table[].log)
    var out = List[List[Int]]()
    var at = 0
    while at < used:  # driver-loop: over the logged plans
        var count = Int(log[at])
        var ids = List[Int]()
        for r in range(count):  # driver-loop: over a logged plan's ranks
            ids.append(Int(log[at + 1 + r]))
        out.append(ids^)
        at += 1 + count
    _words(table[].log_used)[0] = Int64(0)
    Atomic.store[ordering=Ordering.RELEASE](lock, Int64(0))
    return out^
