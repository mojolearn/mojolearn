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
slot's devices, so the slot data needs no lock. Entering the same plan again
on one thread (the driver binds it in each binding it reaches) adds a level;
a different plan on the same thread is refused by name. No floating-point value is read or written here.

THE RULE (docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md): only binding
entries (a family's public native entry) read the thread-scoped plan, once,
with `device_plan_current(<family setting name>)`. Everything below the entry
takes the resulting `DevicePlan` value as an argument and calls no
`device_plan_*` reader. The readers below stay for the entries and for
callers not yet moved.
"""
from std.atomic import Atomic, Ordering

from core.shared_slot import shared_slot_check, shared_slot_word
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
    # depths[s]: how many times the owning thread entered this same plan
    # (a driver binds it once per binding; where `_Global` storage is shared
    # by name across the process's DSOs, those enters meet in one slot).
    # ids[s * DEVICE_PLAN_MAX_DEVICES + r]: the physical device of rank r.
    # active[0]: the number of claimed slots (a fast path for the common
    # case of no plan anywhere in the process).
    # The lists are sized once here and never resized, so their storage
    # addresses are stable for the process.
    # first field: the layout word every accessor checks (core/shared_slot.mojo)
    var layout: Int
    var owners: List[Int64]
    var counts: List[Int64]
    var masks: List[Int64]
    var depths: List[Int64]
    var ids: List[Int64]
    var active: List[Int64]
    # The placement log the verifier's two-device witness reads
    # (`device_plan_log_take`, python/mojolearn/_verify_par.py): every bound
    # plan's [count, id0, id1, ...], appended under `log_lock`, until full.
    var log: List[Int64]
    var log_used: List[Int64]
    var log_lock: List[Int64]
    # used[r]: 1 once an owner loop opened the device of rank r under a bound
    # plan (`DevicePlan.device`, `device_plan_device`) since the last
    # `device_plan_ranks_take`. Every owner loop asks for a rank's device
    # exactly where it opens that rank's DeviceContext for its shard, so the
    # number of set words is the number of ranks (logical or physical
    # devices) that received work. Read by the one-GPU multi-device gate
    # (tools/logical_shard_identity.py) through `device_plan_log`; it fails a
    # 2- or 3-device arm that ran on fewer than two ranks (lane
    # perf3-cluster-gp-gate). Integer flags only.
    var used: List[Int64]

    def __init__(out self):
        self.layout = shared_slot_word[_DevicePlanTable](_DEVICE_PLAN_LAYOUT)
        self.owners = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.counts = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.masks = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.depths = List[Int64](length=DEVICE_PLAN_SLOTS, fill=Int64(0))
        self.ids = List[Int64](length=DEVICE_PLAN_SLOTS * DEVICE_PLAN_MAX_DEVICES, fill=Int64(0))
        self.active = List[Int64](length=1, fill=Int64(0))
        self.log = List[Int64](length=DEVICE_PLAN_LOG_WORDS, fill=Int64(0))
        self.log_used = List[Int64](length=1, fill=Int64(0))
        self.log_lock = List[Int64](length=1, fill=Int64(0))
        self.used = List[Int64](length=DEVICE_PLAN_MAX_DEVICES, fill=Int64(0))


#: Layout version of `_DevicePlanTable` (core/shared_slot.mojo rule). V2:
#: the table gained `used` (lane perf3-cluster-gp-gate); V3: the layout word.
comptime _DEVICE_PLAN_LAYOUT = 3
comptime _DEVICE_PLAN_NAME_IDN: StaticString = "MojolearnDevicePlanIdenticalV3"
comptime _DEVICE_PLAN_NAME_OTHER: StaticString = "MojolearnDevicePlanOtherV3"
# typed StaticString arms: a conditional of two literals is a String value,
# which shared_slot_check (StaticString) does not take
comptime _DEVICE_PLAN_NAME: StaticString = _DEVICE_PLAN_NAME_IDN if GLOBAL_NUMERIC_MODE == 1 else _DEVICE_PLAN_NAME_OTHER
comptime _DEVICE_PLAN_TABLE = _Global[
    StorageType=_DevicePlanTable,
    # a new name per layout version, so a DSO built with an older layout
    # never shares storage with this one; the layout word refuses one that
    # forgot to bump it
    name=_DEVICE_PLAN_NAME,
    init_fn=_DevicePlanTable.__init__,
]


def _plan_table() raises -> UnsafePointer[_DevicePlanTable, MutAnyOrigin]:
    var table = _DEVICE_PLAN_TABLE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    shared_slot_check(table[].layout, shared_slot_word[_DevicePlanTable](_DEVICE_PLAN_LAYOUT), _DEVICE_PLAN_NAME)
    return table


def _thread_id() -> Int64:
    # pthread_t is an unsigned long on Linux and a pointer on macOS: either
    # way a nonzero word that identifies the live calling thread.
    return Int64(Int(external_call["pthread_self", UInt]()))


def _words(mut list: List[Int64]) -> MutPointer[Int64, MutUntrackedOrigin]:
    return rebind[MutPointer[Int64, MutUntrackedOrigin]](list.unsafe_ptr())


def _own_slot() raises -> Int:
    """The calling thread's slot, or -1 when it has no plan."""
    var table = _plan_table()
    var active = _words(table[].active)
    if Atomic.load[ordering=Ordering.ACQUIRE](active) == 0:
        return -1
    var owners = _words(table[].owners)
    var me = _thread_id()
    for s in range(DEVICE_PLAN_SLOTS):  # driver-loop: over the fixed plan slots
        if Atomic.load[ordering=Ordering.ACQUIRE](owners + s) == me:
            return s
    return -1


#: THE LOGICAL-DEVICE TEST OPTION (lane multi-gpu-logical-shard-gate,
#: Andrew 2026-11-06). Set to "1" only by tools/logical_shard_identity.py
#: (the release's one-GPU multi-device gate); not a user feature and not
#: documented. With it, a device list may name the same physical device more
#: than once (`devices=(0, 0, 0)`): each rank is still its own owner, opens
#: its own `DeviceContext(device_id=...)` (its own stream/queue) and moves its
#: data through the same peer_clone / transfer_bytes calls a real N-GPU run
#: uses. Only the three distinctness refusals consult it (here,
#: bindings/parallel_plan_door.mojo `plan_devices`,
#: bindings/parallel_public_native.mojo `_devices`); no dispatcher, split or
#: fold reads it, so the logical run takes exactly the real multi-GPU path.
comptime DEVICE_PLAN_LOGICAL_VARIABLE = "MOJOLEARN_DEVICE_PLAN_LOGICAL"


def device_plan_logical() -> Bool:
    """True when the logical-device test option is set (repeated device
    indices admitted)."""
    return String(getenv(DEVICE_PLAN_LOGICAL_VARIABLE)) == "1"


def device_plan_enter(devices: List[Int], families: Int = DEVICE_PLAN_WORKER) raises:
    """Bind `devices` (rank r runs on physical device `devices[r]`) to the
    calling thread until `device_plan_exit`, for the owner families in
    `families` (DEVICE_PLAN_* bits)."""
    var count = len(devices)
    if count < 1 or count > DEVICE_PLAN_MAX_DEVICES:
        raise Error("ValueError: a device plan names 1.." + String(DEVICE_PLAN_MAX_DEVICES) + " devices, got " + String(count))
    var logical = device_plan_logical()
    for i in range(count):  # driver-loop: over the selected device indices
        if devices[i] < 0:
            raise Error("ValueError: devices must be distinct nonnegative integer indices")
        if logical:
            continue
        for j in range(i):  # driver-loop: over the earlier selected device indices
            if devices[j] == devices[i]:
                raise Error("ValueError: devices must be distinct nonnegative integer indices")
    var table = _plan_table()
    var held = _own_slot()
    if held >= 0:
        # The same plan entered again (the next binding of the same driver
        # call): one more level. Any other plan would nest two drivers.
        var same = Int(_words(table[].counts)[held]) == count and Int(_words(table[].masks)[held]) == families
        if same:
            for r in range(count):  # driver-loop: over the plan's ranks
                if Int(_words(table[].ids)[held * DEVICE_PLAN_MAX_DEVICES + r]) != devices[r]:
                    same = False
        if not same:
            raise Error("RuntimeError: a device plan is already bound to this thread; multi-GPU drivers do not nest")
        _words(table[].depths)[held] = _words(table[].depths)[held] + 1
        return
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
    _words(table[].depths)[slot] = Int64(1)
    _ = Atomic[DType.int64].fetch_add(_words(table[].active), Int64(1))
    _log_plan(devices)


def device_plan_exit() raises:
    """Release the calling thread's plan."""
    var slot = _own_slot()
    if slot < 0:
        raise Error("RuntimeError: no device plan is bound to this thread")
    var table = _plan_table()
    var depth = Int(_words(table[].depths)[slot])
    if depth > 1:
        _words(table[].depths)[slot] = Int64(depth - 1)
        return
    _words(table[].depths)[slot] = Int64(0)
    _words(table[].counts)[slot] = Int64(0)
    _words(table[].masks)[slot] = Int64(0)
    _ = Atomic[DType.int64].fetch_add(_words(table[].active), Int64(-1))
    Atomic.store[ordering=Ordering.RELEASE](_words(table[].owners) + slot, Int64(0))


def device_plan_size() raises -> Int:
    """The number of devices in the calling thread's plan, 0 without one."""
    var slot = _own_slot()
    if slot < 0:
        return 0
    var table = _plan_table()
    return Int(_words(table[].counts)[slot])


def _mark_rank(rank: Int) raises:
    """Record that an owner loop opened rank `rank`'s device (see `used`)."""
    if rank < 0 or rank >= DEVICE_PLAN_MAX_DEVICES:
        return
    var table = _plan_table()
    Atomic.store[ordering=Ordering.RELEASE](_words(table[].used) + rank, Int64(1))


def device_plan_ranks_take() raises -> Int:
    """The number of distinct ranks whose device an owner loop opened under a
    bound plan since the last take, and a cleared record."""
    var table = _plan_table()
    var used = _words(table[].used)
    var ranks = 0
    for r in range(DEVICE_PLAN_MAX_DEVICES):  # driver-loop: over the fixed rank flags
        if Atomic.load[ordering=Ordering.ACQUIRE](used + r) != 0:
            ranks += 1
        Atomic.store[ordering=Ordering.RELEASE](used + r, Int64(0))
    return ranks


def _plan_device(slot: Int, rank: Int) raises -> Int:
    """The physical device of owner `rank` in the bound plan of `slot`."""
    var table = _plan_table()
    var count = Int(_words(table[].counts)[slot])
    if rank < 0 or rank >= count:
        raise Error("device plan: owner rank " + String(rank) + " outside a plan of " + String(count) + " devices")
    return Int(_words(table[].ids)[slot * DEVICE_PLAN_MAX_DEVICES + rank])


def device_plan_device(rank: Int) raises -> Int:
    """The physical device of owner `rank`: the plan's entry while a plan is
    bound to the calling thread, else `rank` (the worker-process meaning,
    where the visibility mask made rank r the r-th selected device). Under a
    bound plan the rank is recorded as having received work (`used`)."""
    var slot = _own_slot()
    if slot < 0:
        return rank
    var device = _plan_device(slot, rank)
    _mark_rank(rank)
    return device


def device_plan_root() raises -> Int:
    """The calling thread's plan's first device (its root), -1 without a plan."""
    var slot = _own_slot()
    if slot < 0:
        return -1
    var table = _plan_table()
    return Int(_words(table[].ids)[slot * DEVICE_PLAN_MAX_DEVICES])


def device_plan_setting(name: String, default: String = "") raises -> String:
    """What an owner loop parses as its device count: the plan's size as text
    while a plan driving `name`'s family is bound to the calling thread, else
    `getenv(name, default)`."""
    var slot = _own_slot()
    if slot >= 0:
        var table = _plan_table()
        if (Int(_words(table[].masks)[slot]) & _family_bit(name)) != 0:
            return String(Int(_words(table[].counts)[slot]))
    return String(getenv(name, default))


struct DevicePlan(Copyable, Movable):
    """The device plan one family entry runs under, read once from the
    calling thread (`device_plan_current`) and passed down as a value.

    `setting`: the raw text an owner loop parses as its device count, exactly
    what `device_plan_setting(name, default)` returned at the entry.
    `count`: the device count that setting names, >= 1 ("" or text that is
    not a positive integer counts as 1; callers that need the old parse
    error parse `setting` themselves).
    `root`: `device_plan_root()` at the entry: the plan's first device, -1
    when no plan was bound (use `device(0)` for the device of rank 0).
    `devices`: the device of each rank in plan order: the bound plan's ids,
    else 0..count-1 (without a plan the rank is the device).
    `bound`: whether a plan was bound to the thread at the entry.
    No floating-point value is read or written here."""

    var setting: String
    var count: Int
    var root: Int
    var devices: List[Int]
    var bound: Bool

    def __init__(out self, setting: String, count: Int, root: Int, var devices: List[Int], bound: Bool):
        self.setting = setting
        self.count = count
        self.root = root
        self.devices = devices^
        self.bound = bound

    def single(self) -> Bool:
        """Exactly one device: `setting == "" or setting == "1"`, the literal
        test `solver/impl/cd.mojo:1715` and `:1728` apply to
        `device_plan_setting("MOJOLEARN_SOLVER_DEVICE_COUNT")`. Same truth
        table as that string test, so a gate moved onto it keeps its bits."""
        return self.setting == "" or self.setting == "1"

    def device(self, rank: Int) raises -> Int:
        """The physical device of owner `rank`, as `device_plan_device(rank)`
        answered at the entry: the plan's entry while a plan was bound (an
        out-of-range rank raises the same error), else `rank`."""
        if not self.bound:
            return rank
        var count = len(self.devices)
        if rank < 0 or rank >= count:
            raise Error("device plan: owner rank " + String(rank) + " outside a plan of " + String(count) + " devices")
        # Every owner loop asks here where it opens this rank's context for
        # its shard: record the rank as having received work (`used`).
        _mark_rank(rank)
        return self.devices[rank]


def _setting_count(setting: String) -> Int:
    # "" (no setting) is one device; a positive integer is that many; any
    # other text counts as one here (callers that raised on it keep parsing
    # `DevicePlan.setting` themselves, so their behavior is unchanged).
    if setting == "":
        return 1
    try:
        var n = Int(setting)
        if n >= 1:
            return n
    except:
        pass
    return 1


def device_plan_current(name: String, default: String = "") raises -> DevicePlan:
    """The calling thread's device plan for the family whose count setting is
    `name` (e.g. "MOJOLEARN_SVM_DEVICE_COUNT"), read once through
    `device_plan_setting`, `device_plan_size`, `device_plan_device` and
    `device_plan_root`. Binding entries call this; code below them takes the
    value. `default` is what `device_plan_setting(name, default)` falls back
    to without a plan or variable."""
    var setting = device_plan_setting(name, default)
    var count = _setting_count(setting)
    var size = device_plan_size()
    var devices = List[Int]()
    if size > 0:
        # Read the ids without recording work (`_plan_device`, not
        # `device_plan_device`): building the value opens no rank.
        var slot = _own_slot()
        for r in range(size):  # driver-loop: over the plan's ranks
            devices.append(_plan_device(slot, r))
    else:
        for r in range(count):  # driver-loop: over the plan's ranks
            devices.append(r)
    return DevicePlan(setting, count, device_plan_root(), devices^, size > 0)


def plan_for_single() -> DevicePlan:
    """A one-device plan for callers outside any plan and any family entry:
    setting "", count 1, no plan bound (rank 0 is device 0)."""
    var devices = List[Int]()
    devices.append(0)
    return DevicePlan(String(""), 1, -1, devices^, False)


def _log_lock() raises -> MutPointer[Int64, MutUntrackedOrigin]:
    var table = _plan_table()
    var lock = _words(table[].log_lock)
    while True:  # driver-loop: spin on the placement log's lock word
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](lock, expected, Int64(1)):
            return lock
    return lock


def _log_plan(devices: List[Int]) raises:
    """Append one bound plan to the placement log (dropped when full)."""
    var table = _plan_table()
    var lock = _log_lock()
    var used = Int(_words(table[].log_used)[0])
    if used + 1 + len(devices) <= DEVICE_PLAN_LOG_WORDS:
        var log = _words(table[].log)
        log[used] = Int64(len(devices))
        for r in range(len(devices)):  # driver-loop: over the plan's ranks
            log[used + 1 + r] = Int64(devices[r])
        _words(table[].log_used)[0] = Int64(used + 1 + len(devices))
    Atomic.store[ordering=Ordering.RELEASE](lock, Int64(0))


def device_plan_log_take() raises -> List[List[Int]]:
    """The plans bound since the last take, in order, and an empty log."""
    var table = _plan_table()
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
