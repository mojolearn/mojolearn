# SPDX-License-Identifier: Apache-2.0
"""THE ONE HOST THREAD SPLIT: `sync_parallelize` with a PINNED floating-point
environment (lane cpu, 2026-09-27, DEVIATION 5900).

EVERY host thread split in the tree goes through this module.
`tools/check_host_parallel_sites.py` (run by `pixi run check-host-parallel`)
refuses any `.mojo` file outside it that calls `sync_parallelize` directly.
This module also absorbs lane/algos-linear's `core/host_fp_env.mojo`
(`host_ieee_fp_enter` / `host_ieee_fp_leave` around each task body, main
0b7b6d5c1): `host_parallelize` below does the same job at the split rather
than inside each body, so that file and its calls in
`core/classical_host_predict.mojo` and `glm/estimator.mojo` were removed
when the two merged, and no second environment module is ever added.

MEASURED, NOT ASSUMED. On an x86 RunPod box (AMD EPYC 7352, the pinned
toolchain) the calling thread's MXCSR reads 0x1fa0 (the IEEE default) and
every worker thread `sync_parallelize` dispatches onto reads 0x9fe0: the
Mojo runtime's workers run with FTZ (bit 15) and DAZ (bit 6) set. A host
path therefore computed a subnormal result as a subnormal when its rows ran
on the calling thread and as a signed zero when they ran on a worker. The
bits depended on the THREAD COUNT and on which rows a task owned:
`LogisticRegression.predict_proba` (float64 `1 / (1 + exp(709.39))` =
8.22e-309) read DIVERGENT CPU against CUDA on the `dupes` fixture at the
default thread count and IDENTICAL at MOJOLEARN_CPU_THREADS=1, and the
GPU binding's serial host link agreed with the one-thread run.

THE MOVE IS PIN. There are exactly two entries:

- `host_parallelize(func, n)`: every task runs in the CALLING thread's
  environment. It reads MXCSR (x86) or FPCR (Arm) on the caller; each task
  installs it, runs the body and restores the worker's own value, so the
  runtime's pool is left as it was found. This is the entry for every host
  loop. The caller's environment is the one a one-task (serial) run uses,
  so a split's bits no longer depend on the task count.
- `host_parallelize_pool_env(func, n)`: the tasks keep the runtime WORKER's
  environment (FTZ+DAZ on x86). Only the GBDT fit's host regions use it
  (`gbdt/train.mojo`, `gbdt/resident_model.mojo`, `gbdt/host/gbdt_oracle.mojo`):
  every recorded GBDT column (CUDA, AMD, Apple and CPU) was computed with
  their border search and staging on pool workers, and pinning the caller's
  environment there moved 74 `denormal`-fixture cells of the gbdt,
  cross-val and saved-model lanes on the CUDA column (measured at 762f811cc).
  Moving GBDT to the caller's environment is a column re-record and is the
  trees lane's call (the GBDT environment question). Its bits do not depend on the task count: a pool task runs
  with FTZ+DAZ even at n = 1. OPEN, named: gbdt_oracle's serial small-fit
  arm (`n_rows * n_features < 2^18`) runs on the calling thread.

Float32 arithmetic spelled through `ftz` (`checks/numerics.mojo`, an integer
test) reads the same under either mode; what the pin fixes is everything
that is not: float64 host arithmetic (the probability links,
`portable_exp64`/`portable_log64`) and any float32 operation whose operand
or result can be subnormal before its flush. A target with neither register
(none today) runs the tasks as `sync_parallelize` does.

CELLS THAT MOVED (DEVIATION 5900; each was thread-count dependent before,
none moved at MOJOLEARN_CPU_THREADS=1): `logistic-unpenalized-no-intercept`
on `dupes` (infer, batch), CPU column only; it now equals the CUDA column.

`host_fp_env` / `host_fp_env_set` are exported for a caller that owns its
own threads.

NESTED REGIONS (lane gbdt-host-par, 2026-11-21). What `sync_parallelize`
does when it is called from inside a task of the same pool, read from the
published sources (`max/algorithm/backend/cpu/parallelize.mojo`, the
runtime's `TaskGroup.wait`), not from a run: with more than one work item
it never runs them inline. It queues every item on the one process pool
and BLOCKS the calling worker in the runtime's wait until they are done.
So an inner split from a pool task parks that worker; when the outer split
holds every worker (one task per core, the member set's plan) no worker is
left to run the inner items unless the runtime's wait helps, which nothing
documents, and a pool that does help still runs outer and inner tasks
interleaved on the same threads. Only one work item runs on the caller.

`host_in_region()` is how a split learns that it may be inside a pool
task: both entries count the regions that are open on the pool (a split
with more than one task, from the launch to the join) in one process-wide
word shared by every binding. A caller that sizes its own split asks it
and takes ONE task when a region is open, which both entries run on the
calling thread, in the environment the outer task already installed (the
caller's, the one the inner tasks would have been pinned to). The count is
per process, not per thread: it is never zero inside a pool task (no
deadlock, never more runnable threads than cores), and a top-level split
on another thread that overlaps an open region also runs on its caller
(serial for that one split, the same bits).
"""
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget
from std.memory import stack_allocation

from std.atomic import Atomic, Ordering

from max.algorithm import sync_parallelize

from core.launch_log import activity_host_split
from core.shared_slot import SharedSlot


struct _HostRegions(Defaultable, Movable):
    """`open[0]`: how many host splits of more than one task are between
    their launch and their join, in the whole process (module note, NESTED
    REGIONS). One word, read and written with atomic operations only."""

    var open: List[Int64]

    def __init__(out self):
        self.open = List[Int64](length=1, fill=Int64(0))


comptime _HOST_REGIONS = SharedSlot[_HostRegions, "MojolearnHostRegionsV1"]


def _host_regions_word() -> Int:
    """The address of the open-region word, or 0 when the process slot could
    not be made (then no region is counted and none is seen)."""
    try:
        var st = _HOST_REGIONS.get_or_create_ptr()
        return Int(st[].open.unsafe_ptr())
    except:
        return 0


@always_inline
def _host_region_step(word: Int, step: Int64):
    if word != 0:
        _ = Atomic[Int64].fetch_add(
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=word), step
        )


def host_in_region() -> Bool:
    """True while any host split of more than one task is open in the
    process, so always True on a thread that is running a pool task. A
    caller that sizes its own split takes one task when it is True (module
    note, NESTED REGIONS). One slot read and one atomic load per call: ask
    once per split, never per row."""
    var word = _host_regions_word()
    if word == 0:
        return False
    return Atomic.load[ordering=Ordering.ACQUIRE](
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=word)
    ) != Int64(0)


@always_inline
def host_fp_env() -> UInt64:
    """The calling thread's floating-point control word: MXCSR on x86,
    FPCR on Arm, 0 elsewhere."""
    comptime if CompilationTarget.is_x86():
        var p = stack_allocation[1, UInt32]()
        p.store(0, UInt32(0))
        llvm_intrinsic["llvm.x86.sse.stmxcsr", NoneType](p)
        return UInt64(p.load(0))
    elif CompilationTarget.has_neon():
        return UInt64(llvm_intrinsic["llvm.aarch64.get.fpcr", Int64]())
    else:
        return UInt64(0)


@always_inline
def host_fp_env_set(v: UInt64):
    """Install a control word `host_fp_env` read (no-op elsewhere). The
    write is an intrinsic with side effects on memory, so the task's loads
    and the arithmetic that reads them stay after it."""
    comptime if CompilationTarget.is_x86():
        var p = stack_allocation[1, UInt32]()
        p.store(0, UInt32(v))
        llvm_intrinsic["llvm.x86.sse.ldmxcsr", NoneType](p)
    elif CompilationTarget.has_neon():
        llvm_intrinsic["llvm.aarch64.set.fpcr", NoneType](Int64(v))


def host_parallelize[FuncType: def(Int) -> None](ref func: FuncType, n: Int):
    """`sync_parallelize(func, n)`, every task in the caller's floating-point
    environment (DEVIATION 5900). The work split, the task count and the
    join are `sync_parallelize`'s, unchanged.

    One task (or none) runs on the calling thread without touching the pool
    (lane perf-host-grain): the caller's environment is the one every task
    is pinned to, so the bits are the pool's."""
    if n <= 0:
        return
    # lane fx-verify-trace: the activity record of a verification call counts
    # this split (one slot read and one Bool test when no verification asked;
    # a split is never inside a loop over rows or elements, it IS that loop)
    activity_host_split(n)
    if n == 1:
        func(0)
        return
    # THE NESTED RULE, FOR EVERY CALLER (orchestrator, 2026-11-25): while a
    # region is open the tasks run one after another on the calling thread,
    # which is already in the environment they would be pinned to. An inner
    # split queued on the same pool parks its worker (module note, NESTED
    # REGIONS), and the member route now runs fits of every family inside
    # pool tasks, so the rule cannot be left to each family's own sizing.
    # One slot read and one atomic load per split, never per row.
    if host_in_region():
        for i in range(n):  # driver-loop: over the tasks of one nested split
            func(i)
        return
    var env = host_fp_env()

    def task(i: Int) {imm func, imm env}:
        var saved = host_fp_env()
        host_fp_env_set(env)
        func(i)
        host_fp_env_set(saved)

    # the region is open from the launch to the join (NESTED REGIONS);
    # sync_parallelize does not raise, so the word is always stepped back
    var regions = _host_regions_word()
    _host_region_step(regions, Int64(1))
    sync_parallelize(task, n)
    _host_region_step(regions, Int64(-1))


def host_nested_units(units: Int) -> Int:
    """The task count a caller's own rule gave, under the nested rule
    (module note, NESTED REGIONS): one while a region is open, else `units`
    (at least one). For a split whose tasks each take a share of a range, so
    one task takes all of it."""
    if host_in_region():
        return 1
    return max(units, 1)


def host_parallelize_items[FuncType: def(Int) -> None](ref func: FuncType, n: Int):
    """`host_parallelize(func, n)` for a split whose task count IS its item
    count (one column, one permutation per task), under the nested rule:
    while a region is open the items run one after another on the calling
    thread, which is already in the environment the tasks would be pinned
    to (module note, NESTED REGIONS)."""
    # `host_parallelize` applies the nested rule itself (2026-11-25); this name
    # stays for the callers that state "one item per task".
    host_parallelize(func, n)


def host_parallelize_pool_env[FuncType: def(Int) -> None](ref func: FuncType, n: Int):
    """`sync_parallelize(func, n)` with the runtime WORKER's environment
    (FTZ+DAZ on x86), for the GBDT fit's host regions only; see the module
    note. Every other host loop uses `host_parallelize`. While another
    region is open (the caller may be a pool task: a member's fit) the items
    run one after another on the calling thread, in the same environment
    (module note, NESTED REGIONS)."""
    if n > 0:
        activity_host_split(n)
    if n <= 1:
        # one work item runs on the caller with the pool's flush-to-zero
        # scope (sync_parallelize's own one-item arm); no region is open
        sync_parallelize(func, n)
        return
    if host_in_region():
        # NESTED REGIONS: a region is open, so this caller may be a pool
        # task. Every item runs on the calling thread, each through
        # sync_parallelize's one-item arm, which runs it on the caller with
        # the pool's flush-to-zero scope: the bits of a pool task, no
        # second launch on the pool.
        for i in range(n):  # driver-loop: over the work items of one nested split

            def one(unit: Int) {imm func, imm i}:
                func(i)

            sync_parallelize(one, 1)
        return
    var regions = _host_regions_word()
    _host_region_step(regions, Int64(1))
    sync_parallelize(func, n)
    _host_region_step(regions, Int64(-1))


comptime HOST_NARROW_CHUNK = 65536


def host_narrow_codes_i32(src_address: Int, dst_address: Int, count: Int):
    """Narrow `count` int64 class codes at `src_address` into int32 at
    `dst_address`, split across the host pool in chunks of
    `HOST_NARROW_CHUNK` words (lane fx-audit, audit-neural finding 1: the
    member predicts' serial narrowing loop). A count of one chunk or less runs
    on the calling thread (`host_parallelize` with one task), so a small
    predict pays no pool launch; the chunk size is a memory-bandwidth grain,
    not a shape rule. Codes are class indices, so the narrowing is exact."""
    var units = (count + HOST_NARROW_CHUNK - 1) // HOST_NARROW_CHUNK

    def task(unit: Int) {imm src_address, imm dst_address, imm count}:
        var src = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=src_address)
        var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst_address)
        var stop = min(count, (unit + 1) * HOST_NARROW_CHUNK)
        for i in range(unit * HOST_NARROW_CHUNK, stop):  # native data loop (one host-pool chunk of codes, not a driver loop)
            dst[i] = Int32(src[i])

    host_parallelize(task, units)


def host_widen_index_i64(src_address: Int, dst_address: Int, count: Int):
    """Widen `count` int32 row numbers at `src_address` into int64 at
    `dst_address`, split across the host pool in chunks of
    `HOST_NARROW_CHUNK` words, as `host_narrow_codes_i32` (lane fx-audit 10:
    a meta-estimator's int32 draws handed to a member that reads int64 row
    numbers). Exact."""
    var units = (count + HOST_NARROW_CHUNK - 1) // HOST_NARROW_CHUNK

    def task(unit: Int) {imm src_address, imm dst_address, imm count}:
        var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=src_address)
        var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst_address)
        var stop = min(count, (unit + 1) * HOST_NARROW_CHUNK)
        for i in range(unit * HOST_NARROW_CHUNK, stop):  # native data loop (one host-pool chunk of row numbers, not a driver loop)
            dst[i] = Int64(src[i])

    host_parallelize(task, units)
