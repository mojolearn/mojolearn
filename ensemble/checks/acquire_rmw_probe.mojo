# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the cross-block mutex actually establish an acquire edge? Probe, then decide.

    pixi run mojo run -I . ensemble/checks/acquire_rmw_probe.mojo
    pixi run mojo run -I . -D MOJOLEARN_PROBE_ACQUIRE_CAS=1 ensemble/checks/acquire_rmw_probe.mojo

STANDING_ORDERS rule 9: a capability claim gets a probe before it constrains a
design. Two claims are at stake and they are NOT the same claim.

  1. CAN an acquire ordering be used on a read-modify-write on this column?
     **ANSWERED for gfx942, 2026-09-16: YES.** The acquire arm compiled and ran
     (`acquire_arm_compiled True`, exact), and `_mojolearn_rf.so` built with
     `-D MOJOLEARN_RF_ACQUIRE_CAS=1`. Recorded as `column_has_acquire_rmw` in
     `ensemble/checks/atomic_matrix.mojo`.

  2. DOES the shipped spelling lose updates? `split.mojo` DEVIATION 106 claims
     "the synchronizes-with edge is a load-acquire observing a store-release".
     That holds when the acquire-load and the claiming compare-exchange observe
     the SAME release. They need not: an RMW must read the latest value in the
     coherence order, a plain acquire-load must not. A thread can leave the spin
     on a STALE zero from release R1, then win the RELAXED claim on the zero
     from a later release R2, having performed no acquire that synchronizes with
     R2 -- leaving the previous holder's PLAIN store to the protected payload
     unordered against this thread's PLAIN read of it.

WHAT THIS PROBE DOES. One mutex, one payload word, `BLOCKS` blocks, each taking
the lock `ROUNDS` times. Under the lock it does a PLAIN read-modify-write --
the shape of `_publish_to_global`, which reads `split[node]`, merges, and stores
it back. A sound edge ends at exactly `BLOCKS * ROUNDS`; every lost update is a
shortfall.

WHY `ROUNDS` AND `HOLD` EXIST, and this is the correction that produced them.
The first version acquired ONCE per block with a two-instruction critical
section, and measured the shipped spelling as EXACT (512/512 on gfx942) while
its unlocked sabotage lost 508. That is a NULL, not a clearance: with one
acquisition the spin almost never loops, and the hypothesised window REQUIRES it
to loop -- the load must observe one release while the claim lands on another.
`ROUNDS` makes the spin contend repeatedly; `HOLD` widens the critical section
so a competing release can land inside it.

`HOLD` MUST NOT BE DELETABLE. A hold loop whose result is discarded is exactly
what a compiler removes, which would silently restore the narrow section and
reproduce the old null for a reason unrelated to the hypothesis. So the hold
accumulates FROM the payload and is STORED BACK into a sink the host reads, and
the host checks the sink is non-zero. If the sink is zero the hold was elided
and the run says so instead of reporting a shortfall.

AND IT SABOTAGES ITS OWN COMPARISON. A probe whose check cannot fail reports OK
for a kernel that did nothing, so the last arm runs UNLOCKED at the same
`ROUNDS` and its shortfall must be large. If the unlocked arm comes out exact,
the cell is not contending and NOTHING above it means anything.
"""

from std.atomic import Atomic, Ordering
from max.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext

from checks.kernel_matrix import TARGET_COLUMN, column_name

comptime BLOCK = 64
comptime BLOCKS = 512
comptime ROUNDS = 64
comptime HOLD = 256
comptime SPINS = 1 << 22
comptime WANT = BLOCKS * ROUNDS
#: The acquire arm is opt-in so a column that cannot legalize it still builds.
comptime PROBE_ACQUIRE = is_defined["MOJOLEARN_PROBE_ACQUIRE_CAS"]()


@always_inline
def _hold(seed: Int32) -> Int32:
    """A hold whose result is OBSERVED, so it cannot be optimised away."""
    var acc = seed
    for _h in range(HOLD):
        acc = acc ^ (acc << 1) ^ Int32(1)
    return acc


def _relaxed_kernel(
    mutex: MutPointer[Int32, MutAnyOrigin],
    payload: MutPointer[Int32, MutAnyOrigin],
    sink: MutPointer[Int32, MutAnyOrigin],
):
    """THE SHIPPED SPELLING: acquire-load spin, weak RELAXED claim, release store."""
    if Int(thread_idx.x) != 0:
        return
    var local = Int32(0)
    for _r in range(ROUNDS):
        var guard = 0
        while guard < SPINS:
            guard += 1
            if Atomic.load[ordering = Ordering.ACQUIRE](mutex) != Int32(0):
                continue
            var expected = Int32(0)
            if Atomic.compare_exchange[
                success_ordering = Ordering.RELAXED,
                failure_ordering = Ordering.RELAXED,
                weak=True,
            ](mutex, expected, Int32(1)):
                break
        # PLAIN read-modify-write under the lock, widened by an observed hold.
        var v = payload.unsafe_load(0)
        local = local ^ _hold(v)
        payload.unsafe_store(0, v + Int32(1))
        Atomic.store[ordering = Ordering.RELEASE](mutex, Int32(0))
    _ = Atomic.fetch_add(sink.unsafe_offset(0), local | Int32(1))


def _acquire_kernel(
    mutex: MutPointer[Int32, MutAnyOrigin],
    payload: MutPointer[Int32, MutAnyOrigin],
    sink: MutPointer[Int32, MutAnyOrigin],
):
    """THE PROPOSED SPELLING: identical but for the ordering ON THE CLAIM."""
    if Int(thread_idx.x) != 0:
        return
    var local = Int32(0)
    for _r in range(ROUNDS):
        var guard = 0
        while guard < SPINS:
            guard += 1
            if Atomic.load[ordering = Ordering.ACQUIRE](mutex) != Int32(0):
                continue
            var expected = Int32(0)
            if Atomic.compare_exchange[
                success_ordering = Ordering.ACQUIRE,
                failure_ordering = Ordering.RELAXED,
                weak=True,
            ](mutex, expected, Int32(1)):
                break
        var v = payload.unsafe_load(0)
        local = local ^ _hold(v)
        payload.unsafe_store(0, v + Int32(1))
        Atomic.store[ordering = Ordering.RELEASE](mutex, Int32(0))
    _ = Atomic.fetch_add(sink.unsafe_offset(0), local | Int32(1))


def _unlocked_kernel(
    mutex: MutPointer[Int32, MutAnyOrigin],
    payload: MutPointer[Int32, MutAnyOrigin],
    sink: MutPointer[Int32, MutAnyOrigin],
):
    """THE SABOTAGE: the same shape at the same ROUNDS, with NO lock."""
    if Int(thread_idx.x) != 0:
        return
    var local = Int32(0)
    for _r in range(ROUNDS):
        var v = payload.unsafe_load(0)
        local = local ^ _hold(v)
        payload.unsafe_store(0, v + Int32(1))
    _ = Atomic.fetch_add(sink.unsafe_offset(0), local | Int32(1))


def _report(name: StringSlice, got: Int, sink: Int) -> Int:
    print(
        String(name)
        + ": got "
        + String(got)
        + " want "
        + String(WANT)
        + " shortfall "
        + String(WANT - got)
        + " sink "
        + String(sink)
    )
    return got


def _run_relaxed(ctx: DeviceContext) raises -> Int:
    var mtx = ctx.enqueue_create_buffer[DType.int32](1)
    var pay = ctx.enqueue_create_buffer[DType.int32](1)
    var snk = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(mtx, Int32(0))
    ctx.enqueue_memset(pay, Int32(0))
    ctx.enqueue_memset(snk, Int32(0))
    ctx.synchronize()
    ctx.enqueue_function[_relaxed_kernel](
        mtx.unsafe_ptr(), pay.unsafe_ptr(), snk.unsafe_ptr(),
        grid_dim=BLOCKS, block_dim=BLOCK,
    )
    ctx.synchronize()
    var hp = ctx.enqueue_create_host_buffer[DType.int32](1)
    var hs = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hp, src_buf=pay)
    ctx.enqueue_copy(dst_buf=hs, src_buf=snk)
    ctx.synchronize()
    var got = _report("relaxed", Int(hp.unsafe_ptr().unsafe_load(0)),
                      Int(hs.unsafe_ptr().unsafe_load(0)))
    _ = mtx^
    _ = pay^
    _ = snk^
    _ = hp^
    _ = hs^
    return got


def _run_unlocked(ctx: DeviceContext) raises -> Int:
    var mtx = ctx.enqueue_create_buffer[DType.int32](1)
    var pay = ctx.enqueue_create_buffer[DType.int32](1)
    var snk = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(mtx, Int32(0))
    ctx.enqueue_memset(pay, Int32(0))
    ctx.enqueue_memset(snk, Int32(0))
    ctx.synchronize()
    ctx.enqueue_function[_unlocked_kernel](
        mtx.unsafe_ptr(), pay.unsafe_ptr(), snk.unsafe_ptr(),
        grid_dim=BLOCKS, block_dim=BLOCK,
    )
    ctx.synchronize()
    var hp = ctx.enqueue_create_host_buffer[DType.int32](1)
    var hs = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hp, src_buf=pay)
    ctx.enqueue_copy(dst_buf=hs, src_buf=snk)
    ctx.synchronize()
    var got = _report("unlocked(SABOTAGE)", Int(hp.unsafe_ptr().unsafe_load(0)),
                      Int(hs.unsafe_ptr().unsafe_load(0)))
    _ = mtx^
    _ = pay^
    _ = snk^
    _ = hp^
    _ = hs^
    return got


def _run_acquire(ctx: DeviceContext) raises -> Int:
    """Guarded at the CALL, not at the definition: Mojo rejects a module-scope
    `comptime if`, which is what failed to parse on leg 13.

    KNOWN LIMITATION, stated rather than papered over: `_acquire_kernel` is
    DEFINED unconditionally, so on a column that cannot legalize an acquire
    ordering on an RMW (Apple, per DEVIATION 106) this file may fail to build
    for a reason that has nothing to do with what it measures. On gfx942 the
    ordering is MEASURED to legalize, which is the column this probe is for."""
    comptime if not PROBE_ACQUIRE:
        return -1
    var mtx = ctx.enqueue_create_buffer[DType.int32](1)
    var pay = ctx.enqueue_create_buffer[DType.int32](1)
    var snk = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(mtx, Int32(0))
    ctx.enqueue_memset(pay, Int32(0))
    ctx.enqueue_memset(snk, Int32(0))
    ctx.synchronize()
    ctx.enqueue_function[_acquire_kernel](
        mtx.unsafe_ptr(), pay.unsafe_ptr(), snk.unsafe_ptr(),
        grid_dim=BLOCKS, block_dim=BLOCK,
    )
    ctx.synchronize()
    var hp = ctx.enqueue_create_host_buffer[DType.int32](1)
    var hs = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hp, src_buf=pay)
    ctx.enqueue_copy(dst_buf=hs, src_buf=snk)
    ctx.synchronize()
    var got = _report("acquire", Int(hp.unsafe_ptr().unsafe_load(0)),
                      Int(hs.unsafe_ptr().unsafe_load(0)))
    _ = mtx^
    _ = pay^
    _ = snk^
    _ = hp^
    _ = hs^
    return got


def main() raises:
    var ctx = DeviceContext()
    print("column " + String(column_name(TARGET_COLUMN)))
    print(
        "blocks "
        + String(BLOCKS)
        + " block_dim "
        + String(BLOCK)
        + " rounds "
        + String(ROUNDS)
        + " hold "
        + String(HOLD)
        + " want "
        + String(WANT)
    )
    print("acquire_arm_compiled " + String(PROBE_ACQUIRE))

    var relaxed = _run_relaxed(ctx)
    var acquired = _run_acquire(ctx)
    var unlocked = _run_unlocked(ctx)

    print("")
    # THE SABOTAGE. Without it, "relaxed was exact" is indistinguishable from
    # "nothing contended", and the probe would report OK for a cell that never
    # raced.
    if unlocked >= WANT:
        print(
            "CONTROL FAILED: the UNLOCKED arm reached "
            + String(unlocked)
            + " of "
            + String(WANT)
            + ", so this cell is not contending and NOTHING above is evidence."
        )
        return
    print(
        "control OK: the unlocked arm lost "
        + String(WANT - unlocked)
        + " updates, so the cell contends."
    )
    if relaxed < WANT:
        print(
            "RELAXED CLAIM LOSES UPDATES: "
            + String(WANT - relaxed)
            + " of "
            + String(WANT)
            + ". The shipped mutex does not protect a plain"
            + " read-modify-write on this column."
        )
    else:
        print(
            "relaxed claim was EXACT on this run. A NULL, not a clearance:"
            + " evidence it holds under THIS contention pattern, not proof the"
            + " edge is formally established."
        )
    if PROBE_ACQUIRE:
        if acquired >= WANT:
            print("ACQUIRE CLAIM EXACT: the acquire-ordering RMW compiles, runs, loses nothing.")
        else:
            print(
                "ACQUIRE CLAIM ALSO LOST "
                + String(WANT - acquired)
                + ": the ordering is not the whole story."
            )
