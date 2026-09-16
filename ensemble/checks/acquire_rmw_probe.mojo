# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the cross-block mutex actually establish an acquire edge? Probe, then decide.

    pixi run mojo run -I . ensemble/checks/acquire_rmw_probe.mojo

STANDING_ORDERS rule 9: a capability claim gets a probe before it constrains a
design. Two claims are at stake here and they are NOT the same claim.

  1. CAN an acquire ordering be used on a read-modify-write on this column?
     `ensemble/decisiontree/batched_levelalgo/split.mojo` DEVIATION 106 records
     that the Apple backend "rejects `acquire` success ordering on a
     compare-exchange" by name. Nothing in this tree measures AMD.

  2. DOES the shipped spelling lose updates? DEVIATION 106 claims "the
     synchronizes-with edge is a load-acquire observing a store-release". That
     holds when the acquire-load and the claiming compare-exchange observe the
     SAME release. They need not. A read-modify-write must read the latest value
     in the coherence order; a plain acquire-load must not. So a thread can exit
     the spin on a STALE zero from release R1, then win the RELAXED
     compare-exchange on the zero from a later release R2, having never
     performed an acquire that synchronizes with R2 -- leaving the previous
     holder's PLAIN store to the protected payload unordered against this
     thread's PLAIN read of it.

WHAT THIS PROBE DOES. One mutex, one payload word, `BLOCKS` blocks. Thread 0 of
each block takes the lock, reads the payload with a PLAIN load, adds one, writes
it back with a PLAIN store, and releases. The critical section is deliberately a
non-atomic read-modify-write, because that is exactly the shape of
`_publish_to_global`: it reads `split[node]`, merges, and stores it back.

With a sound edge the payload ends at exactly BLOCKS. Every lost update shows up
as a shortfall, and the shortfall is the number of times the protection failed.

THE ARMS
  relaxed  the spelling that ships today (ACQUIRE load spin + weak RELAXED claim)
  acquire  the same protocol with the claim itself carrying ACQUIRE

A COMPILE failure on the acquire arm is a RESULT, not an error: it answers claim
1 as "no" for this column and changes the shape of any fix. It is therefore
behind `-D MOJOLEARN_PROBE_ACQUIRE_CAS=1`, so the relaxed arm still builds and
runs on a column that cannot legalize the other.

AND IT SABOTAGES ITS OWN COMPARISON. A probe whose check cannot fail reports OK
for a kernel that did nothing, so the last arm runs an UNLOCKED kernel whose
shortfall must be large. If the unlocked arm comes out exact, the machine is not
contending and NOTHING above this line means anything.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

from checks.kernel_matrix import TARGET_COLUMN, column_name

comptime BLOCK = 64
comptime BLOCKS = 512
comptime SPINS = 1 << 22
#: The acquire arm is opt-in so a column that cannot legalize it still builds.
comptime PROBE_ACQUIRE = is_defined["MOJOLEARN_PROBE_ACQUIRE_CAS"]()


def _relaxed_kernel(
    mutex: MutPointer[Int32, MutAnyOrigin],
    payload: MutPointer[Int32, MutAnyOrigin],
):
    """THE SHIPPED SPELLING: acquire-load spin, weak RELAXED claim, release store."""
    if Int(thread_idx.x) != 0:
        return
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
    # PLAIN read-modify-write under the lock -- the shape of _publish_to_global.
    var v = payload.unsafe_load(0)
    payload.unsafe_store(0, v + Int32(1))
    Atomic.store[ordering = Ordering.RELEASE](mutex, Int32(0))


def _unlocked_kernel(
    mutex: MutPointer[Int32, MutAnyOrigin],
    payload: MutPointer[Int32, MutAnyOrigin],
):
    """THE SABOTAGE: no lock at all. Its shortfall proves the cell contends."""
    if Int(thread_idx.x) != 0:
        return
    var v = payload.unsafe_load(0)
    payload.unsafe_store(0, v + Int32(1))


def _run(
    ctx: DeviceContext,
    name: StringSlice,
    unlocked: Bool,
) raises -> Int:
    var mtx = ctx.enqueue_create_buffer[DType.int32](1)
    var pay = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(mtx, Int32(0))
    ctx.enqueue_memset(pay, Int32(0))
    ctx.synchronize()
    if unlocked:
        ctx.enqueue_function[_unlocked_kernel](
            mtx.unsafe_ptr(), pay.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
        )
    else:
        ctx.enqueue_function[_relaxed_kernel](
            mtx.unsafe_ptr(), pay.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
        )
    ctx.synchronize()
    var host = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=host, src_buf=pay)
    ctx.synchronize()
    var got = Int(host.unsafe_ptr().unsafe_load(0))
    print(
        String(name)
        + ": got "
        + String(got)
        + " want "
        + String(BLOCKS)
        + " shortfall "
        + String(BLOCKS - got)
    )
    _ = mtx^
    _ = pay^
    _ = host^
    return got


comptime if PROBE_ACQUIRE:

    def _acquire_kernel(
        mutex: MutPointer[Int32, MutAnyOrigin],
        payload: MutPointer[Int32, MutAnyOrigin],
    ):
        """THE PROPOSED SPELLING: the CLAIM itself carries ACQUIRE."""
        if Int(thread_idx.x) != 0:
            return
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
        payload.unsafe_store(0, v + Int32(1))
        Atomic.store[ordering = Ordering.RELEASE](mutex, Int32(0))

    def _run_acquire(ctx: DeviceContext) raises -> Int:
        var mtx = ctx.enqueue_create_buffer[DType.int32](1)
        var pay = ctx.enqueue_create_buffer[DType.int32](1)
        ctx.enqueue_memset(mtx, Int32(0))
        ctx.enqueue_memset(pay, Int32(0))
        ctx.synchronize()
        ctx.enqueue_function[_acquire_kernel](
            mtx.unsafe_ptr(), pay.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
        )
        ctx.synchronize()
        var host = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_buf=host, src_buf=pay)
        ctx.synchronize()
        var got = Int(host.unsafe_ptr().unsafe_load(0))
        print(
            "acquire: got "
            + String(got)
            + " want "
            + String(BLOCKS)
            + " shortfall "
            + String(BLOCKS - got)
        )
        _ = mtx^
        _ = pay^
        _ = host^
        return got


def main() raises:
    var ctx = DeviceContext()
    print("column " + String(column_name(TARGET_COLUMN)))
    print("blocks " + String(BLOCKS) + " block_dim " + String(BLOCK))
    print("acquire_arm_compiled " + String(PROBE_ACQUIRE))

    var relaxed = _run(ctx, "relaxed", False)

    var acquired = -1
    comptime if PROBE_ACQUIRE:
        acquired = _run_acquire(ctx)

    # THE SABOTAGE. Without it, "relaxed was exact" is indistinguishable from
    # "nothing contended", and the whole probe would report OK for a cell that
    # never raced.
    var unlocked = _run(ctx, "unlocked(SABOTAGE)", True)

    print("")
    if unlocked >= BLOCKS:
        print(
            "CONTROL FAILED: the UNLOCKED arm reached "
            + String(unlocked)
            + " of "
            + String(BLOCKS)
            + ", so this cell is not contending and NOTHING above is evidence."
        )
        return
    print(
        "control OK: the unlocked arm lost "
        + String(BLOCKS - unlocked)
        + " updates, so the cell contends."
    )
    if relaxed < BLOCKS:
        print(
            "RELAXED CLAIM LOSES UPDATES: "
            + String(BLOCKS - relaxed)
            + " of "
            + String(BLOCKS)
            + ". The shipped mutex does not protect a plain"
            + " read-modify-write on this column."
        )
    else:
        print(
            "relaxed claim was exact on this run. That is evidence it works"
            + " HERE, not proof the edge is formally established; a rare"
            + " window may need more contention to show."
        )
    comptime if PROBE_ACQUIRE:
        if acquired >= BLOCKS:
            print("ACQUIRE CLAIM EXACT: the acquire-ordering RMW compiles, runs, and loses nothing.")
        else:
            print(
                "ACQUIRE CLAIM ALSO LOST "
                + String(BLOCKS - acquired)
                + ": the ordering is not the whole story."
            )
