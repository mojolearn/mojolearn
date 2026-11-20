# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Contention check of the production mutex helper, with a REAL stock arm.

WHAT IT DETECTS, AND WHY THE FINAL COUNT IS THE ORDERING DETECTOR. Every holder reads the
count out of the payload, widens the critical section, writes count + 1 back, and
releases. If the ordering edge is missing, a holder can be served a payload line that
predates the previous holder's write, read a STALE count, and write back a value that
erases the previous holder's increment. The two payload words are written together and
read together, so a stale read is SELF-CONSISTENT and the checksum still passes. The lost
increment shows up only in the FINAL COUNT. That is the same shape as the random forest's
lost candidate, which is the point.

WHY THIS VERSION CONTENDS HARDER THAN THE FIRST ONE. The 2026-09-16 gfx942 run of the
earlier version passed with the ordering hole FULLY PRESENT (the repair it was built with
emitted no instructions at all). Two blocks and 128 claims is not enough contention for
the spin loop to loop, and the hypothesised window needs it to. `lane/rf-score-weighted-
nondeterminism`'s leg 14 learned the same lesson from its own probe and rewrote it for the
same reason. BLOCKS and ROUNDS are parameters here so a leg can sweep them, and the
critical section carries a HOLD whose result is stored where the host reads it, so a
compiler cannot quietly narrow the section back.

A PASS OF THE REPAIRED ARM ALONE PROVES NOTHING. Run the stock arm
(`-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1`) on the same box and look at it first. If it does
not lose an increment, this check has not reached the window and its repaired-arm pass is
a NULL, not a clearance. Report it as a null.
"""
from std.atomic import Atomic, Ordering
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from core.device_mutex import claim_device_mutex

comptime CHECKSUM_KEY = Int32(0x12345678)


def contend[
    skip_write: Bool, available: Int, held: Int, rounds: Int, hold: Int
](
    mutex: MutPointer[Int32, MutAnyOrigin],
    payload: MutPointer[Int32, MutAnyOrigin],
):
    """One claimant per block. Thread 0 does the claiming, as at every production site."""
    if thread_idx.x != 0:
        return
    var sink = Int32(0)
    for _ in range(rounds):
        claim_device_mutex(mutex, Int32(available), Int32(held))

        var count = payload[unsafe_offset=0]
        if payload[unsafe_offset=1] != (count ^ CHECKSUM_KEY):
            # A TORN read. Self-consistent stale reads do not land here; they land in
            # the final count. This catches the other failure mode.
            payload[unsafe_offset=2] = Int32(1)

        # Widen the critical section so the other blocks' spins actually loop. The
        # result is stored, so this cannot be optimized away into nothing.
        for h in range(hold):
            sink = sink ^ (count + Int32(h) + Int32(block_idx.x))

        comptime if not skip_write:
            count += 1
            payload[unsafe_offset=0] = count
            payload[unsafe_offset=1] = count ^ CHECKSUM_KEY

        Atomic.store[ordering = Ordering.RELEASE](mutex, Int32(available))
    payload[unsafe_offset=3] = sink


def check[
    skip_write: Bool, available: Int, held: Int, blocks: Int, rounds: Int, hold: Int
](ctx: DeviceContext) raises -> Bool:
    var mutex = ctx.enqueue_create_buffer[DType.int32](1)
    var data = ctx.enqueue_create_buffer[DType.int32](4)
    var host = ctx.enqueue_create_host_buffer[DType.int32](4)
    host[0] = Int32(0)
    host[1] = CHECKSUM_KEY
    host[2] = Int32(0)
    host[3] = Int32(0)
    ctx.enqueue_memset(mutex, Int32(available))
    ctx.enqueue_copy(dst_buf=data, src_buf=host)
    ctx.enqueue_function[contend[skip_write, available, held, rounds, hold]](
        mutex.unsafe_ptr(), data.unsafe_ptr(), grid_dim=blocks, block_dim=32,
    )
    ctx.enqueue_copy(dst_buf=host, src_buf=data)
    ctx.synchronize()

    var want = Int32(blocks * rounds)
    var got = host[0]
    var ok = (
        got == want
        and host[1] == (want ^ CHECKSUM_KEY)
        and host[2] == Int32(0)
    )
    print(
        "mutex", available, "->", held,
        "blocks", blocks, "rounds", rounds, "hold", hold,
        "skip-write", skip_write,
        "count", got, "want", want, "lost", want - got,
        "torn", host[2], "accepted", ok,
    )
    _ = mutex^
    _ = data^
    _ = host^
    return ok


def main() raises:
    var ctx = DeviceContext()

    # The production state pairs: the forest and ExtraTrees `0 -> 1`, and the fused kNN
    # consumer's `-2 -> -1`.
    var failures = 0
    for _ in range(4):
        if not check[False, 0, 1, 64, 64, 64](ctx):
            failures += 1
        if not check[False, -2, -1, 64, 64, 64](ctx):
            failures += 1

    # A wider critical section and more claimants, which is where a missing edge is most
    # likely to show. Kept separate so a loss here is attributable.
    for _ in range(4):
        if not check[False, 0, 1, 128, 32, 256](ctx):
            failures += 1

    # SABOTAGE. The skip-write arm must be REJECTED, which is what makes the count
    # assertion above a check that can fail rather than a check that cannot.
    if check[True, 0, 1, 64, 64, 64](ctx):
        raise Error("skip-write sabotage was NOT detected; this check cannot fail")

    if failures != 0:
        # A stock arm is EXPECTED to land here. That is the result, not an error.
        print("LOST-UPDATE ARMS:", failures)
        raise Error("mutex handoff lost at least one update")
    print("PASS device_mutex: both state pairs, no lost updates, sabotage rejected")
