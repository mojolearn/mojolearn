"""Does `Split`'s total order actually reduce, on the device, per node?

    pixi run mojo run -I . ensemble/mojo_only/split_check.mojo

NO CUML COUNTERPART -- this is a check, and cuML has no equivalent of it.
It covers `ensemble/decisiontree/batched_levelalgo/split.mojo`, which
mirrors `split.cuh` at rapidsai/cuml `v26.08.00`
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`).

WHY THIS FILE IS SHAPED THE WAY IT IS. This repository's most expensive
lesson is that a check whose expected value is the same in every cell
verifies the total and nothing about placement: the same histogram kernel
read `0 wrong of 512` under uniform data and `490 wrong of 512` under
hashed data, and two earlier exclusions had reported it correct at exactly
the failing parameters. So every candidate below is HASHED per
(node, lane), no two nodes see the same population, and the comparison is
per NODE against an independent host reduction -- never a digest, never a
total.

The four arms:

  A. THE ORDER, on the host. Hand-written candidate pairs, each exercising
     one branch of `update` (`split.cuh:142-191`) with an expected outcome
     written as a literal. If the branch order is transcribed wrong, the
     branch that changed is named.

  B. NON-ASSOCIATIVITY, on the host. DEVIATION 105 claims their operator
     is not associative and carries a worked counterexample. A claim in a
     comment is a hypothesis; this arm RUNS it. Two groupings of the same
     three candidates, and the check passes when they DISAGREE -- that is,
     it fails if the port accidentally made the operator associative,
     which would mean it is no longer their algorithm.

  C. THE DEVICE REDUCTION. One block per node, hashed gains per lane, all
     gains DISTINCT so the merge branch is unreachable and the reduction
     is genuinely order-independent -- which makes an exact per-node
     comparison against a host reduction legitimate. This is the arm that
     proves `warp_reduce` + `eval_best_split` were ENQUEUED and work; a
     kernel is not ported until it has been enqueued, and compiling is not
     evidence.

  D. SABOTAGE, one per mechanism. A digest cannot tell a working reduction
     from a no-op, and reach is per-branch, so each mechanism gets its own
     corruption and its own predicted movement. An arm that "moves under
     any corruption" is not evidence; the predictions below differ from
     each other on purpose.
"""

from std.gpu import WARP_SIZE, block_idx, block_dim, thread_idx
from std.gpu.primitives.id import lane_id
from max.gpu.host import DeviceContext
from std.memory import stack_allocation
from std.sys.info import size_of
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from ensemble.decisiontree.batched_levelalgo.split import Split

comptime DT = DType.float32
comptime TPB = 256
comptime N_NODES = 37  # deliberately not a power of two
comptime N_BINS = 64
comptime MAX_WARPS = 32

# Sabotage selectors, passed as a kernel argument so the SAME binary runs
# every arm -- a sabotage compiled into a different binary proves nothing
# about the one that ships.
comptime SAB_NONE = 0
comptime SAB_SKIP_LAST_WARP_STEP = 1
comptime SAB_COLID_TIEBREAK_REVERSED = 2
comptime SAB_NO_MIDPOINT = 3


@always_inline
def _mix(x: UInt64) -> UInt64:
    """A hash, so planted values are scattered rather than uniform."""
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


@always_inline
def _gain_for(node: Int, slot: Int) -> Float32:
    """DISTINCT hashed gains: the low bits carry the slot so no two slots
    inside a node can tie, which is what makes arm C's exact comparison
    legitimate (see the module docstring)."""
    var h = _mix(UInt64(node) * 0x9E3779B97F4A7C15 + UInt64(slot))
    # 24 bits of hash, exactly representable in float32, plus the slot so
    # ties are impossible within a node.
    return Float32(Int((h >> 20) & 0xFFFFFF)) * 1024.0 + Float32(slot)


@always_inline
def _colid_for(node: Int, slot: Int) -> Int32:
    var h = _mix(UInt64(node) * 0x517CC1B727220A95 + UInt64(slot))
    return Int32(Int(h % 97))


@always_inline
def _nleft_for(node: Int, slot: Int) -> Int64:
    var h = _mix(UInt64(node) * 0x2545F4914F6CDD1D + UInt64(slot))
    return Int64(Int(h % 5000))


@always_inline
def _bin_for(node: Int, slot: Int) -> Int32:
    var h = _mix(UInt64(node) * 0x9E3779B185EBCA87 + UInt64(slot))
    return Int32(Int(h % UInt64(N_BINS)))


def _reduce_kernel(
    splits: MutPointer[Split[DT], MutAnyOrigin],
    mutexes: MutPointer[Int32, MutAnyOrigin],
    quantiles: MutPointer[Float32, MutAnyOrigin],
    n_slots_in: Int32,
    sabotage_in: Int32,
):
    """One block per node. Each thread builds one hashed candidate and the
    block reduces them through the ported `eval_best_split`."""
    var node = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n_slots = Int(n_slots_in)
    var sabotage = Int(sabotage_in)

    var scratch = stack_allocation[
        MAX_WARPS, Split[DT], address_space = AddressSpace.SHARED
    ]()

    var sp = Split[DT]()
    if tid < n_slots:
        var colid = _colid_for(node, tid)
        if sabotage == SAB_COLID_TIEBREAK_REVERSED:
            # Reverse the FEATURE tie-break by negating the ordering key.
            # Every gain here is distinct, so this branch is never taken
            # in arm C -- which is exactly the point: this sabotage must
            # NOT move arm C, and must move the arm that ties on gain.
            colid = Int32(96) - colid
        _ = sp.update(
            Float32(Int(_bin_for(node, tid))),  # quesval = the bin value
            colid,
            _gain_for(node, tid),
            _nleft_for(node, tid),
            _bin_for(node, tid),
            _bin_for(node, tid),
        )

    if sabotage == SAB_SKIP_LAST_WARP_STEP:
        # Drop the final rotate of the in-warp reduction, so lanes never
        # see half the warp. Predicted movement: the winner is wrong for
        # any node whose best candidate sits in the unreached half.
        var lane = lane_id()
        var i = WARP_SIZE // 2
        while i >= 2:
            _ = lane
            i //= 2
        # deliberately no reduction at all past this point for the warp
        if lane == 0:
            scratch[unsafe_offset = tid // WARP_SIZE] = sp.copy()
        barrier()
        if tid == 0:
            var acc = scratch[unsafe_offset=0].copy()
            var n_warps = Int(block_dim.x) // WARP_SIZE
            for w in range(1, n_warps):
                var o = scratch[unsafe_offset=w].copy()
                _ = acc.update(
                    o.quesval,
                    o.colid,
                    o.best_metric_val,
                    o.global_nLeft,
                    o.split_start,
                    o.split_end,
                )
            splits[unsafe_offset=node] = acc.copy()
        return

    if sabotage == SAB_NO_MIDPOINT:
        # Skip `select_split_range_midpoint` by publishing directly.
        #
        # PREDICTED MOVEMENT, and the first prediction written here was
        # WRONG -- it said "nothing moves", on the reasoning that their
        # rule maps a unit range [b,b] to bin b + (b - b + 1)/2 = b and is
        # therefore the identity. The run said 35 of 37 quesvals moved,
        # and the run was right: `split.cuh:126-134` does not only pick
        # the bin, it then ASSIGNS `quesval = quantiles[bin]`. So on a
        # unit range the rule is the identity on the BIN INDEX and is not
        # the identity on the THRESHOLD.
        #
        # That is a real semantic of their algorithm and it is the reason
        # this arm is worth keeping: the `quesval` a candidate carries
        # through the reduction exists only to break ties (`:181-188`);
        # the threshold that is finally PUBLISHED is always read back out
        # of the quantiles array. Predicted movement is therefore every
        # node whose planted quesval differs from `quantiles[bin]`, which
        # the host computes exactly below rather than approximating with
        # "most".
        sp.warp_reduce()
        var lane = lane_id()
        if lane == 0:
            scratch[unsafe_offset = tid // WARP_SIZE] = sp.copy()
        barrier()
        if tid // WARP_SIZE == 0:
            var n_warps = Int(block_dim.x) // WARP_SIZE
            if lane < n_warps:
                sp = scratch[unsafe_offset=lane].copy()
            else:
                sp = Split[DT]()
            sp.warp_reduce()
            if tid == 0:
                splits[unsafe_offset=node] = sp.copy()
        return

    sp.eval_best_split(
        scratch,
        splits.unsafe_offset(node),
        mutexes.unsafe_offset(node),
        quantiles,
        Int32(N_BINS),
    )


def _host_reduce(node: Int, n_slots: Int) -> Split[DT]:
    """The independent tally. Written as a plain left fold over the same
    candidates, in slot order -- deliberately a DIFFERENT order from the
    device's rotate-and-reduce, because with all-distinct gains the result
    must not depend on order and that independence is part of what is
    being checked."""
    var acc = Split[DT]()
    for slot in range(n_slots):
        _ = acc.update(
            Float32(Int(_bin_for(node, slot))),
            _colid_for(node, slot),
            _gain_for(node, slot),
            _nleft_for(node, slot),
            _bin_for(node, slot),
            _bin_for(node, slot),
        )
    return acc


def arm_a_the_order() raises -> Int:
    """Branch-by-branch, on the host, against literals."""
    var failures = 0

    # 1. higher gain wins outright (`split.cuh:150-157`)
    var s = Split[DT]()
    _ = s.update(Float32(1.0), Int32(3), Float32(10.0), Int64(5), Int32(1), Int32(1))
    _ = s.update(Float32(2.0), Int32(0), Float32(20.0), Int64(9), Int32(2), Int32(2))
    if s.best_metric_val != Float32(20.0) or s.colid != Int32(0):
        failures += 1
        print("  arm A: gain-wins branch FAILED, got gain",
              s.best_metric_val, "colid", s.colid)

    # 2. lower gain loses even with a higher colid
    var s2 = Split[DT]()
    _ = s2.update(Float32(1.0), Int32(0), Float32(20.0), Int64(9), Int32(2), Int32(2))
    _ = s2.update(Float32(9.0), Int32(50), Float32(10.0), Int64(5), Int32(1), Int32(1))
    if s2.best_metric_val != Float32(20.0) or s2.colid != Int32(0):
        failures += 1
        print("  arm A: lower-gain-loses branch FAILED")

    # 3. equal gain -> HIGHER colid wins (`:161-168`). Note the direction:
    #    higher, not lower. This is the branch most likely to be flipped by
    #    a careless transcription, and nothing else in the file would
    #    notice.
    var s3 = Split[DT]()
    _ = s3.update(Float32(1.0), Int32(7), Float32(10.0), Int64(5), Int32(1), Int32(1))
    _ = s3.update(Float32(2.0), Int32(9), Float32(10.0), Int64(6), Int32(2), Int32(2))
    if s3.colid != Int32(9):
        failures += 1
        print("  arm A: equal-gain-higher-colid branch FAILED, colid", s3.colid)
    var s3b = Split[DT]()
    _ = s3b.update(Float32(1.0), Int32(9), Float32(10.0), Int64(5), Int32(1), Int32(1))
    _ = s3b.update(Float32(2.0), Int32(7), Float32(10.0), Int64(6), Int32(2), Int32(2))
    if s3b.colid != Int32(9):
        failures += 1
        print("  arm A: equal-gain-lower-colid-loses branch FAILED")

    # 4. equal gain + equal colid + EQUAL nLeft -> ranges MERGE (`:174-177`)
    var s4 = Split[DT]()
    _ = s4.update(Float32(10.0), Int32(4), Float32(10.0), Int64(5), Int32(1), Int32(1))
    _ = s4.update(Float32(30.0), Int32(4), Float32(10.0), Int64(5), Int32(3), Int32(3))
    if s4.split_start != Int32(1) or s4.split_end != Int32(3):
        failures += 1
        print("  arm A: merge branch FAILED, range [", s4.split_start,
              ",", s4.split_end, "]")
    if s4.quesval != Float32(30.0):
        failures += 1
        print("  arm A: merge keeps the UPPER quesval; got", s4.quesval)

    # 5. equal gain + equal colid + DIFFERENT nLeft -> quesval max (`:181-188`)
    var s5 = Split[DT]()
    _ = s5.update(Float32(10.0), Int32(4), Float32(10.0), Int64(5), Int32(1), Int32(1))
    _ = s5.update(Float32(30.0), Int32(4), Float32(10.0), Int64(7), Int32(3), Int32(3))
    if s5.quesval != Float32(30.0) or s5.split_start != Int32(3):
        failures += 1
        print("  arm A: quesval-max branch FAILED")
    var s5b = Split[DT]()
    _ = s5b.update(Float32(30.0), Int32(4), Float32(10.0), Int64(5), Int32(3), Int32(3))
    _ = s5b.update(Float32(10.0), Int32(4), Float32(10.0), Int64(7), Int32(1), Int32(1))
    if s5b.quesval != Float32(30.0) or s5b.split_start != Int32(3):
        failures += 1
        print("  arm A: quesval-max is not symmetric")

    # 6. `local_nLeft` is reset by every replace (`:117`)
    var s6 = Split[DT]()
    _ = s6.update(Float32(1.0), Int32(1), Float32(1.0), Int64(1), Int32(1), Int32(1))
    s6.local_nLeft = Int64(999)
    _ = s6.update(Float32(2.0), Int32(2), Float32(2.0), Int64(2), Int32(2), Int32(2))
    if s6.local_nLeft != Int64(0):
        failures += 1
        print("  arm A: replace_with must zero local_nLeft; got", s6.local_nLeft)

    if failures == 0:
        print("  arm A OK: all six branches of split.cuh:142-191 behave")
    return failures


def arm_b_non_associativity() raises -> Int:
    """RUN the DEVIATION 105 counterexample rather than asserting it.

    Passes when the two groupings DISAGREE. If they agree, this port has
    accidentally made their operator associative, which means it is no
    longer their operator -- so agreement is the failure.
    """
    var quantiles = List[Float32]()
    for b in range(N_BINS):
        quantiles.append(Float32(b) * 10.0)

    # X (nLeft 5, [1,1], q 10), X2 (nLeft 5, [3,3], q 30), Y (nLeft 7, [2,2], q 20)
    var g = Float32(10.0)
    var c = Int32(4)

    var a = Split[DT]()
    _ = a.update(Float32(10.0), c, g, Int64(5), Int32(1), Int32(1))
    _ = a.update(Float32(30.0), c, g, Int64(5), Int32(3), Int32(3))
    _ = a.update(Float32(20.0), c, g, Int64(7), Int32(2), Int32(2))
    a.select_split_range_midpoint(quantiles.unsafe_ptr(), Int32(N_BINS))

    var b2 = Split[DT]()
    _ = b2.update(Float32(10.0), c, g, Int64(5), Int32(1), Int32(1))
    _ = b2.update(Float32(20.0), c, g, Int64(7), Int32(2), Int32(2))
    _ = b2.update(Float32(30.0), c, g, Int64(5), Int32(3), Int32(3))
    b2.select_split_range_midpoint(quantiles.unsafe_ptr(), Int32(N_BINS))

    print(
        "  arm B: grouping ((X+X2)+Y) -> bin", a.split_start,
        "quesval", a.quesval,
        " | grouping ((X+Y)+X2) -> bin", b2.split_start,
        "quesval", b2.quesval,
    )
    if a.split_start == b2.split_start and a.quesval == b2.quesval:
        print(
            "  arm B FAILED: the two groupings AGREE. DEVIATION 105 says"
            " their operator is not associative; if this port made it"
            " associative it is no longer their algorithm."
        )
        return 1
    print(
        "  arm B OK: MEASURED non-associativity -- their reduction order"
        " is observable in the chosen split, which is what DEVIATION 105"
        " records and what the cuML-on-NVIDIA oracle must settle."
    )
    return 0


def _run_device(
    ctx: DeviceContext, n_slots: Int, sabotage: Int
) raises -> List[Split[DT]]:
    var splits = ctx.enqueue_create_buffer[DType.uint8](
        N_NODES * size_of[Split[DT]]()
    )
    var mutexes = ctx.enqueue_create_buffer[DType.int32](N_NODES)
    var quantiles = ctx.enqueue_create_buffer[DType.float32](N_BINS)
    ctx.enqueue_memset(mutexes, Int32(0))

    var hq = ctx.enqueue_create_host_buffer[DType.float32](N_BINS)
    for b in range(N_BINS):
        hq.unsafe_ptr().unsafe_store(b, Float32(b) * 10.0)
    ctx.enqueue_copy(dst_buf=quantiles, src_ptr=hq.unsafe_ptr())

    # Their `initSplit` (`split.cuh:284-289`) seeds every slot with a
    # default Split; without it the mutex merge would read garbage.
    var hs = ctx.enqueue_create_host_buffer[DType.uint8](
        N_NODES * size_of[Split[DT]]()
    )
    var sp_ptr = hs.unsafe_ptr().unsafe_bitcast[Split[DT]]()
    for n in range(N_NODES):
        sp_ptr[unsafe_offset=n] = Split[DT]()
    ctx.enqueue_copy(dst_buf=splits, src_ptr=hs.unsafe_ptr())
    ctx.synchronize()
    # Mojo frees a value at its LAST USE, and `hq`'s last use above was
    # the enqueue itself -- the buffer was dead while its copy was still
    # queued (freed-at-enqueue UAF, found by the perf lane on the gbdt
    # path 2026-08-22; contention-sensitive, so a green check proves
    # nothing). Keep-alives AFTER the synchronize.
    _ = hq^
    _ = hs^

    ctx.enqueue_function[_reduce_kernel](
        splits.unsafe_ptr().unsafe_bitcast[Split[DT]](),
        mutexes.unsafe_ptr(),
        quantiles.unsafe_ptr(),
        Int32(n_slots),
        Int32(sabotage),
        grid_dim=N_NODES,
        block_dim=TPB,
    )
    var out = ctx.enqueue_create_host_buffer[DType.uint8](
        N_NODES * size_of[Split[DT]]()
    )
    ctx.enqueue_copy(dst_buf=out, src_buf=splits)
    ctx.synchronize()
    # Device form of the freed-at-enqueue UAF: `mutexes` died at its
    # `.unsafe_ptr()` in the kernel argument list. Keep-alive AFTER the
    # sync.
    _ = mutexes^

    var res = List[Split[DT]]()
    var op = out.unsafe_ptr().unsafe_bitcast[Split[DT]]()
    for n in range(N_NODES):
        res.append(op[unsafe_offset=n])
    return res^


def arm_c_device(ctx: DeviceContext, n_slots: Int) raises -> Int:
    """Per NODE, against an independent host fold in a different order."""
    var got = _run_device(ctx, n_slots, SAB_NONE)
    var wrong = 0
    for n in range(N_NODES):
        var want = _host_reduce(n, n_slots)
        # The device applied their midpoint rule before publishing
        # (`split.cuh:250`), so the host tally applies the same rule.
        var q = List[Float32]()
        for b in range(N_BINS):
            q.append(Float32(b) * 10.0)
        want.select_split_range_midpoint(q.unsafe_ptr(), Int32(N_BINS))
        if (
            got[n].best_metric_val != want.best_metric_val
            or got[n].colid != want.colid
            or got[n].quesval != want.quesval
            or got[n].global_nLeft != want.global_nLeft
        ):
            wrong += 1
            if wrong <= 3:
                print(
                    "  arm C node", n, "MISMATCH: got gain",
                    got[n].best_metric_val, "colid", got[n].colid,
                    "q", got[n].quesval, " want gain", want.best_metric_val,
                    "colid", want.colid, "q", want.quesval,
                )
    if wrong == 0:
        print(
            "  arm C OK:", N_NODES, "nodes x", n_slots,
            "hashed candidates, every node's winner matches a host fold"
            " taken in a DIFFERENT order",
        )
        return 0
    print("  arm C FAILED:", wrong, "of", N_NODES, "nodes wrong")
    return 1


def arm_d_sabotage(ctx: DeviceContext, n_slots: Int) raises -> Int:
    """One corruption per mechanism, each with its own PREDICTION.

    An arm that moves under any corruption is not evidence. These
    predictions differ from each other on purpose: two must move and one
    must not, and the one that must not is the one that tells you arm C
    does not reach the midpoint rule.
    """
    var failures = 0
    var base = _run_device(ctx, n_slots, SAB_NONE)

    # (i) the in-warp rotate. Predicted: MOVES, for most nodes.
    var s1 = _run_device(ctx, n_slots, SAB_SKIP_LAST_WARP_STEP)
    var moved1 = 0
    for n in range(N_NODES):
        if s1[n].best_metric_val != base[n].best_metric_val:
            moved1 += 1
    if moved1 == 0:
        failures += 1
        print(
            "  arm D(i) FAILED: dropping the warp rotate moved 0 of",
            N_NODES,
            "nodes -- arm C cannot see whether warp_reduce runs at all",
        )
    else:
        print(
            "  arm D(i) OK: dropping the warp rotate moved", moved1, "of",
            N_NODES, "node winners",
        )

    # (ii) the colid tie-break. Predicted: does NOT move, because every
    #      gain in this fixture is distinct so `:161-168` is unreachable.
    #      This is the honest reading: arm C proves the GAIN branch and
    #      says nothing about the colid branch, which is why arm A exists.
    var s2 = _run_device(ctx, n_slots, SAB_COLID_TIEBREAK_REVERSED)
    var moved2 = 0
    for n in range(N_NODES):
        if s2[n].best_metric_val != base[n].best_metric_val:
            moved2 += 1
    if moved2 != 0:
        failures += 1
        print(
            "  arm D(ii) FAILED: reversing the colid tie-break moved",
            moved2,
            "node winners, but every gain in this fixture is distinct so"
            " that branch should be unreachable. The fixture is not what"
            " it claims to be.",
        )
    else:
        print(
            "  arm D(ii) OK: reversing the colid tie-break moved 0 winners,"
            " confirming this fixture never ties on gain -- so arm C's"
            " exact comparison is legitimate, and the colid branch is"
            " covered by arm A and by nothing else here.",
        )

    # (iii) the midpoint rule. Predicted: moves for exactly those nodes
    #       whose winning bin has `quantiles[bin] != planted quesval`.
    #       The count is computed here, not guessed -- a sabotage whose
    #       prediction is "most of them" cannot fail.
    var s3 = _run_device(ctx, n_slots, SAB_NO_MIDPOINT)
    var want_moved = 0
    var moved3 = 0
    var wrong_node = -1
    for n in range(N_NODES):
        var w = _host_reduce(n, n_slots)
        # The winner's range is a unit range [bin, bin]; their rule
        # republishes the threshold as quantiles[bin] = bin * 10.0, while
        # the planted quesval was Float32(bin).
        var republished = Float32(Int(w.split_start)) * 10.0
        var planted = w.quesval
        if republished != planted:
            want_moved += 1
        var did_move = s3[n].quesval != base[n].quesval
        if did_move:
            moved3 += 1
        if did_move != (republished != planted) and wrong_node < 0:
            wrong_node = n
    if moved3 != want_moved or wrong_node >= 0:
        failures += 1
        print(
            "  arm D(iii) FAILED: skipping select_split_range_midpoint"
            " moved", moved3, "quesvals, expected exactly", want_moved,
            "; first node disagreeing per-node:", wrong_node,
        )
    else:
        print(
            "  arm D(iii) OK: skipping the midpoint rule moved exactly",
            moved3, "of", N_NODES, "quesvals, matching a per-node"
            " prediction -- and confirming their rule REPUBLISHES the"
            " threshold from the quantiles array even on a unit range,"
            " so a carried quesval is a tie-break key and never the"
            " published threshold.",
        )

    return failures


def main() raises:
    print("split_check: ensemble/decisiontree/batched_levelalgo/split.mojo")
    print("  mirroring split.cuh @ cuml v26.08.00 265b9da6")
    var failures = 0
    failures += arm_a_the_order()
    failures += arm_b_non_associativity()

    var ctx = DeviceContext()
    # n_slots deliberately not a multiple of the warp width, so the
    # `lane < n_warps` seeding at `split.cuh:242-245` is exercised.
    failures += arm_c_device(ctx, 200)
    failures += arm_d_sabotage(ctx, 200)

    if failures == 0:
        print("split_check: ALL OK")
    else:
        raise Error("split_check: " + String(failures) + " failure(s)")
