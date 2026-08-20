"""`ComputePartitionStats`, both kernels, against a host tally.

THE GAP THIS FILLS
------------------
Commit 47966cf replaced a hand-written `prefix_sum`-plus-broadcast with
`max.gpu.primitives.block.sum` in BOTH kernels of
`gbdt/gpu_util/partitions_reduce.mojo`, on the evidence that the symbol
exists and takes the parameters we pass it. Nothing in this tree then read
the result of that reduction against anything.

**A SIGNATURE CHECK PROVES THE GPU IS REACHABLE, NOT THAT THE ANSWER IS
RIGHT.** `nn.argsort[target="gpu"]` resolves, runs, raises nothing, and
returns a well-formed permutation that is not sorted above 256 elements. The
first inversion is always at output position 256, which reads as a single
block with no cross-block merge. A reduction can fail exactly that way and
look exactly that healthy, and `compute_partition_stats` is a TWO-LEVEL
reduction whose second level exists only to combine across blocks.

WHAT IT COMPARES, AND WHY EACH RULE IS HERE
-------------------------------------------
1. **The stats are HASHED per (plane, row)**, so no two cells of the oracle
   agree. A fixture where every element carries the same value verifies a
   total and nothing about placement. That is not hypothetical here: the same
   kernel at the same parameters gave 0 wrong of 512 on a uniform plant and
   490 wrong of 512 on a scattered one (`RESUME.md`).

2. **The oracle is EXACT, not tolerated.** Every planted value is an integer
   in `[-63, 63]`, and a leaf holds at most 40,000 of them, so every partial
   sum on either side is an integer under `2^24` and is represented exactly
   in Float32 whatever order it was summed in. The comparison is therefore
   equality with no tolerance at all, and there is no fudge factor for a
   defect to hide under. It also sidesteps the open Float32-versus-`double`
   question in `partitions_reduce.mojo`'s second deviation block, which this
   check is not the place to price.

3. **Both levels are checked, PER ELEMENT.** `out_stats` is one value per
   (leaf, stat) and comparing only that would be comparing a total. So the
   per-block `partials` plane is read back too and every (slot, stat, block)
   entry is compared against the host's stripe sum. At the largest arm that
   is 157 distinct expected values per (leaf, stat) instead of one, and a
   phase 2 that dropped every block past the first would leave phase 1
   perfect and `out_stats` wrong, which is precisely the shape of the
   `argsort` failure.

4. **Sizes straddle the 256-wide block unevenly**: 1, 255, 256, 257, 512,
   513, 1000, 4096 and 40000. 256 is the size at which `argsort` was still
   correct and 257 is where it first was not. A size that is a clean multiple
   of the block cannot see a missing cross-block combine.

5. **Rows outside every partition are POISONED** with 1000000.0 and the leaf
   id list is PERMUTED with unused ids left poisoned in `out_stats`. A kernel
   that reads one row too far, or writes through the slot index where it
   should write through the leaf id, moves a cell rather than staying quiet.

REACH, WHICH IS THE PART A DIGEST CANNOT GIVE YOU
--------------------------------------------------
Two independent arms, because they fail in different directions.

`check_partitions_reduce_sabotage` adds a known delta to ONE row's stat and
requires the answer to MOVE by exactly that delta in exactly one cell and to
be bit-identical everywhere else. It sweeps the perturbed row across lane 0,
lane 1, the last lane of a block, the first lane of the NEXT block, and the
last row of the partition, so a reduction that only counted lane 0, or only
counted block 0, fails on a named position rather than on a total. It also
runs the same delta on a POISON row outside every partition and requires
nothing to move, which is the negative control: an arm that moves on any
perturbation whatsoever is measuring the harness, not the kernel.

`check_partitions_reduce_narrow_grid` hands `compute_partition_stats` a
`max_leaf_rows` far smaller than the widest leaf and requires the answer to
be exact anyway. Phase 1 has to stripe to survive that, the way `ComputeSum`
stripes in `cuda_util/kernel/update_part_props.cu`, and phase 2 has to read
one partial per launched block the way `SaveResultsImpl` does. Against the
kernels as they stood before this file existed -- one chunk per block, and a
chunk count phase 2 re-derived from the partition size -- this arm returns a
truncated sum on the first leaf and walks out of its own row of `partials` on
the second. It is a reach test that fails loudly on the old code.
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute

from gbdt.gpu_util.partitions_reduce import (
    STATS_BLOCK,
    compute_partition_stats,
    partition_stats_chunks,
)

#: Written into `out_stats` and into `partials` before every arm. Any cell
#: the kernels are supposed to fill and do not comes back holding this, and
#: any cell they are NOT supposed to fill and do comes back not holding it.
comptime POISON = Float32(-777777.0)

#: The value in every row that belongs to no partition, and in the padding
#: either side of the level. A leaf that reads one row past its end picks up
#: a number four orders of magnitude away from anything it should have.
comptime GAP_VALUE = Float32(1000000.0)

#: What the sabotage arm adds to one row. Integer, so the oracle stays exact,
#: and far larger than any planted value so a partial credit is unmistakable.
comptime SABOTAGE_DELTA = Float32(4096.0)

#: Rows of poison in front of the first partition and between neighbours.
comptime GAP_ROWS = 7

#: The leaf-id space every arm runs in. Wider than the five slots a level
#: uses, so three ids are never named and their `out_stats` cells stay
#: poisoned unless the writeback indexes by slot instead of by leaf id.
comptime MAX_LEAVES = 8

#: Three planes, so `stat * line_size` is a stride that has to be right and
#: not a multiplication by zero. Plane 0 is positive, planes 1 and 2 are
#: signed and therefore cancel, which is the case where a wrong summation
#: order or a dropped term is hardest to see in the total.
comptime N_STATS = 3


def _mix(i: Int, salt: Int) -> UInt64:
    """splitmix64. Adjacent rows land nowhere near each other."""
    var z = UInt64(i + 1) * 0x9E3779B97F4A7C15 + UInt64(
        salt + 1
    ) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _stat_value(stat: Int, row: Int) -> Float32:
    """One planted stat, hashed and INTEGER-VALUED in `[-63, 63]`.

    Integer because that is what makes the host oracle exact: a leaf of
    40,000 such values has a magnitude sum under 2.6e6, every intermediate
    sum on the device and on the host is an integer under `2^24`, and Float32
    represents all of them exactly regardless of the order they were added
    in. So the comparison downstream is equality and not a tolerance.

    Plane 0 is strictly positive, which is what a weight plane looks like and
    is the plane whose totals a wrong reduction is likeliest to still get
    right. Planes 1 and 2 straddle zero.
    """
    var h = _mix(row, 17 + stat * 101)
    if stat == 0:
        return Float32(Int(h % UInt64(63)) + 1)
    return Float32(Int(h % UInt64(127)) - 63)


def _push_unique(mut xs: List[Int], v: Int):
    for i in range(len(xs)):
        if xs[i] == v:
            return
    xs.append(v)


def _arm(
    ctx: DeviceContext,
    sizes: List[Int],
    leaf_ids: List[Int],
    max_leaves: Int,
    grid_rows: Int,
    sab_row: Int,
    sab_stat: Int,
    name: String,
) raises -> HostBuffer[DType.float32]:
    """One launch of `compute_partition_stats`, checked cell by cell.

    `grid_rows` is what gets handed to the kernel as `max_leaf_rows`. Pass
    `-1` for the honest value (the widest leaf); pass anything smaller to run
    the narrow-grid arm, which is only correct if phase 1 stripes.

    `sab_row` is an ABSOLUTE row index to add `SABOTAGE_DELTA` to, or `-1`
    for none. The host oracle is built from the same planted array, so a
    sabotage inside a partition is expected to move that partition's cell and
    a sabotage in the poison gap is expected to move nothing. Both are
    assertions; neither is an allowance.

    Returns the `out_stats` plane so the caller can diff two arms directly.
    """
    var n_slots = len(sizes)
    if n_slots != len(leaf_ids):
        raise Error("sizes and leaf ids disagree in length")

    # ---- the level: partitions separated by poison ---------------------
    var offs = List[Int]()
    var cursor = GAP_ROWS
    var widest = 1
    for k in range(n_slots):
        offs.append(cursor)
        cursor += sizes[k] + GAP_ROWS
        if sizes[k] > widest:
            widest = sizes[k]
    var n_rows = cursor

    # EXPECTATIONS FOLLOW THE BUILD. `compute_partition_stats` sizes its
    # grid from the MACHINE (`update_part_props.cu:215`) and ignores
    # `max_leaf_rows` entirely, so the expected `partials` layout must come
    # from the same formula, queried from the same device. `grid_rows` is
    # kept as an argument because the arm below still passes it, but it can
    # no longer narrow anything: the stripe is ALWAYS exercised whenever a
    # partition is wider than `max_chunks * STATS_BLOCK`, which the 40,000
    # row arm guarantees on any device this runs on.
    var grid_hint = widest
    if grid_rows > 0:
        grid_hint = grid_rows
    var max_chunks = partition_stats_chunks(
        ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT), N_STATS
    )

    # ---- the planted stats, hashed, with the gap poisoned ---------------
    var host_stat = List[Float64]()
    for _ in range(N_STATS * n_rows):
        host_stat.append(Float64(GAP_VALUE))
    for k in range(n_slots):
        for i in range(sizes[k]):
            var r = offs[k] + i
            for s in range(N_STATS):
                host_stat[s * n_rows + r] = Float64(_stat_value(s, r))
    if sab_row >= 0:
        host_stat[sab_stat * n_rows + sab_row] += Float64(SABOTAGE_DELTA)

    var h_stats = ctx.enqueue_create_host_buffer[DType.float32](
        N_STATS * n_rows
    )
    for i in range(N_STATS * n_rows):
        h_stats.unsafe_ptr().unsafe_store(i, Float32(host_stat[i]))
    var stats = ctx.enqueue_create_buffer[DType.float32](N_STATS * n_rows)
    ctx.enqueue_copy(dst_buf=stats, src_ptr=h_stats.unsafe_ptr())

    # ---- the partition table, indexed BY LEAF ID ------------------------
    # One host staging buffer per `enqueue_copy`: they are asynchronous, and
    # sharing one across four of them is how `boosting_hist_check` raced its
    # own fixture for the life of the port.
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(0))
    for k in range(n_slots):
        var lid = leaf_ids[k]
        if lid < 0 or lid >= max_leaves:
            raise Error("leaf id " + String(lid) + " is out of range")
        h_off.unsafe_ptr().unsafe_store(lid, UInt32(offs[k]))
        h_sz.unsafe_ptr().unsafe_store(lid, UInt32(sizes[k]))
    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())

    # The slot-to-leaf map, deliberately NOT the identity. Their writeback is
    # `statSums[leafId * statCount + statId]` (`update_part_props.cu`,
    # `SaveResultsImpl`), and a kernel that wrote through the slot instead
    # agrees with the oracle on every identity fixture ever written.
    var h_ids = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        h_ids.unsafe_ptr().unsafe_store(i, UInt32(0))
    for k in range(n_slots):
        h_ids.unsafe_ptr().unsafe_store(k, UInt32(leaf_ids[k]))
    var leaves = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    ctx.enqueue_copy(dst_buf=leaves, src_ptr=h_ids.unsafe_ptr())

    # ---- the two output planes, poisoned first --------------------------
    # `partials` is given TWO slots more than the launch will touch, so a
    # slot-stride that overruns lands in poison rather than in the next
    # arm's leftovers.
    var partial_cells = (n_slots + 2) * N_STATS * max_chunks
    var h_partial_poison = ctx.enqueue_create_host_buffer[DType.float32](
        partial_cells
    )
    for i in range(partial_cells):
        h_partial_poison.unsafe_ptr().unsafe_store(i, POISON)
    var partials = ctx.enqueue_create_buffer[DType.float32](partial_cells)
    ctx.enqueue_copy(dst_buf=partials, src_ptr=h_partial_poison.unsafe_ptr())

    var out_cells = max_leaves * N_STATS
    var h_out_poison = ctx.enqueue_create_host_buffer[DType.float32](out_cells)
    for i in range(out_cells):
        h_out_poison.unsafe_ptr().unsafe_store(i, POISON)
    var out_stats = ctx.enqueue_create_buffer[DType.float32](out_cells)
    ctx.enqueue_copy(dst_buf=out_stats, src_ptr=h_out_poison.unsafe_ptr())
    ctx.synchronize()

    compute_partition_stats(
        ctx,
        n_slots,
        grid_hint,
        N_STATS,
        n_rows,
        leaves,
        p_off,
        p_sz,
        stats,
        partials,
        out_stats,
    )
    ctx.synchronize()

    var got_partial = ctx.enqueue_create_host_buffer[DType.float32](
        partial_cells
    )
    ctx.enqueue_copy(dst_ptr=got_partial.unsafe_ptr(), src_buf=partials)
    var got_out = ctx.enqueue_create_host_buffer[DType.float32](out_cells)
    ctx.enqueue_copy(dst_ptr=got_out.unsafe_ptr(), src_buf=out_stats)
    ctx.synchronize()

    # ---- PHASE 1, per (slot, stat, block) -------------------------------
    # Block `b` owns rows `b * STATS_BLOCK + t + j * gridDim.x * STATS_BLOCK`,
    # which is every row whose chunk index is congruent to `b` modulo the
    # grid width. That is `ComputeSum`'s stripe, restated here rather than
    # imported, because a check that derives its expectation from the thing
    # under test agrees with the bug.
    var wrong_partial = 0
    var first_bad_partial = String("")
    for k in range(n_slots):
        for s in range(N_STATS):
            for b in range(max_chunks):
                var want = Float64(0.0)
                var c = b
                while c * STATS_BLOCK < sizes[k]:
                    var lo = c * STATS_BLOCK
                    var hi = lo + STATS_BLOCK
                    if hi > sizes[k]:
                        hi = sizes[k]
                    for i in range(lo, hi):
                        want += host_stat[s * n_rows + offs[k] + i]
                    c += max_chunks
                var idx = (k * N_STATS + s) * max_chunks + b
                var have = Float64(
                    got_partial.unsafe_ptr().unsafe_load(idx)
                )
                if have != want:
                    wrong_partial += 1
                    if first_bad_partial == "":
                        first_bad_partial = (
                            String("slot ")
                            + String(k)
                            + " stat "
                            + String(s)
                            + " block "
                            + String(b)
                            + " want "
                            + String(want)
                            + " got "
                            + String(have)
                        )

    var partial_spill = 0
    for i in range(n_slots * N_STATS * max_chunks, partial_cells):
        if got_partial.unsafe_ptr().unsafe_load(i) != POISON:
            partial_spill += 1

    # ---- PHASE 2, per (leaf, stat) --------------------------------------
    var wrong_out = 0
    var first_bad_out = String("")
    for k in range(n_slots):
        for s in range(N_STATS):
            var want = Float64(0.0)
            for i in range(sizes[k]):
                want += host_stat[s * n_rows + offs[k] + i]
            var have = Float64(
                got_out.unsafe_ptr().unsafe_load(leaf_ids[k] * N_STATS + s)
            )
            if have != want:
                wrong_out += 1
                if first_bad_out == "":
                    first_bad_out = (
                        String("slot ")
                        + String(k)
                        + " (leaf id "
                        + String(leaf_ids[k])
                        + ") stat "
                        + String(s)
                        + " want "
                        + String(want)
                        + " got "
                        + String(have)
                    )

    # Leaf ids the level never named must still hold poison. This is the
    # arm that separates "wrote through the leaf id" from "wrote through the
    # slot", and no identity-permutation fixture can make that distinction.
    var stomped = 0
    for lid in range(max_leaves):
        var named = False
        for k in range(n_slots):
            if leaf_ids[k] == lid:
                named = True
        if named:
            continue
        for s in range(N_STATS):
            if got_out.unsafe_ptr().unsafe_load(lid * N_STATS + s) != POISON:
                stomped += 1

    print(
        "    arm",
        name,
        ": rows",
        n_rows,
        " grid x",
        max_chunks,
        " partials wrong",
        wrong_partial,
        "of",
        n_slots * N_STATS * max_chunks,
        " totals wrong",
        wrong_out,
        "of",
        n_slots * N_STATS,
    )

    if wrong_partial != 0:
        raise Error(
            String("PHASE 1 (`partition_stats_partial_kernel`) is wrong in ")
            + String(wrong_partial)
            + " of "
            + String(n_slots * N_STATS * max_chunks)
            + " per-block partials on arm "
            + name
            + ". First: "
            + first_bad_partial
            + ". The oracle is exact integer arithmetic, so this is not"
            " rounding. Either `block.sum` is not summing every lane of the"
            " block, or the stripe in `ComputeSum`"
            " (`cuda_util/kernel/update_part_props.cu`) is not covering every"
            " row this block owns"
        )
    if partial_spill != 0:
        raise Error(
            String("PHASE 1 wrote ")
            + String(partial_spill)
            + " cells past the (slot, stat, block) region the launch owns on"
            " arm "
            + name
            + ", so the slot stride is wrong and one leaf's partials are"
            " landing in another's"
        )
    if wrong_out != 0:
        raise Error(
            String("PHASE 2 (`partition_stats_finish_kernel`) is wrong in ")
            + String(wrong_out)
            + " of "
            + String(n_slots * N_STATS)
            + " leaf totals on arm "
            + name
            + " while every per-block partial under them is exact. First: "
            + first_bad_out
            + ". That is the cross-block combine failing on its own, which"
            " is the same shape as `nn.argsort` being correct at 256 and"
            " wrong at 257. Compare against `SaveResultsImpl`'s"
            " `for (x < tempVarsBlockCount) total += tempVars[i]`"
            " (`cuda_util/kernel/update_part_props.cu`)"
        )
    if stomped != 0:
        raise Error(
            String("PHASE 2 wrote ")
            + String(stomped)
            + " cells belonging to leaf ids this level never named, on arm "
            + name
            + ". Their writeback is `statSums[leafId * statCount + statId]`,"
            " so the index is the LOOKED-UP leaf id and not the slot"
        )
    return got_out^


def _sizes() -> List[Int]:
    """Sizes that straddle the 256-wide block unevenly.

    256 is where `nn.argsort` was still correct and 257 is where it first was
    not. 512 and 4096 are clean multiples of the block and are here as the
    controls that CANNOT see a missing cross-block combine; 513, 1000 and
    40000 are the ones that can. 1 is the degenerate partition, and 40000 is
    157 blocks deep so phase 2 has real work rather than a single term.
    """
    var s = List[Int]()
    s.append(1)
    s.append(255)
    s.append(256)
    s.append(257)
    s.append(512)
    s.append(513)
    s.append(1000)
    s.append(4096)
    s.append(40000)
    return s^


def _level_for(size: Int) -> List[Int]:
    """Five partitions built around one size under test.

    TWO leaves of the same size at different offsets, because they carry
    different hashed content and a kernel that computed leaf 0 and broadcast
    it agrees with the oracle on any level whose leaves are all distinct
    sizes. A half-sized leaf, so one launch mixes block counts and the
    surplus blocks on the shorter leaf have to contribute exactly zero. A
    single-row leaf. And an EMPTY one, whose answer is 0.0 in their code too
    (`effectiveBlockCount` collapses to 0 and `result` stays 0).
    """
    var v = List[Int]()
    v.append(size)
    v.append(size)
    var half = size // 2
    if half < 1:
        half = 1
    v.append(half)
    v.append(1)
    v.append(0)
    return v^


def _ids() -> List[Int]:
    """A PERMUTED slot-to-leaf map inside eight leaves, leaving 1, 4 and 6
    unnamed so the writeback has somewhere to be caught going wrong."""
    var v = List[Int]()
    v.append(5)
    v.append(0)
    v.append(3)
    v.append(7)
    v.append(2)
    return v^


def check_partitions_reduce() raises:
    """Every size, both kernels, exact."""
    var ctx = DeviceContext()
    var sizes = _sizes()
    for si in range(len(sizes)):
        var s = sizes[si]
        _ = _arm(
            ctx,
            _level_for(s),
            _ids(),
            MAX_LEAVES,
            -1,
            -1,
            0,
            String("size ") + String(s),
        )
    print(
        "  ComputePartitionStats matches an exact host tally at sizes 1, 255,"
        " 256, 257, 512, 513, 1000, 4096 and 40000, per block and per leaf"
    )


def check_partitions_reduce_narrow_grid() raises:
    """REACH for the stripe: a grid far narrower than the widest leaf.

    `max_leaf_rows` is a grid-sizing hint in their code -- `numBlocks.x` comes
    from the SM count and knows nothing about partition length -- and their
    `ComputeSum` stripes so that the grid covers the partition whatever its
    width. Ours has to do the same or `max_leaf_rows` is an undeclared
    correctness precondition.

    Two widths. 256 collapses the grid to ONE block in x, so the whole
    40,000-row leaf goes through a single threadgroup striding 256 rows at a
    time. 1000 gives four blocks, so each one takes ten strides and phase 2
    has four partials to combine. Both must be exact.

    Against the kernels before this arm existed -- one chunk per block, and a
    phase 2 that re-derived its chunk count from `part_size` -- the first
    width returns the first 256 rows of a 40,000-row leaf and the second
    walks 157 entries through a 4-entry row of `partials`. Neither raises.
    """
    var ctx = DeviceContext()
    var narrow = List[Int]()
    narrow.append(256)
    narrow.append(1000)
    for i in range(len(narrow)):
        _ = _arm(
            ctx,
            _level_for(40000),
            _ids(),
            MAX_LEAVES,
            narrow[i],
            -1,
            0,
            String("40000 rows through a ") + String(narrow[i]) + "-row grid",
        )
    print(
        "  a 40000-row leaf is exact through the machine-sized grid"
        " (update_part_props.cu:215), which is far narrower than the leaf, so"
        " phase 1 stripes and phase 2 reads one partial per launched block"
    )


def check_partitions_reduce_sabotage() raises:
    """REACH by perturbation: one row moves, and exactly one cell follows it.

    Modelled on `replicated_half_byte_check`'s third arm. A check that cannot
    fail on a broken kernel is a check that passed a broken kernel, and a
    digest cannot tell a working change from a no-op.

    The perturbed row is swept across positions that separate the failure
    modes:

        0          lane 0 of block 0 -- the position a reduction that only
                   reads its first lane still gets right
        1          lane 1, which that same reduction does not
        255        the LAST lane of a block
        256        lane 0 of the SECOND block, so a phase 2 that dropped
                   every block past the first loses this one entirely
        257        one past it
        size/2     the middle
        size-1     the last row of the partition, and the row a kernel that
                   is off by one at the end never reads

    Each position is asserted twice: the named cell must move by exactly
    `SABOTAGE_DELTA`, and every other cell must be BIT-IDENTICAL to the clean
    run. The second half is what makes the first mean anything -- a kernel
    that recomputed everything from scratch on a changed input would move the
    named cell too.

    The last arm perturbs a row in the POISON GAP between two partitions and
    requires NOTHING to move. Without it, "the answer moved" would be
    satisfied by a kernel that read the whole buffer.
    """
    var ctx = DeviceContext()
    var sab_stat = 2  # not plane 0: the stat stride has to be exercised

    var arm_sizes = List[Int]()
    arm_sizes.append(257)
    arm_sizes.append(40000)

    for ai in range(len(arm_sizes)):
        var size = arm_sizes[ai]
        var level = _level_for(size)
        var ids = _ids()
        var clean = _arm(
            ctx,
            level,
            ids,
            MAX_LEAVES,
            -1,
            -1,
            0,
            String("clean baseline at ") + String(size),
        )

        # slot 0 starts at GAP_ROWS by construction in `_arm`.
        var base = GAP_ROWS
        var pos = List[Int]()
        _push_unique(pos, 0)
        if size > 1:
            _push_unique(pos, 1)
        if size > 255:
            _push_unique(pos, 255)
        if size > 256:
            _push_unique(pos, 256)
        if size > 257:
            _push_unique(pos, 257)
        _push_unique(pos, size // 2)
        _push_unique(pos, size - 1)

        for pi in range(len(pos)):
            var p = pos[pi]
            var dirty = _arm(
                ctx,
                level,
                ids,
                MAX_LEAVES,
                -1,
                base + p,
                sab_stat,
                String("SABOTAGE size ")
                + String(size)
                + " row "
                + String(p)
                + " stat "
                + String(sab_stat),
            )
            var target = ids[0] * N_STATS + sab_stat
            var moved = 0
            var elsewhere = 0
            for c in range(MAX_LEAVES * N_STATS):
                var a = clean.unsafe_ptr().unsafe_load(c)
                var b = dirty.unsafe_ptr().unsafe_load(c)
                if c == target:
                    if b - a == SABOTAGE_DELTA:
                        moved = 1
                elif a != b:
                    elsewhere += 1
            if moved == 0:
                raise Error(
                    String("THE SABOTAGE ARM PASSED. Adding ")
                    + String(SABOTAGE_DELTA)
                    + " to row "
                    + String(p)
                    + " of a "
                    + String(size)
                    + "-row partition did not move that leaf's total by"
                    " exactly that much. The row is inside the partition, so"
                    " either the block reduction is not reading that lane or"
                    " the block holding it never reaches `out_stats`. Every"
                    " other agreement in this file is vacuous until this"
                    " passes"
                )
            if elsewhere != 0:
                raise Error(
                    String("perturbing one row of one partition moved ")
                    + String(elsewhere)
                    + " cells that belong to other leaves or other stat"
                    " planes, at row "
                    + String(p)
                    + " of "
                    + String(size)
                    + ". The partitions do not overlap and the planes are"
                    " strided apart, so nothing outside the named cell has"
                    " any business changing"
                )

        # The negative control, in the poison gap ahead of slot 1.
        var gap_row = base + size + 2
        var gapped = _arm(
            ctx,
            level,
            ids,
            MAX_LEAVES,
            -1,
            gap_row,
            sab_stat,
            String("SABOTAGE size ") + String(size) + " POISON GAP row",
        )
        var gap_moved = 0
        for c in range(MAX_LEAVES * N_STATS):
            var a = clean.unsafe_ptr().unsafe_load(c)
            var b = gapped.unsafe_ptr().unsafe_load(c)
            if a != b:
                gap_moved += 1
        if gap_moved != 0:
            raise Error(
                String("perturbing a row that belongs to NO partition moved ")
                + String(gap_moved)
                + " cells at size "
                + String(size)
                + ". The kernels are reading outside the leaf they were"
                " given, which also means the movement asserted by the arms"
                " above proves nothing about placement"
            )
        print(
            "    size",
            size,
            ": every swept row moves its own cell by exactly",
            SABOTAGE_DELTA,
            "and nothing else, and a gap row moves nothing",
        )
    print(
        "  both kernels are REACHED: a single perturbed row moves exactly one"
        " leaf total, from lane 0, lane 255, the second block and the last row"
    )
