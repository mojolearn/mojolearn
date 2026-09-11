# SPDX-License-Identifier: Apache-2.0
"""Check the small-k selector's trial arms against each other and the host.

DEVIATION 2497 (C4, block-uniform trip count), 2498 (C1, block head-bound
rejection) and 2515 (C2, warp-scope group bound), both bounds measured
NEGATIVE on the H100 2026-09-11 and kept as the record, and 2517 (deferred
insertion: a four-slot register queue per lane, the K-chain run only at
warp-uniform drains; NEGATIVE too), and 2521 (`capk`: the K-specialized
kernel instantiated with CAP = K, a list of exactly k slots; `capk_selp`:
CAP = K plus a branch-free min/max carry chain; both NEUTRAL), and 2523
(`warpbound_guard`: C2's warp bound composed with the vote guard of
DEVIATION 2522, the admission `pending < min(threshold, bound)` inside a
warp-uniform ballot branch so a step where no lane admits skips the chain;
`warpbound_guard1`: the same with the bound refreshed every batch instead
of every second). Needs a build with
`-D MOJOLEARN_KNN_SELECT_TRIAL=1`;
without it the launch refuses every non-default arm and this check raises,
which is the point: it cannot pass on a binary that has only one arm.

Planted tile: every cell a splitmix64 hash of (row, column, seed) quantized to
61 distinct values so value ties are dense and the index tie-break decides;
exact matches (+0.0) planted at fixed columns including 2047 / 2048 (the
first batch boundary), the row's middle and its last column; a signed zero
and a subnormal beside them; and one row of all-equal values (pure index
ties). The oracle is the long-rows check's exhaustive host rank over
composite keys.

Two properties, k = 10 and 15:

1. arms: `baseline`, `uniform`, `headbound`, `warpbound`, `deferred`,
   `capk`, `capk_selp`, `warpbound_guard` and `warpbound_guard1` give
   identical (value bits, index) output and match the host oracle, on
   lengths chosen so that (length - 1792) mod 2048 is in 1..255 (the
   per-thread trip count of the baseline diverges across threads: 1793,
   2047, 3940, 4095, 65281, 65535, 65536 + 3940) and at 65535, 65536, 65537;
   65536 and 69476 run all four head-bound refreshes (after batches 1, 4,
   12, 28) and sixteen warp-bound refreshes (every second batch from batch
   2); 1793, 2047 and 3940 have at most one batch, so the warpbound arm
   never refreshes there and must equal the uniform arm by construction.
   For the deferred arm the same lengths exercise: a partition with no
   batch (1793, 2047: the tail loop alone, eager inserts from an empty
   queue), one batch plus a tail (3940, 4095), full-queue drains inside a
   batch (the first batches of every row admit every element, so the
   queues fill every four elements), batch-end drains with partial queues
   (the later batches), a tail after 32 batches (65537, 69476) and the
   all-equal row 2 (every key admitted by index order alone). For the
   capk arms the k = 10 and 15 cases ARE the K-specialized buckets
   (`smallk_bucket_kernel[10, 10, ...]`, `[15, 15, ...]`), so every list
   is exactly k deep: every lane's list is full with no spare slot from
   the first batch on (256 elements per lane per batch, 61 distinct
   values, so every lane admits more than k keys and its carry runs off
   the end of a k-deep list), the pop's shift ends at slot k - 1, and the
   all-equal row 2 has each of lanes 0 .. k - 1 holding one winner. This
   check needs the specialization row (NVIDIA) or
   `-D MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON=1`: on the generic bucket
   the capk arms raise by design. For the warpbound_guard arms the same
   lengths exercise: no refresh at all (1793, 2047, 3940, 4095: at most
   one batch, and the first refresh needs two, so the arm is the voteguard
   form by construction, ballot and all), sixteen (guard: batches 2, 4,
   .., 32) or thirty-one (guard1: batches 2 .. 32) refreshes on 65,536
   columns, the same on 65,537 with a one-column tail, and a thirty-third
   batch on 69,476 (guard1 refreshes there too, guard does not) with a
   1,892-column tail, the ballot on every element step of every batch,
   and the all-equal row 2 (the bound is a real key from the first
   refresh on and every lane's gate is the block's index order).
2. reach: each arm's SABOTAGE instantiation changes at least one output
   cell on a 65,536-column row, so a green property 1 is a green on nine
   arms that actually ran. The capk arms carry the uniform arm's flip
   (their scan is the uniform loop), which proves their launch branch and
   loop ran; that the list is k deep is read from the kernel stats leg's
   register count, not from a flip. The warpbound sabotage rejects every
   non-negative-distance key after its first refresh (4,096 columns), so
   the planted +0.0 at length / 2 and length - 1 must vanish from rows 0
   and 1. The deferred sabotage skips the newest queued key at every
   drain; the planted +0.0 at length - 1 is the last element of the last
   batch, hence the newest in lane 255's queue at that batch's drain, and
   must vanish from rows 0 and 1. The warpbound_guard arms carry C2's
   bit-63 sabotage in the refresh and nothing else (the voteguard flip is
   compiled out under the bound): on those instantiations the bound's
   only consumer is the guard's admission predicate, so the planted +0.0
   at length / 2 and length - 1 vanishing from rows 0 and 1 proves the
   refresh ran AND its result reached the guarded path; the cells that
   survive were inserted by the guarded chain, the only insertion path in
   that loop form.

RUN OWED (any GPU box, never the Mac):
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_SELECT_TRIAL=1 \\
        -I . neighbors/checks/knn_selector_arms_check.mojo
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.select_radix_identical import composite_key
from neighbors.checks.select_smallk_identical_candidate import (
    SMALLK_ARM_BASELINE,
    SMALLK_ARM_CAPK,
    SMALLK_ARM_CAPK_SELP,
    SMALLK_ARM_DEFERRED,
    SMALLK_ARM_HEADBOUND,
    SMALLK_ARM_SABOTAGE,
    SMALLK_ARM_UNIFORM,
    SMALLK_ARM_WARPBOUND,
    SMALLK_ARM_WARPBOUND_GUARD,
    SMALLK_ARM_WARPBOUND_GUARD1,
    SMALLK_SELECT_TRIAL,
    smallk_select_launch,
)

comptime ROWS = 3


def splitmix64(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> UInt64(31))


def plant_bits(row: Int, col: Int, length: Int) -> UInt32:
    """Hashed, quantized, with the planted specials."""
    if row == 2:
        return UInt32(1065353216)  # 1.0 everywhere: pure index ties
    var h = splitmix64(UInt64(row) * UInt64(1000003) + UInt64(col) * UInt64(7919) + UInt64(17))
    var bits = UInt32(1065353216 + Int(h % UInt64(61)))
    # Exact matches (+0.0) at the batch boundary, the middle and the end.
    if col == 3 or col == 2047 or col == 2048 or col == length // 2 or col == length - 1:
        bits = 0
    if col == 7:
        bits = UInt32(2147483648)  # -0.0
    if col == 11:
        bits = 1  # smallest positive subnormal
    if row == 1 and (col == 5 or col == 2049):
        bits = 0  # more zeros in row 1: a bigger exact-match tie group
    return bits


def host_rank(host_ptr: MutPointer[Float32, MutAnyOrigin], row: Int, length: Int, rank: Int, previous: UInt64) -> Int:
    var best = UInt64(18446744073709551615)
    var selected = -1
    for col in range(length):
        var key = composite_key(host_ptr.unsafe_load(row * length + col), UInt32(col), True)
        if (rank == 0 or key > previous) and key < best:
            best = key
            selected = col
    return selected


def run_arm(
    ctx: DeviceContext, values_ptr: MutPointer[Float32, MutAnyOrigin],
    length: Int, k: Int, arm: Int,
    mut got_d: List[UInt32], mut got_i: List[UInt32],
) raises:
    var distances = ctx.enqueue_create_buffer[DType.float32](ROWS * k)
    var indices = ctx.enqueue_create_buffer[DType.uint32](ROWS * k)
    var host_d = ctx.enqueue_create_host_buffer[DType.float32](ROWS * k)
    var host_i = ctx.enqueue_create_host_buffer[DType.uint32](ROWS * k)
    smallk_select_launch(
        ctx, values_ptr,
        distances.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        indices.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        ROWS, length, k, True, arm,
    )
    ctx.enqueue_copy(dst_ptr=host_d.unsafe_ptr(), src_buf=distances)
    ctx.enqueue_copy(dst_ptr=host_i.unsafe_ptr(), src_buf=indices)
    ctx.synchronize()
    got_d.clear()
    got_i.clear()
    for cell in range(ROWS * k):
        got_d.append(bitcast[DType.uint32](host_d.unsafe_ptr().unsafe_load(cell)))
        got_i.append(host_i.unsafe_ptr().unsafe_load(cell))
    # Device buffers are freed at their last use; hold them past the copy.
    _ = distances^
    _ = indices^
    _ = host_d^
    _ = host_i^


def cells_differ(a_d: List[UInt32], a_i: List[UInt32], b_d: List[UInt32], b_i: List[UInt32]) -> Int:
    var n = 0
    for cell in range(len(a_d)):
        if a_d[cell] != b_d[cell] or a_i[cell] != b_i[cell]:
            n += 1
    return n


def check_case(length: Int, k: Int, reach: Bool) raises:
    with DeviceContext() as ctx:
        var host = ctx.enqueue_create_host_buffer[DType.float32](ROWS * length)
        ctx.synchronize()
        for row in range(ROWS):
            for col in range(length):
                host.unsafe_ptr().unsafe_store(row * length + col, bitcast[DType.float32](plant_bits(row, col, length)))
        var values = ctx.enqueue_create_buffer[DType.float32](ROWS * length)
        ctx.enqueue_copy(dst_buf=values, src_ptr=host.unsafe_ptr())
        var values_ptr = values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

        var base_d: List[UInt32] = []
        var base_i: List[UInt32] = []
        var uni_d: List[UInt32] = []
        var uni_i: List[UInt32] = []
        var hb_d: List[UInt32] = []
        var hb_i: List[UInt32] = []
        var wb_d: List[UInt32] = []
        var wb_i: List[UInt32] = []
        var df_d: List[UInt32] = []
        var df_i: List[UInt32] = []
        var ck_d: List[UInt32] = []
        var ck_i: List[UInt32] = []
        var cs_d: List[UInt32] = []
        var cs_i: List[UInt32] = []
        var wg_d: List[UInt32] = []
        var wg_i: List[UInt32] = []
        var wg1_d: List[UInt32] = []
        var wg1_i: List[UInt32] = []
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_BASELINE, base_d, base_i)
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_UNIFORM, uni_d, uni_i)
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_HEADBOUND, hb_d, hb_i)
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_WARPBOUND, wb_d, wb_i)
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_DEFERRED, df_d, df_i)
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_CAPK, ck_d, ck_i)
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_CAPK_SELP, cs_d, cs_i)
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_WARPBOUND_GUARD, wg_d, wg_i)
        run_arm(ctx, values_ptr, length, k, SMALLK_ARM_WARPBOUND_GUARD1, wg1_d, wg1_i)

        # 1a. The baseline against the exhaustive host rank.
        var host_ptr = host.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for row in range(ROWS):
            var previous = UInt64(0)
            for rank in range(k):
                var selected = host_rank(host_ptr, row, length, rank, previous)
                var cell = row * k + rank
                if selected < 0 or base_i[cell] != UInt32(selected):
                    raise Error("baseline index differs from the host rank at length " + String(length) + " k " + String(k))
                var expected = bitcast[DType.uint32](host_ptr.unsafe_load(row * length + selected))
                if base_d[cell] != expected:
                    raise Error("baseline value bits differ from the tile cell at length " + String(length) + " k " + String(k))
                previous = composite_key(host_ptr.unsafe_load(row * length + selected), UInt32(selected), True)
        # 1b. C4, C1, C2, deferred, capk, capk_selp, warpbound_guard and
        # warpbound_guard1 against the baseline, cell for cell.
        var d_uni = cells_differ(base_d, base_i, uni_d, uni_i)
        var d_hb = cells_differ(base_d, base_i, hb_d, hb_i)
        var d_wb = cells_differ(base_d, base_i, wb_d, wb_i)
        var d_df = cells_differ(base_d, base_i, df_d, df_i)
        var d_ck = cells_differ(base_d, base_i, ck_d, ck_i)
        var d_cs = cells_differ(base_d, base_i, cs_d, cs_i)
        var d_wg = cells_differ(base_d, base_i, wg_d, wg_i)
        var d_wg1 = cells_differ(base_d, base_i, wg1_d, wg1_i)
        if d_uni != 0:
            raise Error("uniform arm differs from baseline in " + String(d_uni) + " cells at length " + String(length) + " k " + String(k))
        if d_hb != 0:
            raise Error("headbound arm differs from baseline in " + String(d_hb) + " cells at length " + String(length) + " k " + String(k))
        if d_wb != 0:
            raise Error("warpbound arm differs from baseline in " + String(d_wb) + " cells at length " + String(length) + " k " + String(k))
        if d_df != 0:
            raise Error("deferred arm differs from baseline in " + String(d_df) + " cells at length " + String(length) + " k " + String(k))
        if d_ck != 0:
            raise Error("capk arm differs from baseline in " + String(d_ck) + " cells at length " + String(length) + " k " + String(k))
        if d_cs != 0:
            raise Error("capk_selp arm differs from baseline in " + String(d_cs) + " cells at length " + String(length) + " k " + String(k))
        if d_wg != 0:
            raise Error("warpbound_guard arm differs from baseline in " + String(d_wg) + " cells at length " + String(length) + " k " + String(k))
        if d_wg1 != 0:
            raise Error("warpbound_guard1 arm differs from baseline in " + String(d_wg1) + " cells at length " + String(length) + " k " + String(k))
        print("SELECTOR_ARMS_CASE_PASS", length, k, ROWS * k)

        # 2. Reach: every arm's sabotage must move at least one cell.
        if reach:
            var sab_d: List[UInt32] = []
            var sab_i: List[UInt32] = []
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_BASELINE | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_base = cells_differ(base_d, base_i, sab_d, sab_i)
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_UNIFORM | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_uni = cells_differ(base_d, base_i, sab_d, sab_i)
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_HEADBOUND | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_hb = cells_differ(base_d, base_i, sab_d, sab_i)
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_WARPBOUND | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_wb = cells_differ(base_d, base_i, sab_d, sab_i)
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_DEFERRED | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_df = cells_differ(base_d, base_i, sab_d, sab_i)
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_CAPK | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_ck = cells_differ(base_d, base_i, sab_d, sab_i)
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_CAPK_SELP | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_cs = cells_differ(base_d, base_i, sab_d, sab_i)
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_WARPBOUND_GUARD | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_wg = cells_differ(base_d, base_i, sab_d, sab_i)
            run_arm(ctx, values_ptr, length, k, SMALLK_ARM_WARPBOUND_GUARD1 | SMALLK_ARM_SABOTAGE, sab_d, sab_i)
            var f_wg1 = cells_differ(base_d, base_i, sab_d, sab_i)
            if f_base == 0 or f_uni == 0 or f_hb == 0 or f_wb == 0 or f_df == 0 or f_ck == 0 or f_cs == 0 or f_wg == 0 or f_wg1 == 0:
                raise Error(
                    "REACH NOT PROVEN: sabotage flipped baseline " + String(f_base)
                    + ", uniform " + String(f_uni) + ", headbound " + String(f_hb)
                    + ", warpbound " + String(f_wb) + ", deferred " + String(f_df)
                    + ", capk " + String(f_ck) + ", capk_selp " + String(f_cs)
                    + ", warpbound_guard " + String(f_wg) + ", warpbound_guard1 " + String(f_wg1) + " cells"
                )
            print("SELECTOR_ARMS_REACH_PASS", length, k, f_base, f_uni, f_hb, f_wb, f_df, f_ck, f_cs, f_wg, f_wg1)
        _ = host^
        _ = values^


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("selector arms check requires IDENTICAL")
    comptime if not SMALLK_SELECT_TRIAL:
        raise Error("selector arms check needs -D MOJOLEARN_KNN_SELECT_TRIAL=1: without it only the default arm exists")
    # (length - 1792) mod 2048 in 1..255, plus 65535 / 65536 / 65537.
    var lengths: List[Int] = [1793, 2047, 3940, 4095, 65281, 65535, 65536, 65537, 65536 + 3940]
    var counts: List[Int] = [10, 15]
    var cases = 0
    for length in lengths:
        for k in counts:
            check_case(length, k, length == 65536)
            cases += 1
    print("KNN SELECTOR ARMS PASS", "cases", cases)
