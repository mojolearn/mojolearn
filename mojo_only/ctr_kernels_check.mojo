"""The ten CTR elementwise kernels, on device, cell by cell.

    pixi run check-ctr-kernels

Every kernel in `gbdt/ctrs/kernel/ctr_calcers.mojo` is launched on a
DeviceContext and its whole output is compared against a host tally
computed a second way in this file. There is no total, no digest and no
tolerance: an `==` per cell for the integer kernels, an exact `==` on the
float ones, because every float here is a divide of small exactly
representable values and a rounding difference would mean a different
expression, not a different order.

## Fixture rules, each from a bug this repository already paid for

* **SCATTERED VALUES, NEVER UNIFORM OR CONSECUTIVE.** Bins, indices,
  targets and weights are all hashed. A uniform fixture verifies the total
  and nothing about placement -- the histogram lesson in `RESUME.md`, where
  the same kernel reported 0 wrong of 512 under uniform bins and 490 wrong
  under scattered ones.
* **SIZES ARE NOT MULTIPLES OF THE BLOCK GEOMETRY.** `CTR_BLOCK_SIZE` is
  256 and two kernels take four elements per thread, so the sizes below are
  chosen to leave a ragged tail in every launcher: a bounds test that only
  ever runs on a full block is not a bounds test.
* **BOTH SIDES OF EVERY SWITCH** (`PORTING_RULES.md` rule 8). The two
  nullable-pointer kernels run with and without their index map; the
  border-mask extractor runs start and end; the binarized-target stats run
  Borders and Buckets.
* **NEGATIVE ZERO IS PLANTED ON PURPOSE.** `FillBinarizedTargetsStats`
  re-applies the weight's sign with `ExtractSignBit`, a BIT read, and a
  segment start whose weight is 0.0 arrives as -0.0. Under `weight < 0`
  that flag is silently dropped and the segmented scan downstream loses a
  segment boundary. The fixture plants -0.0 weights at segment starts so a
  `< 0` rewrite fails here rather than in a model three stages later.
* **BIT 31 IS PLANTED; BIT 30 IS NOT, AND THAT IS A STATED GAP.**
  `TIndexWrapper::Index()` masks THIRTY bits (`0x3FFFFFFF`), not 31, so a
  reader that used `0x7FFFFFFF` would agree with this file at every size it
  can run. Distinguishing the two masks needs an index with bit 30 set,
  which means either a fixture above 2^30 rows or a deliberately
  out-of-range index word -- and the scatter kernels would then write out
  of bounds rather than fail a comparison. The constant is documented at
  its source in `gbdt/ctrs/index_wrapper.mojo` and is UNGATED here. Saying
  so is the point: an ungated constant that nobody has written down is the
  one that moves.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast

from gbdt.ctrs.index_wrapper import (
    CTR_INDEX_MASK,
    CTR_SEGMENT_START_BIT,
    index_of,
    is_segment_start,
)
from gbdt.ctrs.kernel.ctr_calcers import (
    launch_compute_non_weighted_bin_freq_ctr,
    launch_compute_weighted_bin_freq_ctr,
    launch_extract_border_masks,
    launch_fill_binarized_targets_stats,
    launch_gather_trivial_weights,
    launch_make_means,
    launch_make_means_and_scatter,
    launch_merge_bins,
    launch_update_borders_mask,
    launch_write_mask,
)


#: Ragged against 256 and against 256 * 4.
comptime CK_SIZE = 2611


def _h(x: Int, salt: Int) -> Int:
    """A scattered non-monotone map. Not a ramp, on purpose."""
    var v = ((x + 1) * 2654435761 + salt * 40503) % 1000003
    if v < 0:
        v += 1000003
    return v


def _u32_dev(
    ctx: DeviceContext, values: List[UInt32]
) raises -> DeviceBuffer[DType.uint32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.uint32](n)
    var host = ctx.enqueue_create_host_buffer[DType.uint32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    return buf^


def _f32_dev(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    return buf^


def _u8_dev(
    ctx: DeviceContext, values: List[UInt8]
) raises -> DeviceBuffer[DType.uint8]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.uint8](n)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    return buf^


def _read_u32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.uint32], n: Int
) raises -> List[UInt32]:
    var host = ctx.enqueue_create_host_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[UInt32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    return out^


def _read_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    return out^


def _cmp_f32(
    mut failures: List[String],
    what: String,
    got: List[Float32],
    want: List[Float32],
) -> Int:
    """Compare and RECORD rather than raise.

    Recording is what makes the sabotage proof possible in one run: a
    comparator that raises on the first mismatch can only ever demonstrate
    that ONE of the sixteen comparisons has teeth. With every mismatch
    collected, a single run that perturbs all ten kernels prints all
    sixteen names, which is per-comparison evidence.
    """
    var wrong = 0
    var first = -1
    for i in range(len(want)):
        var a = bitcast[DType.uint32](got[i])
        var b = bitcast[DType.uint32](want[i])
        if a != b:
            wrong += 1
            if first < 0:
                first = i
    if wrong != 0:
        failures.append(
            what
            + ": "
            + String(wrong)
            + " of "
            + String(len(want))
            + " cells wrong, first at "
            + String(first)
            + " (got "
            + String(got[first])
            + ", want "
            + String(want[first])
            + ")"
        )
    return len(want)


def _cmp_u32(
    mut failures: List[String],
    what: String,
    got: List[UInt32],
    want: List[UInt32],
) -> Int:
    """`_cmp_f32`'s integer twin; see there for why it records."""
    var wrong = 0
    var first = -1
    for i in range(len(want)):
        if got[i] != want[i]:
            wrong += 1
            if first < 0:
                first = i
    if wrong != 0:
        failures.append(
            what
            + ": "
            + String(wrong)
            + " of "
            + String(len(want))
            + " cells wrong, first at "
            + String(first)
            + " (got "
            + String(got[first])
            + ", want "
            + String(want[first])
            + ")"
        )
    return len(want)


def _permuted_indices(n: Int, flag_every: Int) raises -> List[UInt32]:
    """A scattered permutation of `0..n-1` carrying segment flags in bit 31.

    Fisher-Yates rather than a stride, so the permutation has no period a
    strided bug could line up with. Bit 30 stays clear -- see the file
    docstring for why that is a stated gap rather than an oversight.
    """
    var perm = List[Int]()
    for i in range(n):
        perm.append(i)
    # deterministic Fisher-Yates, scattered rather than strided
    var state = UInt64(0x9E3779B97F4A7C15)
    for i in range(n - 1, 0, -1):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var j = Int((state >> 33) % UInt64(i + 1))
        var t = perm[i]
        perm[i] = perm[j]
        perm[j] = t

    var out = List[UInt32]()
    for i in range(n):
        var word = UInt32(perm[i])
        if _h(i, 3) % flag_every == 0:
            word |= CTR_SEGMENT_START_BIT
        out.append(word)
    return out^


def check_ctr_kernels() raises:
    print("CTR elementwise kernels on device, cell by cell:")
    var ctx = DeviceContext()
    var n = CK_SIZE
    var cells = 0
    var failures = List[String]()

    var indices = _permuted_indices(n, 7)
    var d_indices = _u32_dev(ctx, indices)

    # --- GatherTrivialWeights, both mask arms ---------------------------
    var first_zero = UInt32(n * 2 // 3)
    for arm in range(2):
        var write_mask_flag = arm == 0
        var dst = ctx.enqueue_create_buffer[DType.float32](n)
        launch_gather_trivial_weights(
            ctx, d_indices, n, first_zero, write_mask_flag, dst
        )
        ctx.synchronize()
        var got = _read_f32(ctx, dst, n)
        var want = List[Float32]()
        for i in range(n):
            var v = Float32(1.0) if index_of(indices[i]) < first_zero else (
                Float32(0.0)
            )
            if write_mask_flag and is_segment_start(indices[i]):
                want.append(-v)
            else:
                want.append(v)
        cells += _cmp_f32(
            failures,
            String("GatherTrivialWeights(mask=")
            + String(write_mask_flag)
            + String(")"),
            got,
            want,
        )
        # the -0.0 the sign-bit reader downstream depends on must be
        # present, or the FillBinarizedTargetsStats case below is vacuous
        if write_mask_flag:
            var neg_zeros = 0
            for i in range(n):
                if bitcast[DType.uint32](got[i]) == UInt32(0x80000000):
                    neg_zeros += 1
            if neg_zeros == 0:
                failures.append(
                    String("GatherTrivialWeights produced no -0.0; the")
                    + String(" fixture cannot exercise ExtractSignBit")
                )

    # --- WriteMask -------------------------------------------------------
    var base = List[Float32]()
    for i in range(n):
        base.append(Float32(_h(i, 11) % 997) / Float32(97.0))
    var wm = _f32_dev(ctx, base)
    launch_write_mask(ctx, d_indices, n, wm)
    ctx.synchronize()
    var wm_got = _read_f32(ctx, wm, n)
    var wm_want = List[Float32]()
    for i in range(n):
        if is_segment_start(indices[i]):
            wm_want.append(-base[i])
        else:
            wm_want.append(base[i])
    cells += _cmp_f32(failures, String("WriteMask"), wm_got, wm_want)

    # --- the two bin-freq kernels ---------------------------------------
    #
    # `bins` here is a segment id per SORTED position, so it is
    # non-decreasing, which is what both kernels' callers guarantee.
    var seg_ids = List[UInt32]()
    var seg_offsets = List[UInt32]()
    var current = UInt32(0)
    for i in range(n):
        if i > 0 and _h(i, 23) % 5 == 0:
            current += UInt32(1)
        if len(seg_offsets) <= Int(current):
            seg_offsets.append(UInt32(i))
        seg_ids.append(current)
    seg_offsets.append(UInt32(n))
    var n_segments = Int(current) + 1

    var bin_sums = List[Float32]()
    for b in range(n_segments):
        bin_sums.append(Float32(_h(b, 31) % 500) + Float32(0.25))
    var d_bins = _u32_dev(ctx, seg_ids)
    var d_bin_sums = _f32_dev(ctx, bin_sums)
    var d_offsets = _u32_dev(ctx, seg_offsets)

    var total_weight = Float32(n)
    for arm in range(2):
        var has_map = arm == 0
        var dst = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_memset(dst, Float32(-7.0))
        launch_compute_weighted_bin_freq_ctr(
            ctx,
            d_indices,
            has_map,
            d_bins,
            d_bin_sums,
            total_weight,
            Float32(0.0),
            Float32(1.0),
            dst,
            n,
        )
        ctx.synchronize()
        var got = _read_f32(ctx, dst, n)
        var want = List[Float32]()
        for _ in range(n):
            want.append(Float32(-7.0))
        for i in range(n):
            var slot: Int
            if has_map:
                slot = Int(index_of(indices[i]))
            else:
                slot = i
            want[slot] = (bin_sums[Int(seg_ids[i])] + Float32(0.0)) / (
                total_weight + Float32(1.0)
            )
        cells += _cmp_f32(
            failures,
            String("WeightedBinFreqCtrs(map=") + String(has_map) + String(")"),
            got,
            want,
        )

    for arm in range(2):
        var has_map = arm == 0
        var dst = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_memset(dst, Float32(-7.0))
        launch_compute_non_weighted_bin_freq_ctr(
            ctx,
            d_indices,
            has_map,
            d_bins,
            d_offsets,
            n,
            Float32(0.0),
            Float32(1.0),
            dst,
        )
        ctx.synchronize()
        var got = _read_f32(ctx, dst, n)
        var want = List[Float32]()
        for _ in range(n):
            want.append(Float32(-7.0))
        for i in range(n):
            var slot: Int
            if has_map:
                slot = Int(index_of(indices[i]))
            else:
                slot = i
            var b = Int(seg_ids[i])
            var count = seg_offsets[b + 1] - seg_offsets[b]
            want[slot] = (Float32(count) + Float32(0.0)) / (
                Float32(n) + Float32(1.0)
            )
        cells += _cmp_f32(
            failures,
            String("NonWeightedBinFreqCtrs(map=")
            + String(has_map)
            + String(")"),
            got,
            want,
        )

    # --- UpdateBordersMask ----------------------------------------------
    #
    # All three of their tests must be reachable, so the fixture carries
    # pre-set flags (test 1), bin changes (test 2), and equal bins whose
    # PREVIOUS-tensor bins differ (test 3). The third is the one a
    # single-feature fixture cannot reach.
    var prev_bins = List[UInt32]()
    for r in range(n):
        prev_bins.append(UInt32(_h(r, 41) % 3))
    var ub_indices = indices.copy()
    var d_ub = _u32_dev(ctx, ub_indices)
    var d_prev = _u32_dev(ctx, prev_bins)
    launch_update_borders_mask(ctx, d_bins, d_prev, d_ub, n)
    ctx.synchronize()
    var ub_got = _read_u32(ctx, d_ub, n)
    var ub_want = List[UInt32]()
    var by_test = [0, 0, 0]
    for i in range(n):
        var cur = ub_indices[i]
        var mask = is_segment_start(cur)
        if mask:
            by_test[0] += 1
        if not mask:
            mask = i == 0 or seg_ids[i] != seg_ids[i - 1]
            if mask:
                by_test[1] += 1
        if not mask:
            mask = prev_bins[Int(index_of(cur))] != prev_bins[
                Int(index_of(ub_indices[i - 1]))
            ]
            if mask:
                by_test[2] += 1
        if mask:
            ub_want.append(cur | CTR_SEGMENT_START_BIT)
        else:
            ub_want.append(cur)
    cells += _cmp_u32(failures, String("UpdateBordersMask"), ub_got, ub_want)
    for t in range(3):
        if by_test[t] == 0:
            raise Error(
                "UpdateBordersMask test "
                + String(t + 1)
                + " was never the deciding one; the fixture does not reach"
                " that branch"
            )
    print(
        "  UpdateBordersMask: all three tests decided at least once",
        by_test[0], by_test[1], by_test[2],
    )

    # --- MergeBins -------------------------------------------------------
    var mb_a = List[UInt32]()
    var mb_b = List[UInt32]()
    for i in range(n):
        mb_a.append(UInt32(_h(i, 53) % 61))
        mb_b.append(UInt32(_h(i, 59) % 7))
    var d_mb_a = _u32_dev(ctx, mb_a)
    var d_mb_b = _u32_dev(ctx, mb_b)
    launch_merge_bins(ctx, d_mb_a, d_mb_b, UInt32(3), n)
    ctx.synchronize()
    var mb_got = _read_u32(ctx, d_mb_a, n)
    var mb_want = List[UInt32]()
    for i in range(n):
        mb_want.append((mb_a[i] << UInt32(3)) | mb_b[i])
    cells += _cmp_u32(failures, String("MergeBins"), mb_got, mb_want)

    # --- ExtractBorderMasks, both arms ----------------------------------
    for arm in range(2):
        var start_segment = arm == 0
        var dst = ctx.enqueue_create_buffer[DType.uint32](n)
        launch_extract_border_masks(ctx, d_indices, dst, n, start_segment)
        ctx.synchronize()
        var got = _read_u32(ctx, dst, n)
        var want = List[UInt32]()
        for i in range(n):
            var flag: Bool
            if start_segment:
                flag = is_segment_start(indices[i])
            else:
                if i + 1 < n:
                    flag = is_segment_start(indices[i + 1])
                else:
                    flag = True
            want.append(UInt32(1) if flag else UInt32(0))
        cells += _cmp_u32(
            failures,
            String("ExtractBorderMasks(start=")
            + String(start_segment)
            + String(")"),
            got,
            want,
        )

    # --- FillBinarizedTargetsStats, both arms ----------------------------
    #
    # Weights come from GatherTrivialWeights with the mask on, so the -0.0
    # planted above is the input here and `ExtractSignBit` is under test
    # rather than assumed.
    var weights = List[Float32]()
    for i in range(n):
        var v = Float32(1.0) if index_of(indices[i]) < first_zero else (
            Float32(0.0)
        )
        if is_segment_start(indices[i]):
            weights.append(-v)
        else:
            weights.append(v)
    var targets = List[UInt8]()
    for i in range(n):
        targets.append(UInt8(_h(i, 67) % 3))
    var d_weights = _f32_dev(ctx, weights)
    var d_targets = _u8_dev(ctx, targets)
    for arm in range(2):
        var borders = arm == 0
        var bin_index = UInt32(1)
        var dst = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_memset(dst, Float32(0.0))
        launch_fill_binarized_targets_stats(
            ctx, d_targets, d_weights, n, dst, bin_index, borders
        )
        ctx.synchronize()
        var got = _read_f32(ctx, dst, n)
        var want = List[Float32]()
        for i in range(n):
            var w = weights[i]
            var hit: Bool
            if borders:
                hit = UInt32(targets[i]) > bin_index
            else:
                hit = UInt32(targets[i]) == bin_index
            var mag = w if w >= Float32(0.0) else -w
            var v = mag * (Float32(1.0) if hit else Float32(0.0))
            if (bitcast[DType.uint32](w) >> 31) != UInt32(0):
                v = -v
            want.append(v)
        cells += _cmp_f32(
            failures,
            String("FillBinarizedTargetsStats(borders=")
            + String(borders)
            + String(")"),
            got,
            want,
        )

    # --- MakeMeans -------------------------------------------------------
    var sums = List[Float32]()
    var denom = List[Float32]()
    for i in range(n):
        sums.append(Float32(_h(i, 71) % 64))
        denom.append(Float32(_h(i, 73) % 64))
    var d_sums = _f32_dev(ctx, sums)
    var d_denom = _f32_dev(ctx, denom)
    launch_make_means(ctx, d_sums, d_denom, n, Float32(0.5), Float32(1.0))
    ctx.synchronize()
    var mm_got = _read_f32(ctx, d_sums, n)
    var mm_want = List[Float32]()
    for i in range(n):
        mm_want.append(
            (sums[i] + Float32(0.5)) / (denom[i] + Float32(1.0))
        )
    cells += _cmp_f32(failures, String("MakeMeans"), mm_got, mm_want)

    # --- MakeMeansAndScatter, both arms ---------------------------------
    var d_sums2 = _f32_dev(ctx, sums)
    for arm in range(2):
        var has_map = arm == 0
        var dst = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_memset(dst, Float32(-3.0))
        launch_make_means_and_scatter(
            ctx,
            d_sums2,
            d_denom,
            n,
            Float32(1.0),
            Float32(1.0),
            d_indices,
            has_map,
            CTR_INDEX_MASK,
            dst,
        )
        ctx.synchronize()
        var got = _read_f32(ctx, dst, n)
        var want = List[Float32]()
        for _ in range(n):
            want.append(Float32(-3.0))
        for i in range(n):
            var slot: Int
            if has_map:
                slot = Int(indices[i] & CTR_INDEX_MASK)
            else:
                slot = i
            want[slot] = (sums[i] + Float32(1.0)) / (
                denom[i] + Float32(1.0)
            )
        cells += _cmp_f32(
            failures,
            String("MakeMeansAndScatter(map=")
            + String(has_map)
            + String(")"),
            got,
            want,
        )

    if len(failures) > 0:
        var msg = String("CTR kernel comparisons FAILED:")
        for i in range(len(failures)):
            msg += String("\n    ") + failures[i]
        raise Error(msg)
    print(
        "  ten kernels, sixteen launches,",
        cells,
        "cells compared one at a time against a host tally, all exact",
    )


def main() raises:
    check_ctr_kernels()
