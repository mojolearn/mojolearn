"""The CTR block ON THE DEVICE, and the permutation that decides its answer.

    pixi run check-ctr-device

`mojo_only/ctr_check.mojo` gates the HOST calcers against an independent
O(n^2) tally. This file gates the DEVICE ones -- the radix-sorted bin
ordering, the segmented scan, and `THistoryBasedCtrCalcerGpu` -- and then
gates the thing neither of those can see on its own: that the CTR ESTIMATION
PERMUTATION is reached, and that it CHANGES THE ANSWER for the better.

## Five sections

0. **THE PRIMITIVES THIS ROUND ADDED**, each on its own hostile fixture:
   `ScanVector<ui32>` on BOTH arms of its `inclusive` switch,
   `GatherWithMask` at `ui32` and `ui8`, and `ScatterWithMask`. They are all
   reached through the bin builder below, but the builder can drive only one
   shape of each, and a parameter that selects behaviour is a parameter the
   checks enumerate (`PORTING_RULES.md` 8). A third of the map words carry
   the top bit, because a fixture with clean indices cannot tell
   `map[i] & mask` from `map[i]` and the mask is the whole reason those two
   kernels exist.

1. **THE BIN ORDERING**, `TCtrBinBuilderGpu` against `TCtrBinBuilder`, the
   whole `Indices` word compared cell by cell -- row id AND the bit-31
   segment flag. Swept over every cardinality that crosses a boundary:
   1 (which must raise), 2, 3, 15, 16, 17, 31, 32, 254, 255, 256, 1000,
   4096. The 15/16/17 band is not decoration: `ReorderBins` sorts
   `IntLog2(uniqueValues)` bits, so 16 categories sort 4 bits and 17 sort 5,
   and a fold-count bug at exactly that step has already shipped once in
   this repository (`one_hot_cardinality_check.mojo`).

   Category codes are HASHED, never a ramp and never uniform. A uniform
   fixture makes every segment the same length and hides a stride bug; a
   ramp makes the sort the identity and hides the sort entirely.

   Stability is checked SEPARATELY from correctness of the segments,
   because sortedness cannot see a tie that moved -- and the tie order IS
   the permutation, which is the whole point of the block.

2. **THE ORDERED STATISTIC**, `THistoryBasedCtrCalcerGpu` against two
   references at once: the host calcer, and an independent O(n^2) tally
   that walks the permutation directly and shares no code with either -- no
   sort, no flags, no scan. All three default priors, and the three columns
   are required to DIFFER, because three identical columns would mean the
   prior loop ran once and copied.

3. **THE PERMUTATION MATTERS, MEASURED.** A gate that still passes with the
   identity substituted has proven nothing about a permutation, so this
   section makes row order produce a visibly WORSE estimator and asserts on
   the difference in both directions.

   The fixture is sorted by target. In ROW order, the rows preceding row
   `r` in its category are the rows with SMALLER targets, so the ordered
   statistic is a monotone function of the row's own target -- the
   statistic leaks the label it is supposed to be estimating without it.
   Under the CTR estimation permutation the preceding rows are a random
   subset of the category, so it does not. The leak is measured as

       mean(ctr | binarized target 1) - mean(ctr | binarized target 0)

   and the check requires the identity arm to be LARGE and the permuted arm
   to be NEAR ZERO. Either assertion failing is a red run: an implementation
   that ignores the order fails the first, and one that permutes wrongly (or
   not at all) fails the second.

4. **`train(cat_features=...)` UNDER THE GPU DEFAULT.** Four columns per
   categorical feature -- three `Borders` priors and one `FeatureFreq` --
   and a fit that runs. Before this round `train()` raised on any Borders
   config, so this section is the wiring's only reach check.
"""

from max.gpu.host import DeviceContext

from gbdt.ctrs.ctr import CTR_BORDERS, CTR_FEATURE_FREQ, TCtrConfig
from gbdt.ctrs.ctr_binarization import (
    build_binarized_target,
    build_target_borders,
)
from gbdt.ctrs.ctr_bins_builder import TCtrBinBuilder, TCtrBinBuilderGpu
from gbdt.ctrs.ctr_calcers import (
    THistoryBasedCtrCalcer,
    THistoryBasedCtrCalcerGpu,
    compute_simple_ctrs_gpu,
)
from gbdt.ctrs.index_wrapper import index_of, is_segment_start
from gbdt.data.permutation import (
    DEFAULT_PERMUTATION_COUNT,
    get_identity_permutation,
    get_permutation,
)
from gbdt.gpu_util.kernel.scan import launch_scan_vector_u32
from gbdt.gpu_util.kernel.transform import (
    launch_gather_with_mask_u32,
    launch_gather_with_mask_u8,
    launch_scatter_with_mask_u32,
)
from gbdt.options.catboost_options import TCatFeatureParams
from gbdt.train import train


def _hashed(x: Int, salt: Int) -> Int:
    """Scattered, not strided: a stride has a period a strided bug can line
    up with."""
    var v = ((x + 1) * 2654435761 + salt * 40503) % 1000003
    if v < 0:
        v += 1000003
    return v


def _cardinalities() -> List[Int]:
    """Every step of `policy_for_fold_count`, both byte boundaries, and the
    15/16/17 band `IntLog2` changes bit width across."""
    return [2, 3, 15, 16, 17, 31, 32, 254, 255, 256, 1000, 4096]


def _codes(n: Int, k: Int, salt: Int) -> List[UInt32]:
    var out = List[UInt32]()
    for r in range(n):
        out.append(UInt32(_hashed(r, salt) % k))
    return out^


# ---------------------------------------------------------------------------
# 0. the three primitives this round added, gated on their own
# ---------------------------------------------------------------------------


def _check_primitives(ctx: DeviceContext, mut failures: List[String]) raises:
    """`ScanVector<ui32>`, `GatherWithMask` and `ScatterWithMask`, directly.

    They are reached through the bin builder, but the builder can only
    exercise ONE shape of each: a simple CTR feeds `ComputeCurrentBins` an
    array whose only end flag is the last one, and the gather's map is
    whatever the sort produced. So each is also driven here on a hostile
    fixture, both arms of the scan's `inclusive` switch included
    (`PORTING_RULES.md` 8: a parameter that selects behaviour is a
    parameter the checks enumerate).

    `n` is prime and is a multiple of neither block (512 for the scan, 256
    for the transforms), so every kernel runs a ragged tail. The map is an
    affine permutation mod that prime, which is a bijection, so a scatter
    accounts for every slot; and the TOP BIT IS SET on a scattered subset
    of the map words, because the whole reason these two exist is the mask
    -- a fixture with clean indices cannot tell `map[i] & mask` from
    `map[i]`.

    MEASURED, and it is why this section is not redundant: deleting the
    scan's phase-3 block carry moves 3489 of 4001 cells HERE and leaves
    section 1 completely green. For a SIMPLE ctr the only end flag
    `ComputeCurrentBins` sees is the last one, so its exclusive scan is all
    zeros and every block's carry is zero -- the device-wide half of that
    scan is unreachable through the builder until tree CTRs land.
    """
    var n = 4001
    var mask = UInt32(0x3FFFFFFF)

    var values = List[UInt32]()
    var src = List[UInt32]()
    var src8 = List[UInt8]()
    var perm = List[Int]()
    var map_words = List[UInt32]()
    for i in range(n):
        values.append(UInt32(_hashed(i, 5) % 13))
        src.append(UInt32(_hashed(i, 11) % 1000003))
        src8.append(UInt8(_hashed(i, 23) % 251))
        var p = (i * 1103515245 + 12345) % n
        perm.append(p)
        var w = UInt32(p)
        if _hashed(i, 41) % 3 == 0:
            w |= UInt32(0x80000000)
        map_words.append(w)

    var h_u32 = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var h_u8 = ctx.enqueue_create_host_buffer[DType.uint8](n)

    var d_values = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_out = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_src = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_map = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_src8 = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_dst8 = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_block_sums = ctx.enqueue_create_buffer[DType.uint32](
        (n + 512 - 1) // 512
    )

    # ONE HOST STAGING BUFFER PER `enqueue_copy`. They are asynchronous
    # (`PORTING.md` 12), and the first draft of this section reused a single
    # buffer for all four uploads -- which failed loudly, on all four
    # primitives at once, before any of them had a bug.
    var up_values = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var up_src = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var up_map = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var up_src8 = ctx.enqueue_create_host_buffer[DType.uint8](n)
    for i in range(n):
        up_values.unsafe_ptr().unsafe_store(i, values[i])
        up_src.unsafe_ptr().unsafe_store(i, src[i])
        up_map.unsafe_ptr().unsafe_store(i, map_words[i])
        up_src8.unsafe_ptr().unsafe_store(i, src8[i])
    ctx.enqueue_copy(dst_buf=d_values, src_ptr=up_values.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_src, src_ptr=up_src.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_map, src_ptr=up_map.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_src8, src_ptr=up_src8.unsafe_ptr())
    ctx.synchronize()

    # --- ScanVector, both arms ---
    for arm in range(2):
        var inclusive = arm == 1
        launch_scan_vector_u32(
            ctx, n, inclusive, d_values, d_out, d_block_sums
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=h_u32.unsafe_ptr(), src_buf=d_out)
        ctx.synchronize()
        var running = UInt32(0)
        var wrong = 0
        var first = -1
        for i in range(n):
            var want: UInt32
            if inclusive:
                running += values[i]
                want = running
            else:
                want = running
                running += values[i]
            if h_u32.unsafe_ptr().unsafe_load(i) != want:
                wrong += 1
                if first < 0:
                    first = i
        if wrong != 0:
            failures.append(
                String("ScanVector ")
                + (String("inclusive") if inclusive else String("exclusive"))
                + String(": ")
                + String(wrong)
                + String(" of ")
                + String(n)
                + String(" cells differ from a host running sum, first at ")
                + String(first)
            )

    # --- GatherWithMask, ui32 and ui8 ---
    launch_gather_with_mask_u32(ctx, d_out, d_src, d_map, n, mask)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h_u32.unsafe_ptr(), src_buf=d_out)
    ctx.synchronize()
    var gather_wrong = 0
    for i in range(n):
        if h_u32.unsafe_ptr().unsafe_load(i) != src[perm[i]]:
            gather_wrong += 1
    if gather_wrong != 0:
        failures.append(
            String("GatherWithMask<ui32>: ")
            + String(gather_wrong)
            + String(" of ")
            + String(n)
            + String(" cells are not src[map[i] & 0x3FFFFFFF]")
        )

    launch_gather_with_mask_u8(ctx, d_dst8, d_src8, d_map, n, mask)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h_u8.unsafe_ptr(), src_buf=d_dst8)
    ctx.synchronize()
    var gather8_wrong = 0
    for i in range(n):
        if h_u8.unsafe_ptr().unsafe_load(i) != src8[perm[i]]:
            gather8_wrong += 1
    if gather8_wrong != 0:
        failures.append(
            String("GatherWithMask<ui8>: ")
            + String(gather8_wrong)
            + String(" of ")
            + String(n)
            + String(" cells wrong; this is GetGatheredBinSample")
        )

    # --- ScatterWithMask ---
    launch_scatter_with_mask_u32(ctx, d_out, d_src, d_map, n, mask)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h_u32.unsafe_ptr(), src_buf=d_out)
    ctx.synchronize()
    var scatter_wrong = 0
    for i in range(n):
        if h_u32.unsafe_ptr().unsafe_load(perm[i]) != src[i]:
            scatter_wrong += 1
    if scatter_wrong != 0:
        failures.append(
            String("ScatterWithMask<ui32>: ")
            + String(scatter_wrong)
            + String(" of ")
            + String(n)
            + String(" cells are not dst[map[i] & 0x3FFFFFFF] = src[i]")
        )

    print(
        "  primitives: ScanVector both arms, GatherWithMask ui32 and ui8,"
        " ScatterWithMask -- every cell against a host tally, top bit set"
        " on a third of the map words"
    )


# ---------------------------------------------------------------------------
# 1. the bin ordering
# ---------------------------------------------------------------------------


def _check_bin_ordering(ctx: DeviceContext, mut failures: List[String]) raises:
    var cards = _cardinalities()
    var checked = 0
    var n = 2003

    # A NON-IDENTITY order, so a builder that ignored its argument and used
    # row order would disagree here rather than agreeing by accident.
    var order = get_permutation(n, DEFAULT_PERMUTATION_COUNT - 1, 1).fill_order()

    for ci in range(len(cards)):
        var k = cards[ci]
        var codes = _codes(n, k, 13 + ci)

        var host = TCtrBinBuilder(order.copy())
        host.add_cat_feature_bins(codes, k)

        var gpu = TCtrBinBuilderGpu(ctx, order)
        gpu.add_cat_feature_bins(ctx, codes, k)
        var got = gpu.read_indices(ctx)

        var label = String("k=") + String(k)

        var wrong = 0
        var first = -1
        for i in range(n):
            if got[i] != host.indices[i]:
                wrong += 1
                if first < 0:
                    first = i
        if wrong != 0:
            failures.append(
                label
                + String(": ")
                + String(wrong)
                + String(" of ")
                + String(n)
                + String(" index words differ from the host builder's,")
                + String(" first at sorted position ")
                + String(first)
                + String(" (device row ")
                + String(index_of(got[first]))
                + String(" flag ")
                + String(1 if is_segment_start(got[first]) else 0)
                + String(", host row ")
                + String(index_of(host.indices[first]))
                + String(" flag ")
                + String(1 if is_segment_start(host.indices[first]) else 0)
                + String(")")
            )

        # --- sortedness, computed and reported SEPARATELY ---
        var descending = 0
        for i in range(1, n):
            if Int(codes[Int(index_of(got[i]))]) < Int(
                codes[Int(index_of(got[i - 1]))]
            ):
                descending += 1
        if descending != 0:
            failures.append(
                label
                + String(": the device ordering is not sorted by bin -- ")
                + String(descending)
                + String(" descending steps")
            )

        # --- STABILITY, which is the half sortedness cannot see ---
        # Within one category the surviving order must be the CTR
        # estimation permutation's. `position[row]` is where the order puts
        # the row; inside a bin those positions must ascend.
        var position = List[Int]()
        for _ in range(n):
            position.append(0)
        for p in range(n):
            position[Int(order[p])] = p
        var unstable = 0
        for i in range(1, n):
            var row = Int(index_of(got[i]))
            var prev = Int(index_of(got[i - 1]))
            if codes[row] == codes[prev] and position[row] < position[prev]:
                unstable += 1
        if unstable != 0:
            failures.append(
                label
                + String(": ")
                + String(unstable)
                + String(" ties are out of PERMUTATION order; the radix")
                + String(" sort must be stable or every Borders value is")
                + String(" silently randomized")
            )

        # --- the segment flags, against the definition ---
        var flag_wrong = 0
        for i in range(n):
            var want_flag: Bool
            if i == 0:
                want_flag = True
            else:
                want_flag = codes[Int(index_of(got[i]))] != codes[
                    Int(index_of(got[i - 1]))
                ]
            if is_segment_start(got[i]) != want_flag:
                flag_wrong += 1
        if flag_wrong != 0:
            failures.append(
                label
                + String(": ")
                + String(flag_wrong)
                + String(" segment flags disagree with the bin changing;")
                + String(" UpdateBordersMask flags i == 0 unconditionally")
            )

        # a fixture with one segment cannot see a reset
        var segments = 0
        for i in range(n):
            if is_segment_start(got[i]):
                segments += 1
        if segments < 2:
            failures.append(
                label
                + String(": the ordering has ")
                + String(segments)
                + String(" segments; a one-segment fixture cannot see a")
                + String(" scan that never resets")
            )
        checked += 1

    # k = 1 is their `CB_ENSURE(uniqueValues > 1, "useless catFeature")`
    var ones = List[UInt32]()
    for _ in range(n):
        ones.append(UInt32(0))
    var refused = False
    try:
        var b = TCtrBinBuilderGpu(ctx, order)
        b.add_cat_feature_bins(ctx, ones, 1)
    except:
        refused = True
    if not refused:
        failures.append(
            String("a single-category feature must raise their 'useless")
            + String(" catFeature found'")
        )

    print(
        "  bin ordering:", checked, "cardinalities, every index word equal"
        " to the host builder's, sorted, STABLE in permutation order, flags"
        " equal to the definition",
    )


# ---------------------------------------------------------------------------
# 2. the ordered statistic
# ---------------------------------------------------------------------------


def _borders_configs() raises -> List[TCtrConfig]:
    var params = TCatFeatureParams.default()
    var configs = params.simple_ctr_configs()
    var out = List[TCtrConfig]()
    for i in range(len(configs)):
        if configs[i].ctr_type == CTR_BORDERS:
            out.append(configs[i])
    return out^


def _check_ordered_statistic(
    ctx: DeviceContext, mut failures: List[String]
) raises:
    var configs = _borders_configs()
    var prior_num = [Float32(0.0), Float32(0.5), Float32(1.0)]
    var cards = [2, 3, 16, 17, 257]
    var n = 1511
    var checked = 0

    for ci in range(len(cards)):
        var k = cards[ci]
        var codes = _codes(n, k, 29 + ci)
        var codes_int = List[Int]()
        for r in range(n):
            codes_int.append(Int(codes[r]))

        # a SCATTERED binarized target, so the running sums differ per row
        var target = List[UInt8]()
        for r in range(n):
            target.append(UInt8(1) if _hashed(r, 19) % 3 == 0 else UInt8(0))

        var order = get_permutation(n, 2, 1).fill_order()

        # the DEVICE arm
        var gpu_builder = TCtrBinBuilderGpu(ctx, order)
        gpu_builder.add_cat_feature_bins(ctx, codes, k)
        var gpu_calcer = THistoryBasedCtrCalcerGpu(ctx, gpu_builder)
        gpu_calcer.set_binarized_sample(ctx, target)
        var got = gpu_calcer.visit_cat_feature_ctr(ctx, configs)

        # the HOST arm, same order
        var host_builder = TCtrBinBuilder(order.copy())
        host_builder.add_cat_feature_bins(codes, k)
        var host_calcer = THistoryBasedCtrCalcer(host_builder)
        host_calcer.set_binarized_sample(target.copy())
        var host = host_calcer.visit_cat_feature_ctr(configs)

        var label = String("k=") + String(k)
        if len(got) != 3 or len(host) != 3:
            failures.append(
                label
                + String(": three priors must give three columns; device ")
                + String(len(got))
                + String(", host ")
                + String(len(host))
            )
            return

        # THE INDEPENDENT TALLY, O(n^2), sharing no code with either arm.
        # `position[row]` is where the CTR ESTIMATION PERMUTATION puts the
        # row; "before" means a smaller position, never a smaller row id.
        var position = List[Int]()
        for _ in range(n):
            position.append(0)
        for p in range(n):
            position[Int(order[p])] = p

        for pi in range(3):
            var wrong_tally = 0
            var wrong_host = 0
            var first = -1
            for r in range(n):
                var count = Float32(0.0)
                var hits = Float32(0.0)
                for q in range(n):
                    if q != r and codes_int[q] == codes_int[r] and (
                        position[q] < position[r]
                    ):
                        count += Float32(1.0)
                        if target[q] > UInt8(0):
                            hits += Float32(1.0)
                var want = (hits + prior_num[pi]) / (count + Float32(1.0))
                if got[pi][r] != want:
                    wrong_tally += 1
                    if first < 0:
                        first = r
                if got[pi][r] != host[pi][r]:
                    wrong_host += 1
            if wrong_tally != 0:
                failures.append(
                    label
                    + String(" prior ")
                    + String(prior_num[pi])
                    + String(": ")
                    + String(wrong_tally)
                    + String(" of ")
                    + String(n)
                    + String(" device values differ from an independent")
                    + String(" O(n^2) tally over the permutation, first at")
                    + String(" row ")
                    + String(first)
                    + String(" (got ")
                    + String(got[pi][first])
                    + String(")")
                )
            if wrong_host != 0:
                failures.append(
                    label
                    + String(" prior ")
                    + String(prior_num[pi])
                    + String(": ")
                    + String(wrong_host)
                    + String(" of ")
                    + String(n)
                    + String(" device values differ from the host calcer")
                )

        # three priors, three DIFFERENT columns
        for pi in range(1, 3):
            var same = True
            for r in range(n):
                if got[pi][r] != got[0][r]:
                    same = False
                    break
            if same:
                failures.append(
                    label
                    + String(": prior ")
                    + String(prior_num[pi])
                    + String(" produced the same column as prior 0.0")
                )

        # a fixture whose values are all equal proves nothing about
        # placement
        var distinct = 0
        for r in range(1, n):
            if got[0][r] != got[0][0]:
                distinct += 1
        if distinct == 0:
            failures.append(
                label + String(": every device value is identical")
            )
        checked += 1

    print(
        "  ordered statistic:", checked, "cardinalities x 3 priors, every"
        " row equal to an independent O(n^2) tally over the PERMUTATION and"
        " to the host calcer",
    )


# ---------------------------------------------------------------------------
# 3. the permutation matters, measured
# ---------------------------------------------------------------------------


def _leak(column: List[Float32], target: List[UInt8]) -> Float64:
    """`mean(ctr | target bin 1) - mean(ctr | target bin 0)`.

    How much of its OWN label a row's ordered statistic reveals. An honest
    ordered target statistic scores this near zero when the category carries
    no information about the target; one computed in a target-correlated
    order scores it high, which is the leak that makes row order a different
    estimator rather than a slower one.
    """
    var s1 = Float64(0.0)
    var s0 = Float64(0.0)
    var n1 = 0
    var n0 = 0
    for r in range(len(column)):
        if target[r] > UInt8(0):
            s1 += Float64(column[r])
            n1 += 1
        else:
            s0 += Float64(column[r])
            n0 += 1
    if n1 == 0 or n0 == 0:
        return Float64(0.0)
    return s1 / Float64(n1) - s0 / Float64(n0)


def _check_permutation_matters(
    ctx: DeviceContext, mut failures: List[String]
) raises:
    var n = 4001
    var k = 17
    var configs = _borders_configs()

    # SORTED BY TARGET, which is the case the identity order is worst at and
    # the case a caller of this library can hand us at any time: their own
    # pipeline shuffles the pool at load and this port has no such stage.
    var y = List[Float32]()
    for r in range(n):
        y.append(Float32(r) / Float32(n))
    var codes = _codes(n, k, 7)

    var borders = build_target_borders(
        y, TCatFeatureParams.default().target_binarization
    )
    var target = build_binarized_target(y, borders)

    var ones = 0
    for r in range(n):
        if target[r] > UInt8(0):
            ones += 1
    if ones == 0 or ones == n:
        failures.append(
            String("the binarized target is constant; this fixture cannot")
            + String(" measure a leak")
        )
        return

    var identity_order = get_identity_permutation(n).fill_order()
    var shuffled_order = get_permutation(
        n, DEFAULT_PERMUTATION_COUNT - 1, 1
    ).fill_order()

    var row_arm = compute_simple_ctrs_gpu(
        ctx, codes, k, configs, target, identity_order
    )
    var perm_arm = compute_simple_ctrs_gpu(
        ctx, codes, k, configs, target, shuffled_order
    )

    var leak_row = _leak(row_arm[0], target)
    var leak_perm = _leak(perm_arm[0], target)

    var moved = 0
    for r in range(n):
        if row_arm[0][r] != perm_arm[0][r]:
            moved += 1

    print(
        "    target-sorted fixture,", n, "rows,", k, "categories,",
        ones, "rows above the target border",
    )
    print("    leak in ROW order       :", leak_row)
    print("    leak in the PERMUTATION :", leak_perm)
    print("    cells that moved        :", moved, "of", n)

    # REACH. If the order argument were ignored, these two columns would be
    # identical and nothing else in this file would notice.
    if moved < (9 * n) // 10:
        failures.append(
            String("only ")
            + String(moved)
            + String(" of ")
            + String(n)
            + String(" values changed between row order and the CTR")
            + String(" estimation permutation; the order argument is not")
            + String(" reaching the ordering")
        )

    # DIRECTION, both ways, so neither arm can be swapped for the other.
    if leak_row < Float64(0.10):
        failures.append(
            String("row order leaked only ")
            + String(leak_row)
            + String(" of the target on a target-SORTED fixture; the")
            + String(" fixture is supposed to make row order the worst")
            + String(" possible estimator, so this means the ordered")
            + String(" statistic is not reading the order at all")
        )
    # 0.01, and the bound is CALIBRATED rather than guessed: this fixture
    # is deterministic (fixed rows, fixed permutation id), the shipped
    # arm measures -0.0027, and the tightest real defect seen here -- an
    # INCLUSIVE segmented scan, which lets a row see its own target --
    # measures +0.0224. A bound of 0.03 was tried first and let that one
    # through; the number below is what catches it with margin on both
    # sides.
    if leak_perm > Float64(0.01) or leak_perm < Float64(-0.01):
        failures.append(
            String("the CTR estimation permutation leaked ")
            + String(leak_perm)
            + String(" of the target, which row order leaks ")
            + String(leak_row)
            + String(" of; a correct permutation makes the preceding rows")
            + String(" of a category independent of this row's label")
        )
    print(
        "  permutation matters: row order leaks the label, the permutation"
        " does not, and every value moved",
    )


# ---------------------------------------------------------------------------
# 4. train() under the GPU default
# ---------------------------------------------------------------------------


def _check_train_default(
    ctx: DeviceContext, mut failures: List[String]
) raises:
    var n = 900
    var k = 23

    var codes = _codes(n, k, 71)
    # a target that is a function of the CATEGORY, so a fit that cannot see
    # the categorical column cannot fit it
    var y = List[Float32]()
    for r in range(n):
        var c = Int(codes[r])
        y.append(Float32(_hashed(c, 3) % 100) / Float32(100.0))

    var x = List[Float32]()
    for r in range(n):
        x.append(Float32(codes[r]))
    for r in range(n):
        x.append(Float32(_hashed(r, 91) % 50) / Float32(50.0))

    var cat_flags: List[Bool] = [True, False]

    var default_params: List[TCatFeatureParams] = [
        TCatFeatureParams.default()
    ]
    var tm = train(
        ctx, x, y, n, 2,
        border_count=32,
        n_estimators=20,
        max_depth=4,
        cat_features=cat_flags,
        cat_feature_params=default_params,
    )

    print(
        "    default(): columns", len(tm.fold_counts),
        "of which ctr", tm.ctr_column_count,
        "-- first loss", tm.losses[0], "last", tm.losses[len(tm.losses) - 1],
    )
    if tm.ctr_column_count != 4:
        failures.append(
            String("their GPU simple_ctr default is Borders at three priors")
            + String(" plus FeatureFreq, so ONE categorical feature must")
            + String(" produce FOUR columns; got ")
            + String(tm.ctr_column_count)
        )
    if len(tm.fold_counts) != 5:
        failures.append(
            String("one numeric column plus four ctr columns is five; got ")
            + String(len(tm.fold_counts))
        )
    if tm.losses[len(tm.losses) - 1] >= tm.losses[0]:
        failures.append(
            String("the fit did not reduce its loss (")
            + String(tm.losses[0])
            + String(" -> ")
            + String(tm.losses[len(tm.losses) - 1])
            + String("); a default categorical fit must train")
        )

    # the frequency-only surface still ships one column, so the switch has
    # both sides exercised
    var freq_params: List[TCatFeatureParams] = [
        TCatFeatureParams.feature_freq_only()
    ]
    var tm2 = train(
        ctx, x, y, n, 2,
        border_count=32,
        n_estimators=5,
        max_depth=4,
        cat_features=cat_flags,
        cat_feature_params=freq_params,
    )
    print(
        "    feature_freq_only(): columns", len(tm2.fold_counts),
        "of which ctr", tm2.ctr_column_count,
    )
    if tm2.ctr_column_count != 1:
        failures.append(
            String("feature_freq_only() must produce ONE ctr column; got ")
            + String(tm2.ctr_column_count)
        )

    # --- AND THE PERMUTATION REACHES train(), ON A FIXTURE THAT CAN SEE IT
    #
    # The fixture above cannot: its target is a pure function of the
    # category, so the ordered statistic is near 0 or near 1 whatever the
    # order, and the quantized columns come out the same. The one below is
    # the opposite -- rows SORTED BY TARGET and a category that carries no
    # information about it, with the categorical column as the ONLY feature.
    # In row order the CTR leaks the label, so the training loss collapses
    # to something no honest estimator can reach; under the CTR estimation
    # permutation it cannot.
    var m = 4001
    var leak_codes = _codes(m, 29, 137)
    var leak_y = List[Float32]()
    for r in range(m):
        leak_y.append(Float32(r) / Float32(m))
    var leak_x = List[Float32]()
    for r in range(m):
        leak_x.append(Float32(leak_codes[r]))
    var leak_flags: List[Bool] = [True]

    var shipped_fit = train(
        ctx, leak_x, leak_y, m, 1,
        border_count=32, n_estimators=30, max_depth=4,
        cat_features=leak_flags, cat_feature_params=default_params,
    )
    var identity_fit = train(
        ctx, leak_x, leak_y, m, 1,
        border_count=32, n_estimators=30, max_depth=4,
        cat_features=leak_flags, cat_feature_params=default_params,
        ctr_estimation_permutation_id=0,
    )
    var shipped_loss = shipped_fit.losses[len(shipped_fit.losses) - 1]
    var identity_loss = identity_fit.losses[len(identity_fit.losses) - 1]
    print(
        "    leak fixture (target-sorted, uninformative category, ctr"
        " columns only):",
    )
    print("      train loss under the PERMUTATION:", shipped_loss)
    print("      train loss in ROW order         :", identity_loss)
    if identity_loss >= shipped_loss * Float64(0.9):
        failures.append(
            String("train(ctr_estimation_permutation_id=0) reached train")
            + String(" loss ")
            + String(identity_loss)
            + String(" against the shipped id ")
            + String(DEFAULT_PERMUTATION_COUNT - 1)
            + String("'s ")
            + String(shipped_loss)
            + String("; on a target-sorted fixture row order LEAKS the")
            + String(" label and must fit far better in sample. These being")
            + String(" close means the permutation argument is not reaching")
            + String(" the CTR ordering")
        )
    print("  train(): the categorical path runs under CatBoost's GPU default")


def _section_end(name: String, before: Int, failures: List[String]):
    """Print the running failure count after every section.

    Each section's own summary line prints unconditionally, so a run read
    from the top looks green until the raise at the bottom. Under a
    sabotage that is exactly the wrong impression, and this line is what
    stops it.
    """
    var got = len(failures) - before
    if got == 0:
        print("    [", name, "OK ]")
    else:
        print("    [", name, "FAILED,", got, "findings ]")


def check_ctr_device() raises:
    print("CTR block on the device:")
    var ctx = DeviceContext()
    var failures = List[String]()
    var mark = 0
    _check_primitives(ctx, failures)
    _section_end("primitives", mark, failures)
    mark = len(failures)
    _check_bin_ordering(ctx, failures)
    _section_end("bin ordering", mark, failures)
    mark = len(failures)
    _check_ordered_statistic(ctx, failures)
    _section_end("ordered statistic", mark, failures)
    mark = len(failures)
    _check_permutation_matters(ctx, failures)
    _section_end("permutation matters", mark, failures)
    mark = len(failures)
    _check_train_default(ctx, failures)
    _section_end("train()", mark, failures)
    if len(failures) > 0:
        var msg = String("device CTR checks FAILED:")
        for i in range(len(failures)):
            msg += String("\n    ") + failures[i]
        raise Error(msg)
    print("  all five sections green")


def main() raises:
    check_ctr_device()
