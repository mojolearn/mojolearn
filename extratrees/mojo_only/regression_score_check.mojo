"""Does the device publish a REGRESSION key the split reduction can rank by?

    pixi run mojo run -I . extratrees/mojo_only/regression_score_check.mojo

NO CUML COUNTERPART -- this is a check. It covers the regression arm of
`extratrees/ported/decisiontree/batched_levelalgo/kernels/
builder_kernels_impl.mojo::node_feature_score_finalize_kernel` and the
`regression_key` arithmetic it shares with `node_feature_score_host`, which
carry DEVIATION BLOCKS 189-193.

WHAT WAS BROKEN AND WHAT THIS FILE HAS TO PROVE
-----------------------------------------------
The finalize kernel published `(0, 0)` for every regression candidate, so
`split_reduce_kernel`'s exact key was absent, EVERY comparison tied, the order
degenerated to `Split.update`, and `Split.best_metric_val` is a constant `0.0`
on the device path (DEVIATION 182) -- so the winner of a regression node was
decided by FEATURE INDEX. Not approximately; entirely.

It now publishes cuML's own MSE gain as an exact rational in two `Int64`s.
This check has to establish four separate things, because they can fail
independently:

  1. the device computes what the host oracle computes, per cell, bit for bit;
  2. that value is the rational DEVIATION 189 says it is, recomputed here in
     `Int128` down a DIFFERENT code path (explicit power-of-two division, not
     a shift), so a sign or shift defect cannot agree with itself;
  3. its ORDER is sklearn's MSE proxy's order -- checked against
     `fixed_point.mojo::mse_proxy_exact`, which is another file's transcription
     of `_criterion.pyx:944-973` and therefore an authority rather than a
     mirror;
  4. the whole reduction, on the device, picks the same winner a host fold
     picks, on real scored candidates.

WHY THE VALUE CANNOT SIMPLY BE sklearn's PROXY, MEASURED RATHER THAN ARGUED
---------------------------------------------------------------------------
`num = sum_L^2 n_R + sum_R^2 n_L` over sums bounded by the shipped `2^30`
fixed-point slot wraps `Int64` at TEN ROWS in a node -- arm D measures it -- and
wraps NEGATIVE, which `compare_exact_key`'s sign split then ranks below every
other candidate. Arm G measures the two ways of narrowing it against a
`Float64` ground truth and reports how many orderings each one loses. The route
this file's subject took was chosen by those numbers; arm G is where they are
recomputed every run rather than quoted from a commit message.

WHAT IS DELIBERATELY ADVERSARIAL
--------------------------------
1. **Mixed-sign labels with a non-zero mean**, so the fixed-point sums REACH
   the slot (a zero-mean target cancels and every bound in sight goes slack)
   while the sign paths stay live. The key's numerator is a SQUARE and is
   therefore never negative -- see arm B, which asserts that and reports it,
   because DEVIATION 167 claims the opposite.
2. **A second label plane that VIOLATES the fixed-point contract**, so
   DEVIATION 193's refusal is a branch this check reaches rather than a branch
   it describes. Without it the slot-guard sabotage would have no target.
3. **Adjacent-`n_left` near-tie pairs** in arm C: candidate pairs built so that
   their `|A|` values land in the SAME truncation bucket and their denominators
   differ by one row. That is the only construction that can make the published
   key disagree with the exact proxy, and the check reports the rate instead of
   claiming there is none.
4. **Column-major storage, shuffled row_ids, gaps between nodes, `fslot` that
   is not `col`, and `work_item.idx` that is not `nid`** -- the same five
   confusions `score_kernel_check` builds, kept because the key is read out of
   accumulators those confusions would corrupt.
5. **A ragged batch** straddling the block width, so the cross-block
   `Atomic.fetch_add` is what the key is computed from.

THE ARMS
--------
  P. PRECONDITION -- the range pass this check feeds the score pass.
  A. PER CELL -- device against `node_feature_score_host`, every field.
  B. PER CELL, THE KEY -- against an `Int128` recomputation down a different
     path, plus its sign, plus its agreement with `mse_proxy_exact`'s ORDER.
  C. THE ORDER -- over hashed candidates and over adversarial near-ties,
     counted both ways.
  D. THE BOUND -- the ten-row wrap MEASURED, `regression_key_bound_ok` exact
     below and refused above, and the device's own refusal observed.
  E. END TO END -- `split_reduce_kernel` on these candidates against a host
     fold, per node.
  F. SABOTAGE -- one per MECHANISM, each a kernel ARGUMENT so every arm is the
     shipping binary. Each must turn arm A red.
  G. THE ROUTE -- the published key, the exact `Int128` proxy, a narrowed
     accumulator and a shifted sum, all four scored against a `Float64` ground
     truth. This is the measurement that chose the design.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from std.gpu import WARP_SIZE
from max.gpu.host import DeviceContext

from extratrees.mojo_only.fixed_point import (
    accumulator_bits_for,
    ceil_log2,
    choose_scale,
    compare_mse_proxy_exact,
    mse_proxy_exact,
    quantize,
    SLOT_BITS,
)
from extratrees.mojo_only.fixtures import (
    cell_hash,
    shaped_dataset,
    signed_unit,
    SALT_LABEL,
    SALT_X,
    SALT_Y,
    SHAPE_CONSTANT,
    SHAPE_HASHED,
    SHAPE_NEAR_CONST_EQUAL,
    SHAPE_NEGATIVE,
    SHAPE_ONE_ODD_ROW,
    SHAPE_OUTLIER,
    SHAPE_SPANS_ZERO,
    SHAPE_TWO_VALUED,
)
from extratrees.mojo_only.pcg_rng import key_for
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.batched_levelalgo.split import (
    compare_exact_key,
    ExactKey,
    Split,
    SplitExact,
    split_reduce_init_kernel,
    split_reduce_kernel,
    split_reduce_shared_bytes,
    SPLIT_SAB_NONE,
)
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    WorkloadInfo,
)
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    build_workload_info,
    FeatureRange,
    feature_range_at,
    mse_gain_from_exact_totals,
    node_feature_is_constant,
    node_feature_min_max,
    node_feature_range_decode_kernel,
    node_feature_range_init_kernel,
    node_feature_range_kernel,
    node_feature_score_finalize_kernel,
    node_feature_score_host,
    node_feature_score_init_kernel,
    node_feature_score_kernel,
    RANGE_SAB_NONE,
    regression_key_bound_ok,
    regression_key_shift,
    REGRESSION_KEY_BITS,
    REGRESSION_SUM_BITS,
    ScoredCandidate,
    scored_candidate_at,
    SCORE_SAB_NONE,
    SCORE_SAB_REG_DEN_ONE,
    SCORE_SAB_REG_NO_CENTER,
    SCORE_SAB_REG_NO_SHIFT,
    SCORE_SAB_REG_NO_SLOT_GUARD,
    SCORE_STATUS_CONSTANT,
    SCORE_STATUS_MISSING_REFUSED,
    SCORE_STATUS_REGRESSION_REFUSED,
    SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF,
    SCORE_STATUS_SCORED,
    SCORE_STATUS_UNVISITED,
    TPB_DEFAULT,
)


comptime TPB = TPB_DEFAULT
comptime FIN_TPB = 64
comptime RED_TPB = 128
"""The reduction's block width. A whole number of warps on every target --
`split_reduce_kernel` comptime-asserts it, because `nWarps` silently becomes
zero on a 64-wide wavefront otherwise."""

comptime MAX_ACC = 2
"""Regression uses ONE accumulator; the comptime bound is 2 so that a kernel
looping to `MAX_ACC` instead of `n_acc` would be visible."""
comptime N_ACC = 1

comptime N_ROWS = 512
comptime N_COLS = 8
comptime MIN_SAMPLES_LEAF = 1
comptime TREE_ID = 9
comptime SEED: UInt64 = 0xB0A710A5

comptime BIG_NODE = 1
"""The node whose labels leave the fixed-point slot in the second plane. It has
THREE rows, so every split of it is 1|2 or 2|1 and every one of them puts a
sum outside the slot -- a predictable refusal set."""

comptime OVER_SLOT: Int32 = 1200000000
"""Just over the `2^30` slot and nowhere near a wrap: the refusal must be a
PRECONDITION test, not an overflow test."""

comptime BIG_POS: Int32 = 2140000000
comptime BIG_NEG: Int32 = -1070000000
"""Just inside `Int32` so the ACCUMULATION is still sound and only the KEY's
precondition is violated -- otherwise the sabotage would be testing the wrong
thing. At three rows these also make the unguarded numerator actually WRAP
(|A| reaches 6.42e9, `|A| >> 1` is 3.21e9, and its square is 1.03e19 against
Int64's 9.22e18), so DEVIATION 193's guard is observed catching a wrap rather
than an inconvenience."""


def _vendor() -> String:
    comptime if has_apple_gpu_accelerator():
        return "Apple"
    elif has_nvidia_gpu_accelerator():
        return "NVIDIA"
    elif has_amd_gpu_accelerator():
        return "AMD"
    else:
        return "unknown vendor"


def _node_ranges() -> List[InstanceRange]:
    """A ragged batch against TPB = 128: 1, 1, 1, 2 and 1 blocks, with GAPS
    between the nodes so a kernel that walked from the previous node's end
    would read rows belonging to no node. Node 1 is the three-row node arm D
    and arm F need."""
    var out = List[InstanceRange]()
    out.append(InstanceRange(Int32(5), Int32(40)))
    out.append(InstanceRange(Int32(60), Int32(3)))
    out.append(InstanceRange(Int32(70), Int32(128)))
    out.append(InstanceRange(Int32(210), Int32(200)))
    out.append(InstanceRange(Int32(430), Int32(1)))
    return out^


def _shuffled_row_ids(n_rows: Int) -> List[Int32]:
    var ids = List[Int32]()
    for r in range(n_rows):
        ids.append(Int32(r))
    for i in range(n_rows - 1, 0, -1):
        var h = UInt32(i) * 2654435761
        h ^= h >> 15
        var j = Int(h % UInt32(i + 1))
        var t = ids[i]
        ids[i] = ids[j]
        ids[j] = t
    return ids^


def _build_columns() -> List[Float32]:
    """COLUMN-MAJOR, `data[col * M + row]` (`dataset.h:24`). Six of
    `fixtures.mojo`'s shapes, so a constant column and an outlier column are
    both present and the CONSTANT and SCORED statuses both occur."""
    var fixture = shaped_dataset(SEED, N_ROWS, _shape_list())
    var flat = List[Float32](length=N_COLS * N_ROWS, fill=Float32(0.0))
    for c in range(N_COLS):
        for r in range(N_ROWS):
            flat[c * N_ROWS + r] = fixture.value(r, c)
    return flat^


def _shape_list() -> List[Int]:
    """The column shapes, CHOSEN rather than taken as the first `N_COLS` of
    `all_shapes()`. That is not cherry-picking: six of them must SCORE or the
    key has no cells to be checked in (the first run of this check took the
    first six shapes, four of which are near-constant, and arm B was left with
    nine ordered pairs). The two skipped shapes are kept so that the CONSTANT
    status still occurs beside the scored ones."""
    var shapes = List[Int]()
    shapes.append(SHAPE_HASHED)
    shapes.append(SHAPE_CONSTANT)
    shapes.append(SHAPE_TWO_VALUED)
    shapes.append(SHAPE_OUTLIER)
    shapes.append(SHAPE_NEGATIVE)
    shapes.append(SHAPE_NEAR_CONST_EQUAL)
    shapes.append(SHAPE_SPANS_ZERO)
    shapes.append(SHAPE_ONE_ODD_ROW)
    return shapes^


def _colids(n_nodes: Int) -> List[Int32]:
    """Each node's own permutation of the columns, so `fslot` and `col` are
    different numbers. 5 and 6 are coprime."""
    var out = List[Int32]()
    for nid in range(n_nodes):
        for fslot in range(N_COLS):
            out.append(Int32((fslot * 5 + nid) % N_COLS))
    return out^


# ---------------------------------------------------------------------------
# The INDEPENDENT recomputation of the key. Arm B compares the shipped
# `regression_key` against this, and it is written to share nothing with it:
# `Int128` throughout, an explicit division by a power of two instead of a
# shift, and the magnitude taken after the division rather than before.
# ---------------------------------------------------------------------------
def _key_reference(
    sum_left: Int64, sum_total: Int64, n_left: Int, n_right: Int, row_count: Int
) raises -> Tuple[Int128, Int128]:
    var sl = Int128(sum_left)
    var sr = Int128(sum_total) - sl
    var nl = Int128(n_left)
    var nr = Int128(n_right)
    var a = sl * nr - sr * nl
    var j = ceil_log2(row_count) + REGRESSION_SUM_BITS - REGRESSION_KEY_BITS
    if j < 0:
        j = 0
    var divisor = Int128(1)
    for _ in range(j):
        divisor *= Int128(2)
    # Truncation toward zero: divide the SIGNED value, then take the
    # magnitude. `regression_key` takes the magnitude first and shifts. The
    # two agree exactly iff the truncation really is toward zero, which is the
    # point of writing them differently.
    var scaled = a // divisor
    if a < Int128(0) and scaled * divisor != a:
        scaled += Int128(1)
    if scaled < Int128(0):
        scaled = -scaled
    return (scaled * scaled, nl * nr)


def _cmp_rational(
    a_num: Int128, a_den: Int128, b_num: Int128, b_den: Int128
) -> Int:
    """Order two rationals with positive denominators. Separate from
    `compare_mse_proxy_exact` only so that arm C's two sides do not share a
    comparator."""
    if a_den <= Int128(0) and b_den <= Int128(0):
        return 0
    if a_den <= Int128(0):
        return -1
    if b_den <= Int128(0):
        return 1
    var l = a_num * b_den
    var r = b_num * a_den
    if l > r:
        return 1
    if l < r:
        return -1
    return 0


def _scale_for_bits(magnitude: Float64, row_count: Int, bits: Int) -> Float64:
    """`choose_scale`'s snap, at an ARBITRARY accumulator width.

    Arm G needs the scale a narrower accumulator would produce and
    `choose_scale` derives its width from the row count, so the snap is
    repeated here rather than the function being bent. Same rule: multiply, do
    not divide, so the chosen power is a function of the magnitude's BITS
    (`fixed_point.mojo` records why that mattered).
    """
    if magnitude <= 0.0:
        return 1.0
    var limit = Float64((1 << bits) - 1 - row_count)
    if limit <= 0.0:
        return 0.0
    var snapped = 1.0
    if magnitude <= limit:
        while magnitude * (snapped * 2.0) <= limit:
            snapped *= 2.0
    else:
        while magnitude * snapped > limit:
            snapped /= 2.0
    return snapped


def _status_name(s: Int32) -> String:
    if s == SCORE_STATUS_UNVISITED:
        return "UNVISITED"
    if s == SCORE_STATUS_SCORED:
        return "SCORED"
    if s == SCORE_STATUS_CONSTANT:
        return "CONSTANT"
    if s == SCORE_STATUS_MISSING_REFUSED:
        return "MISSING_REFUSED"
    if s == SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF:
        return "REJECTED_MIN_SAMPLES_LEAF"
    if s == SCORE_STATUS_REGRESSION_REFUSED:
        return "REGRESSION_REFUSED"
    return "?"


@fieldwise_init
struct ScoreRun(Copyable, Movable):
    var cells: List[ScoredCandidate]


def _oracle(
    flat: List[Float32],
    row_ids: List[Int32],
    labels: List[Int32],
    ranges: List[InstanceRange],
    colids: List[Int32],
    work_items: List[NodeWorkItem],
    extents: List[FeatureRange],
) raises -> List[ScoredCandidate]:
    """`node_feature_score_host` over every cell. A module-level function and
    not a closure: the kernels' oracle is compared against twice, once per
    label plane, and a closure over `main`'s locals cannot be."""
    var out = List[ScoredCandidate]()
    for nid in range(len(ranges)):
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            var col = Int(colids[s])
            out.append(
                node_feature_score_host(
                    rebind[MutPointer[Float32, MutAnyOrigin]](
                        flat.unsafe_ptr()
                    ),
                    rebind[MutPointer[Int32, MutAnyOrigin]](
                        row_ids.unsafe_ptr()
                    ),
                    rebind[MutPointer[Int32, MutAnyOrigin]](
                        labels.unsafe_ptr()
                    ),
                    N_ROWS,
                    Int(ranges[nid].begin),
                    Int(ranges[nid].count),
                    col,
                    extents[s],
                    key_for(
                        SEED,
                        UInt32(TREE_ID),
                        UInt32(Int(work_items[nid].idx)),
                        UInt32(col),
                    ),
                    N_ACC,
                    False,
                    MIN_SAMPLES_LEAF,
                    device_draw=True,
                )
            )
    return out^


def main() raises:
    comptime assert has_accelerator(), "regression_score_check requires a GPU"

    print("========================================================")
    print("regression_score_check -- DEVIATION BLOCKS 189-193")
    print("========================================================")

    var ranges = _node_ranges()
    var n_nodes = len(ranges)
    var n_cells = n_nodes * N_COLS
    var n_acc_cells = n_cells * N_ACC
    var colids = _colids(n_nodes)
    var row_ids = _shuffled_row_ids(N_ROWS)
    var flat = _build_columns()

    for nid in range(n_nodes):
        var seen = List[Int](length=N_COLS, fill=0)
        for fslot in range(N_COLS):
            seen[Int(colids[nid * N_COLS + fslot])] += 1
        for c in range(N_COLS):
            if seen[c] != 1:
                raise Error("the colids table is not a permutation")

    var work_items = List[NodeWorkItem]()
    for nid in range(n_nodes):
        work_items.append(NodeWorkItem(Int32(200 + nid), Int32(4), ranges[nid]))

    # ------------------------------------------------------------------
    # THE LABELS. Mixed sign with a non-zero mean, for the reason
    # `score_kernel_check` recorded: a zero-mean target CANCELS, every partial
    # sum stays tiny, and every bound this file is about goes slack.
    # ------------------------------------------------------------------
    var y = List[Float64]()
    for r in range(N_ROWS):
        var v = Float64(signed_unit(cell_hash(SEED, r, 0, SALT_Y))) + 1.75
        if ((cell_hash(SEED, r, 1, SALT_LABEL) >> 13) & 7) == 0:
            v = -3.25
        y.append(v)
    var magnitude = Float64(0.0)
    for r in range(N_ROWS):
        var a = y[r]
        magnitude += -a if a < 0.0 else a
    var scale = choose_scale(magnitude, N_ROWS)
    var labels_q = List[Int32]()
    var max_q = Int64(0)
    for r in range(N_ROWS):
        var q = quantize(y[r], scale)
        var aq = q if q >= 0 else -q
        if aq > max_q:
            max_q = aq
        labels_q.append(Int32(q))

    # The SECOND plane: the same labels except in node BIG_NODE's three rows,
    # which are pushed outside the fixed-point slot. DEVIATION 193's refusal
    # is a branch, and an unchecked branch is an unreached branch.
    var labels_big = labels_q.copy()
    var big_begin = Int(ranges[BIG_NODE].begin)
    labels_big[Int(row_ids[big_begin])] = BIG_POS
    labels_big[Int(row_ids[big_begin + 1])] = BIG_NEG
    labels_big[Int(row_ids[big_begin + 2])] = BIG_NEG
    # And ONE row of node 0, just over the slot. Node 1 makes the numerator
    # actually WRAP; this one makes the guard refuse WITHOUT a wrap, which is
    # the case a guard written as an overflow test rather than a precondition
    # test would miss. Every split of node 0 has this row on one side or the
    # other, so every scored cell of it is refused.
    labels_big[Int(row_ids[Int(ranges[0].begin)])] = OVER_SLOT

    # The largest partial sum any node of the SHIPPING plane can hold, so the
    # fixture's own conformance to DEVIATION 193's precondition is measured
    # rather than assumed.
    var worst_partial = Int64(0)
    for nid in range(n_nodes):
        var acc = Int64(0)
        var neg = Int64(0)
        for p in range(
            Int(ranges[nid].begin), Int(ranges[nid].begin + ranges[nid].count)
        ):
            var q = Int64(Int(labels_q[Int(row_ids[p])]))
            if q > 0:
                acc += q
            else:
                neg += -q
        if acc > worst_partial:
            worst_partial = acc
        if neg > worst_partial:
            worst_partial = neg

    print("")
    print(
        "[fixture]",
        n_nodes,
        "nodes x",
        N_COLS,
        "columns =",
        n_cells,
        "cells;",
        N_ROWS,
        "rows; fixed-point scale",
        scale,
        "from a magnitude sum of",
        magnitude,
    )
    print(
        "          largest quantized label",
        max_q,
        "; largest one-sided partial sum any node can form",
        worst_partial,
        "against the 2^" + String(REGRESSION_SUM_BITS) + " slot",
        Int64(1) << Int64(REGRESSION_SUM_BITS),
    )
    if worst_partial < (Int64(1) << Int64(24)):
        raise Error(
            "this fixture's partial sums never approach the slot, so nothing"
            " below is testing the width it claims to test. DEFECT IN THE"
            " CHECK."
        )
    for nid in range(n_nodes):
        print(
            "          node",
            nid,
            "rows",
            Int(ranges[nid].count),
            "-> shift j =",
            regression_key_shift(Int(ranges[nid].count)),
        )

    var plan = build_workload_info(work_items, TPB)
    print(
        "          workload_info flattens the batch into",
        plan.n_blocks_dimx,
        "blocks,",
        plan.n_large_nodes,
        "node(s) LARGE",
    )

    var failures = 0

    # ------------------------------------------------------------------
    # The host oracle's input.
    # ------------------------------------------------------------------
    var labels_f = List[Float32](length=N_ROWS, fill=Float32(0.0))
    var dataset = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](flat.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](labels_f.unsafe_ptr()),
        Int32(N_ROWS),
        Int32(N_COLS),
        Int32(N_ROWS),
        Int32(N_COLS),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        Int32(1),
    )
    var extents = List[FeatureRange]()
    for nid in range(n_nodes):
        for fslot in range(N_COLS):
            extents.append(
                node_feature_min_max(
                    dataset, work_items[nid], colids[nid * N_COLS + fslot]
                )
            )

    # ------------------------------------------------------------------
    # The device.
    # ------------------------------------------------------------------
    var ctx = DeviceContext()
    print(
        "[device] accelerator present;",
        _vendor(),
        "-- every arm runs the SAME binary, the sabotage is an argument",
    )

    var d_min = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_max = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_missing = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_merges = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_minkey = ctx.enqueue_create_buffer[DType.uint32](n_cells)
    var d_maxkey = ctx.enqueue_create_buffer[DType.uint32](n_cells)
    var d_data = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_ROWS)
    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_colids = ctx.enqueue_create_buffer[DType.int32](len(colids))
    var d_lab = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_lab_big = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_items = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes * size_of[NodeWorkItem]()
    )
    var d_wl = ctx.enqueue_create_buffer[DType.uint8](
        plan.n_blocks_dimx * size_of[WorkloadInfo]()
    )

    var s_data = ctx.enqueue_create_host_buffer[DType.float32](
        N_COLS * N_ROWS
    )
    var s_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var s_colids = ctx.enqueue_create_host_buffer[DType.int32](len(colids))
    var s_lab = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var s_lab_big = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var s_items = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes * size_of[NodeWorkItem]()
    )
    var s_wl = ctx.enqueue_create_host_buffer[DType.uint8](
        plan.n_blocks_dimx * size_of[WorkloadInfo]()
    )
    ctx.synchronize()

    for i in range(N_COLS * N_ROWS):
        s_data.unsafe_ptr().unsafe_store(i, flat[i])
    for i in range(N_ROWS):
        s_row_ids.unsafe_ptr().unsafe_store(i, row_ids[i])
        s_lab.unsafe_ptr().unsafe_store(i, labels_q[i])
        s_lab_big.unsafe_ptr().unsafe_store(i, labels_big[i])
    for i in range(len(colids)):
        s_colids.unsafe_ptr().unsafe_store(i, colids[i])
    var items_ptr = s_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    for i in range(n_nodes):
        items_ptr[unsafe_offset=i] = work_items[i]
    var wl_ptr = s_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo]()
    for i in range(plan.n_blocks_dimx):
        wl_ptr[unsafe_offset=i] = plan.info[i]

    ctx.enqueue_copy(dst_buf=d_data, src_ptr=s_data.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_row_ids, src_ptr=s_row_ids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_colids, src_ptr=s_colids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_lab, src_ptr=s_lab.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_lab_big, src_ptr=s_lab_big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_items, src_ptr=s_items.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wl, src_ptr=s_wl.unsafe_ptr())
    ctx.synchronize()

    # ------------------------------------------------------------------
    # ARM P -- the range pass, this check's INPUT.
    # ------------------------------------------------------------------
    ctx.enqueue_function[node_feature_range_init_kernel](
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_merges.unsafe_ptr(),
        Int32(n_cells),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )
    ctx.enqueue_function[node_feature_range_kernel[TPB]](
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_merges.unsafe_ptr(),
        d_data.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        d_colids.unsafe_ptr(),
        Int32(N_ROWS),
        Int32(N_COLS),
        Int32(N_COLS),
        Int32(RANGE_SAB_NONE),
        grid_dim=(plan.n_blocks_dimx, N_COLS, 1),
        block_dim=(TPB, 1, 1),
    )
    # DEVIATION 204: the merge produced order-preserving KEYS; this turns
    # them back into the `(min, max)` floats every later pass reads, and
    # applies the empty-cell sentinel.
    ctx.enqueue_function[node_feature_range_decode_kernel](
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        Int32(n_cells),
        Int32(RANGE_SAB_NONE),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )
    var r_min = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
    var r_max = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
    var r_missing = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
    ctx.enqueue_copy(dst_buf=r_min, src_buf=d_min)
    ctx.enqueue_copy(dst_buf=r_max, src_buf=d_max)
    ctx.enqueue_copy(dst_buf=r_missing, src_buf=d_missing)
    ctx.synchronize()

    print("")
    print("[arm P] the range pass, on float BIT PATTERNS")
    var p_bad = 0
    for s in range(n_cells):
        var got = feature_range_at(
            rebind[MutPointer[Float32, MutAnyOrigin]](r_min.unsafe_ptr()),
            rebind[MutPointer[Float32, MutAnyOrigin]](r_max.unsafe_ptr()),
            rebind[MutPointer[Int32, MutAnyOrigin]](r_missing.unsafe_ptr()),
            s // N_COLS,
            s % N_COLS,
            N_COLS,
        )
        var w = extents[s]
        if (
            got.min_value.to_bits() != w.min_value.to_bits()
            or got.max_value.to_bits() != w.max_value.to_bits()
            or got.n_missing != w.n_missing
        ):
            p_bad += 1
    if p_bad == 0:
        print("  arm P OK:", n_cells, "cells; the score pass is fed the DEVICE's ranges")
    else:
        failures += 1
        print("  arm P FAILED:", p_bad, "cells wrong")

    # ------------------------------------------------------------------
    # THE RUNS. Two label planes; on the shipping plane the shipping arm plus
    # three key sabotages, on the violating plane the shipping arm plus the
    # slot-guard sabotage.
    # ------------------------------------------------------------------
    var d_status = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_thresh = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_n_left = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_n_total = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_num = ctx.enqueue_create_buffer[DType.int64](n_cells)
    var d_den = ctx.enqueue_create_buffer[DType.int64](n_cells)
    var d_blocks = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_acc_left = ctx.enqueue_create_buffer[DType.int32](n_acc_cells)
    var d_acc_total = ctx.enqueue_create_buffer[DType.int32](n_acc_cells)
    # DEVIATION 211: the kernels take the tree id PER NODE now; this check's
    # batch is one tree, so every slot is TREE_ID.
    var d_tree_ids = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var h_tree_ids = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    ctx.synchronize()
    for i in range(n_nodes):
        h_tree_ids.unsafe_ptr().unsafe_store(i, Int32(TREE_ID))
    ctx.enqueue_copy(dst_buf=d_tree_ids, src_ptr=h_tree_ids.unsafe_ptr())
    ctx.synchronize()

    var arms = List[Int32]()
    var arm_big = List[Bool]()
    var arm_name = List[String]()
    arms.append(SCORE_SAB_NONE)
    arm_big.append(False)
    arm_name.append("shipping")
    arms.append(SCORE_SAB_REG_NO_SHIFT)
    arm_big.append(False)
    arm_name.append("the node-uniform shift (j forced to 0)")
    arms.append(SCORE_SAB_REG_NO_CENTER)
    arm_big.append(False)
    arm_name.append("cuML's centering (drop the right child's term)")
    arms.append(SCORE_SAB_REG_DEN_ONE)
    arm_big.append(False)
    arm_name.append("the denominator (n_L n_R -> 1)")
    arms.append(SCORE_SAB_NONE)
    arm_big.append(True)
    arm_name.append("shipping, on the CONTRACT-VIOLATING label plane")
    # ^ NOT a sabotage: it is the shipping kernel on the second label plane,
    # and arm A requires it to AGREE with its own oracle. It is in this list
    # only because it needs a run.
    arms.append(SCORE_SAB_REG_NO_SLOT_GUARD)
    arm_big.append(True)
    arm_name.append("the fixed-point slot guard, on the violating plane")
    # THE RESTORE, and it is the last arm on purpose: the shipping kernel run
    # again AFTER every sabotage, into the same buffers, required to be
    # bit-identical to the first shipping run. A sabotage that leaves a buffer
    # dirty would otherwise be indistinguishable from one that works.
    arms.append(SCORE_SAB_NONE)
    arm_big.append(False)
    arm_name.append("RESTORE -- the shipping kernel, rerun after every arm")

    var arm_is_sabotage = List[Bool]()
    for a in range(len(arms)):
        arm_is_sabotage.append(arms[a] != SCORE_SAB_NONE)

    var runs = List[ScoreRun]()
    for a in range(len(arms)):
        var sab = arms[a]
        var lab_ptr = d_lab_big.unsafe_ptr() if arm_big[
            a
        ] else d_lab.unsafe_ptr()
        ctx.enqueue_function[node_feature_score_init_kernel](
            d_status.unsafe_ptr(),
            d_thresh.unsafe_ptr(),
            d_n_left.unsafe_ptr(),
            d_n_total.unsafe_ptr(),
            d_num.unsafe_ptr(),
            d_den.unsafe_ptr(),
            d_blocks.unsafe_ptr(),
            d_acc_left.unsafe_ptr(),
            d_acc_total.unsafe_ptr(),
            Int32(n_cells),
            Int32(n_acc_cells),
            grid_dim=ceildiv(n_acc_cells, 64) + 1,
            block_dim=64,
        )
        comptime acc_kernel = node_feature_score_kernel[TPB, MAX_ACC, False]
        ctx.enqueue_function[acc_kernel](
            d_n_left.unsafe_ptr(),
            d_n_total.unsafe_ptr(),
            d_acc_left.unsafe_ptr(),
            d_acc_total.unsafe_ptr(),
            d_blocks.unsafe_ptr(),
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            d_data.unsafe_ptr(),
            d_row_ids.unsafe_ptr(),
            lab_ptr,
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            d_colids.unsafe_ptr(),
            d_tree_ids.unsafe_ptr(),
            Int32(N_ROWS),
            Int32(N_COLS),
            Int32(N_ACC),
            SEED,
            sab,
            grid_dim=(plan.n_blocks_dimx, N_COLS, 1),
            block_dim=(TPB, 1, 1),
        )
        comptime fin_kernel = node_feature_score_finalize_kernel[MAX_ACC, False]
        ctx.enqueue_function[fin_kernel](
            d_status.unsafe_ptr(),
            d_thresh.unsafe_ptr(),
            d_num.unsafe_ptr(),
            d_den.unsafe_ptr(),
            d_n_left.unsafe_ptr(),
            d_n_total.unsafe_ptr(),
            d_acc_left.unsafe_ptr(),
            d_acc_total.unsafe_ptr(),
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_colids.unsafe_ptr(),
            d_tree_ids.unsafe_ptr(),
            Int32(n_cells),
            Int32(N_COLS),
            Int32(N_ACC),
            SEED,
            Int32(MIN_SAMPLES_LEAF),
            sab,
            grid_dim=ceildiv(n_cells, FIN_TPB),
            block_dim=FIN_TPB,
        )

        var o_status = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var o_thresh = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
        var o_n_left = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var o_n_total = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var o_num = ctx.enqueue_create_host_buffer[DType.int64](n_cells)
        var o_den = ctx.enqueue_create_host_buffer[DType.int64](n_cells)
        var o_left = ctx.enqueue_create_host_buffer[DType.int32](n_acc_cells)
        var o_total = ctx.enqueue_create_host_buffer[DType.int32](n_acc_cells)
        ctx.enqueue_copy(dst_buf=o_status, src_buf=d_status)
        ctx.enqueue_copy(dst_buf=o_thresh, src_buf=d_thresh)
        ctx.enqueue_copy(dst_buf=o_n_left, src_buf=d_n_left)
        ctx.enqueue_copy(dst_buf=o_n_total, src_buf=d_n_total)
        ctx.enqueue_copy(dst_buf=o_num, src_buf=d_num)
        ctx.enqueue_copy(dst_buf=o_den, src_buf=d_den)
        ctx.enqueue_copy(dst_buf=o_left, src_buf=d_acc_left)
        ctx.enqueue_copy(dst_buf=o_total, src_buf=d_acc_total)
        ctx.synchronize()

        var cells = List[ScoredCandidate]()
        for nid in range(n_nodes):
            for fslot in range(N_COLS):
                cells.append(
                    scored_candidate_at(
                        rebind[MutPointer[Int32, MutAnyOrigin]](
                            o_status.unsafe_ptr()
                        ),
                        rebind[MutPointer[Float32, MutAnyOrigin]](
                            o_thresh.unsafe_ptr()
                        ),
                        rebind[MutPointer[Int32, MutAnyOrigin]](
                            o_n_left.unsafe_ptr()
                        ),
                        rebind[MutPointer[Int32, MutAnyOrigin]](
                            o_n_total.unsafe_ptr()
                        ),
                        rebind[MutPointer[Int32, MutAnyOrigin]](
                            o_left.unsafe_ptr()
                        ),
                        rebind[MutPointer[Int32, MutAnyOrigin]](
                            o_total.unsafe_ptr()
                        ),
                        rebind[MutPointer[Int64, MutAnyOrigin]](
                            o_num.unsafe_ptr()
                        ),
                        rebind[MutPointer[Int64, MutAnyOrigin]](
                            o_den.unsafe_ptr()
                        ),
                        nid,
                        fslot,
                        N_COLS,
                        N_ACC,
                    )
                )
        runs.append(ScoreRun(cells^))

    # Mojo frees a buffer at its LAST USE; the launches above read this one
    # and the loop's synchronize must come first.
    _ = d_tree_ids^
    _ = h_tree_ids^

    # ------------------------------------------------------------------
    # The host oracle, for both label planes.
    # ------------------------------------------------------------------
    var want = _oracle(
        flat, row_ids, labels_q, ranges, colids, work_items, extents
    )
    var want_big = _oracle(
        flat, row_ids, labels_big, ranges, colids, work_items, extents
    )
    _ = flat.unsafe_ptr()
    _ = row_ids.unsafe_ptr()
    _ = labels_q.unsafe_ptr()
    _ = labels_big.unsafe_ptr()
    _ = labels_f.unsafe_ptr()

    var base = runs[0].copy()
    var base_big = runs[4].copy()

    # ------------------------------------------------------------------
    # ARM A -- per cell, exact.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm A] per (node, feature) cell against node_feature_score_host:"
        " status, threshold BITS, n_left, n_total, the label sum, and the"
        " published rational"
    )
    var a_bad = 0
    var a_scored = 0
    var a_reported = 0
    for s in range(n_cells):
        if base.cells[s].status == SCORE_STATUS_SCORED:
            a_scored += 1
        if not base.cells[s].matches(want[s]):
            a_bad += 1
            if a_reported < 6:
                a_reported += 1
                print(
                    "  MISMATCH cell",
                    s,
                    "got status",
                    _status_name(base.cells[s].status),
                    "num",
                    base.cells[s].gini_num,
                    "den",
                    base.cells[s].gini_den,
                    "| want",
                    _status_name(want[s].status),
                    "num",
                    want[s].gini_num,
                    "den",
                    want[s].gini_den,
                )
    var a_bad_big = 0
    for s in range(n_cells):
        if not base_big.cells[s].matches(want_big[s]):
            a_bad_big += 1
    if a_bad == 0 and a_bad_big == 0:
        print(
            "  arm A OK:",
            n_cells,
            "cells on each of two label planes identical to the host"
            " transcription;",
            a_scored,
            "of them SCORED",
        )
    else:
        failures += 1
        print(
            "  arm A FAILED:",
            a_bad,
            "cells wrong on the shipping plane,",
            a_bad_big,
            "on the violating plane",
        )
    if a_scored < 8:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK ***: only",
            a_scored,
            "cells reached the key at all",
        )

    # ------------------------------------------------------------------
    # ARM B -- the KEY itself, per cell.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm B] the published pair against an Int128 recomputation down a"
        " different path, its sign, and its agreement with"
        " mse_proxy_exact's ORDER"
    )
    var b_bad = 0
    var b_checked = 0
    var b_negative = 0
    var b_pairs = 0
    var b_pair_bad = 0
    var b_pair_tie = 0
    for nid in range(n_nodes):
        var count = Int(ranges[nid].count)
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            var c = base.cells[s].copy()
            if c.status != SCORE_STATUS_SCORED:
                continue
            b_checked += 1
            var nl = Int(c.n_left)
            var nr = Int(c.n_total) - nl
            var want_key = _key_reference(
                Int64(Int(c.acc_left[0])),
                Int64(Int(c.acc_total[0])),
                nl,
                nr,
                count,
            )
            if (
                Int128(c.gini_num) != want_key[0]
                or Int128(c.gini_den) != want_key[1]
            ):
                b_bad += 1
                if b_bad <= 4:
                    print(
                        "  KEY MISMATCH cell",
                        s,
                        "published",
                        c.gini_num,
                        "/",
                        c.gini_den,
                        "reference",
                        want_key[0],
                        "/",
                        want_key[1],
                    )
            if c.gini_num < Int64(0):
                b_negative += 1
    # The ORDER, per PAIR, within each node, against sklearn's proxy as
    # `fixed_point.mojo` transcribes it. That file is the authority here: it is
    # `_criterion.pyx:944-973` in Int128 and it shares no line with the key.
    for nid in range(n_nodes):
        for i in range(N_COLS):
            for j in range(i + 1, N_COLS):
                var ca = base.cells[nid * N_COLS + i].copy()
                var cb = base.cells[nid * N_COLS + j].copy()
                if (
                    ca.status != SCORE_STATUS_SCORED
                    or cb.status != SCORE_STATUS_SCORED
                ):
                    continue
                var pa = mse_proxy_exact(
                    Int64(Int(ca.acc_left[0])),
                    Int(ca.n_left),
                    Int64(Int(ca.acc_total[0] - ca.acc_left[0])),
                    Int(ca.n_total - ca.n_left),
                )
                var pb = mse_proxy_exact(
                    Int64(Int(cb.acc_left[0])),
                    Int(cb.n_left),
                    Int64(Int(cb.acc_total[0] - cb.acc_left[0])),
                    Int(cb.n_total - cb.n_left),
                )
                var want_ord = compare_mse_proxy_exact(pa[0], pa[1], pb[0], pb[1])
                if want_ord == 0:
                    continue
                b_pairs += 1
                var got_ord = Int(
                    compare_exact_key(
                        ExactKey(ca.gini_num, ca.gini_den, Int32(1)),
                        ExactKey(cb.gini_num, cb.gini_den, Int32(1)),
                    )
                )
                if got_ord == 0:
                    b_pair_tie += 1
                elif got_ord != want_ord:
                    b_pair_bad += 1
    if b_bad == 0:
        print(
            "  the pair matches the Int128 reference in all",
            b_checked,
            "scored cells -- and the reference divides by a power of two and"
            " takes the magnitude AFTERWARDS, so truncation toward zero is"
            " what is being agreed on",
        )
    else:
        failures += 1
        print("  arm B FAILED:", b_bad, "of", b_checked, "keys wrong")
    print(
        "  the numerator was negative in",
        b_negative,
        "of",
        b_checked,
        "cells, and it CANNOT be: it is a square. DEVIATION 167 says an MSE"
        " numerator 'is not sign-constrained' and that sentence is FALSE for"
        " this key and for mse_proxy_exact alike -- both are sums of squares"
        " times counts. Reported as an OPEN doc item.",
    )
    if b_negative != 0:
        failures += 1
        print("  arm B FAILED: a square came back negative, so it wrapped")
    print(
        "  the ORDER against mse_proxy_exact:",
        b_pairs,
        "strictly-ordered in-node pairs,",
        b_pair_bad,
        "inverted,",
        b_pair_tie,
        "collapsed to a tie",
    )
    if b_pairs < 10:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK ***: too few ordered pairs to say"
            " anything"
        )
    if b_pair_bad != 0:
        failures += 1
        print("  arm B FAILED: the published key inverted a real ordering")

    # ------------------------------------------------------------------
    # ARM C -- the order, on synthetic candidates including adversarial
    # near-ties. The fixture above cannot build a near-tie on purpose; this
    # can, and a check that only ever sees well-separated candidates has not
    # tested the truncation at all.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm C] the ORDER over synthetic candidates -- hashed, then"
        " adversarial near-ties built to land in the SAME truncation bucket"
    )
    var c_sizes = List[Int]()
    c_sizes.append(1024)
    c_sizes.append(65536)
    c_sizes.append(1 << 20)
    for si in range(len(c_sizes)):
        var n = c_sizes[si]
        var j = regression_key_shift(n)
        var slot = Int64(1) << Int64(REGRESSION_SUM_BITS - 1)

        # (a) HASHED, well separated, and SHARING THE NODE'S TOTAL.
        #
        # THE SHARED TOTAL IS NOT A CONVENIENCE, and this check learned it the
        # hard way: its first run drew an independent `sum_total` per candidate
        # and counted 66 of 1128 "inversions" at n = 1024. Nothing was wrong
        # with the key. DEVIATION 189 drops `sum_T^2/n` because it is a NODE
        # constant, so candidates carrying different node totals are not
        # comparable by this key and never occur -- `split_reduce_kernel`
        # reduces one node per `block_idx.y`. The fixture was the defect;
        # sub-arm (c) below keeps that fact measured instead of merely fixed.
        var node_total = Int64(
            Int((cell_hash(SEED, si, 3, SALT_LABEL) >> 19) % UInt64(1 << 29))
        ) - Int64(1 << 28)
        var hashed_pairs = 0
        var hashed_bad = 0
        var cand_sl = List[Int64]()
        var cand_nl = List[Int]()
        for t in range(48):
            var h = cell_hash(SEED, t, si, SALT_X)
            var nl = 1 + Int(h % UInt64(n - 1))
            var sl = Int64(Int((h >> 20) % UInt64(1 << 29))) - Int64(1 << 28)
            cand_nl.append(nl)
            cand_sl.append(sl)
        for i in range(len(cand_nl)):
            for k in range(i + 1, len(cand_nl)):
                var nla = cand_nl[i]
                var nlb = cand_nl[k]
                var pa = mse_proxy_exact(
                    cand_sl[i], nla, node_total - cand_sl[i], n - nla
                )
                var pb = mse_proxy_exact(
                    cand_sl[k], nlb, node_total - cand_sl[k], n - nlb
                )
                var w = compare_mse_proxy_exact(pa[0], pa[1], pb[0], pb[1])
                if w == 0:
                    continue
                hashed_pairs += 1
                var ka = _key_reference(cand_sl[i], node_total, nla, n - nla, n)
                var kb = _key_reference(cand_sl[k], node_total, nlb, n - nlb, n)
                if _cmp_rational(ka[0], ka[1], kb[0], kb[1]) != w:
                    hashed_bad += 1

        # (b) ADVERSARIAL: adjacent n_left, and the second candidate's `A`
        # placed inside the first one's truncation bucket. Same node total.
        var adv_pairs = 0
        var adv_bad = 0
        var adv_tie = 0
        for t in range(4000):
            var h = cell_hash(SEED, t, si + 50, SALT_LABEL)
            var nla = 2 + Int(h % UInt64(n - 3))
            var nlb = nla + 1 if (h >> 3) % 2 == 0 else nla - 1
            if nlb < 1 or nlb >= n:
                continue
            var st = node_total
            var sla = Int64(
                Int(
                    (cell_hash(SEED, t, si + 50, SALT_X) >> 11)
                    % UInt64(1 << 29)
                )
            ) - Int64(1 << 28)
            var aa = sla * Int64(n - nla) - (st - sla) * Int64(nla)
            var abs_aa = aa if aa >= 0 else -aa
            # `A_b = sl_b * n - st * nl_b`, so solve for the `sl_b` whose `A`
            # lands on |A_a| and walk a few units either side of it.
            var slb = (abs_aa + st * Int64(nlb)) // Int64(n)
            for d in range(-2, 3):
                var use = slb + Int64(d)
                if use > slot or use < -slot:
                    continue
                var pa = mse_proxy_exact(sla, nla, st - sla, n - nla)
                var pb = mse_proxy_exact(use, nlb, st - use, n - nlb)
                var w = compare_mse_proxy_exact(pa[0], pa[1], pb[0], pb[1])
                if w == 0:
                    continue
                adv_pairs += 1
                var ka = _key_reference(sla, st, nla, n - nla, n)
                var kb = _key_reference(use, st, nlb, n - nlb, n)
                var g = _cmp_rational(ka[0], ka[1], kb[0], kb[1])
                if g == 0:
                    adv_tie += 1
                elif g != w:
                    adv_bad += 1

        # (c) THE CONTROL for (a): the SAME candidates with a per-candidate
        # node total. This must be BADLY wrong -- if it is not, the fixture
        # cannot tell a node-local key from a global one and (a) proves
        # nothing.
        var cross_pairs = 0
        var cross_bad = 0
        for i in range(len(cand_nl)):
            for k in range(i + 1, len(cand_nl)):
                var nla = cand_nl[i]
                var nlb = cand_nl[k]
                var sta = cand_sl[i] + Int64(i) * Int64(1 << 20)
                var stb = cand_sl[k] + Int64(k) * Int64(1 << 20)
                var pa = mse_proxy_exact(
                    cand_sl[i], nla, sta - cand_sl[i], n - nla
                )
                var pb = mse_proxy_exact(
                    cand_sl[k], nlb, stb - cand_sl[k], n - nlb
                )
                var w = compare_mse_proxy_exact(pa[0], pa[1], pb[0], pb[1])
                if w == 0:
                    continue
                cross_pairs += 1
                var ka = _key_reference(cand_sl[i], sta, nla, n - nla, n)
                var kb = _key_reference(cand_sl[k], stb, nlb, n - nlb, n)
                if _cmp_rational(ka[0], ka[1], kb[0], kb[1]) != w:
                    cross_bad += 1

        print(
            "  n =",
            n,
            " j =",
            j,
            ":",
            hashed_pairs,
            "hashed pairs,",
            hashed_bad,
            "inverted |",
            adv_pairs,
            "adversarial near-tie pairs,",
            adv_bad,
            "inverted,",
            adv_tie,
            "tied | cross-node control:",
            cross_bad,
            "of",
            cross_pairs,
            "inverted",
        )
        if hashed_pairs < 100 or adv_pairs < 100:
            failures += 1
            print("  *** DEFECT IN THE CHECK ***: too few pairs at n =", n)
        if hashed_bad != 0:
            failures += 1
            print(
                "  arm C FAILED: the key inverted an ordinary, well-separated"
                " pair of ONE NODE at n =",
                n,
            )
        if cross_bad == 0:
            failures += 1
            print(
                "  *** DEFECT IN THE CHECK ***: the cross-node control did"
                " not go wrong, so sub-arm (a) cannot distinguish a node-local"
                " key from one that ignores the node constant"
            )
    print(
        "  the adversarial inversions are REAL and are DEVIATION 192's"
        " measured price: no pair of Int64 fields can hold the exact proxy"
        " above ten rows, so the question is which approximation, and arm G"
        " is where that was decided."
    )

    # ------------------------------------------------------------------
    # ARM D -- the bound.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm D] the bound, MEASURED: sklearn's proxy in Int64, the published"
        " key's precondition, and the device's own refusal"
    )
    var slot_max = Int64(1) << Int64(REGRESSION_SUM_BITS)
    var m = slot_max - Int64(1)
    var d_wrap_at = 0
    for n in range(2, 40):
        var nr = Int64(n - 1)
        var exact = Int128(m) * Int128(m) * Int128(nr)
        var as_i64 = m * m * nr
        if Int128(as_i64) != exact:
            d_wrap_at = n
            print(
                "  sklearn's proxy numerator at |sum_L| = 2^30-1, sum_R = 0,"
                " n_L = 1:"
            )
            print("      n =", n - 1, "  true", Int128(m) * Int128(m) * Int128(nr - 1), " Int64 agrees")
            print("      n =", n, "  true", exact)
            print("      Int64 holds up to  ", Int64.MAX)
            print("      the Int64 form reads", as_i64, "-- WRAPPED, and NEGATIVE")
            break
    if d_wrap_at != 10:
        failures += 1
        print(
            "  arm D FAILED: the wrap is at n =",
            d_wrap_at,
            "and DEVIATION 190 records 10. One of the two is wrong.",
        )
    else:
        print(
            "  so publishing sklearn's proxy is unavailable above NINE ROWS at"
            " the shipped scale, which is the whole reason for DEVIATION 189"
        )

    # The published key's precondition, computed in Int128 rather than
    # asserted, at the boundary and on both sides of it.
    var bound_bad = 0
    for si in range(6):
        var n = 1 << (2 * si + 2)
        if not regression_key_bound_ok(n, Int(slot_max)):
            bound_bad += 1
            print("  bound FAILED at n =", n, "with |sum| = 2^30 (must hold)")
        if regression_key_bound_ok(n, Int(slot_max * Int64(2))):
            bound_bad += 1
            print("  bound FAILED at n =", n, "with |sum| = 2^31 (must refuse)")
    if bound_bad != 0:
        failures += 1
        print("  arm D FAILED: regression_key_bound_ok disagrees with itself")
    else:
        print(
            "  regression_key_bound_ok: exact at |sum| = 2^30 and refused at"
            " 2^31, at six node sizes from 4 to 4,194,304 -- computed in"
            " Int128, not asserted"
        )

    # The DEVICE's refusal, from the violating plane.
    var refused = 0
    var refused_elsewhere = 0
    var unguarded_scored = 0
    var unguarded_wrapped = 0
    for s in range(n_cells):
        if base_big.cells[s].status == SCORE_STATUS_REGRESSION_REFUSED:
            refused += 1
            if s // N_COLS != BIG_NODE and s // N_COLS != 0:
                refused_elsewhere += 1
        if (
            runs[5].cells[s].status == SCORE_STATUS_SCORED
            and base_big.cells[s].status == SCORE_STATUS_REGRESSION_REFUSED
        ):
            unguarded_scored += 1
            if runs[5].cells[s].gini_num < Int64(0):
                unguarded_wrapped += 1
    print(
        "  on the contract-violating plane the device REFUSED",
        refused,
        "cells (",
        refused_elsewhere,
        "of them outside the two nodes whose labels leave the slot, which"
        " must be 0); with the guard sabotaged those same cells published a"
        " key instead,",
        unguarded_wrapped,
        "of",
        unguarded_scored,
        "of them with a NEGATIVE numerator -- an actual Int64 wrap, observed"
        " in the shipping kernel",
    )
    if refused == 0 or refused_elsewhere != 0:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK ***: the refusal did not fire exactly"
            " where the fixture violates the contract"
        )
    if unguarded_wrapped == 0:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK ***: the unguarded arm never actually"
            " wrapped, so the guard is checked against nothing"
        )

    # ------------------------------------------------------------------
    # ARM E -- end to end through split_reduce_kernel.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm E] the DEVICE reduction over these candidates against a host"
        " fold, per node"
    )
    var cand = List[SplitExact]()
    for nid in range(n_nodes):
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            var c = base.cells[s].copy()
            # `score_to_candidate_kernel`'s policy (DEVIATION 182), replicated
            # here rather than imported: `builder.mojo` is another session's
            # file this run must not depend on compiling.
            if c.status == SCORE_STATUS_SCORED:
                cand.append(
                    SplitExact(
                        Split(
                            c.threshold,
                            colids[s],
                            Float32(0.0),
                            c.n_left,
                        ),
                        ExactKey(c.gini_num, c.gini_den, Int32(1)),
                    )
                )
            else:
                cand.append(SplitExact())

    var host_best = List[SplitExact]()
    for nid in range(n_nodes):
        var acc = SplitExact()
        for fslot in range(N_COLS):
            _ = acc.update(cand[nid * N_COLS + fslot], SPLIT_SAB_NONE)
        host_best.append(acc)

    var n_c = len(cand)
    var c_q = ctx.enqueue_create_buffer[DType.float32](n_c)
    var c_c = ctx.enqueue_create_buffer[DType.int32](n_c)
    var c_m = ctx.enqueue_create_buffer[DType.float32](n_c)
    var c_l = ctx.enqueue_create_buffer[DType.int32](n_c)
    var c_nu = ctx.enqueue_create_buffer[DType.int64](n_c)
    var c_de = ctx.enqueue_create_buffer[DType.int64](n_c)
    var c_v = ctx.enqueue_create_buffer[DType.int32](n_c)
    var e_q = ctx.enqueue_create_buffer[DType.float32](n_nodes)
    var e_c = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var e_m = ctx.enqueue_create_buffer[DType.float32](n_nodes)
    var e_l = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var e_nu = ctx.enqueue_create_buffer[DType.int64](n_nodes)
    var e_de = ctx.enqueue_create_buffer[DType.int64](n_nodes)
    var e_v = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var e_mg = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var e_nw = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var e_mx = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var e_nb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var e_nc = ctx.enqueue_create_buffer[DType.int32](n_nodes)

    var t_q = ctx.enqueue_create_host_buffer[DType.float32](n_c)
    var t_c = ctx.enqueue_create_host_buffer[DType.int32](n_c)
    var t_m = ctx.enqueue_create_host_buffer[DType.float32](n_c)
    var t_l = ctx.enqueue_create_host_buffer[DType.int32](n_c)
    var t_nu = ctx.enqueue_create_host_buffer[DType.int64](n_c)
    var t_de = ctx.enqueue_create_host_buffer[DType.int64](n_c)
    var t_v = ctx.enqueue_create_host_buffer[DType.int32](n_c)
    var t_nb = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    var t_nc = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    ctx.synchronize()
    for i in range(n_c):
        t_q.unsafe_ptr().unsafe_store(i, cand[i].split.quesval)
        t_c.unsafe_ptr().unsafe_store(i, cand[i].split.colid)
        t_m.unsafe_ptr().unsafe_store(i, cand[i].split.best_metric_val)
        t_l.unsafe_ptr().unsafe_store(i, cand[i].split.n_left)
        t_nu.unsafe_ptr().unsafe_store(i, cand[i].key.num)
        t_de.unsafe_ptr().unsafe_store(i, cand[i].key.den)
        t_v.unsafe_ptr().unsafe_store(i, cand[i].key.valid)
    for i in range(n_nodes):
        t_nb.unsafe_ptr().unsafe_store(i, Int32(i * N_COLS))
        t_nc.unsafe_ptr().unsafe_store(i, Int32(N_COLS))
    ctx.enqueue_copy(dst_buf=c_q, src_ptr=t_q.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_c, src_ptr=t_c.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_m, src_ptr=t_m.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_l, src_ptr=t_l.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_nu, src_ptr=t_nu.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_de, src_ptr=t_de.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_v, src_ptr=t_v.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=e_nb, src_ptr=t_nb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=e_nc, src_ptr=t_nc.unsafe_ptr())
    ctx.enqueue_memset(e_mx, Int32(0))
    ctx.synchronize()

    ctx.enqueue_function[split_reduce_init_kernel](
        e_q.unsafe_ptr(),
        e_c.unsafe_ptr(),
        e_m.unsafe_ptr(),
        e_l.unsafe_ptr(),
        e_nu.unsafe_ptr(),
        e_de.unsafe_ptr(),
        e_v.unsafe_ptr(),
        e_mg.unsafe_ptr(),
        e_nw.unsafe_ptr(),
        Int32(n_nodes),
        grid_dim=ceildiv(n_nodes, 64),
        block_dim=64,
    )
    ctx.enqueue_function[split_reduce_kernel[RED_TPB]](
        e_q.unsafe_ptr(),
        e_c.unsafe_ptr(),
        e_m.unsafe_ptr(),
        e_l.unsafe_ptr(),
        e_nu.unsafe_ptr(),
        e_de.unsafe_ptr(),
        e_v.unsafe_ptr(),
        e_mg.unsafe_ptr(),
        e_nw.unsafe_ptr(),
        e_mx.unsafe_ptr(),
        c_q.unsafe_ptr(),
        c_c.unsafe_ptr(),
        c_m.unsafe_ptr(),
        c_l.unsafe_ptr(),
        c_nu.unsafe_ptr(),
        c_de.unsafe_ptr(),
        c_v.unsafe_ptr(),
        e_nb.unsafe_ptr(),
        e_nc.unsafe_ptr(),
        Int32(1),
        Int32(SPLIT_SAB_NONE),
        grid_dim=(1, n_nodes, 1),
        block_dim=(RED_TPB, 1, 1),
    )
    var u_q = ctx.enqueue_create_host_buffer[DType.float32](n_nodes)
    var u_c = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    var u_l = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    var u_nu = ctx.enqueue_create_host_buffer[DType.int64](n_nodes)
    var u_de = ctx.enqueue_create_host_buffer[DType.int64](n_nodes)
    var u_v = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    ctx.enqueue_copy(dst_buf=u_q, src_buf=e_q)
    ctx.enqueue_copy(dst_buf=u_c, src_buf=e_c)
    ctx.enqueue_copy(dst_buf=u_l, src_buf=e_l)
    ctx.enqueue_copy(dst_buf=u_nu, src_buf=e_nu)
    ctx.enqueue_copy(dst_buf=u_de, src_buf=e_de)
    ctx.enqueue_copy(dst_buf=u_v, src_buf=e_v)
    ctx.synchronize()

    var e_bad = 0
    var e_real = 0
    var e_distinct_cols = List[Int]()
    for nid in range(n_nodes):
        var w = host_best[nid].copy()
        var got_col = u_c.unsafe_ptr().unsafe_load(nid)
        if w.split.colid != Int32(-1):
            e_real += 1
            var already = False
            for k in range(len(e_distinct_cols)):
                if e_distinct_cols[k] == Int(got_col):
                    already = True
            if not already:
                e_distinct_cols.append(Int(got_col))
        if (
            got_col != w.split.colid
            or u_q.unsafe_ptr().unsafe_load(nid).to_bits()
            != w.split.quesval.to_bits()
            or u_l.unsafe_ptr().unsafe_load(nid) != w.split.n_left
            or u_nu.unsafe_ptr().unsafe_load(nid) != w.key.num
            or u_de.unsafe_ptr().unsafe_load(nid) != w.key.den
            or u_v.unsafe_ptr().unsafe_load(nid) != w.key.valid
        ):
            e_bad += 1
            print(
                "  WINNER MISMATCH node",
                nid,
                "device colid",
                got_col,
                "num",
                u_nu.unsafe_ptr().unsafe_load(nid),
                "| host colid",
                w.split.colid,
                "num",
                w.key.num,
            )
    if e_bad == 0:
        print(
            "  arm E OK:",
            n_nodes,
            "nodes reduced on the device pick the same winner as the host"
            " fold, field for field, of which",
            e_real,
            "had a real candidate; the winners span",
            len(e_distinct_cols),
            "distinct columns",
        )
    else:
        failures += 1
        print("  arm E FAILED:", e_bad, "of", n_nodes, "winners differ")
    if e_real < 3 or len(e_distinct_cols) < 2:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK ***: too few real winners, or every"
            " node picked the same column, so the reduction is not being"
            " asked anything"
        )

    # A CONTROL, and it is the point of the whole task: with the key switched
    # off -- which is what the finalize kernel published before this change --
    # the reduction has nothing to rank by and the winner is the highest
    # `colid` among scored cells. If that is the SAME answer as above, this
    # fixture cannot tell the two apart and arm E proves nothing.
    var blind = List[SplitExact]()
    for nid in range(n_nodes):
        var acc = SplitExact()
        for fslot in range(N_COLS):
            var c = cand[nid * N_COLS + fslot].copy()
            _ = acc.update(
                SplitExact(c.split, ExactKey()), SPLIT_SAB_NONE
            )
        blind.append(acc)
    var differs = 0
    for nid in range(n_nodes):
        if blind[nid].split.colid != host_best[nid].split.colid:
            differs += 1
    print(
        "  control -- the OLD behaviour (no key, every comparison ties,"
        " Split.update decides by colid) picks a different column in",
        differs,
        "of",
        n_nodes,
        "nodes",
    )
    if differs == 0:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK ***: the key changes nothing on this"
            " fixture, so arm E would pass with the key removed"
        )

    # ------------------------------------------------------------------
    # ARM F -- the sabotages.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm F] sabotage, one per MECHANISM, each a kernel ARGUMENT so every"
        " arm is the shipping binary"
    )
    for a in range(1, len(arms)):
        if not arm_is_sabotage[a]:
            continue
        var moved = 0
        for s in range(n_cells):
            var expected = (
                want_big[s].copy() if arm_big[a] else want[s].copy()
            )
            if not runs[a].cells[s].matches(expected):
                moved += 1
        print("  -", arm_name[a])
        if moved == 0:
            failures += 1
            print(
                "      *** DEFECT IN THE CHECK ***: 0 of",
                n_cells,
                "cells moved. Arm A cannot see this mechanism; the FIXTURE is"
                " what needs fixing.",
            )
        else:
            print("      RED as required:", moved, "of", n_cells, "cells wrong")

    # ------------------------------------------------------------------
    # ARM G -- the route, decided by measurement.
    # ------------------------------------------------------------------
    var restore_bad = 0
    for s in range(n_cells):
        if not runs[len(arms) - 1].cells[s].matches(base.cells[s]):
            restore_bad += 1
    if restore_bad == 0:
        print(
            "  - RESTORE: the shipping arm rerun after every sabotage is"
            " bit-identical to the shipping arm before them, in all",
            n_cells,
            "cells",
        )
    else:
        failures += 1
        print(
            "  - RESTORE FAILED:",
            restore_bad,
            "cells differ from the first shipping run, so a sabotage left"
            " state behind",
        )

    print("")
    print(
        "[arm G] the ROUTE: four keys scored against a Float64 ground truth,"
        " counting ordered candidate pairs each one gets BACKWARDS"
    )
    var g_sizes = List[Int]()
    g_sizes.append(1 << 16)
    g_sizes.append(1 << 20)
    for gi in range(len(g_sizes)):
        var n = g_sizes[gi]
        var n_cand = 20
        var yy = List[Float64]()
        var mag = Float64(0.0)
        for r in range(n):
            var v = Float64(signed_unit(cell_hash(SEED, r, gi, SALT_Y))) + 0.75
            yy.append(v)
            mag += -v if v < 0.0 else v
        var bits30 = accumulator_bits_for(n)
        var s30 = _scale_for_bits(mag, n, bits30)
        # Route 1: the narrowest accumulator for which sklearn's proxy still
        # fits Int64, `2b + log2(n) <= 63`.
        var b1 = (63 - ceil_log2(n)) // 2
        var s1 = _scale_for_bits(mag, n, b1)
        # Form B: keep sklearn's proxy, shift the SUMS instead of the labels.
        var kb = 0
        while 2 * (bits30 - kb) + ceil_log2(n) > 62:
            kb += 1

        var f_l = List[Float64]()
        var q30_l = List[Int64]()
        var q1_l = List[Int64]()
        var nl_l = List[Int]()
        var f_tot = Float64(0.0)
        var q30_tot = Int64(0)
        var q1_tot = Int64(0)
        for r in range(n):
            f_tot += yy[r]
            q30_tot += quantize(yy[r], s30)
            q1_tot += quantize(yy[r], s1)
        for t in range(n_cand):
            var fl = Float64(0.0)
            var a30 = Int64(0)
            var a1 = Int64(0)
            var nl = 0
            for r in range(n):
                var h = cell_hash(SEED, r, t + 7 * gi, SALT_LABEL)
                # A hashed split whose balance varies with `t`, so the
                # candidates differ in BOTH the sum and the count.
                if Int(h % UInt64(64)) < 4 + 2 * t:
                    nl += 1
                    fl += yy[r]
                    a30 += quantize(yy[r], s30)
                    a1 += quantize(yy[r], s1)
            if nl == 0 or nl == n:
                continue
            f_l.append(fl)
            q30_l.append(a30)
            q1_l.append(a1)
            nl_l.append(nl)

        var tot = 0
        var bad30 = 0
        var bad1 = 0
        var badkey = 0
        var badshift = 0
        for i in range(len(nl_l)):
            for k in range(i + 1, len(nl_l)):
                var nla = nl_l[i]
                var nlb = nl_l[k]
                var nra = n - nla
                var nrb = n - nlb
                var ta = f_l[i] * f_l[i] / Float64(nla) + (
                    f_tot - f_l[i]
                ) * (f_tot - f_l[i]) / Float64(nra)
                var tb = f_l[k] * f_l[k] / Float64(nlb) + (
                    f_tot - f_l[k]
                ) * (f_tot - f_l[k]) / Float64(nrb)
                if ta == tb:
                    continue
                tot += 1
                var w = 1 if ta > tb else -1

                var pa = mse_proxy_exact(
                    q30_l[i], nla, q30_tot - q30_l[i], nra
                )
                var pb = mse_proxy_exact(
                    q30_l[k], nlb, q30_tot - q30_l[k], nrb
                )
                if compare_mse_proxy_exact(pa[0], pa[1], pb[0], pb[1]) != w:
                    bad30 += 1

                var ra = mse_proxy_exact(q1_l[i], nla, q1_tot - q1_l[i], nra)
                var rb = mse_proxy_exact(q1_l[k], nlb, q1_tot - q1_l[k], nrb)
                if compare_mse_proxy_exact(ra[0], ra[1], rb[0], rb[1]) != w:
                    bad1 += 1

                var ka = _key_reference(q30_l[i], q30_tot, nla, nra, n)
                var kk = _key_reference(q30_l[k], q30_tot, nlb, nrb, n)
                if _cmp_rational(ka[0], ka[1], kk[0], kk[1]) != w:
                    badkey += 1

                var sa_l = q30_l[i] >> Int64(kb)
                var sa_r = (q30_tot - q30_l[i]) >> Int64(kb)
                var sb_l = q30_l[k] >> Int64(kb)
                var sb_r = (q30_tot - q30_l[k]) >> Int64(kb)
                var xa = mse_proxy_exact(sa_l, nla, sa_r, nra)
                var xb = mse_proxy_exact(sb_l, nlb, sb_r, nrb)
                if compare_mse_proxy_exact(xa[0], xa[1], xb[0], xb[1]) != w:
                    badshift += 1

        print(
            "  n =",
            n,
            ":",
            tot,
            "ordered pairs against Float64 truth --",
        )
        print(
            "      exact Int128 proxy at b =",
            bits30,
            "(the ceiling, and it does NOT fit Int64):",
            bad30,
            "backwards",
        )
        print(
            "      THIS FILE'S published key (Int64):              ",
            badkey,
            "backwards",
        )
        print(
            "      route 1, narrowed accumulator b =",
            b1,
            "(Int64):    ",
            bad1,
            "backwards",
        )
        print(
            "      route 2b, sklearn's proxy on sums >>",
            kb,
            "(Int64):",
            badshift,
            "backwards",
        )
        if tot < 20:
            failures += 1
            print("  *** DEFECT IN THE CHECK ***: too few pairs at n =", n)
        if badkey > bad1 or badkey > badshift:
            failures += 1
            print(
                "  *** arm G: THE MEASUREMENT NO LONGER SUPPORTS THE DESIGN"
                " ***. The published key is supposed to be the best of the"
                " three that fit Int64; it is not, on this run. DEVIATION 190"
                " must be revisited, not this line."
            )
        if badkey > bad30:
            failures += 1
            print(
                "  *** arm G: the Int64 key beat by the Int128 form it"
                " approximates by more than nothing ***"
            )

    # The gain helper the caller needs for `split_not_valid` (DEVIATION 193).
    print("")
    print(
        "[arm G'] mse_gain_from_exact_totals, the regression counterpart of"
        " DEVIATION 183's host-side gain"
    )
    var gain_bad = 0
    var gain_nonzero = 0
    var inv_scale = 1.0 / scale
    for s in range(n_cells):
        var c = base.cells[s].copy()
        if c.status != SCORE_STATUS_SCORED:
            continue
        var g = mse_gain_from_exact_totals(
            Int64(Int(c.acc_left[0])),
            Int64(Int(c.acc_total[0])),
            Int(c.n_left),
            Int(c.n_total - c.n_left),
            inv_scale,
        )
        # cuML's MSE gain is a square over positive counts and cannot be
        # negative; `split_not_valid` compares it against
        # `min_impurity_decrease`, so a negative one would leaf every node.
        if not (g >= Float32(0.0)):
            gain_bad += 1
        if g > Float32(0.0):
            gain_nonzero += 1
    if gain_bad == 0 and gain_nonzero > 0:
        print(
            "  OK:",
            gain_nonzero,
            "scored cells have a strictly positive gain and none is negative"
            " or NaN, so cuML's zero-gain gate can be applied on this path",
        )
    else:
        failures += 1
        print(
            "  FAILED:",
            gain_bad,
            "gains negative or NaN,",
            gain_nonzero,
            "strictly positive",
        )

    print("")
    print(
        "[shared memory] split_reduce_kernel at TPB",
        RED_TPB,
        "wants",
        split_reduce_shared_bytes(RED_TPB, WARP_SIZE),
        "bytes with WARP_SIZE",
        WARP_SIZE,
    )

    print("")
    if failures == 0:
        print(
            "regression_score_check: PASS -- the device publishes a"
            " regression key the reduction can rank by, it is cuML's own MSE"
            " gain as an exact rational, its order is sklearn's proxy's order"
            " on every ordinary pair, and its price is measured rather than"
            " claimed"
        )
    else:
        print("regression_score_check: FAIL --", failures, "arm(s) red")
        raise Error("regression_score_check failed")
