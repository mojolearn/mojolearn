"""Do the DRAW and the SCORE kernels agree with the host, cell by cell, exactly?

    pixi run mojo run -I . extratrees/mojo_only/score_kernel_check.mojo

NO CUML COUNTERPART -- this is a check. It covers
`extratrees/ported/decisiontree/batched_levelalgo/kernels/
builder_kernels_impl.mojo::node_feature_score_kernel` and
`::node_feature_score_finalize_kernel`, which are steps 2, 3 and 4 of
DEVIATION 137 and carry DEVIATION BLOCKS 170-175. Their INPUT is the range
kernel's output, and this check produces that input by RUNNING the range
kernel, so the two steps are chained here rather than mocked.

WHY THE COMPARISON CAN BE EXACT, WITH NO TOLERANCE AT ALL
---------------------------------------------------------
Everything the score pass accumulates is an INTEGER: class counts for Gini,
and -- by DEVIATION 135's ruling -- fixed-point label sums for regression,
quantized once on the host so the device never converts a float at all.
Integer addition is associative and exact, so the device's per-thread strided
accumulation, its block reductions and its cross-block `atomicAdd` are OBLIGED
to produce the same value as a sequential host loop over `row_ids`, not merely
a close one. The threshold is not accumulated: it is the same pure function of
the same key (DEVIATION 130) evaluated twice, so it is compared on FLOAT BIT
PATTERNS. A tolerance anywhere here would hide a defect rather than absorb
float noise.

WHAT IS DELIBERATELY ADVERSARIAL ABOUT THE FIXTURE
--------------------------------------------------
1. **Column-major storage**, `dataset.h:24`, against a row-major fixture: the
   transpose is done here, so a `row * N + col` misread lands on a different
   cell and not on a different layout of the same cell.
2. **Shuffled `row_ids`, nodes that do not start at slot 0, and gaps between
   nodes**, so a slot index and a row id are different numbers and a kernel
   that walked from the previous node's end would read rows belonging to no
   node.
3. **`fslot` is not `col`**: each node's sampled-column list is its own
   permutation of all 15 columns.
4. **`work_item.idx` is not `nid`**: the node ids in the batch start at 100.
   The DRAW keys on `idx` (DEVIATION 130) and the range pass never reads it,
   so a kernel that keyed on the batch index would draw different thresholds
   without changing anything else.
5. **A RAGGED batch, both block paths present and NAMED by the device.** Node
   row counts are 0, 1, 3, 127, 128, 129 and 1000 against a 128-thread block.
   `out_n_blocks` is written only by the accumulate kernel, so it is the
   device's own report of the path -- and it is 0 for a skipped cell, which is
   how this check proves a constant feature was SKIPPED rather than scored to
   zero.
6. **The adversarial column shapes from `fixtures.mojo`**, including the
   near-constant triple one ulp below / at / above `FEATURE_THRESHOLD`. The
   `at` column is the only thing in this file that can catch the constant
   test's comparison direction, and it is why that sabotage has a
   one-cell-wide target rather than a broad one.
7. **Three columns built HERE with a span of a few ULPs at a base of 2^20.**
   sklearn's `:653-654` guard -- a draw landing exactly on `max` becomes `min`
   -- is measured by `range_draw_check` to fire on about 1.5% of draws across
   span magnitudes, and NEVER on the shaped columns. At base 2^20 the ulp is
   0.125, so a span of k ulps makes the guard fire with probability about
   1/(2k), and these columns put it inside this fixture. They also make rows
   land EXACTLY on the drawn threshold, which is the only way to see the
   difference between `<=` and `<`.
8. **Two NaN columns**, scattered and total, because DEVIATION 136's refusal
   is a branch and an unchecked branch is an unreached branch. `fixtures.mojo`
   is not modified; these live here, as they do in `range_kernel_check`.
9. **Three classes with unequal, hashed counts.** A per-class accumulator
   whose expected value is the same in every class verifies the total and
   nothing about placement.

THE ARMS
--------
  P. PRECONDITION -- the range kernel's output equals `node_feature_min_max`
     per cell, on bit patterns. This check's input has to be right before its
     subject can be wrong.
  A. PER-CELL, exact, both objectives: status, threshold bits, n_left,
     n_total, every accumulator, and the exact Gini rational.
  B. THE PATH, from `out_n_blocks`: single-block and multi-block nodes named
     and counted, and every skipped cell showing zero blocks.
  C. THE STATUSES, enumerated: all four legal outcomes must occur, and
     `UNVISITED` must not.
  D. SABOTAGE, one per MECHANISM, each a kernel ARGUMENT so every arm runs the
     SAME binary that ships, and each stating its prediction BEFORE the count.
     A sabotage that does not turn arm A red is a defect in this FIXTURE.
  E. THE PUBLISHED WIDTH -- the `Int64` Gini numerator's row bound, MEASURED
     by wrapping it, not asserted from the algebra (DEVIATION BLOCK 175).
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from max.gpu.host import DeviceContext

from extratrees.mojo_only.fixed_point import choose_scale, quantize
from extratrees.mojo_only.fixtures import (
    all_shapes,
    cell_hash,
    float_from_bits,
    shape_name,
    shaped_dataset,
    signed_unit,
    SALT_LABEL,
    SALT_X,
    SALT_Y,
)
from extratrees.mojo_only.pcg_rng import (
    key_for,
    PCGenerator,
    SplitKey,
    uniform_float,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    WorkloadInfo,
)
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    build_workload_info,
    draw_threshold,
    FEATURE_THRESHOLD,
    draw_threshold_raw,
    empty_scored_candidate,
    FeatureRange,
    feature_range_at,
    node_feature_is_constant,
    node_feature_min_max,
    node_feature_range_init_kernel,
    node_feature_range_kernel,
    node_feature_score_finalize_kernel,
    node_feature_score_host,
    node_feature_score_init_kernel,
    node_feature_score_kernel,
    RANGE_SAB_NONE,
    score_row_bound_ok,
    ScoredCandidate,
    scored_candidate_at,
    SCORE_MAX_ROWS_EXACT,
    SCORE_SAB_BLOCK0_ONLY,
    SCORE_SAB_CONSTANT_STRICT,
    SCORE_SAB_FLOAT_ACCUM,
    SCORE_SAB_NO_MAX_GUARD,
    SCORE_SAB_NO_ROW_IDS,
    SCORE_SAB_NONE,
    SCORE_SAB_SCALE_X2,
    SCORE_SAB_SIDE_INVERTED,
    SCORE_SAB_STRICT_LESS,
    SCORE_SAB_TOTAL_IS_LEFT,
    SCORE_STATUS_CONSTANT,
    SCORE_STATUS_MISSING_REFUSED,
    SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF,
    SCORE_STATUS_SCORED,
    SCORE_STATUS_UNVISITED,
    TPB_DEFAULT,
)


comptime TPB = TPB_DEFAULT
comptime MAX_ACC = 4
"""The comptime accumulator bound this check instantiates the kernel at
(DEVIATION 172). Three classes plus one unused slot: a bound EQUAL to the class
count would not notice a kernel that looped to `MAX_ACC` instead of `n_acc`."""

comptime N_ROWS = 2048
comptime N_CLASSES = 3
comptime MIN_SAMPLES_LEAF = 1
"""sklearn's default (`_classes.py`, `min_samples_leaf=1`)."""

comptime MIN_SAMPLES_LEAF_ALT = 2
"""AND A SECOND VALUE, because at 1 the rejection branch of
`_splitter.pyx:664-666` CANNOT FIRE in this formulation, which the first run of
this check discovered by finding zero cells in that state.

The proof, which is short: after sklearn's `:653-654` guard the threshold lies
in `[min, max)`. The rows holding `min` satisfy `value <= threshold`, so
`n_left >= 1`; the rows holding `max` do not, so `n_right >= 1`. Both children
are non-empty by construction, so `min_samples_leaf = 1` rejects nothing. At 2
the branch is live -- the outlier and one-odd-row columns put exactly one row
on one side -- so the check runs BOTH, per rule 8."""
comptime TREE_ID = 5
comptime QNAN_BITS: UInt32 = 0x7FC00000
"""A quiet NaN as a BIT PATTERN, for the reason `range_kernel_check` gives:
`0.0 / 0.0` is a constant the compiler may fold differently at different
optimisation levels."""

comptime COL_SOME_NAN = 10
comptime COL_ALL_NAN = 11
comptime COL_ULP2 = 12
comptime COL_ULP4 = 13
comptime COL_ULP8 = 14
comptime N_COLS = 15

comptime ULP_BASE: Float32 = 1048576.0
"""2^20, where the float32 ulp is exactly 0.125."""
comptime ULP_STEP: Float32 = 0.125


@fieldwise_init
struct ScoreRun(Copyable, Movable):
    """One launch pair's output, read back and reassembled per cell."""

    var cells: List[ScoredCandidate]
    var n_blocks: List[Int32]


def _node_ranges() -> List[InstanceRange]:
    """The RAGGED batch, the same shape `range_kernel_check` uses: against
    TPB = 128 these are 1, 1, 1, 1, 1, 2 and 8 blocks, straddling the boundary
    on both sides (127 / 128 / 129), including a node of ONE row and a node of
    NONE, and leaving gaps between the nodes."""
    var out = List[InstanceRange]()
    out.append(InstanceRange(Int32(7), Int32(1)))
    out.append(InstanceRange(Int32(11), Int32(3)))
    out.append(InstanceRange(Int32(20), Int32(127)))
    out.append(InstanceRange(Int32(200), Int32(128)))
    out.append(InstanceRange(Int32(400), Int32(129)))
    out.append(InstanceRange(Int32(600), Int32(1000)))
    out.append(InstanceRange(Int32(1900), Int32(0)))
    return out^


def _shuffled_row_ids(n_rows: Int) -> List[Int32]:
    """A Fisher-Yates shuffle over a deterministic hash, so a slot index and
    the row id it holds are different numbers."""
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


def _ulp_column_value(seed: UInt64, row: Int, col: Int, n_ulps: Int) -> Float32:
    """A column at base 2^20 spanning `n_ulps` float32 ulps.

    Rows 0 and 1 are pinned to the two ends so min and max are a closed form,
    exactly as `fixtures.mojo`'s two-valued shapes are; the rest are scattered
    on a hash so placement matters. See the module docstring, point 7: this is
    the column that makes sklearn's `== max -> min` guard fire and that puts
    rows exactly ON the drawn threshold.
    """
    if row == 0:
        return ULP_BASE
    if row == 1:
        return ULP_BASE + ULP_STEP * Float32(n_ulps)
    var h = cell_hash(seed, row, col, SALT_X)
    return ULP_BASE + ULP_STEP * Float32(Int((h >> 37) % UInt64(n_ulps + 1)))


def _build_columns(seed: UInt64) -> List[Float32]:
    """The COLUMN-MAJOR feature plane, `data[col * M + row]` (`dataset.h:24`).

    Ten shaped columns from `fixtures.mojo`, two NaN columns and three
    ulp-span columns built here. `fixtures.mojo` is not modified.
    """
    var shapes = all_shapes()
    var fixture = shaped_dataset(seed, N_ROWS, shapes)
    var qnan = float_from_bits(QNAN_BITS)
    var flat = List[Float32](length=N_COLS * N_ROWS, fill=Float32(0.0))
    for c in range(fixture.n_cols):
        for r in range(N_ROWS):
            flat[c * N_ROWS + r] = fixture.value(r, c)
    for r in range(N_ROWS):
        # Scattered on a hash bit, not blocked: placement has to matter.
        var h = cell_hash(seed, r, COL_SOME_NAN, SALT_X)
        if (h >> 33) % 5 == 0:
            flat[COL_SOME_NAN * N_ROWS + r] = qnan
        else:
            flat[COL_SOME_NAN * N_ROWS + r] = signed_unit(h) * 50.0
        flat[COL_ALL_NAN * N_ROWS + r] = qnan
        flat[COL_ULP2 * N_ROWS + r] = _ulp_column_value(seed, r, COL_ULP2, 2)
        flat[COL_ULP4 * N_ROWS + r] = _ulp_column_value(seed, r, COL_ULP4, 4)
        flat[COL_ULP8 * N_ROWS + r] = _ulp_column_value(seed, r, COL_ULP8, 8)
    return flat^


def _colids(n_nodes: Int) -> List[Int32]:
    """Each node samples ALL the columns, in its OWN permutation, so `fslot`
    and `col` are different numbers. 7 and 15 are coprime, so every row of the
    table is a permutation -- which the caller CHECKS rather than assumes."""
    var out = List[Int32]()
    for nid in range(n_nodes):
        for fslot in range(N_COLS):
            out.append(Int32((fslot * 7 + nid * 4) % N_COLS))
    return out^


def _guard_census(
    ranges: List[InstanceRange],
    colids: List[Int32],
    extents: List[FeatureRange],
    seed: UInt64,
    work_items: List[NodeWorkItem],
) -> Int:
    """How many cells of this fixture make sklearn's `:653-654` guard fire.

    A count, not an assertion: the caller uses it to pick a seed and then FAILS
    if it is zero, because a sabotage aimed at a branch no cell reaches is a
    defect in the fixture and not a pass.
    """
    var n = 0
    for nid in range(len(ranges)):
        var count = Int(ranges[nid].count)
        var node_id = UInt32(Int(work_items[nid].idx))
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            var col = Int(colids[s])
            var e = extents[s]
            if e.n_missing != 0:
                continue
            if node_feature_is_constant(e, Int32(count)):
                continue
            var key = key_for(seed, UInt32(TREE_ID), node_id, UInt32(col))
            if draw_threshold_raw(key, e) == e.max_value:
                n += 1
    return n


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
    return "?"


def _col_name(col: Int, shapes: List[Int]) -> String:
    if col == COL_SOME_NAN:
        return "scattered NaN"
    if col == COL_ALL_NAN:
        return "all NaN"
    if col == COL_ULP2:
        return "2-ulp span at 2^20"
    if col == COL_ULP4:
        return "4-ulp span at 2^20"
    if col == COL_ULP8:
        return "8-ulp span at 2^20"
    return shape_name(shapes[col])


def _vendor() -> String:
    """Which GPU family this run used. A HOST-side check (`has_*`), not a
    target check (`is_*`): the kernel is one GPU-agnostic source and nothing in
    it branches on the vendor."""
    comptime if has_apple_gpu_accelerator():
        return "Apple"
    elif has_nvidia_gpu_accelerator():
        return "NVIDIA"
    elif has_amd_gpu_accelerator():
        return "AMD"
    else:
        return "unknown vendor"


def _sabotage_verdict(
    name: String,
    prediction: String,
    run: ScoreRun,
    want: List[ScoredCandidate],
    n_cells: Int,
) -> Int:
    """A sabotage must turn arm A RED. If it does not, the FIXTURE is the
    defect -- it means arm A cannot see the mechanism at all."""
    var wrong = 0
    for s in range(n_cells):
        if not run.cells[s].matches(want[s]):
            wrong += 1
    print("  -", name)
    print("      predicted:", prediction)
    if wrong == 0:
        print(
            "      *** DEFECT IN THE CHECK ***: 0 of",
            n_cells,
            "cells moved. Arm A cannot see this mechanism; the fixture is what"
            " needs fixing, not the kernel.",
        )
        return 1
    print("      RED as required:", wrong, "of", n_cells, "cells wrong")
    return 0


def _moved_exactly(
    name: String,
    run: ScoreRun,
    base: ScoreRun,
    expected_moved: List[Bool],
) -> Int:
    """A sabotage whose prediction is 'most of them' cannot fail. These predict
    an exact SET of cells, computed rather than guessed."""
    var wrong_shape = 0
    var moved = 0
    var want_n = 0
    for s in range(len(expected_moved)):
        var did_move = not run.cells[s].matches(base.cells[s])
        if did_move:
            moved += 1
        if expected_moved[s]:
            want_n += 1
        if did_move != expected_moved[s]:
            wrong_shape += 1
    if wrong_shape == 0:
        print(
            "  -",
            name,
            "moved EXACTLY the",
            moved,
            "cells predicted, and no others",
        )
        return 0
    print(
        "  -",
        name,
        "SHAPE WRONG:",
        moved,
        "cells moved,",
        want_n,
        "predicted,",
        wrong_shape,
        "cells disagree with the prediction",
    )
    return 1


def check_objective[
    CLASSIFICATION: Bool
](
    ctx: DeviceContext,
    p_min: MutPointer[Float32, MutAnyOrigin],
    p_max: MutPointer[Float32, MutAnyOrigin],
    p_missing: MutPointer[Int32, MutAnyOrigin],
    p_data: MutPointer[Float32, MutAnyOrigin],
    p_row_ids: MutPointer[Int32, MutAnyOrigin],
    p_labels: MutPointer[Int32, MutAnyOrigin],
    p_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    p_wl: MutPointer[WorkloadInfo, MutAnyOrigin],
    p_colids: MutPointer[Int32, MutAnyOrigin],
    h_data: MutPointer[Float32, MutAnyOrigin],
    h_row_ids: MutPointer[Int32, MutAnyOrigin],
    h_labels: MutPointer[Int32, MutAnyOrigin],
    extents: List[FeatureRange],
    colids: List[Int32],
    work_items: List[NodeWorkItem],
    ranges: List[InstanceRange],
    n_blocks_dimx: Int,
    seed: UInt64,
    n_acc: Int,
    shapes: List[Int],
) raises -> Int:
    """Everything for ONE objective: the host oracle, the shipping run, every
    sabotage, and arms A, B, C and D.

    The objective is a COMPTIME parameter because cuML's is: they instantiate
    `computeSplitKernel` once per objective template rather than branching in
    the kernel. Both instantiations are enumerated by `main`, per rule 8 -- a
    parameter that selects a kernel path is a parameter the checks enumerate.
    """
    var n_nodes = len(ranges)
    var n_cells = n_nodes * N_COLS
    var n_acc_cells = n_cells * n_acc
    var failures = 0
    var tag = "classification" if CLASSIFICATION else "regression"

    print("")
    print("========================================================")
    print("[objective]", tag, "-- n_acc =", n_acc)
    print("========================================================")

    # ------------------------------------------------------------------
    # The host oracle, per cell. `node_feature_score_host` is the sequential
    # transcription and it is the authority here.
    # ------------------------------------------------------------------
    var want = List[ScoredCandidate]()
    for nid in range(n_nodes):
        var begin = Int(ranges[nid].begin)
        var count = Int(ranges[nid].count)
        var node_id = UInt32(Int(work_items[nid].idx))
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            var col = Int(colids[s])
            var key = key_for(seed, UInt32(TREE_ID), node_id, UInt32(col))
            want.append(
                node_feature_score_host(
                    h_data,
                    h_row_ids,
                    h_labels,
                    N_ROWS,
                    begin,
                    count,
                    col,
                    extents[s],
                    key,
                    n_acc,
                    CLASSIFICATION,
                    MIN_SAMPLES_LEAF,
                )
            )

    # ------------------------------------------------------------------
    # The device. One allocation set; one launch PAIR per arm, all of the
    # SAME binary, the arm selected by a kernel argument.
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
    ctx.synchronize()

    var arms = List[Int32]()
    var arm_msl = List[Int]()
    arms.append(SCORE_SAB_NONE)
    arms.append(SCORE_SAB_CONSTANT_STRICT)
    arms.append(SCORE_SAB_NO_MAX_GUARD)
    arms.append(SCORE_SAB_SIDE_INVERTED)
    arms.append(SCORE_SAB_STRICT_LESS)
    arms.append(SCORE_SAB_NO_ROW_IDS)
    arms.append(SCORE_SAB_BLOCK0_ONLY)
    arms.append(SCORE_SAB_TOTAL_IS_LEFT)
    comptime if not CLASSIFICATION:
        # The two fixed-point mechanisms only exist on the regression path.
        arms.append(SCORE_SAB_SCALE_X2)
        arms.append(SCORE_SAB_FLOAT_ACCUM)
    for _ in range(len(arms)):
        arm_msl.append(MIN_SAMPLES_LEAF)
    # The LAST arm is not a sabotage: it is the shipping kernel at the other
    # `min_samples_leaf`, so that `_splitter.pyx:664-666` is exercised at a
    # value where it can fire. See MIN_SAMPLES_LEAF_ALT.
    arms.append(SCORE_SAB_NONE)
    arm_msl.append(MIN_SAMPLES_LEAF_ALT)
    var alt_arm = len(arms) - 1

    var runs = List[ScoreRun]()
    for a in range(len(arms)):
        var sab = arms[a]
        comptime init_kernel = node_feature_score_init_kernel
        ctx.enqueue_function[init_kernel](
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
        comptime acc_kernel = node_feature_score_kernel[
            TPB, MAX_ACC, CLASSIFICATION
        ]
        ctx.enqueue_function[acc_kernel](
            d_n_left.unsafe_ptr(),
            d_n_total.unsafe_ptr(),
            d_acc_left.unsafe_ptr(),
            d_acc_total.unsafe_ptr(),
            d_blocks.unsafe_ptr(),
            p_min,
            p_max,
            p_missing,
            p_data,
            p_row_ids,
            p_labels,
            p_items,
            p_wl,
            p_colids,
            Int32(N_ROWS),
            Int32(N_COLS),
            Int32(n_acc),
            seed,
            Int32(TREE_ID),
            sab,
            grid_dim=(n_blocks_dimx, N_COLS, 1),
            block_dim=(TPB, 1, 1),
        )
        comptime fin_kernel = node_feature_score_finalize_kernel[
            MAX_ACC, CLASSIFICATION
        ]
        ctx.enqueue_function[fin_kernel](
            d_status.unsafe_ptr(),
            d_thresh.unsafe_ptr(),
            d_num.unsafe_ptr(),
            d_den.unsafe_ptr(),
            d_n_left.unsafe_ptr(),
            d_n_total.unsafe_ptr(),
            d_acc_left.unsafe_ptr(),
            d_acc_total.unsafe_ptr(),
            p_min,
            p_max,
            p_missing,
            p_items,
            p_colids,
            Int32(n_cells),
            Int32(N_COLS),
            Int32(n_acc),
            seed,
            Int32(TREE_ID),
            Int32(arm_msl[a]),
            sab,
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )

        var o_status = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var o_thresh = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
        var o_n_left = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var o_n_total = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var o_num = ctx.enqueue_create_host_buffer[DType.int64](n_cells)
        var o_den = ctx.enqueue_create_host_buffer[DType.int64](n_cells)
        var o_blocks = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var o_left = ctx.enqueue_create_host_buffer[DType.int32](n_acc_cells)
        var o_total = ctx.enqueue_create_host_buffer[DType.int32](n_acc_cells)
        ctx.enqueue_copy(dst_buf=o_status, src_buf=d_status)
        ctx.enqueue_copy(dst_buf=o_thresh, src_buf=d_thresh)
        ctx.enqueue_copy(dst_buf=o_n_left, src_buf=d_n_left)
        ctx.enqueue_copy(dst_buf=o_n_total, src_buf=d_n_total)
        ctx.enqueue_copy(dst_buf=o_num, src_buf=d_num)
        ctx.enqueue_copy(dst_buf=o_den, src_buf=d_den)
        ctx.enqueue_copy(dst_buf=o_blocks, src_buf=d_blocks)
        ctx.enqueue_copy(dst_buf=o_left, src_buf=d_acc_left)
        ctx.enqueue_copy(dst_buf=o_total, src_buf=d_acc_total)
        ctx.synchronize()

        var cells = List[ScoredCandidate]()
        var blocks = List[Int32]()
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
                        n_acc,
                    )
                )
        for s in range(n_cells):
            blocks.append(o_blocks.unsafe_ptr().unsafe_load(s))
        runs.append(ScoreRun(cells^, blocks^))

    var base = runs[0].copy()

    # ------------------------------------------------------------------
    # ARM A -- per cell, exact, no tolerance.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm A]", tag, "-- per (node, feature) cell against"
        " node_feature_score_host: status, threshold BITS, n_left, n_total,"
        " every accumulator, the exact rational"
    )
    var ok = 0
    var bad = 0
    var reported = 0
    for nid in range(n_nodes):
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            if base.cells[s].matches(want[s]):
                ok += 1
            else:
                bad += 1
                if reported < 8:
                    reported += 1
                    var col = Int(colids[s])
                    print(
                        "  MISMATCH node",
                        nid,
                        "fslot",
                        fslot,
                        "col",
                        col,
                        "(",
                        _col_name(col, shapes),
                        ") got status",
                        _status_name(base.cells[s].status),
                        "thr",
                        base.cells[s].threshold,
                        "nL",
                        base.cells[s].n_left,
                        "nT",
                        base.cells[s].n_total,
                        "num",
                        base.cells[s].gini_num,
                        "| want status",
                        _status_name(want[s].status),
                        "thr",
                        want[s].threshold,
                        "nL",
                        want[s].n_left,
                        "nT",
                        want[s].n_total,
                        "num",
                        want[s].gini_num,
                    )
    if bad == 0:
        print(
            "  arm A OK:",
            ok,
            "of",
            n_cells,
            "cells identical to the host transcription, with the threshold"
            " compared on float BIT PATTERNS",
        )
    else:
        failures += 1
        print("  arm A FAILED:", bad, "of", n_cells, "cells wrong")

    # ------------------------------------------------------------------
    # ARM B -- which path each node took, from the DEVICE's own report.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm B]", tag, "-- the path each cell took, as reported by the"
        " accumulate kernel (out_n_blocks), not inferred"
    )
    var single = 0
    var multi = 0
    var skipped = 0
    var path_bad = 0
    for nid in range(n_nodes):
        var count = Int(ranges[nid].count)
        var want_blocks = ceildiv(count, TPB)
        if want_blocks < 1:
            want_blocks = 1
        var visited_cells = 0
        var skipped_cells = 0
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            var st = want[s].status
            var expect: Int
            if st == SCORE_STATUS_CONSTANT or st == SCORE_STATUS_MISSING_REFUSED:
                expect = 0
                skipped_cells += 1
            else:
                expect = want_blocks
                visited_cells += 1
            if Int(base.n_blocks[s]) != expect:
                path_bad += 1
        skipped += skipped_cells
        if want_blocks == 1:
            single += 1
        else:
            multi += 1
        var label = "SINGLE-BLOCK" if want_blocks == 1 else "MULTI-BLOCK"
        print(
            "  node",
            nid,
            "(idx",
            work_items[nid].idx,
            ") rows",
            count,
            "->",
            want_blocks,
            "block(s),",
            label,
            "--",
            visited_cells,
            "cells accumulated by exactly",
            want_blocks,
            "block(s),",
            skipped_cells,
            "skipped with ZERO blocks",
        )
    if path_bad == 0 and single > 0 and multi > 0 and skipped > 0:
        print(
            "  arm B OK:",
            single,
            "nodes took the single-block path,",
            multi,
            "the multi-block path, and",
            skipped,
            "cells were SKIPPED (0 blocks) rather than scored to zero",
        )
    else:
        failures += 1
        print(
            "  arm B FAILED: path_bad",
            path_bad,
            "single",
            single,
            "multi",
            multi,
            "skipped",
            skipped,
        )

    # ------------------------------------------------------------------
    # ARM C -- every status enumerated. Rule 8: a path no check ran is a path
    # no check speaks for.
    # ------------------------------------------------------------------
    print("")
    print("[arm C]", tag, "-- the four legal statuses, counted from the device")
    var n_unvisited = 0
    var n_scored = 0
    var n_constant = 0
    var n_missing = 0
    var n_rejected = 0
    for s in range(n_cells):
        var st = base.cells[s].status
        if st == SCORE_STATUS_UNVISITED:
            n_unvisited += 1
        elif st == SCORE_STATUS_SCORED:
            n_scored += 1
        elif st == SCORE_STATUS_CONSTANT:
            n_constant += 1
        elif st == SCORE_STATUS_MISSING_REFUSED:
            n_missing += 1
        elif st == SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF:
            n_rejected += 1
    print(
        "  SCORED",
        n_scored,
        " CONSTANT",
        n_constant,
        " MISSING_REFUSED",
        n_missing,
        " REJECTED_MIN_SAMPLES_LEAF",
        n_rejected,
        " UNVISITED",
        n_unvisited,
    )
    if (
        n_unvisited == 0
        and n_scored > 0
        and n_constant > 0
        and n_missing > 0
        and n_rejected == 0
    ):
        print(
            "  arm C OK: SCORED, CONSTANT and MISSING_REFUSED all occurred, no"
            " cell was left UNVISITED, and REJECTED_MIN_SAMPLES_LEAF occurred"
            " ZERO times -- which is not a gap but a PROOF: at"
            " min_samples_leaf = 1 the guarded threshold lies in [min, max),"
            " so the min rows go left and the max rows go right and neither"
            " child can be empty. Arm C2 runs the value where it CAN fire."
        )
    else:
        failures += 1
        print(
            "  arm C FAILED: UNVISITED must be 0; SCORED, CONSTANT and"
            " MISSING_REFUSED must not be; and REJECTED must be 0 at"
            " min_samples_leaf = 1, because a threshold in [min, max) cannot"
            " empty a child"
        )

    # ------------------------------------------------------------------
    # ARM C2 -- the same kernels at min_samples_leaf = 2, where
    # `_splitter.pyx:664-666` can fire. Rule 8: a parameter that selects a
    # path is a parameter the checks enumerate.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm C2]", tag, "-- the shipping kernel at min_samples_leaf =",
        MIN_SAMPLES_LEAF_ALT,
        ", where the rejection branch is reachable"
    )
    var want_alt = List[ScoredCandidate]()
    for nid in range(n_nodes):
        var begin2 = Int(ranges[nid].begin)
        var count2 = Int(ranges[nid].count)
        var node_id2 = UInt32(Int(work_items[nid].idx))
        for fslot in range(N_COLS):
            var s2 = nid * N_COLS + fslot
            var col2 = Int(colids[s2])
            var key2 = key_for(seed, UInt32(TREE_ID), node_id2, UInt32(col2))
            want_alt.append(
                node_feature_score_host(
                    h_data,
                    h_row_ids,
                    h_labels,
                    N_ROWS,
                    begin2,
                    count2,
                    col2,
                    extents[s2],
                    key2,
                    n_acc,
                    CLASSIFICATION,
                    MIN_SAMPLES_LEAF_ALT,
                )
            )
    var alt_bad = 0
    var alt_rejected = 0
    for s in range(n_cells):
        if not runs[alt_arm].cells[s].matches(want_alt[s]):
            alt_bad += 1
        if (
            runs[alt_arm].cells[s].status
            == SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF
        ):
            alt_rejected += 1
    if alt_bad == 0 and alt_rejected > 0:
        print(
            "  arm C2 OK:",
            n_cells,
            "cells identical to the host at min_samples_leaf =",
            MIN_SAMPLES_LEAF_ALT,
            "and",
            alt_rejected,
            "of them took the REJECTED_MIN_SAMPLES_LEAF branch",
        )
    else:
        failures += 1
        print(
            "  arm C2 FAILED:",
            alt_bad,
            "cells wrong,",
            alt_rejected,
            "rejected -- a rejection branch no cell reaches is a branch this"
            " check cannot speak for",
        )

    # ------------------------------------------------------------------
    # ARM D -- sabotage, one per mechanism, same binary, kernel argument.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm D]", tag, "-- sabotage, one per MECHANISM, selected by a kernel"
        " argument so every arm runs the binary that ships"
    )
    for a in range(1, len(arms)):
        if arm_msl[a] != MIN_SAMPLES_LEAF:
            continue
        var sab = arms[a]
        if sab == SCORE_SAB_CONSTANT_STRICT:
            failures += _sabotage_verdict(
                "the constant test's comparison DIRECTION (`max <= min + 1e-7`"
                " -> `max < min + 1e-7`, _splitter.pyx:616-617)",
                "moves ONLY the cells whose spread is EXACTLY"
                " FEATURE_THRESHOLD in float32 -- the near_const_equal column,"
                " in every node with rows",
                runs[a],
                want,
                n_cells,
            )
        elif sab == SCORE_SAB_NO_MAX_GUARD:
            failures += _sabotage_verdict(
                "sklearn's `threshold == max -> min` guard"
                " (_splitter.pyx:653-654)",
                "moves ONLY the cells where the RAW draw landed exactly on"
                " max; a threshold equal to max sends every row left",
                runs[a],
                want,
                n_cells,
            )
        elif sab == SCORE_SAB_SIDE_INVERTED:
            failures += _sabotage_verdict(
                "the left/right assignment (`value <= threshold` goes left,"
                " builder_kernels_impl.cuh:65-66, inverted to `>`)",
                "moves nearly every scored cell; no exact set is predicted"
                " because predicting it would mean re-implementing the"
                " sabotaged kernel on the host and then checking that",
                runs[a],
                want,
                n_cells,
            )
        elif sab == SCORE_SAB_STRICT_LESS:
            failures += _sabotage_verdict(
                "the BOUNDARY of that assignment (`<=` -> `<`, so a row EQUAL"
                " to the threshold goes right)",
                "moves ONLY the cells where some row sits exactly ON the drawn"
                " threshold -- which is what the ulp-span columns are for",
                runs[a],
                want,
                n_cells,
            )
        elif sab == SCORE_SAB_NO_ROW_IDS:
            failures += _sabotage_verdict(
                "the row_ids indirection (`row_ids[i]` -> `i`, for the feature"
                " value AND the label)",
                "moves nearly every scored cell: row_ids is a shuffle, so slot"
                " i and row row_ids[i] are different rows",
                runs[a],
                want,
                n_cells,
            )
        elif sab == SCORE_SAB_BLOCK0_ONLY:
            failures += _sabotage_verdict(
                "the cross-block atomicAdd merge (publish only from"
                " offset_blockid == 0)",
                "moves EXACTLY the scored cells of the two MULTI-BLOCK nodes:"
                " block 0 sees at most TPB rows, so n_total alone gives it"
                " away",
                runs[a],
                want,
                n_cells,
            )
        elif sab == SCORE_SAB_TOTAL_IS_LEFT:
            failures += _sabotage_verdict(
                "the node TOTALS covering every row (accumulate totals over"
                " left-going rows only, so cuML's `total - left` recovery of"
                " the right child, objectives.cuh:72-73, gives zero)",
                "moves EXACTLY the cells where some accumulator differs"
                " between left and total",
                runs[a],
                want,
                n_cells,
            )
        elif sab == SCORE_SAB_SCALE_X2:
            failures += _sabotage_verdict(
                "the fixed-point SCALE (device accumulates 2 * labels_q, i.e."
                " a scale that did not reach the device intact)",
                "moves EXACTLY the visited cells whose label sum is nonzero on"
                " one side or the other",
                runs[a],
                want,
                n_cells,
            )
        elif sab == SCORE_SAB_FLOAT_ACCUM:
            failures += _sabotage_verdict(
                "the fixed-point ACCUMULATOR itself (sum the labels in"
                " Float32 and truncate once, which is what DEVIATION 135"
                " ruled out)",
                "moves the cells whose nodes hold enough rows for a 24-bit"
                " mantissa to lose a low bit; no exact set is predicted"
                " because that set is a statement about float32 rounding of a"
                " particular thread partition, not about the mechanism",
                runs[a],
                want,
                n_cells,
            )
    # ------------------------------------------------------------------
    # ARM D' -- the narrow sabotages moved the cells they predicted and no
    # others. A sabotage whose prediction is "most of them" cannot fail.
    # ------------------------------------------------------------------
    print("")
    print(
        "[arm D']", tag, "-- the narrow sabotages moved EXACTLY the cells"
        " predicted, computed rather than guessed"
    )
    var pred_const = List[Bool]()
    var pred_guard = List[Bool]()
    var pred_block0 = List[Bool]()
    var pred_totals = List[Bool]()
    var pred_strict = List[Bool]()
    var pred_scale = List[Bool]()
    for nid in range(n_nodes):
        var begin = Int(ranges[nid].begin)
        var count = Int(ranges[nid].count)
        var node_id = UInt32(Int(work_items[nid].idx))
        var nblocks = ceildiv(count, TPB)
        if nblocks < 1:
            nblocks = 1
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            var col = Int(colids[s])
            var e = extents[s]
            var visited = (
                want[s].status == SCORE_STATUS_SCORED
                or want[s].status == SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF
            )

            # The constant test's direction: the two forms disagree exactly
            # when the spread is EXACTLY FEATURE_THRESHOLD in float32.
            var strict_const = Int32(count) == e.n_missing or (
                e.max_value < e.min_value + FEATURE_THRESHOLD
                and e.n_missing == 0
            )
            var normal_const = node_feature_is_constant(e, Int32(count))
            pred_const.append(
                e.n_missing == 0 and normal_const != strict_const
            )

            # The `== max -> min` guard: only where the RAW draw hit max.
            var guard = False
            if visited:
                var key = key_for(seed, UInt32(TREE_ID), node_id, UInt32(col))
                guard = draw_threshold_raw(key, e) == e.max_value
            pred_guard.append(guard)

            # The cross-block merge: block 0 sees at most TPB rows.
            pred_block0.append(visited and nblocks > 1)

            # The totals: they move iff some accumulator differs between the
            # left child and the node.
            var totals_move = False
            if visited:
                for k in range(n_acc):
                    if want[s].acc_total[k] != want[s].acc_left[k]:
                        totals_move = True
            pred_totals.append(totals_move)

            # The boundary: a row sitting exactly ON the threshold.
            var on_threshold = False
            if visited:
                for p in range(begin, begin + count):
                    var row = Int(h_row_ids[unsafe_offset=p])
                    if (
                        h_data[unsafe_offset = col * N_ROWS + row]
                        == want[s].threshold
                    ):
                        on_threshold = True
            pred_strict.append(on_threshold)

            # The scale: doubling a zero changes nothing.
            var scale_move = False
            comptime if not CLASSIFICATION:
                if visited and (
                    want[s].acc_left[0] != 0 or want[s].acc_total[0] != 0
                ):
                    scale_move = True
            pred_scale.append(scale_move)

    for a in range(1, len(arms)):
        if arm_msl[a] != MIN_SAMPLES_LEAF:
            continue
        var sab = arms[a]
        if sab == SCORE_SAB_CONSTANT_STRICT:
            failures += _moved_exactly(
                "constant test direction", runs[a], base, pred_const
            )
        elif sab == SCORE_SAB_NO_MAX_GUARD:
            failures += _moved_exactly(
                "the == max -> min guard", runs[a], base, pred_guard
            )
        elif sab == SCORE_SAB_BLOCK0_ONLY:
            failures += _moved_exactly(
                "the cross-block merge", runs[a], base, pred_block0
            )
        elif sab == SCORE_SAB_TOTAL_IS_LEFT:
            failures += _moved_exactly(
                "the node totals", runs[a], base, pred_totals
            )
        elif sab == SCORE_SAB_STRICT_LESS:
            failures += _moved_exactly(
                "the <= / < boundary", runs[a], base, pred_strict
            )
        elif sab == SCORE_SAB_SCALE_X2:
            failures += _moved_exactly(
                "the fixed-point scale", runs[a], base, pred_scale
            )

    return failures


def main() raises:
    comptime assert has_accelerator(), "this check must run on a GPU"

    var shapes = all_shapes()
    var ranges = _node_ranges()
    var n_nodes = len(ranges)
    var n_cells = n_nodes * N_COLS
    var colids = _colids(n_nodes)
    var row_ids = _shuffled_row_ids(N_ROWS)

    # The colids table must be a PERMUTATION per node -- if it were not, some
    # column would be tested twice and another never, and every count below
    # would still look plausible. Checked, not assumed.
    for nid in range(n_nodes):
        var seen = List[Int](length=N_COLS, fill=0)
        for fslot in range(N_COLS):
            seen[Int(colids[nid * N_COLS + fslot])] += 1
        for c in range(N_COLS):
            if seen[c] != 1:
                raise Error(
                    "the colids table is not a permutation for node "
                    + String(nid)
                    + "; column "
                    + String(c)
                    + " appears "
                    + String(seen[c])
                    + " times"
                )

    var work_items = List[NodeWorkItem]()
    for nid in range(n_nodes):
        # `idx` is the node's id IN THE TREE and deliberately is not the batch
        # index: the DRAW keys on it (DEVIATION 130) and the range pass never
        # reads it, so a kernel that keyed on `nid` would be invisible to
        # everything except the threshold.
        work_items.append(NodeWorkItem(Int32(100 + nid), Int32(3), ranges[nid]))

    # ---------------------------------------------------------------------
    # THE SEED IS CHOSEN, AND WHY THAT IS NOT CHERRY-PICKING. sklearn's
    # `:653-654` guard fires on a draw that lands exactly on `max`;
    # `range_draw_check` measures that at about 1.5% of draws across span
    # magnitudes and at ZERO on the shaped columns. A fixture that does not
    # reach the branch cannot check it, so the seed is chosen to REACH it --
    # the ulp-span columns are what make it reachable at all -- and the count
    # is asserted below, so this check FAILS rather than silently skipping if
    # the arithmetic ever changes. Nothing about the comparison depends on the
    # seed: every cell is compared against the host, whatever it holds.
    # ---------------------------------------------------------------------
    var candidate_seeds = List[UInt64]()
    candidate_seeds.append(0xC0FFEE_123)
    candidate_seeds.append(0xA11CE)
    candidate_seeds.append(0x51DE)
    candidate_seeds.append(0xBEEF01)
    candidate_seeds.append(0xD15EA5E)
    candidate_seeds.append(0x1234567)
    candidate_seeds.append(0x7E57)
    candidate_seeds.append(0xFACADE)

    var seed = candidate_seeds[0]
    var guard_cells = -1
    for si in range(len(candidate_seeds)):
        var s_try = candidate_seeds[si]
        var flat_try = _build_columns(s_try)
        var labels_try = List[Float32](length=N_ROWS, fill=Float32(0.0))
        var ds_try = Dataset(
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                flat_try.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                labels_try.unsafe_ptr()
            ),
            Int32(N_ROWS),
            Int32(N_COLS),
            Int32(N_ROWS),
            Int32(N_COLS),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](
                row_ids.unsafe_ptr()
            ),
            Int32(1),
        )
        var ext_try = List[FeatureRange]()
        for nid in range(n_nodes):
            for fslot in range(N_COLS):
                ext_try.append(
                    node_feature_min_max(
                        ds_try, work_items[nid], colids[nid * N_COLS + fslot]
                    )
                )
        var g = _guard_census(ranges, colids, ext_try, s_try, work_items)
        _ = flat_try.unsafe_ptr()
        _ = labels_try.unsafe_ptr()
        if g > guard_cells:
            guard_cells = g
            seed = s_try
        if g >= 2:
            break

    # ---------------------------------------------------------------------
    # The fixture, at the chosen seed.
    # ---------------------------------------------------------------------
    var flat = _build_columns(seed)

    # The labels. Both planes come from their OWN salt, so neither is a
    # function of any feature or of the row index. The three classes are
    # deliberately SKEWED (about 50 / 33 / 17): a per-class accumulator whose
    # expected value is the same in every class verifies the total and nothing
    # about placement.
    #
    # THE REGRESSION TARGET IS DELIBERATELY NOT ZERO-MEAN, and that is a
    # correction this check made to itself. With `signed_unit` alone the label
    # sum CANCELS: a node of 1000 rows holds a sum of a few thousand units
    # where its magnitudes sum to hundreds of thousands, every partial sum
    # stays under 2^24, and a Float32 accumulator is then EXACT -- so the
    # sabotage that exists to falsify DEVIATION 135 moved 0 of 105 cells on the
    # first run. DEVIATION 135's claim is about sums that REACH the slot, so
    # the fixture has to produce them. Here the target is mostly positive with
    # one row in eight strongly negative: the mean is about 1.0, so a 1000-row
    # node's fixed-point sum lands near 2^28 -- four bits past a float32
    # mantissa -- while the negative rows keep the sign path exercised.
    var labels_cls = List[Int32]()
    var y = List[Float32]()
    for r in range(N_ROWS):
        var h = cell_hash(seed, r, 0, SALT_LABEL)
        var k6 = Int((h >> 40) % 6)
        var k = 0
        if k6 >= 3 and k6 < 5:
            k = 1
        elif k6 >= 5:
            k = 2
        labels_cls.append(Int32(k))
        var yv = signed_unit(cell_hash(seed, r, 0, SALT_Y)) + Float32(1.5)
        if ((cell_hash(seed, r, 1, SALT_Y) >> 11) & 7) == 0:
            yv = Float32(-2.5)
        y.append(yv)

    # DEVIATION 135: the scale is chosen ONCE, on the host, from the WHOLE
    # dataset's sum of magnitudes, and the device sees only integers.
    var magnitude = Float64(0.0)
    for r in range(N_ROWS):
        var v = Float64(y[r])
        if v < 0.0:
            v = -v
        magnitude += v
    var scale = choose_scale(magnitude, N_ROWS)
    var labels_reg = List[Int32]()
    var max_q = Int64(0)
    for r in range(N_ROWS):
        var q = quantize(Float64(y[r]), scale)
        var aq = q if q >= 0 else -q
        if aq > max_q:
            max_q = aq
        if aq > Int64(2147483647):
            raise Error(
                "a quantized label does not fit Int32; the scale is wrong"
            )
        labels_reg.append(Int32(q))

    var class_counts = List[Int](length=N_CLASSES, fill=0)
    for r in range(N_ROWS):
        class_counts[Int(labels_cls[r])] += 1

    print(
        "[fixture]",
        n_nodes,
        "nodes x",
        N_COLS,
        "sampled columns =",
        n_cells,
        "cells;",
        N_ROWS,
        "rows,",
        N_CLASSES,
        "classes with counts",
        class_counts[0],
        class_counts[1],
        class_counts[2],
    )
    print(
        "          seed",
        seed,
        "chosen from",
        len(candidate_seeds),
        "candidates because",
        guard_cells,
        "cell(s) make sklearn's == max -> min guard fire",
    )
    print(
        "          fixed-point scale",
        scale,
        "from a magnitude sum of",
        magnitude,
        "; largest quantized label",
        max_q,
        "against the 2^30 slot",
    )
    if guard_cells < 1:
        raise Error(
            "no cell in this fixture makes sklearn's :653-654 guard fire, so"
            " the NO_MAX_GUARD sabotage has no target. That is a DEFECT IN THE"
            " CHECK, not a pass -- widen the ulp-span columns."
        )

    var plan = build_workload_info(work_items, TPB)
    print(
        "          workload_info flattens the batch into",
        plan.n_blocks_dimx,
        "blocks,",
        plan.n_large_nodes,
        "of the nodes are LARGE (> 1 block)",
    )

    # ---------------------------------------------------------------------
    # The host oracle's input: `node_feature_min_max`, per cell.
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # The device. The score kernels' INPUT is the range kernel's OUTPUT: the
    # two steps are chained here, not mocked, so a disagreement in step 1 is
    # visible before step 4 is blamed for it.
    # ---------------------------------------------------------------------
    var ctx = DeviceContext()
    print(
        "[device] accelerator present;",
        _vendor(),
        "-- every arm below runs the SAME binary, the sabotage is an argument",
    )

    var d_min = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_max = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_missing = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_merges = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_mutex = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_data = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_ROWS)
    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_colids = ctx.enqueue_create_buffer[DType.int32](len(colids))
    var d_cls = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_reg = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_items = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes * size_of[NodeWorkItem]()
    )
    var d_wl = ctx.enqueue_create_buffer[DType.uint8](
        plan.n_blocks_dimx * size_of[WorkloadInfo]()
    )

    # One host staging buffer per copy: they are async, and a shared staging
    # buffer would be rewritten under an in-flight copy.
    var s_data = ctx.enqueue_create_host_buffer[DType.float32](N_COLS * N_ROWS)
    var s_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var s_colids = ctx.enqueue_create_host_buffer[DType.int32](len(colids))
    var s_cls = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var s_reg = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
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
        s_cls.unsafe_ptr().unsafe_store(i, labels_cls[i])
        s_reg.unsafe_ptr().unsafe_store(i, labels_reg[i])
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
    ctx.enqueue_copy(dst_buf=d_cls, src_ptr=s_cls.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_reg, src_ptr=s_reg.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_items, src_ptr=s_items.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wl, src_ptr=s_wl.unsafe_ptr())
    ctx.synchronize()

    var failures = 0

    # ---------------------------------------------------------------------
    # ARM P -- the PRECONDITION: step 1's output, from the device.
    # ---------------------------------------------------------------------
    ctx.enqueue_memset(d_mutex, Int32(0))
    ctx.enqueue_function[node_feature_range_init_kernel](
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_merges.unsafe_ptr(),
        Int32(n_cells),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )
    ctx.enqueue_function[node_feature_range_kernel[TPB]](
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_merges.unsafe_ptr(),
        d_mutex.unsafe_ptr(),
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
    var r_min = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
    var r_max = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
    var r_missing = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
    ctx.enqueue_copy(dst_buf=r_min, src_buf=d_min)
    ctx.enqueue_copy(dst_buf=r_max, src_buf=d_max)
    ctx.enqueue_copy(dst_buf=r_missing, src_buf=d_missing)
    ctx.synchronize()

    print("")
    print(
        "[arm P] the range kernel's output -- this check's INPUT -- against"
        " node_feature_min_max, on float BIT PATTERNS"
    )
    var p_bad = 0
    for nid in range(n_nodes):
        for fslot in range(N_COLS):
            var got = feature_range_at(
                rebind[MutPointer[Float32, MutAnyOrigin]](r_min.unsafe_ptr()),
                rebind[MutPointer[Float32, MutAnyOrigin]](r_max.unsafe_ptr()),
                rebind[MutPointer[Int32, MutAnyOrigin]](
                    r_missing.unsafe_ptr()
                ),
                nid,
                fslot,
                N_COLS,
            )
            var w = extents[nid * N_COLS + fslot]
            if (
                got.min_value.to_bits() != w.min_value.to_bits()
                or got.max_value.to_bits() != w.max_value.to_bits()
                or got.n_missing != w.n_missing
            ):
                p_bad += 1
    if p_bad == 0:
        print(
            "  arm P OK:",
            n_cells,
            "cells; the score kernels are fed the DEVICE's ranges, not the"
            " host's",
        )
    else:
        failures += 1
        print("  arm P FAILED:", p_bad, "of", n_cells, "range cells wrong")

    # ---------------------------------------------------------------------
    # The two objectives, each a separate kernel instantiation (rule 8).
    # ---------------------------------------------------------------------
    # `DeviceBuffer.unsafe_ptr()` carries the buffer's own origin; the kernels
    # and the helpers below take `MutAnyOrigin`, which is what a device pointer
    # is. The buffers outlive both calls -- they are `main`'s locals.
    var q_min = rebind[MutPointer[Float32, MutAnyOrigin]](d_min.unsafe_ptr())
    var q_max = rebind[MutPointer[Float32, MutAnyOrigin]](d_max.unsafe_ptr())
    var q_missing = rebind[MutPointer[Int32, MutAnyOrigin]](
        d_missing.unsafe_ptr()
    )
    var q_data = rebind[MutPointer[Float32, MutAnyOrigin]](d_data.unsafe_ptr())
    var q_row_ids = rebind[MutPointer[Int32, MutAnyOrigin]](
        d_row_ids.unsafe_ptr()
    )
    var q_cls = rebind[MutPointer[Int32, MutAnyOrigin]](d_cls.unsafe_ptr())
    var q_reg = rebind[MutPointer[Int32, MutAnyOrigin]](d_reg.unsafe_ptr())
    var q_colids = rebind[MutPointer[Int32, MutAnyOrigin]](
        d_colids.unsafe_ptr()
    )
    var q_items = rebind[MutPointer[NodeWorkItem, MutAnyOrigin]](
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    )
    var q_wl = rebind[MutPointer[WorkloadInfo, MutAnyOrigin]](
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo]()
    )

    failures += check_objective[True](
        ctx,
        q_min,
        q_max,
        q_missing,
        q_data,
        q_row_ids,
        q_cls,
        q_items,
        q_wl,
        q_colids,
        rebind[MutPointer[Float32, MutAnyOrigin]](flat.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](row_ids.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](labels_cls.unsafe_ptr()),
        extents,
        colids,
        work_items,
        ranges,
        plan.n_blocks_dimx,
        seed,
        N_CLASSES,
        shapes,
    )
    failures += check_objective[False](
        ctx,
        q_min,
        q_max,
        q_missing,
        q_data,
        q_row_ids,
        q_reg,
        q_items,
        q_wl,
        q_colids,
        rebind[MutPointer[Float32, MutAnyOrigin]](flat.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](row_ids.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](labels_reg.unsafe_ptr()),
        extents,
        colids,
        work_items,
        ranges,
        plan.n_blocks_dimx,
        seed,
        1,
        shapes,
    )
    _ = flat.unsafe_ptr()
    _ = row_ids.unsafe_ptr()
    _ = labels_cls.unsafe_ptr()
    _ = labels_reg.unsafe_ptr()
    _ = labels_f.unsafe_ptr()

    # ---------------------------------------------------------------------
    # ARM F -- the DRAW's rescale: the barrier is a no-op on the host, and
    # the trap it defuses is measured rather than described.
    #
    # This arm exists because the FIRST RUN of this check failed on it. The
    # device's thresholds came back one ulp (worst case eight, where the sum
    # cancels near zero) from the host's in 9 of 105 cells, and a sweep showed
    # the device value was EXACTLY `fma(res, span, min)`: the GPU backend
    # contracts the multiply into the add, so `pcg_rng.mojo::_product_f32`'s
    # `@no_inline` -- DEVIATION 142's barrier -- is HOST-ONLY.
    # `_uniform_float_unfused` rounds the product through an integer bitcast
    # instead, which is not floating-point dataflow and cannot be contracted
    # through. See DEVIATION BLOCK 173.
    #
    # Two things have to hold and both are checked here, because the fix
    # touches a function `host_splitter.mojo` also calls:
    #   1. no HOST answer moved -- the barrier form and RAFT's own
    #      `uniform_float` agree bit for bit, per cell and over a sweep;
    #   2. the trap is real -- the fused form differs in cells of THIS
    #      fixture, so the barrier is load-bearing and not decoration.
    # ---------------------------------------------------------------------
    print("")
    print(
        "[arm F] the draw's rescale: the bitcast barrier is a bit-exact no-op"
        " on the host, and the fused form it defuses is not"
    )
    var f_cells_differ = 0
    var f_fused_differ = 0
    for nid in range(n_nodes):
        var count = Int(ranges[nid].count)
        var node_id = UInt32(Int(work_items[nid].idx))
        for fslot in range(N_COLS):
            var s = nid * N_COLS + fslot
            var e = extents[s]
            if e.n_missing != 0 or node_feature_is_constant(e, Int32(count)):
                continue
            var col = Int(colids[s])
            var key = key_for(seed, UInt32(TREE_ID), node_id, UInt32(col))
            var g1 = key.generator()
            var raft = uniform_float(g1, e.min_value, e.max_value)
            if draw_threshold_raw(key, e).to_bits() != raft.to_bits():
                f_cells_differ += 1
            var g2 = key.generator()
            var res = g2.next_float()
            var fused = res.fma(e.max_value - e.min_value, e.min_value)
            if fused.to_bits() != raft.to_bits():
                f_fused_differ += 1

    # The same comparison over a SWEEP, so the no-op claim is not a claim about
    # this fixture's ranges: 61 span magnitudes x 6 bases x 64 keys.
    var swept = 0
    var swept_differ = 0
    for e in range(-30, 31):
        var span = Float32(2.0) ** Float32(e)
        for base in [
            Float32(0.0),
            Float32(1.0),
            Float32(-1.0),
            Float32(1e6),
            Float32(-3.25),
            Float32(1048576.0),
        ]:
            var ext = FeatureRange(base, base + span, 0)
            if ext.max_value <= ext.min_value:
                continue
            for node_id in range(64):
                var k = key_for(
                    UInt64(0x51DE), UInt32(1), UInt32(node_id), UInt32(e + 40)
                )
                var g = k.generator()
                var raft = uniform_float(g, ext.min_value, ext.max_value)
                if draw_threshold_raw(k, ext).to_bits() != raft.to_bits():
                    swept_differ += 1
                swept += 1
    print(
        "  host: the barrier form vs RAFT's uniform_float --",
        f_cells_differ,
        "of the fixture's scored cells and",
        swept_differ,
        "of",
        swept,
        "swept draws differ",
    )
    print(
        "  the FUSED form (fma) vs the same reference --",
        f_fused_differ,
        "of the fixture's scored cells differ, which is what the device"
        " produced before the barrier and what arm A would now catch",
    )
    if f_cells_differ != 0 or swept_differ != 0:
        failures += 1
        print(
            "  arm F FAILED: the barrier CHANGED a host answer. It is supposed"
            " to be the identity on an already-rounded float; a difference"
            " means every host check that pins a threshold has moved."
        )
    elif f_fused_differ == 0:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK ***: the fused form agrees everywhere in"
            " this fixture, so arm A could not tell the barrier from its"
            " absence. Widen the ranges until it can."
        )
    else:
        print(
            "  arm F OK: no host answer moved, and the barrier is load-bearing"
            " in",
            f_fused_differ,
            "cells of this fixture",
        )

    # ---------------------------------------------------------------------
    # ARM E -- the published width, MEASURED. DEVIATION BLOCK 175.
    # ---------------------------------------------------------------------
    print("")
    print(
        "[arm E] the Int64 Gini numerator's row bound, measured by wrapping"
        " it rather than asserted from the algebra"
    )
    var half = Int64(1) << Int64(25)  # a node of 2^26 rows, split evenly
    var sq = half * half  # one class: sq_L = nL^2
    var wrapped = sq * half + sq * half
    var exact = Int128(sq) * Int128(half) + Int128(sq) * Int128(half)
    print(
        "  at nL = nR = 2^25 (a node of 2^26 rows, which is exactly"
        " objectives.mojo's MAX_ROWS_EXACT):"
    )
    print("      the true numerator is", exact)
    print("      Int64 holds up to    ", Int64.MAX)
    print("      the Int64 form reads ", wrapped, "-- WRAPPED")
    if Int128(wrapped) == exact or exact <= Int128(Int64.MAX):
        failures += 1
        print(
            "  arm E FAILED: the numerator did not wrap, so DEVIATION 175's"
            " bound is not the bound this check thinks it is"
        )
    else:
        print(
            "  arm E OK: the exact rational needs 77 bits at that node size,"
            " so SCORE_MAX_ROWS_EXACT =",
            SCORE_MAX_ROWS_EXACT,
            "is the bound the published Int64 pair honours -- TIGHTER than"
            " MAX_ROWS_EXACT (2^26), which bounds the Int128 cross-multiply"
            " and not the Int64 field the numerator is stored in",
        )
    var bound_bad = 0
    if not score_row_bound_ok(SCORE_MAX_ROWS_EXACT):
        bound_bad += 1
    if score_row_bound_ok(1 << 22):
        bound_bad += 1
    if score_row_bound_ok(1 << 26):
        bound_bad += 1
    if bound_bad != 0:
        failures += 1
        print("  arm E FAILED: score_row_bound_ok disagrees with its own bound")
    else:
        print(
            "  score_row_bound_ok: 2^21 exact, 2^22 refused, 2^26 refused --"
            " computed in Int128, not asserted"
        )

    print("")
    if failures == 0:
        print(
            "score_kernel_check: PASS --",
            2 * n_cells,
            "cells identical to the host transcription across two objectives,"
            " every status and both block paths named by the device, and every"
            " mechanism sabotaged",
        )
    else:
        print("score_kernel_check: FAIL --", failures, "arm(s) red")
        raise Error("score_kernel_check failed")
