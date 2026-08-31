# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The split PARTITION and the LEAF-VALUE pass, on the device, per cell.

    pixi run mojo run -I . extratrees/mojo_only/partition_leaf_kernel_check.mojo

NO CUML COUNTERPART -- this is a check. It covers the two kernels at the end of
`extratrees/ported/decisiontree/batched_levelalgo/kernels/
builder_kernels_impl.mojo`:

* `node_split_kernel`, their `nodeSplitKernel` (`builder_kernels_impl.cuh
  :89-107`) around `partitionSamples` (`:43-88`);
* `leaf_kernel`, their `leafKernel` (`:391-417`) and the half of
  `SetLeafPredictions` (`builder.cuh:556-599`) that is not host policy.

They carry DEVIATION BLOCKS 176-181.

WHAT THE PARTITION IS CHECKED AGAINST, AND IT IS TWO DIFFERENT THINGS
---------------------------------------------------------------------
1. **The DEFINITION**, which is what `mojo_only/partition_check.mojo` checks
   the host form against and is repeated here against the device's output.
   Two transcriptions share their bugs, so the properties are asserted
   directly: every left slot `<= quesval`, every right slot `>`, nothing
   outside the range moved, the range is a permutation of itself, and the left
   SET is the same for every block width.
2. **The host transcription's `row_ids` ORDER, slot for slot.**
   `partition_samples`' docstring CLAIMS their algorithm is deterministic
   given the block width -- nothing in it depends on warp scheduling -- so the
   device must reproduce not merely an equivalent partition but the identical
   permutation. That claim is a claim until something tests it, and this is
   what tests it.

THE SCAR THIS FIXTURE INHERITS, AND DOES NOT REPEAT
----------------------------------------------------
`partition_check.mojo` records it: with DISTINCT hashed values NO ROW EVER
EQUALS A THRESHOLD, the boundary branch is dead, and sabotaging BOTH
comparison directions left that check GREEN until a nine-level column was
added whose values repeat while their placement stays scattered. Column 1 of
this fixture is that column, and several work items are given a threshold
sitting exactly ON one of its levels. `PART_SAB_LEFT_MISFIT_GE` and
`PART_SAB_RIGHT_MISFIT_LT` are the two arms that would otherwise be dead.

WHAT THE LEAF PASS IS CHECKED AGAINST
--------------------------------------
`leaf_check.mojo` names the three ways this pass is quietly wrong and this
file copies all three: rows are read THROUGH `row_ids` over the node's range
(so `row_ids` is SHUFFLED), every internal node's slot must stay zero, and the
`vector_leaf` stride is `num_outputs` for ALL nodes rather than packed by
leaf. Labels are hashed so leaves have distinct distributions -- a fixture
where every leaf agrees verifies the total and nothing about placement.

The classification arm is compared against BOTH host forms:
`leaf_values_host` (the kernel file's sequential oracle) and
`builder.mojo::set_leaf_predictions_classification` (the transcription of
their `SetLeafPredictions` + `leafKernel`). They are obliged to agree exactly,
so comparing against both makes a drift between the two host forms visible
too.

The regression arm CANNOT be compared against `set_leaf_predictions_regression`
and DEVIATION BLOCK 179 says why: that function accumulates in `Float64`,
which the device does not have. Its oracle is `leaf_values_host` over the same
fixed-point labels, compared on bit patterns; and the disagreement with the
`Float64` form is MEASURED here, per leaf, every run, so the size of the
quantization error is never a guess.

SABOTAGE
--------
One per MECHANISM, each selected by a kernel ARGUMENT so that every arm runs
the SAME binary that ships. Six for the partition (the two comparison
directions, the block scan, the paired swap, the `row_ids` indirection, the
validity guard) and four for the leaf pass (the `IsLeaf` early return, the
`row_ids` indirection, the stride, the normalization). A sabotage that does
not turn an arm red is a defect in THIS FIXTURE and is reported as one.
"""

from std.gpu import WARP_SIZE
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
from extratrees.ported.decisiontree.flatnode import (
    NODE_IS_LEAF,
    SparseTreeNode,
    TreeMetaDataNode,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    set_leaf_predictions_classification,
    set_leaf_predictions_regression,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.batched_levelalgo.split import Split
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    split_not_valid,
)
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    leaf_kernel,
    leaf_values_host,
    node_split_kernel,
    partition_iteration_bound,
    partition_samples,
    LEAF_MAX_OUT_DEFAULT,
    LEAF_SAB_NO_ISLEAF,
    LEAF_SAB_NO_NORMALIZE,
    LEAF_SAB_NO_ROW_IDS,
    LEAF_SAB_NONE,
    LEAF_SAB_STRIDE_ONE,
    LEAF_VISIT_INTERNAL,
    LEAF_VISIT_NONE,
    LEAF_VISIT_PUBLISHED,
    PART_SAB_LEFT_MISFIT_GE,
    PART_SAB_NO_ROW_IDS,
    PART_SAB_NO_SCAN,
    PART_SAB_NO_VALID_GUARD,
    PART_SAB_NONE,
    PART_SAB_RIGHT_MISFIT_LT,
    PART_SAB_UNPAIRED_SWAP,
    PARTITION_OVERRUN,
    PARTITION_SKIPPED,
    PARTITION_UNVISITED,
    TPB_DEFAULT,
)


comptime N_ROWS = 1024
"""Rows in the partition fixture."""

comptime N_COLS = 2
"""Column 0 is hashed and DISTINCT; column 1 has nine repeated levels whose
placement stays scattered. See the module docstring's scar."""

comptime N_LEVELS = 9
comptime LEVEL_STEP: Float32 = 250.0
comptime LEVEL_BASE: Float32 = -1000.0

comptime LEAF_ROWS = 512
comptime LEAF_NODES = 15
comptime LEAF_CLASSES = 4
comptime LEAF_TPB = TPB_DEFAULT
comptime LEAF_MAX_OUT = LEAF_MAX_OUT_DEFAULT

# ARM A's three block widths, WARP-AWARE (2026-08-22, first AMD run): MAX's
# block primitives require block > warp, so the {32, 64, 128} triplet that
# exercises small blocks on Apple/NVIDIA cannot COMPILE on a 64-wide CDNA
# wavefront. The invariant under test -- three distinct widths, identical
# partitions -- is width-agnostic, but ARM C's REACH premise is not: the
# widest block must still leave at least one work item needing a second
# loop iteration, and at 512 this fixture has none (measured on the
# MI325X: 2 multi-iteration items at 128, 1 at 256, 0 at 512 -- ARM C
# correctly refused the 512 triplet). Wide-warp targets therefore test
# {128, 192, 256}: all > warp 64, all warp-multiples, and the largest is
# the width this fixture measurably still multi-iterates at.
comptime PART_TPB_A = 32 if WARP_SIZE <= 32 else 128
comptime PART_TPB_B = 64 if WARP_SIZE <= 32 else 192
comptime PART_TPB_C = 128 if WARP_SIZE <= 32 else 256


def mix32(x: UInt32) -> UInt32:
    """`partition_check.mojo`'s mixer, unchanged, so the two fixtures scatter
    the same way and a value here can be reasoned about from there."""
    var h = x
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


def _vendor() -> String:
    """Which GPU family this run used. A HOST check (`has_*`), not a target
    check (`is_*`): the kernels are one GPU-agnostic source and nothing in
    them branches on the vendor."""
    comptime if has_apple_gpu_accelerator():
        return "Apple"
    elif has_nvidia_gpu_accelerator():
        return "NVIDIA"
    elif has_amd_gpu_accelerator():
        return "AMD"
    else:
        return "unknown"


def _shuffled(n: Int, seed: UInt32) -> List[Int32]:
    """A Fisher-Yates shuffle over a deterministic hash, so a slot index and
    the row id it holds are different numbers for essentially every slot."""
    var ids = List[Int32]()
    for r in range(n):
        ids.append(Int32(r))
    for i in range(n - 1, 0, -1):
        var j = Int(mix32(seed ^ UInt32(i)) % UInt32(i + 1))
        var t = ids[i]
        ids[i] = ids[j]
        ids[j] = t
    return ids^


def _feature_matrix() -> List[Float32]:
    """COLUMN MAJOR, `dataset.h:24`: `data[col * M + row]`.

    Column 0 is hashed noise in roughly `[-1000, 1000)` and is DISTINCT enough
    that essentially no row equals any threshold. Column 1 quantizes onto nine
    levels while keeping the PLACEMENT scattered, which is the adversarial
    shape for the boundary rule: a threshold set exactly ON a level ties many
    rows to it, spread all over the range, and `<=` against `<` decides every
    one of them.
    """
    var flat = List[Float32](length=N_COLS * N_ROWS, fill=Float32(0.0))
    for r in range(N_ROWS):
        var h0 = mix32(0xBEEF ^ UInt32(r * 2654435761))
        flat[0 * N_ROWS + r] = Float32(Int(h0 % 2000000)) / 1000.0 - 1000.0
        var h1 = mix32(0xC0DE ^ UInt32(r * 40503 + 7))
        var lvl = Int(h1 % UInt32(N_LEVELS))
        flat[1 * N_ROWS + r] = Float32(lvl) * LEVEL_STEP + LEVEL_BASE
    return flat^


@fieldwise_init
struct WorkSpec(Copyable, Movable):
    """One work item of the partition batch, before `n_left` is tallied."""

    var begin: Int
    var count: Int
    var col: Int
    var quesval: Float32
    var best_metric: Float32
    var note: String


def _work_specs() -> List[WorkSpec]:
    """The batch. RANGES ARE DISJOINT and do not tile: there are gaps, so a
    kernel that walked from the previous item's end would touch rows belonging
    to no node, and two blocks can never race for a slot.

    Both extremes are present -- `n_left == 0` and `n_left == count`, where
    their `while` body is unreachable -- and so are single-row items, items
    below / at / above the 128-thread block width, and items whose threshold
    sits exactly on a level of the tied column.
    """
    var out = List[WorkSpec]()
    out.append(WorkSpec(0, 1, 0, Float32(0.0), 1.0, "1 row"))
    out.append(WorkSpec(1, 2, 0, Float32(-1001.0), 1.0, "2 rows, n_left = 0"))
    out.append(WorkSpec(3, 17, 1, LEVEL_BASE, 1.0, "17 rows, TIED on level 0"))
    out.append(WorkSpec(20, 127, 0, Float32(123.456), 1.0, "127 rows, one block short"))
    out.append(
        WorkSpec(
            147, 128, 1, LEVEL_BASE + 3.0 * LEVEL_STEP, 1.0,
            "128 rows, exactly one block, TIED on level 3",
        )
    )
    out.append(WorkSpec(275, 129, 0, Float32(500.0), 1.0, "129 rows, one block over"))
    out.append(
        WorkSpec(
            404, 300, 1, LEVEL_BASE + 7.0 * LEVEL_STEP, 1.0,
            "300 rows, TIED on level 7",
        )
    )
    out.append(WorkSpec(704, 1, 0, Float32(1001.0), 1.0, "1 row, n_left = count"))
    out.append(WorkSpec(705, 200, 0, Float32(-500.0), 1.0, "200 rows"))
    out.append(
        WorkSpec(
            905, 119, 1, LEVEL_BASE - 1.0, 1.0, "119 rows, below every level"
        )
    )
    # A gain-refused split. DEVIATION 216 moved the boundary to sklearn's --
    # gain EQUAL to `min_impurity_decrease` now PASSES (builder_check and
    # split_check pin that side) -- so the refusal this spec keeps covered is
    # the one the fit path actually produces: the invalid-candidate
    # MIN_FINITE sentinel, still strictly below any threshold.
    out.append(
        WorkSpec(
            1024 - 96, 96, 0, Float32(0.0), Float32.MIN_FINITE,
            "96 rows, MIN_FINITE gain -> SplitNotValid",
        )
    )
    return out^


def _build_batch(
    flat: List[Float32], row_ids: List[Int32]
) -> Tuple[List[NodeWorkItem], List[Split]]:
    """Turn the specs into `work_items` and `splits`, tallying `n_left`
    independently -- a plain loop over the range through `row_ids`, which is
    NOT how either the host transcription or the kernel computes anything."""
    var specs = _work_specs()
    var items = List[NodeWorkItem]()
    var splits = List[Split]()
    for b in range(len(specs)):
        var s = specs[b].copy()
        var n_left = 0
        for p in range(s.begin, s.begin + s.count):
            if flat[s.col * N_ROWS + Int(row_ids[p])] <= s.quesval:
                n_left += 1
        # `idx` is the node's id IN THE TREE and deliberately does not start
        # at 0: neither kernel reads it, and that is worth stating.
        items.append(
            NodeWorkItem(Int32(200 + b), Int32(4), InstanceRange(Int32(s.begin), Int32(s.count)))
        )
        splits.append(
            Split(s.quesval, Int32(s.col), s.best_metric, Int32(n_left))
        )
    return (items^, splits^)


def _host_partition_batch(
    mut flat: List[Float32],
    row_ids_in: List[Int32],
    items: List[NodeWorkItem],
    splits: List[Split],
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    tpb: Int,
) raises -> List[Int32]:
    """`nodeSplitKernel` on the host: the validity guard (`:98-104`) and then
    `partition_samples`, item by item, in batch order.

    THE GUARD IS APPLIED HERE and not inside `partition_samples`, because that
    is where their kernel has it: `partitionSamples` itself always partitions.
    """
    var row_ids = row_ids_in.copy()
    var labels = List[Float32](length=N_ROWS, fill=Float32(0.0))
    var dataset = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](flat.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](labels.unsafe_ptr()),
        Int32(N_ROWS),
        Int32(N_COLS),
        Int32(N_ROWS),
        Int32(N_COLS),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        Int32(1),
    )
    for b in range(len(items)):
        if split_not_valid(
            splits[b],
            min_impurity_decrease,
            min_samples_leaf,
            items[b].instances.count,
        ):
            continue
        partition_samples(dataset, splits[b], items[b], tpb)
    # `Dataset`'s pointers are `MutUntrackedOrigin`, so the compiler tracks no
    # relationship to the `List`s behind them and would free one after its
    # last syntactic use. `range_draw_check` read freed memory this way.
    _ = labels.unsafe_ptr()
    _ = flat.unsafe_ptr()
    return row_ids^


@fieldwise_init
struct PartitionRun(Copyable, Movable):
    """One launch's outputs, copied back."""

    var row_ids: List[Int32]
    var n_iters: List[Int32]
    var n_swaps: List[Int32]


def main() raises:
    comptime assert has_accelerator(), "this check must run on a GPU"

    var failures = 0
    print("[device] accelerator present;", _vendor(),
          "-- both kernels are ENQUEUED on it, and every arm below runs the"
          " SAME binary")
    print("")

    failures += _partition_section()
    print("")
    failures += _leaf_section()

    print("")
    if failures == 0:
        print("partition_leaf_kernel_check: PASS")
    else:
        print("partition_leaf_kernel_check: FAIL --", failures, "arm(s) red")
        raise Error("partition_leaf_kernel_check failed")


# ===========================================================================
# THE PARTITION
# ===========================================================================


def _launch_partition[
    TPB: Int
](
    ctx: DeviceContext,
    flat: List[Float32],
    row_ids: List[Int32],
    items: List[NodeWorkItem],
    splits: List[Split],
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    sabotage: Int32,
) raises -> PartitionRun:
    """One launch of `node_split_kernel[TPB]`, with fresh inputs.

    `row_ids` is MUTATED by the kernel, so it is re-uploaded from the pristine
    host copy for every arm; an arm that inherited the previous arm's output
    would be checking a different fixture.
    """
    var n_items = len(items)

    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_iters = ctx.enqueue_create_buffer[DType.int32](n_items)
    var d_swaps = ctx.enqueue_create_buffer[DType.int32](n_items)
    var d_data = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_ROWS)
    var d_items = ctx.enqueue_create_buffer[DType.uint8](
        n_items * size_of[NodeWorkItem]()
    )
    var d_splits = ctx.enqueue_create_buffer[DType.uint8](
        n_items * size_of[Split]()
    )

    var h_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](
        N_COLS * N_ROWS
    )
    var h_items = ctx.enqueue_create_host_buffer[DType.uint8](
        n_items * size_of[NodeWorkItem]()
    )
    var h_splits = ctx.enqueue_create_host_buffer[DType.uint8](
        n_items * size_of[Split]()
    )
    ctx.synchronize()

    for i in range(N_ROWS):
        h_row_ids.unsafe_ptr().unsafe_store(i, row_ids[i])
    for i in range(N_COLS * N_ROWS):
        h_data.unsafe_ptr().unsafe_store(i, flat[i])
    var ip = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    for i in range(n_items):
        ip[unsafe_offset=i] = items[i]
    var sp = h_splits.unsafe_ptr().unsafe_bitcast[Split]()
    for i in range(n_items):
        sp[unsafe_offset=i] = splits[i]

    ctx.enqueue_copy(dst_buf=d_row_ids, src_ptr=h_row_ids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_data, src_ptr=h_data.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_items, src_ptr=h_items.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_splits, src_ptr=h_splits.unsafe_ptr())
    # DEVIATION 180: `out_n_iters` must be SEEDED with `PARTITION_UNVISITED`,
    # which is not a legal outcome, so a work item no block served is visible
    # as a reach failure instead of as a plausible zero.
    ctx.enqueue_memset(d_iters, PARTITION_UNVISITED)
    ctx.enqueue_memset(d_swaps, Int32(0))
    ctx.synchronize()

    ctx.enqueue_function[node_split_kernel[TPB]](
        d_row_ids.unsafe_ptr(),
        d_iters.unsafe_ptr(),
        d_swaps.unsafe_ptr(),
        d_data.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_splits.unsafe_ptr().unsafe_bitcast[Split](),
        Int32(N_ROWS),
        min_impurity_decrease,
        min_samples_leaf,
        sabotage,
        grid_dim=(n_items, 1, 1),
        block_dim=(TPB, 1, 1),
    )

    var o_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var o_iters = ctx.enqueue_create_host_buffer[DType.int32](n_items)
    var o_swaps = ctx.enqueue_create_host_buffer[DType.int32](n_items)
    ctx.enqueue_copy(dst_buf=o_row_ids, src_buf=d_row_ids)
    ctx.enqueue_copy(dst_buf=o_iters, src_buf=d_iters)
    ctx.enqueue_copy(dst_buf=o_swaps, src_buf=d_swaps)
    ctx.synchronize()

    var ids = List[Int32]()
    for i in range(N_ROWS):
        ids.append(o_row_ids.unsafe_ptr().unsafe_load(i))
    var it = List[Int32]()
    var sw = List[Int32]()
    for i in range(n_items):
        it.append(o_iters.unsafe_ptr().unsafe_load(i))
        sw.append(o_swaps.unsafe_ptr().unsafe_load(i))
    return PartitionRun(ids^, it^, sw^)


def _definition_failures(
    flat: List[Float32],
    before: List[Int32],
    after: List[Int32],
    items: List[NodeWorkItem],
    splits: List[Split],
    skipped: List[Bool],
) raises -> Tuple[Int, Int]:
    """`partition_check.mojo`'s five properties, against the DEFINITION and
    not against a second transcription. Returns (cells checked, cells wrong).

    1. every slot in `[begin, begin + n_left)` holds a row `<= quesval`;
    2. every slot in `[begin + n_left, end)` holds a row `>`;
    3. `row_ids` outside every item's range is untouched, slot for slot;
    4. the multiset of row ids inside each range is preserved;
    5. an item the guard SKIPPED is untouched, slot for slot.
    """
    var cells = 0
    var bad = 0
    var inside = List[Bool](length=N_ROWS, fill=False)

    for b in range(len(items)):
        var begin = Int(items[b].instances.begin)
        var count = Int(items[b].instances.count)
        var n_left = Int(splits[b].n_left)
        var col = Int(splits[b].colid)
        var q = splits[b].quesval
        for s in range(begin, begin + count):
            inside[s] = True

        if skipped[b]:
            # (5) `nodeSplitKernel`'s early return moves nothing.
            for s in range(begin, begin + count):
                cells += 1
                if after[s] != before[s]:
                    bad += 1
            continue

        # (1) and (2)
        for s in range(begin, begin + n_left):
            cells += 1
            if not (flat[col * N_ROWS + Int(after[s])] <= q):
                bad += 1
        for s in range(begin + n_left, begin + count):
            cells += 1
            if not (flat[col * N_ROWS + Int(after[s])] > q):
                bad += 1

        # (4) counting sort on row id, an independent structure.
        var seen_before = List[Int](length=N_ROWS, fill=0)
        var seen_after = List[Int](length=N_ROWS, fill=0)
        for s in range(begin, begin + count):
            seen_before[Int(before[s])] += 1
            seen_after[Int(after[s])] += 1
        for r in range(N_ROWS):
            cells += 1
            if seen_before[r] != seen_after[r]:
                bad += 1

    # (3) nothing outside any range moved.
    for s in range(N_ROWS):
        if not inside[s]:
            cells += 1
            if after[s] != before[s]:
                bad += 1
    return (cells, bad)


def _left_marks(
    after: List[Int32], items: List[NodeWorkItem], splits: List[Split]
) -> List[Int]:
    """Which row ids ended up on the LEFT of their item. The block width is a
    scheduling parameter and must not be an algorithmic one, so this set must
    be identical across widths even when the ORDER is not required to be."""
    var marks = List[Int](length=N_ROWS, fill=0)
    for b in range(len(items)):
        var begin = Int(items[b].instances.begin)
        for s in range(begin, begin + Int(splits[b].n_left)):
            marks[Int(after[s])] = 1
    return marks^


def _partition_section() raises -> Int:
    var failures = 0
    var ctx = DeviceContext()

    var flat = _feature_matrix()
    var row_ids = _shuffled(N_ROWS, 0xABCD)
    var built = _build_batch(flat, row_ids)
    var items = built[0].copy()
    var splits = built[1].copy()
    var specs = _work_specs()
    var n_items = len(items)

    # Prove the tied column really ties rows to the swept thresholds, rather
    # than asserting it. A fixture that stopped tying would silently return
    # this check to the state that passed two sabotages.
    var tied = 0
    for b in range(n_items):
        if Int(splits[b].colid) != 1:
            continue
        var begin = Int(items[b].instances.begin)
        for s in range(begin, begin + Int(items[b].instances.count)):
            if flat[1 * N_ROWS + Int(row_ids[s])] == splits[b].quesval:
                tied += 1
    print("[fixture]", N_ROWS, "rows,", N_COLS, "columns,", n_items,
          "work items with DISJOINT ranges;", tied,
          "rows sit EXACTLY on their item's threshold")
    # THE BAR IS DERIVED, NOT PICKED. Three items sit on the nine-level
    # column with a threshold ON a level and hold 17 + 128 + 300 = 445 rows,
    # so about `445 / 9 = 49` of them are expected to tie. The first version
    # of this line demanded 50 and the run measured 45: the bar was above the
    # EXPECTATION, which makes it a coin flip on the hash and not a check.
    # It is set to 20 -- less than half the expectation, still far above the
    # zero that would make the two comparison sabotages invisible -- and the
    # real guarantee is arm E, which asserts those two sabotages turn red.
    if tied < 20:
        failures += 1
        print("  FIXTURE DEFECT: too few tied rows -- the <= branch is"
              " untested and both comparison sabotages will be invisible")

    # Two guard settings. The first is the shipping one, under which several
    # items are REJECTED by `split_not_valid`; the second opens the guard so
    # the `n_left == 0` and `n_left == count` items are actually partitioned
    # (their `while` body is unreachable there, and a partition that is
    # silently a no-op passes any count-only check).
    var guard_names = [String("shipping guard (mid = 0.0, msl = 1)"),
                       String("open guard (mid = -1.0, msl = 0)")]
    var guard_mid = [Float32(0.0), Float32(-1.0)]
    var guard_msl = [Int32(1), Int32(0)]

    for g in range(2):
        var skipped = List[Bool]()
        var n_skipped = 0
        for b in range(n_items):
            var sk = split_not_valid(
                splits[b], guard_mid[g], guard_msl[g],
                items[b].instances.count,
            )
            skipped.append(sk)
            if sk:
                n_skipped += 1
        print("")
        print("[guard]", guard_names[g], "--", n_skipped, "of", n_items,
              "items rejected by split_not_valid")

        # ------------------------------------------------------------------
        # ARM A -- the device's row_ids ORDER against the host transcription's,
        # slot for slot, for three block widths.
        # ------------------------------------------------------------------
        var order_cells = 0
        var order_bad = 0
        var def_cells = 0
        var def_bad = 0
        var marks_ref = List[Int]()
        var width_cells = 0
        var width_bad = 0
        var order_ref = List[Int32]()
        var order_width_diff = 0

        for w in range(3):
            var run: PartitionRun
            var tpb: Int
            if w == 0:
                tpb = PART_TPB_A
                run = _launch_partition[PART_TPB_A](
                    ctx, flat, row_ids, items, splits,
                    guard_mid[g], guard_msl[g], PART_SAB_NONE,
                )
            elif w == 1:
                tpb = PART_TPB_B
                run = _launch_partition[PART_TPB_B](
                    ctx, flat, row_ids, items, splits,
                    guard_mid[g], guard_msl[g], PART_SAB_NONE,
                )
            else:
                tpb = PART_TPB_C
                run = _launch_partition[PART_TPB_C](
                    ctx, flat, row_ids, items, splits,
                    guard_mid[g], guard_msl[g], PART_SAB_NONE,
                )

            var want = _host_partition_batch(
                flat, row_ids, items, splits,
                guard_mid[g], guard_msl[g], tpb,
            )
            var moved = 0
            for s in range(N_ROWS):
                order_cells += 1
                if run.row_ids[s] != want[s]:
                    order_bad += 1
                if run.row_ids[s] != row_ids[s]:
                    moved += 1

            var d = _definition_failures(
                flat, row_ids, run.row_ids, items, splits, skipped
            )
            def_cells += d[0]
            def_bad += d[1]

            # ARM D -- the left SET across widths.
            var marks = _left_marks(run.row_ids, items, splits)
            if w == 0:
                marks_ref = marks.copy()
                order_ref = run.row_ids.copy()
            else:
                for r in range(N_ROWS):
                    width_cells += 1
                    if marks[r] != marks_ref[r]:
                        width_bad += 1
                for r in range(N_ROWS):
                    if run.row_ids[r] != order_ref[r]:
                        order_width_diff += 1

            # ARM C -- the path, from the DEVICE's own report.
            var unvisited = 0
            var overrun = 0
            var skipped_reported = 0
            var multi_iter = 0
            var swapped = 0
            var bound_bad = 0
            for b in range(n_items):
                var it = Int(run.n_iters[b])
                if it == Int(PARTITION_UNVISITED):
                    unvisited += 1
                elif it == Int(PARTITION_OVERRUN):
                    overrun += 1
                elif it == Int(PARTITION_SKIPPED):
                    skipped_reported += 1
                    if not skipped[b]:
                        bound_bad += 1
                else:
                    if skipped[b]:
                        bound_bad += 1
                    if it > partition_iteration_bound(
                        Int(items[b].instances.count),
                        Int(splits[b].n_left),
                        tpb,
                    ):
                        bound_bad += 1
                    if it > 1:
                        multi_iter += 1
                if Int(run.n_swaps[b]) > 0:
                    swapped += 1
            print("  tpb", tpb, "-- device report:", skipped_reported,
                  "item(s) SKIPPED by the guard,", multi_iter,
                  "item(s) ran MORE THAN ONE loop iteration,", swapped,
                  "item(s) actually swapped;", moved, "of", N_ROWS,
                  "slots moved")
            if unvisited != 0 or overrun != 0 or bound_bad != 0:
                failures += 1
                print("    ARM C FAILED: unvisited", unvisited, "overrun",
                      overrun, "disagreements with the host guard or the"
                      " derived bound", bound_bad)
            if skipped_reported != n_skipped:
                failures += 1
                print("    ARM C FAILED: the device skipped",
                      skipped_reported, "items, the host guard skipped",
                      n_skipped)
            if g == 1 and multi_iter == 0:
                failures += 1
                print("    ARM C FAILED: no item ran more than one loop"
                      " iteration, so the multi-iteration arm never ran")
            if swapped == 0:
                failures += 1
                print("    ARM C FAILED: no item swapped anything, so the"
                      " swap path never ran and a partition that is silently"
                      " a no-op would pass every arm above")

        if order_bad == 0:
            print("  ARM A OK:", order_cells,
                  "slot comparisons -- the device reproduces the host"
                  " transcription's row_ids ORDER EXACTLY at every width")
        else:
            failures += 1
            print("  ARM A FAILED:", order_bad, "of", order_cells,
                  "slots differ from the host transcription")
        if def_bad == 0:
            print("  ARM B OK:", def_cells,
                  "definition cells -- predicate, permutation, untouched"
                  " outside, untouched when skipped")
        else:
            failures += 1
            print("  ARM B FAILED:", def_bad, "of", def_cells,
                  "definition cells wrong")
        if width_bad == 0:
            print("  ARM D OK:", width_cells,
                  "membership cells -- the left SET is the same at tpb 32, 64"
                  " and 128")
            # MEASURED, and it falsified the sentence that stood here first,
            # which claimed the ORDER was width-dependent. It is not:
            # `order_width_diff` comes back 0, i.e. the whole PERMUTATION is
            # the same at 32, 64 and 128 threads.
            #
            # That is a property of their algorithm and not of this fixture.
            # Consumption is strictly in slot order on BOTH sides -- the
            # compaction packs the currently-flagged misfits by ascending
            # slot and the first `minlen` of them are taken, and a side that
            # was not fully consumed KEEPS its flags rather than rescanning
            # (`:65-66`, the `llen == minlen` guard). So globally the k-th
            # left misfit is always swapped with the k-th right misfit, and
            # `TPB` only decides how many of those pairs are done per
            # iteration.
            #
            # It is PRINTED and not gated. Their header
            # (`builder_kernels_impl.cuh:39-41`) promises determinism given
            # the block width and nothing more, so width-independence is a
            # fact about their code that this file reports rather than a
            # contract it enforces. Arm A is unaffected -- it compares the
            # device against the HOST at each width, and the host form is a
            # different implementation, not a rerun.
            print("       MEASURED:", order_width_diff, "of", N_ROWS,
                  "slots differ between tpb 32 and tpb 64/128 -- the whole"
                  " permutation is block-width INDEPENDENT, which is stronger"
                  " than their header claims and is derived above")
        else:
            failures += 1
            print("  ARM D FAILED:", width_bad, "of", width_cells,
                  "membership cells differ across block widths")

    # ----------------------------------------------------------------------
    # ARM E -- sabotage, one per MECHANISM, same binary, kernel argument.
    # ----------------------------------------------------------------------
    print("")
    print("[arm E] partition sabotage, one per mechanism")
    var skipped_ship = List[Bool]()
    for b in range(n_items):
        skipped_ship.append(
            split_not_valid(splits[b], Float32(0.0), Int32(1),
                            items[b].instances.count)
        )
    var base = _launch_partition[128](
        ctx, flat, row_ids, items, splits, Float32(0.0), Int32(1),
        PART_SAB_NONE,
    )

    var sabs = [
        PART_SAB_LEFT_MISFIT_GE,
        PART_SAB_RIGHT_MISFIT_LT,
        PART_SAB_NO_SCAN,
        PART_SAB_UNPAIRED_SWAP,
        PART_SAB_NO_ROW_IDS,
        PART_SAB_NO_VALID_GUARD,
    ]
    var names = [
        String("the LEFT comparison direction (`> quesval` -> `>=`, :65):"
               " a row ON the threshold is dragged right"),
        String("the RIGHT comparison direction (`<= quesval` -> `<`, :66):"
               " the other half of the same boundary rule"),
        String("the BLOCK SCAN (compact at thread_idx.x instead of the"
               " exclusive prefix, :69-77)"),
        String("the PAIRED SWAP (write only the left half, :86-88): a row is"
               " duplicated and another dropped"),
        String("the row_ids INDIRECTION (`col[row_ids[loff]]` ->"
               " `col[loff]`, :65-66)"),
        String("the VALIDITY GUARD (partition even when SplitNotValid,"
               " :100-104)"),
    ]
    # WHICH ITEM OWNS EACH SLOT, and which items actually have a row sitting
    # ON their threshold. The two comparison sabotages are PREDICTED to move
    # slots only inside the latter: on the distinct column no row equals a
    # threshold, so `>=` and `>` decide identically there. A sabotage that
    # moved cells everywhere would be red for the wrong reason.
    var owner = List[Int](length=N_ROWS, fill=-1)
    var item_tied = List[Bool](length=n_items, fill=False)
    for b in range(n_items):
        var begin = Int(items[b].instances.begin)
        var col = Int(splits[b].colid)
        for s in range(begin, begin + Int(items[b].instances.count)):
            owner[s] = b
            if flat[col * N_ROWS + Int(row_ids[s])] == splits[b].quesval:
                item_tied[b] = True
    var n_tied_items = 0
    for b in range(n_items):
        if item_tied[b]:
            n_tied_items += 1
    print("  ", n_tied_items, "of", n_items,
          "items hold a row sitting exactly on their own threshold")

    for a in range(len(sabs)):
        var run = _launch_partition[128](
            ctx, flat, row_ids, items, splits, Float32(0.0), Int32(1),
            sabs[a],
        )
        var moved = 0
        var moved_untied = 0
        for s in range(N_ROWS):
            if run.row_ids[s] != base.row_ids[s]:
                moved += 1
                if owner[s] < 0 or not item_tied[owner[s]]:
                    moved_untied += 1
        if a < 2:
            if moved_untied != 0:
                failures += 1
                print("  PREDICTION FAILED:", moved_untied,
                      "slot(s) moved OUTSIDE an item with a tied row, so this"
                      " comparison sabotage is red for the wrong reason")
            else:
                print("   (all", moved,
                      "moved slots lie inside an item that holds a row ON its"
                      " threshold, as predicted)")
        var d = _definition_failures(
            flat, row_ids, run.row_ids, items, splits, skipped_ship
        )
        if moved == 0 and d[1] == 0:
            failures += 1
            print("  SABOTAGE DID NOT TURN THE CHECK RED --", names[a])
            print("    that is a defect in THIS FIXTURE, not a pass")
        else:
            print("  red:", d[1], "definition cell(s) wrong,", moved,
                  "slot(s) moved --", names[a])
    # RESTORE and confirm green: the shipping arm is rerun AFTER every
    # sabotage, so a sabotage that leaked into shared state would be visible.
    var restored = _launch_partition[128](
        ctx, flat, row_ids, items, splits, Float32(0.0), Int32(1),
        PART_SAB_NONE,
    )
    var restore_bad = 0
    for s in range(N_ROWS):
        if restored.row_ids[s] != base.row_ids[s]:
            restore_bad += 1
    if restore_bad == 0:
        print("  restored: the shipping arm rerun after all six sabotages is"
              " bit-identical to the shipping arm before them")
    else:
        failures += 1
        print("  RESTORE FAILED:", restore_bad, "slots differ")

    # `Dataset`'s pointers are `MutUntrackedOrigin`; keep the backing buffers
    # alive to here.
    _ = flat.unsafe_ptr()
    _ = row_ids.unsafe_ptr()
    _ = specs.__len__()
    return failures


# ===========================================================================
# THE LEAF PASS
# ===========================================================================


def _leaf_tree() -> Tuple[
    List[SparseTreeNode[DType.float32]], List[InstanceRange]
]:
    """A full binary tree of depth 3: nodes 0-6 internal, 7-14 leaves.

    The leaves' ranges TILE `[0, LEAF_ROWS)` with wildly different counts --
    including a ONE-row leaf (whose probability vector is a single 1.0, where
    the normalization sabotage is correctly invisible) and a ZERO-row leaf
    (whose `SetLeafVector` is `0 / 0`; theirs has no guard either, and what
    the two sides do with that NaN is measured below rather than assumed).
    Internal ranges are the union of their children's, as `Push` leaves them.
    """
    var counts = [1, 0, 200, 3, 130, 60, 45, 73]
    var nodes = List[SparseTreeNode[DType.float32]]()
    var ranges = List[InstanceRange](
        length=LEAF_NODES, fill=InstanceRange(Int32(0), Int32(0))
    )
    for i in range(LEAF_NODES):
        if i < 7:
            nodes.append(
                SparseTreeNode[DType.float32](
                    Int32(i % 2), Float32(0.5), Float32(1.0),
                    Int32(2 * i + 1), Int32(0),
                )
            )
        else:
            nodes.append(
                SparseTreeNode[DType.float32](
                    Int32(0), Float32(0.0), Float32(0.0),
                    NODE_IS_LEAF, Int32(counts[i - 7]),
                )
            )
    var begin = 0
    for i in range(7, LEAF_NODES):
        ranges[i] = InstanceRange(Int32(begin), Int32(counts[i - 7]))
        begin += counts[i - 7]
    for i in range(6, -1, -1):
        var l = ranges[2 * i + 1]
        var r = ranges[2 * i + 2]
        ranges[i] = InstanceRange(l.begin, Int32(Int(l.count) + Int(r.count)))
    return (nodes^, ranges^)


def _launch_leaf[
    CLASSIFICATION: Bool, TPB: Int
](
    ctx: DeviceContext,
    nodes: List[SparseTreeNode[DType.float32]],
    ranges: List[InstanceRange],
    row_ids: List[Int32],
    labels_q: List[Int32],
    num_outputs: Int,
    inv_scale: Float32,
    sabotage: Int32,
) raises -> Tuple[List[Float32], List[Int32]]:
    """One launch of `leaf_kernel`. Returns (`out_leaves`, `out_visit`)."""
    var n_nodes = len(nodes)
    var n_slots = n_nodes * num_outputs

    var d_leaves = ctx.enqueue_create_buffer[DType.float32](n_slots)
    var d_visit = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var d_nodes = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes * size_of[SparseTreeNode[DType.float32]]()
    )
    var d_ranges = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes * size_of[InstanceRange]()
    )
    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](len(row_ids))
    var d_labels = ctx.enqueue_create_buffer[DType.int32](len(labels_q))

    var h_nodes = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes * size_of[SparseTreeNode[DType.float32]]()
    )
    var h_ranges = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes * size_of[InstanceRange]()
    )
    var h_row_ids = ctx.enqueue_create_host_buffer[DType.int32](len(row_ids))
    var h_labels = ctx.enqueue_create_host_buffer[DType.int32](len(labels_q))
    ctx.synchronize()

    var np = h_nodes.unsafe_ptr().unsafe_bitcast[
        SparseTreeNode[DType.float32]
    ]()
    for i in range(n_nodes):
        np[unsafe_offset=i] = nodes[i]
    var rp = h_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange]()
    for i in range(n_nodes):
        rp[unsafe_offset=i] = ranges[i]
    for i in range(len(row_ids)):
        h_row_ids.unsafe_ptr().unsafe_store(i, row_ids[i])
    for i in range(len(labels_q)):
        h_labels.unsafe_ptr().unsafe_store(i, labels_q[i])

    ctx.enqueue_copy(dst_buf=d_nodes, src_ptr=h_nodes.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ranges, src_ptr=h_ranges.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_row_ids, src_ptr=h_row_ids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_labels, src_ptr=h_labels.unsafe_ptr())
    # `builder.cuh:582`'s `cudaMemsetAsync`, and DEVIATION 180's `out_visit`
    # seed. An internal node's leaf slot is written by nothing and the ZERO is
    # its value, so this memset is part of the algorithm, not hygiene.
    ctx.enqueue_memset(d_leaves, Float32(0.0))
    ctx.enqueue_memset(d_visit, LEAF_VISIT_NONE)
    ctx.synchronize()

    ctx.enqueue_function[leaf_kernel[TPB, LEAF_MAX_OUT, CLASSIFICATION]](
        d_leaves.unsafe_ptr(),
        d_visit.unsafe_ptr(),
        d_nodes.unsafe_ptr().unsafe_bitcast[SparseTreeNode[DType.float32]](),
        d_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange](),
        d_row_ids.unsafe_ptr(),
        d_labels.unsafe_ptr(),
        Int32(num_outputs),
        inv_scale,
        sabotage,
        grid_dim=(n_nodes, 1, 1),
        block_dim=(TPB, 1, 1),
    )

    var o_leaves = ctx.enqueue_create_host_buffer[DType.float32](n_slots)
    var o_visit = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    ctx.enqueue_copy(dst_buf=o_leaves, src_buf=d_leaves)
    ctx.enqueue_copy(dst_buf=o_visit, src_buf=d_visit)
    ctx.synchronize()

    var leaves = List[Float32]()
    for i in range(n_slots):
        leaves.append(o_leaves.unsafe_ptr().unsafe_load(i))
    var visit = List[Int32]()
    for i in range(n_nodes):
        visit.append(o_visit.unsafe_ptr().unsafe_load(i))
    return (leaves^, visit^)


def _same_bits(a: Float32, b: Float32) -> Bool:
    """Bit equality, with NaN treated as agreeing with NaN.

    The house rule is `.to_bits()` and no tolerance, and that is what this is
    everywhere a number is finite. The ONE exception is the zero-row leaf,
    where `SetLeafVector` is `0 / 0` on both sides -- theirs has no guard
    either (`objectives.cuh:97-107`) -- and the NaN a host FPU and a GPU
    produce need not carry the same payload. Two NaNs are counted as agreeing,
    and the bits of both are PRINTED, so the difference is reported rather
    than hidden.
    """
    if a != a and b != b:
        return True
    return a.to_bits() == b.to_bits()


def _leaf_section() raises -> Int:
    var failures = 0
    var ctx = DeviceContext()

    var built = _leaf_tree()
    var nodes = built[0].copy()
    var ranges = built[1].copy()
    var row_ids = _shuffled(LEAF_ROWS, 0xBEE5)

    var n_leaves = 0
    var n_internal = 0
    for i in range(LEAF_NODES):
        if nodes[i].left_child_id == NODE_IS_LEAF:
            n_leaves += 1
        else:
            n_internal += 1
    print("[fixture] tree:", LEAF_NODES, "nodes,", n_leaves, "leaves,",
          n_internal, "internal;", LEAF_ROWS,
          "rows behind a SHUFFLED row_ids")

    # ---------------- classification --------------------------------------
    var labels_f = List[Float32]()
    var labels_q = List[Int32]()
    for r in range(LEAF_ROWS):
        var c = Int(mix32(UInt32(r) ^ 0xC0DE) % UInt32(LEAF_CLASSES))
        labels_f.append(Float32(c))
        labels_q.append(Int32(c))

    var want = leaf_values_host(
        rebind[MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin]](
            nodes.unsafe_ptr()
        ),
        rebind[MutPointer[InstanceRange, MutAnyOrigin]](ranges.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](row_ids.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](labels_q.unsafe_ptr()),
        LEAF_NODES,
        LEAF_CLASSES,
        Float32(1.0),
        True,
    )

    # The SECOND host form: `builder.mojo`'s transcription of their
    # `SetLeafPredictions` + `leafKernel`. It is obliged to agree with
    # `leaf_values_host` exactly, so comparing against both makes a drift
    # between the two host forms visible as well.
    var features = List[Float32](length=LEAF_ROWS, fill=Float32(0.0))
    var dataset = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](features.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](labels_f.unsafe_ptr()),
        Int32(LEAF_ROWS),
        Int32(1),
        Int32(LEAF_ROWS),
        Int32(1),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        Int32(LEAF_CLASSES),
    )
    var tree = TreeMetaDataNode[DType.float32](
        Int32(0), Int32(3), Int32(n_leaves), Int32(LEAF_CLASSES),
        List[Float32](), nodes.copy(),
    )
    set_leaf_predictions_classification(dataset, tree, ranges.copy())

    var host_drift = 0
    for i in range(LEAF_NODES * LEAF_CLASSES):
        if not _same_bits(tree.vector_leaf[i], want[i]):
            host_drift += 1
    if host_drift == 0:
        print("  the two HOST forms agree bit for bit on all",
              LEAF_NODES * LEAF_CLASSES, "slots (leaf_values_host vs"
              " builder.mojo::set_leaf_predictions_classification)")
    else:
        failures += 1
        print("  HOST DRIFT:", host_drift, "slots differ BETWEEN THE TWO HOST"
              " FORMS -- the device is not the suspect here")

    var got = _launch_leaf[True, LEAF_TPB](
        ctx, nodes, ranges, row_ids, labels_q, LEAF_CLASSES, Float32(1.0),
        LEAF_SAB_NONE,
    )
    var base_leaves = got[0].copy()
    var base_visit = got[1].copy()

    var cells = 0
    var bad = 0
    var nan_pairs = 0
    for node_id in range(LEAF_NODES):
        for c in range(LEAF_CLASSES):
            var s = node_id * LEAF_CLASSES + c
            cells += 1
            if not _same_bits(base_leaves[s], want[s]):
                bad += 1
                if bad <= 6:
                    print("    MISMATCH node", node_id, "class", c, "got",
                          base_leaves[s], "(bits",
                          base_leaves[s].to_bits(), ") want", want[s],
                          "(bits", want[s].to_bits(), ")")
            elif base_leaves[s] != base_leaves[s]:
                nan_pairs += 1
                print("    NaN pair at node", node_id, "class", c,
                      "-- device bits", base_leaves[s].to_bits(),
                      ", host bits", want[s].to_bits(),
                      "(the zero-row leaf: 0/0 on both sides,"
                      " objectives.cuh:97-107 has no guard either)")
    if bad == 0:
        print("  ARM F OK:", cells,
              "classification leaf slots bit-identical to BOTH host forms;",
              nan_pairs, "of them are the zero-row leaf's NaN")
    else:
        failures += 1
        print("  ARM F FAILED:", bad, "of", cells, "leaf slots wrong")

    # The device's own report of the path each block took.
    var v_internal = 0
    var v_published = 0
    var v_none = 0
    var v_bad = 0
    for node_id in range(LEAF_NODES):
        var v = base_visit[node_id]
        var is_leaf = nodes[node_id].left_child_id == NODE_IS_LEAF
        if v == LEAF_VISIT_NONE:
            v_none += 1
        elif v == LEAF_VISIT_INTERNAL:
            v_internal += 1
            if is_leaf:
                v_bad += 1
        else:
            v_published += 1
            if not is_leaf:
                v_bad += 1
    if v_none == 0 and v_bad == 0 and v_internal > 0 and v_published > 0:
        print("  ARM G OK: the device reports", v_published,
              "node(s) PUBLISHED and", v_internal,
              "node(s) returned early at IsLeaf -- both paths ran, and no"
              " node was left unvisited")
    else:
        failures += 1
        print("  ARM G FAILED: none", v_none, "internal", v_internal,
              "published", v_published, "disagreements", v_bad)

    # Internal slots must be zero, and the report is what distinguishes that
    # from a slot nothing wrote.
    var internal_bad = 0
    for node_id in range(LEAF_NODES):
        if nodes[node_id].left_child_id == NODE_IS_LEAF:
            continue
        for c in range(LEAF_CLASSES):
            if base_leaves[node_id * LEAF_CLASSES + c] != Float32(0.0):
                internal_bad += 1
    if internal_bad == 0:
        print("  ARM H OK: every internal node's", LEAF_CLASSES,
              "slots are still zero (builder_kernels_impl.cuh:403)")
    else:
        failures += 1
        print("  ARM H FAILED:", internal_bad, "internal slots were written")

    # Block-width independence: the accumulation is an exact integer sum, so
    # a different block width is obliged to produce identical bits.
    var got32 = _launch_leaf[True, PART_TPB_A](
        ctx, nodes, ranges, row_ids, labels_q, LEAF_CLASSES, Float32(1.0),
        LEAF_SAB_NONE,
    )
    var width_bad = 0
    for i in range(LEAF_NODES * LEAF_CLASSES):
        if not _same_bits(got32[0][i], base_leaves[i]):
            width_bad += 1
    if width_bad == 0:
        print("  ARM I OK: tpb", PART_TPB_A, "and tpb", LEAF_TPB,
              "produce bit-identical leaves -- the block width is a"
              " scheduling parameter, not an algorithmic one")
    else:
        failures += 1
        print("  ARM I FAILED:", width_bad, "slots differ across block widths")

    # The fixture must actually distinguish leaves.
    var distinct = 0
    for a in range(LEAF_NODES):
        if nodes[a].left_child_id != NODE_IS_LEAF:
            continue
        var unique = True
        for b in range(a):
            if nodes[b].left_child_id != NODE_IS_LEAF:
                continue
            var same = True
            for c in range(LEAF_CLASSES):
                if not _same_bits(
                    base_leaves[a * LEAF_CLASSES + c],
                    base_leaves[b * LEAF_CLASSES + c],
                ):
                    same = False
            if same:
                unique = False
        if unique:
            distinct += 1
    print("  fixture:", distinct, "of", n_leaves,
          "leaves have a distribution no earlier leaf shares")
    if distinct * 2 <= n_leaves:
        failures += 1
        print("  FIXTURE DEFECT: most leaves must have DISTINCT"
              " distributions, or this fixture cannot tell a misplaced leaf"
              " from a correct one")

    # ---------------- regression -------------------------------------------
    print("")
    var reg_labels = List[Float32]()
    var magnitude = Float64(0.0)
    for r in range(LEAF_ROWS):
        # DIVIDED BY 97, NOT BY A POWER OF TWO, AND THAT IS THE POINT. The
        # first version of this fixture used `/ 64.0`, whose values are all
        # exact multiples of the `1 / 65536` scale `choose_scale` picks, so
        # the quantization was LOSSLESS and DEVIATION 179's measurement came
        # back "0.0 disagreement in 7 of 7 leaves" -- a number that said
        # nothing, because the fixture could not produce a rounding. A
        # non-dyadic divisor makes every label round.
        var y = Float32(Int(mix32(UInt32(r) ^ 0xFEED) % 8192)) / 97.0 - 40.0
        reg_labels.append(y)
        magnitude += Float64(y) if y >= 0 else -Float64(y)
    # DEVIATION 135: the scale is chosen ONCE, on the host, from the WHOLE
    # dataset. The device performs no float-to-integer conversion at all.
    var scale = choose_scale(magnitude, LEAF_ROWS)
    var reg_q = List[Int32]()
    for r in range(LEAF_ROWS):
        var q = quantize(Float64(reg_labels[r]), scale)
        if q > Int64(2147483647) or q < Int64(-2147483648):
            raise Error("a quantized label does not fit Int32")
        reg_q.append(Int32(Int(q)))
    var inv_scale = Float32(1.0 / scale)
    print("[regression] fixed-point scale", scale, "; inv_scale", inv_scale)

    var reg_want = leaf_values_host(
        rebind[MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin]](
            nodes.unsafe_ptr()
        ),
        rebind[MutPointer[InstanceRange, MutAnyOrigin]](ranges.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](row_ids.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](reg_q.unsafe_ptr()),
        LEAF_NODES,
        1,
        inv_scale,
        False,
    )
    var reg_got = _launch_leaf[False, LEAF_TPB](
        ctx, nodes, ranges, row_ids, reg_q, 1, inv_scale, LEAF_SAB_NONE,
    )
    var reg_base = reg_got[0].copy()
    var reg_bad = 0
    for i in range(LEAF_NODES):
        if not _same_bits(reg_base[i], reg_want[i]):
            reg_bad += 1
            if reg_bad <= 4:
                print("    MISMATCH node", i, "got", reg_base[i], "want",
                      reg_want[i])
    if reg_bad == 0:
        print("  ARM J OK:", LEAF_NODES,
              "regression leaves bit-identical to leaf_values_host, the"
              " fixed-point oracle DEVIATION 179 names")
    else:
        failures += 1
        print("  ARM J FAILED:", reg_bad, "regression leaves wrong")

    # DEVIATION BLOCK 179's measurement: how far the device's fixed-point leaf
    # is from `set_leaf_predictions_regression`'s Float64 answer. NOT an
    # assertion -- it cannot be zero and the block says so -- but a number
    # that moves if `choose_scale` ever changes.
    var reg_dataset = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](features.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](
            reg_labels.unsafe_ptr()
        ),
        Int32(LEAF_ROWS),
        Int32(1),
        Int32(LEAF_ROWS),
        Int32(1),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        Int32(1),
    )
    var reg_tree = TreeMetaDataNode[DType.float32](
        Int32(0), Int32(3), Int32(n_leaves), Int32(1),
        List[Float32](), nodes.copy(),
    )
    set_leaf_predictions_regression(reg_dataset, reg_tree, ranges.copy())
    var worst = Float64(0.0)
    var worst_node = -1
    var exact = 0
    var compared = 0
    for i in range(LEAF_NODES):
        if nodes[i].left_child_id != NODE_IS_LEAF:
            continue
        if Int(ranges[i].count) == 0:
            continue
        compared += 1
        var d = Float64(reg_base[i]) - Float64(reg_tree.vector_leaf[i])
        if d < 0:
            d = -d
        if reg_base[i].to_bits() == reg_tree.vector_leaf[i].to_bits():
            exact += 1
        if d > worst:
            worst = d
            worst_node = i
    print("  MEASURED (DEVIATION 179): over", compared,
          "non-empty regression leaves, the device's fixed-point value equals"
          " builder.mojo's Float64 value bit for bit in", exact,
          "of them; the largest disagreement is", worst, "at node",
          worst_node)

    # ---------------- leaf sabotage ----------------------------------------
    print("")
    print("[arm K] leaf sabotage, one per mechanism, same binary")
    var leaf_sabs = [
        LEAF_SAB_NO_ISLEAF,
        LEAF_SAB_NO_ROW_IDS,
        LEAF_SAB_STRIDE_ONE,
        LEAF_SAB_NO_NORMALIZE,
    ]
    var leaf_names = [
        String("the IsLeaf early return (:403): every node's slot is filled"),
        String("the row_ids indirection (:409-411): the slot index is used as"
               " a row id"),
        String("the vector_leaf STRIDE (`node_id + c` instead of"
               " `node_id * num_outputs + c`, decisiontree.cuh:218)"),
        String("the NORMALIZATION (skip `/ total` for Gini and `/ count` for"
               " MSE, objectives.cuh:97-107 and :259-264)"),
    ]
    for a in range(len(leaf_sabs)):
        var run = _launch_leaf[True, LEAF_TPB](
            ctx, nodes, ranges, row_ids, labels_q, LEAF_CLASSES,
            Float32(1.0), leaf_sabs[a],
        )
        var moved = 0
        for i in range(LEAF_NODES * LEAF_CLASSES):
            if not _same_bits(run[0][i], want[i]):
                moved += 1
        if moved == 0:
            failures += 1
            print("  SABOTAGE DID NOT TURN THE CHECK RED --", leaf_names[a])
            print("    that is a defect in THIS FIXTURE, not a pass")
        else:
            print("  red:", moved, "of", LEAF_NODES * LEAF_CLASSES,
                  "leaf slots wrong --", leaf_names[a])

    # The stride sabotage at `num_outputs == 1` is the same arithmetic and is
    # PREDICTED to be invisible. Asserting that it is, rather than skipping
    # it, is what keeps the classification result above meaningful: it says
    # the arm above turned red because of the STRIDE and not because the
    # sabotage argument moves something else.
    var reg_stride = _launch_leaf[False, LEAF_TPB](
        ctx, nodes, ranges, row_ids, reg_q, 1, inv_scale, LEAF_SAB_STRIDE_ONE,
    )
    var reg_stride_moved = 0
    for i in range(LEAF_NODES):
        if not _same_bits(reg_stride[0][i], reg_base[i]):
            reg_stride_moved += 1
    if reg_stride_moved == 0:
        print("  predicted-invisible: the STRIDE sabotage moves nothing at"
              " num_outputs == 1, where `node_id + c` and"
              " `node_id * 1 + c` are the same index")
    else:
        failures += 1
        print("  PREDICTION FAILED: the stride sabotage moved",
              reg_stride_moved, "regression leaves, so the classification arm"
              " above did not turn red for the reason claimed")

    var reg_norm = _launch_leaf[False, LEAF_TPB](
        ctx, nodes, ranges, row_ids, reg_q, 1, inv_scale,
        LEAF_SAB_NO_NORMALIZE,
    )
    var reg_norm_moved = 0
    for i in range(LEAF_NODES):
        if not _same_bits(reg_norm[0][i], reg_base[i]):
            reg_norm_moved += 1
    if reg_norm_moved > 0:
        print("  red:", reg_norm_moved, "of", LEAF_NODES,
              "regression leaves wrong -- the MSE normalization"
              " (objectives.cuh:259-264)")
    else:
        failures += 1
        print("  SABOTAGE DID NOT TURN THE CHECK RED -- the MSE"
              " normalization; that is a defect in THIS FIXTURE")

    # RESTORE and confirm green.
    var restored = _launch_leaf[True, LEAF_TPB](
        ctx, nodes, ranges, row_ids, labels_q, LEAF_CLASSES, Float32(1.0),
        LEAF_SAB_NONE,
    )
    var restore_bad = 0
    for i in range(LEAF_NODES * LEAF_CLASSES):
        if not _same_bits(restored[0][i], base_leaves[i]):
            restore_bad += 1
    if restore_bad == 0:
        print("  restored: the shipping arm rerun after every sabotage is"
              " bit-identical to the shipping arm before them")
    else:
        failures += 1
        print("  RESTORE FAILED:", restore_bad, "slots differ")

    # `Dataset`'s pointers are `MutUntrackedOrigin`; keep the backing buffers
    # alive to here.
    _ = features.unsafe_ptr()
    _ = labels_f.unsafe_ptr()
    _ = labels_q.unsafe_ptr()
    _ = reg_labels.unsafe_ptr()
    _ = reg_q.unsafe_ptr()
    _ = row_ids.unsafe_ptr()
    _ = nodes.unsafe_ptr()
    _ = ranges.unsafe_ptr()
    return failures
