"""Do the four RF device kernels put the right number in the right cell?

    tools/with_build_lock.sh pixi run mojo run -I . \\
        ensemble/mojo_only/builder_kernels_check.mojo

NO CUML COUNTERPART -- this is a check, and cuML has no equivalent of it.
It covers
`ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`,
which mirrors `kernels/builder_kernels_impl.cuh` at rapidsai/cuml
`v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`).

WHY IT IS SHAPED THIS WAY. This repository's most expensive lesson is that
a check whose expected value is the same in every cell verifies the TOTAL
and nothing about PLACEMENT: the same histogram kernel read `0 wrong of
512` under uniform data and `490 wrong of 512` under hashed data, and two
earlier checks had reported it correct at exactly the failing parameters.
A sibling scar: a conservation check could not see a tree that never
split. And a third, from this same round: a lane's sabotage ran GREEN
because the check imported the very formula it was checking.

So, everywhere below:

  * EVERY planted value is HASHED per (row, column) or per (node, cell).
    No two nodes see the same population and no two cells hold the same
    number.
  * EVERY expected value is computed on the HOST, in a DIFFERENT SPELLING
    from the device code -- the bin search is a LINEAR scan against their
    binary one, the partition is a stable two-pass over a list against
    their segmented scan, the cdf is a running total against their
    block-scan-with-chunk-carry.
  * EVERY comparison is PER CELL, and a failure names the first cell.
  * The batch of nodes is RAGGED -- 1, 5, 128, 300, 37, 2, 200 and 64
    instances -- which is the whole reason `WorkloadInfo` exists. 300 and
    200 take three and two blocks at TPB 128, so the cross-block atomic
    flush is exercised; 1 and 2 are single-row and two-row nodes; 128 is
    exactly block-aligned and 37 is not.
  * Each of the two histogram arms -- shared memory (their default) and
    global memory (their fallback) -- is its own NAMED arm. An opt-in
    path is an unchecked path.

THE ARMS

  A. HISTOGRAM, GLOBAL ARM (`:294`, `:317-318`, `:336-342`). Hashed data,
     hashed labels, per-column `n_bins` differing from `max_n_bins`, three
     classes -- so a transposed `label * n_bins + b` offset moves cells and
     a wrong `max_n_bins` stride moves whole columns.
  B. HISTOGRAM, SHARED ARM (`:322-333`, `:344-351`). Same fixture, same
     expected array, fresh histogram. The two arms must agree cell for
     cell with each other AND with the host.
  C. pdf -> cdf AND THE SPLIT SCORING (`:373-393`). Run at TPB 32 with
     `n_bins = 100`, so `ceildiv(100, 32) * 32 = 128` and the scan runs
     FOUR chunks with a ragged last one -- the chunk carry their
     `total_aggregate` exists for. The cdf is compared per cell against a
     host running total, and the published split against an ANALYTIC
     winner planted per node.
  D. THE NODE PARTITION (`:144-212`). Placement per ROW, not counts: a
     partition with the right counts and the wrong order passes a count
     check and fails the algorithm. Includes a node whose split is
     INVALID, which their writer skips and their copy-back must leave
     alone.
  E. THE LEAF PASS (`:213-241`). Distinct leaf values per node, a
     pre-poisoned output buffer, and nodes that are NOT leaves which must
     come back still poisoned.
  F. SABOTAGE, one per mechanism, each with its own PREDICTED movement.

Fixtures are analytic and hashed. No real dataset appears anywhere.
"""

from std.math import ceildiv
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.objectives import (
    CRITERION_GINI,
    ClassificationObjectiveFunction,
)
from ensemble.decisiontree.batched_levelalgo.quantiles import Quantiles
from ensemble.decisiontree.batched_levelalgo.split import Split
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    SharedMemoryConfig,
    WorkloadInfo,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    ClassificationObjective,
    DeviceArgs,
    FindBestSplitsArgs,
    HistogramArgs,
    LeafArgs,
    NodeSplitArgs,
    NodeSplitScratch,
    launch_build_histograms_kernel,
    launch_find_best_splits_kernel,
    launch_leaf_kernel,
    launch_node_split_kernel,
)
from ensemble.flatnode import SparseTreeNode

comptime DT = DType.float32
comptime LT = DType.int32
comptime BinT = ClassificationBin
comptime ObjT = ClassificationObjective[DT, LT, BinT]

comptime TPB = 128
comptime N_CLASSES = 3
comptime N_COLS = 4
comptime N_SAMPLED_COLS = 3
comptime MAX_N_BINS = 8
comptime COL_START = 1
comptime GRID_Y = 2
comptime N_NODES = 8


# --------------------------------------------------------------------------
# Hashing. One mixer, used for every planted value, so no two cells agree by
# construction. Deliberately NOT the hash `random_utils.mojo` ports -- an
# expected value must not come from the code under test.
# --------------------------------------------------------------------------


def h32(x_in: UInt32) -> UInt32:
    var x = x_in
    x = x ^ UInt32(0x9E3779B9)
    x = x * UInt32(0x85EBCA6B)
    x = x ^ (x >> 13)
    x = x * UInt32(0xC2B2AE35)
    x = x ^ (x >> 16)
    return x


def node_counts() -> List[Int]:
    """The ragged batch. 300 and 200 need more than one block at TPB 128;
    1 and 2 are degenerate; 128 is exactly block-aligned; 37 is not."""
    return [1, 5, 128, 300, 37, 2, 200, 64]


def node_begins(counts: List[Int]) -> List[Int]:
    var out = List[Int]()
    var acc = 0
    for i in range(len(counts)):
        out.append(acc)
        acc += counts[i]
    return out^


def build_workload_info(counts: List[Int], tpb: Int) -> List[WorkloadInfo]:
    """`Builder::updateWorkloadInfo`, `builder.cuh:393-407`.

    Their body:

        n_blocks_per_node = max(ceildiv(count, TPB), 1)
        for b in 0..n_blocks_per_node: {int(i), b, n_blocks_per_node}

    Transcribed here because the check must build the same tiling the
    builder will, and `builder.mojo` is another lane's file.
    """
    var out = List[WorkloadInfo]()
    for i in range(len(counts)):
        var nbpn = ceildiv(counts[i], tpb)
        if nbpn < 1:
            nbpn = 1
        for b in range(nbpn):
            out.append(WorkloadInfo(Int32(i), Int32(b), Int32(nbpn)))
    return out^


def bins_per_col() -> List[Int]:
    """Per-column `n_bins`, all DIFFERENT and all below `MAX_N_BINS`, so a
    kernel that uses `max_n_bins` where their code uses `n_bins` -- or the
    other way round -- moves cells rather than staying lucky."""
    return [5, 7, 4, 6]


def quantile_at(col: Int, b: Int) -> Float32:
    """Strictly increasing per column, hashed spacing, hashed offset. The
    array is col-major with stride `MAX_N_BINS` (`:319`)."""
    var base = Float32(Int(h32(UInt32(col * 977 + 13)) % 100)) / 10.0
    var acc = base
    for j in range(b + 1):
        acc += 0.25 + Float32(Int(h32(UInt32(col * 31 + j * 7 + 5)) % 40)) / 20.0
    return acc


def data_at(row: Int, col: Int) -> Float32:
    """Hashed per (row, col), spread across and beyond the quantile range
    so both of `lower_bound`'s clamps are reached."""
    return Float32(Int(h32(UInt32(row * 2654 + col * 104729 + 1)) % 2000)) / 100.0


def label_at(row: Int) -> Int32:
    return Int32(Int(h32(UInt32(row * 7919 + 3)) % N_CLASSES))


def row_id_at(i: Int, n_rows: Int) -> Int32:
    """Hashed, WITH REPLACEMENT -- which is what `raft::random::uniformInt`
    produces at the root (`randomforest.cuh:140-142`). Not the identity, so
    a kernel that forgets the `dataset.row_ids[i]` indirection at `:337`
    reads the wrong rows."""
    return Int32(Int(h32(UInt32(i * 40503 + 17)) % UInt32(n_rows)))


def column_sample_at(nid: Int, slot: Int) -> Int32:
    """Hashed per (node, slot), so no two nodes read the same column and a
    dropped `nid * n_sampled_cols` stride is visible."""
    return Int32(Int(h32(UInt32(nid * 611 + slot * 37 + 9)) % N_COLS))


def host_lower_bound(
    quantiles: List[Float32], col: Int, n_bins: Int, x: Float32
) -> Int:
    """THEIR `:118-133` SEMANTICS, SPELLED AS A LINEAR SCAN.

    "Returns the lowest index in `array` whose value is greater or equal to
    `element`. Values outside the quantile range are clamped to the edge
    bins: values below the first quantile return 0, and values above the
    last quantile return len - 1."

    Written as a forward scan on purpose: the device runs a binary search
    with `end = len - 1`, and a check that ran the same binary search would
    agree with a transcription error on both sides.
    """
    for idx in range(n_bins):
        if quantiles[col * MAX_N_BINS + idx] >= x:
            return idx
    return n_bins - 1


# --------------------------------------------------------------------------
# THE UPLOAD HELPERS, AND WHY THEY EXIST.
#
# MEASURED THIS SESSION, and it cost an arm that had already gone green
# once: **Mojo destroys a value at its LAST USE, not at the end of scope,
# and `enqueue_copy` is ASYNCHRONOUS.** So
#
#     var h = ctx.enqueue_create_host_buffer[...](n)
#     ...fill h...
#     ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
#     ...launch a kernel that reads d...
#
# frees `h` immediately after `h.unsafe_ptr()` -- the last use -- and the
# copy then reads freed host memory. It is a RACE, so it passes sometimes:
# arm C below read a correct `column_samples` on one run and a garbage one
# (n_bins came back as 201325 and as -1) on the next, with no source change
# in between, which is exactly the shape of failure this repository's rule
# about instruments not being trustworthy exists for.
#
# The fix is to force the last use PAST the synchronize, which is what the
# `_ = h^` line below does. Every host-to-device upload in this file goes
# through these three functions so the pattern exists once.
# --------------------------------------------------------------------------


def upload_i32(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.int32], src: List[Int32]
) raises:
    var h = ctx.enqueue_create_host_buffer[DType.int32](len(src))
    for i in range(len(src)):
        h[i] = src[i]
    ctx.enqueue_copy(dst_buf=dst, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def upload_f32(
    ctx: DeviceContext, mut dst: DeviceBuffer[DT], src: List[Float32]
) raises:
    var h = ctx.enqueue_create_host_buffer[DT](len(src))
    for i in range(len(src)):
        h[i] = src[i]
    ctx.enqueue_copy(dst_buf=dst, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def upload_u32(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint32],
    src: List[UInt32],
) raises:
    var h = ctx.enqueue_create_host_buffer[DType.uint32](len(src))
    for i in range(len(src)):
        h[i] = src[i]
    ctx.enqueue_copy(dst_buf=dst, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def upload_structs[
    T: Copyable & Deinitable
](
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.uint8], src: List[T]
) raises:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](
        len(src) * size_of[T]()
    )
    var p = h.unsafe_ptr().unsafe_bitcast[T]()
    for i in range(len(src)):
        p[unsafe_offset=i] = src[i].copy()
    ctx.enqueue_copy(dst_buf=dst, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


# --------------------------------------------------------------------------
# The fixture. One ragged batch of nodes, planted once and reused, because
# every arm must see the SAME population for the cross-arm comparisons to
# mean anything.
# --------------------------------------------------------------------------


struct Fixture(Movable):
    var counts: List[Int]
    var begins: List[Int]
    var wl: List[WorkloadInfo]
    var n_rows: Int
    var n_blocks_dimx: Int
    var quantiles_host: List[Float32]

    var data: DeviceBuffer[DT]
    var labels: DeviceBuffer[LT]
    var sample_weight: DeviceBuffer[DType.float32]
    var row_ids: DeviceBuffer[DType.int32]
    var q_arr: DeviceBuffer[DT]
    var q_nbins: DeviceBuffer[DType.int32]
    var work_items: DeviceBuffer[DType.uint8]
    var workload_info: DeviceBuffer[DType.uint8]
    var column_samples: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        self.counts = node_counts()
        self.begins = node_begins(self.counts)
        self.wl = build_workload_info(self.counts, TPB)
        self.n_blocks_dimx = len(self.wl)
        var total = 0
        for i in range(len(self.counts)):
            total += self.counts[i]
        self.n_rows = total

        var nb = bins_per_col()

        # ---- data, column-major (row_stride 1, col_stride n_rows) -------
        var l_data = List[Float32]()
        for c in range(N_COLS):
            for r in range(total):
                l_data.append(data_at(r, c))
        self.data = ctx.enqueue_create_buffer[DT](total * N_COLS)
        upload_f32(ctx, self.data, l_data)

        var l_lab = List[Int32]()
        for r in range(total):
            l_lab.append(label_at(r))
        self.labels = ctx.enqueue_create_buffer[LT](total)
        upload_i32(ctx, self.labels, l_lab)

        self.sample_weight = ctx.enqueue_create_buffer[DType.float32](1)
        var l_sw = List[Float32]()
        l_sw.append(1.0)
        upload_f32(ctx, self.sample_weight, l_sw)

        var l_rid = List[Int32]()
        for i in range(total):
            l_rid.append(row_id_at(i, total))
        self.row_ids = ctx.enqueue_create_buffer[DType.int32](total)
        upload_i32(ctx, self.row_ids, l_rid)

        # ---- quantiles, col-major with stride MAX_N_BINS ----------------
        self.quantiles_host = List[Float32]()
        for c in range(N_COLS):
            for b in range(MAX_N_BINS):
                self.quantiles_host.append(quantile_at(c, b))
        self.q_arr = ctx.enqueue_create_buffer[DT](N_COLS * MAX_N_BINS)
        upload_f32(ctx, self.q_arr, self.quantiles_host)

        var l_nb = List[Int32]()
        for c in range(N_COLS):
            l_nb.append(Int32(nb[c]))
        self.q_nbins = ctx.enqueue_create_buffer[DType.int32](N_COLS)
        upload_i32(ctx, self.q_nbins, l_nb)

        # ---- work items -------------------------------------------------
        var l_wi = List[NodeWorkItem]()
        for i in range(N_NODES):
            l_wi.append(
                NodeWorkItem(
                    i,
                    Int32(0),
                    InstanceRange(self.begins[i], self.counts[i]),
                )
            )
        self.work_items = ctx.enqueue_create_buffer[DType.uint8](
            N_NODES * size_of[NodeWorkItem]()
        )
        upload_structs(ctx, self.work_items, l_wi)

        # ---- workload info ----------------------------------------------
        self.workload_info = ctx.enqueue_create_buffer[DType.uint8](
            self.n_blocks_dimx * size_of[WorkloadInfo]()
        )
        upload_structs(ctx, self.workload_info, self.wl)

        # ---- column samples ----------------------------------------------
        var l_cs = List[Int32]()
        for n in range(N_NODES):
            for sl in range(N_SAMPLED_COLS):
                l_cs.append(column_sample_at(n, sl))
        self.column_samples = ctx.enqueue_create_buffer[DType.int32](
            N_NODES * N_SAMPLED_COLS
        )
        upload_i32(ctx, self.column_samples, l_cs)

    def dataset(mut self) -> DatasetView[DT, LT]:
        return DatasetView[DT, LT](
            self.data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            self.labels.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
            self.sample_weight.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
            Int64(self.n_rows),
            Int64(N_COLS),
            Int64(1),
            Int64(self.n_rows),
            Int32(self.n_rows),
            Int32(N_SAMPLED_COLS),
            self.row_ids.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
            Int32(N_CLASSES),
            False,
        )

    def quantiles(mut self) -> Quantiles[DT]:
        return Quantiles[DT](
            self.q_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            self.q_nbins.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
        )

    def wi_ptr(mut self) -> MutPointer[NodeWorkItem, MutUntrackedOrigin]:
        return (
            self.work_items.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[NodeWorkItem]()
        )

    def wl_ptr(mut self) -> MutPointer[WorkloadInfo, MutUntrackedOrigin]:
        return (
            self.workload_info.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[WorkloadInfo]()
        )

    def cs_ptr(mut self) -> MutPointer[Int32, MutUntrackedOrigin]:
        return self.column_samples.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()


def expected_histogram(fx: Fixture) -> List[UInt32]:
    """The tally their `:336-342` must produce, computed here row by row
    with the LINEAR bin search and an explicit `label * n_bins + b` offset.

    Layout is theirs (`:315`):
    `(nid * gridDim.y + blockIdx.y) * max_n_bins * n_classes`, and within
    a (node, column) block, `label * n_bins + bin` (`bins.cuh:26`). Note
    the stride is `max_n_bins`, the offset uses `n_bins`; a check that
    used one for the other would agree with a kernel that did the same.
    """
    var nb = bins_per_col()
    var out = List[UInt32]()
    var total_cells = N_NODES * GRID_Y * MAX_N_BINS * N_CLASSES
    for _ in range(total_cells):
        out.append(UInt32(0))
    for nid in range(N_NODES):
        for cy in range(GRID_Y):
            var col = Int(column_sample_at(nid, COL_START + cy))
            var n_bins = nb[col]
            var base = (nid * GRID_Y + cy) * MAX_N_BINS * N_CLASSES
            for i in range(fx.counts[nid]):
                var row = Int(row_id_at(fx.begins[nid] + i, fx.n_rows))
                var x = data_at(row, col)
                var b = host_lower_bound(fx.quantiles_host, col, n_bins, x)
                var lab = Int(label_at(row))
                out[base + lab * n_bins + b] += UInt32(1)
    return out^


@fieldwise_init
struct ArmResult(ImplicitlyCopyable, Movable):
    """What an arm MEASURED, not whether it passed.

    Arm F needs the counts, not a verdict: a sabotage is only evidence if
    the movement it produces is the movement that was PREDICTED, and
    "non-zero" is not a prediction. `bucket_a` and `bucket_b` are the two
    structural splits each arm can distinguish -- see the arm's own
    docstring for what they hold.
    """

    var wrong: Int
    var bucket_a: Int
    var bucket_b: Int


# --------------------------------------------------------------------------
# ARMS A and B -- the histogram, both arms.
# --------------------------------------------------------------------------


def run_histogram[
    sabotage: Int = 0
](
    ctx: DeviceContext,
    mut fx: Fixture,
    use_global: Bool,
    mut hists: DeviceBuffer[DType.uint32],
    mut blob: DeviceArgs[HistogramArgs[ObjT]],
) raises:
    hists.enqueue_fill(0)
    ctx.synchronize()
    var obj = ClassificationObjectiveFunction[DT, LT, BinT](
        Int32(N_CLASSES), Int32(1), CRITERION_GINI
    )
    launch_build_histograms_kernel[TPB=TPB, sabotage=sabotage](
        ctx,
        hists.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[BinT](),
        MAX_N_BINS,
        fx.dataset(),
        fx.quantiles(),
        fx.wi_ptr(),
        COL_START,
        fx.cs_ptr(),
        obj,
        fx.wl_ptr(),
        fx.n_blocks_dimx,
        GRID_Y,
        SharedMemoryConfig(use_global, 0),
        blob,
    )
    ctx.synchronize()


def arm_histogram[
    sabotage: Int = 0
](
    ctx: DeviceContext,
    mut fx: Fixture,
    use_global: Bool,
    quiet: Bool = False,
) raises -> ArmResult:
    """Per cell, never a total.

    `bucket_a` counts wrong cells belonging to nodes the builder gave ONE
    block (counts 1, 5, 128, 37, 2, 64); `bucket_b` counts wrong cells in
    the nodes it gave more than one (counts 300 -> 3 blocks and 200 -> 2).
    That split is what distinguishes a broken `offset_blockid` from a
    broken histogram.
    """
    var name = String("GLOBAL") if use_global else String("SHARED")
    var cells = N_NODES * GRID_Y * MAX_N_BINS * N_CLASSES
    var hists = ctx.enqueue_create_buffer[DType.uint32](cells)
    var blob = DeviceArgs[HistogramArgs[ObjT]](ctx)
    var expected = expected_histogram(fx)

    hists.enqueue_fill(0)
    ctx.synchronize()
    var obj = ClassificationObjectiveFunction[DT, LT, BinT](
        Int32(N_CLASSES), Int32(1), CRITERION_GINI
    )
    launch_build_histograms_kernel[TPB=TPB, sabotage=sabotage](
        ctx,
        hists.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[BinT](),
        MAX_N_BINS,
        fx.dataset(),
        fx.quantiles(),
        fx.wi_ptr(),
        COL_START,
        fx.cs_ptr(),
        obj,
        fx.wl_ptr(),
        fx.n_blocks_dimx,
        GRID_Y,
        SharedMemoryConfig(use_global, 0),
        blob,
    )
    ctx.synchronize()

    var wrong = 0
    var single = 0
    var multi = 0
    var first = -1
    var got0 = UInt32(0)
    var want0 = UInt32(0)
    var planted = 0
    for i in range(len(expected)):
        if expected[i] != UInt32(0):
            planted += 1
    with hists.map_to_host() as h:
        for i in range(cells):
            if h[i] != expected[i]:
                if first < 0:
                    first = i
                    got0 = h[i]
                    want0 = expected[i]
                wrong += 1
                var nid = i // (GRID_Y * MAX_N_BINS * N_CLASSES)
                if ceildiv(fx.counts[nid], TPB) > 1:
                    multi += 1
                else:
                    single += 1
    if not quiet:
        if wrong != 0:
            print(
                "  arm", name, "FAILED:", wrong, "of", cells,
                "cells wrong; first at flat index", first,
                "got", got0, "want", want0,
            )
        else:
            print(
                "  arm", name, "OK: matches the host tally in all", cells,
                "cells (", planted, "non-zero ), over 8 ragged nodes x",
                GRID_Y, "columns",
            )
    _ = blob^
    _ = hists^
    return ArmResult(wrong, single, multi)


# --------------------------------------------------------------------------
# ARM C -- pdf -> cdf and the split scoring.
#
# Run at TPB 32 with n_bins 100 ON PURPOSE. `pdf_to_cdf` pads its loop to
# `ceildiv(n_bins, TPB) * TPB` (`:273`) and carries `total_aggregate`
# across chunks (`:280`); at 100 bins and 32 threads that is FOUR chunks
# with a ragged last one, so the carry is exercised and a check confined to
# the first chunk would be blind. cuML's own launches use TPB 128 with
# max_n_bins <= 128, i.e. exactly one chunk -- their configuration cannot
# reach this bug and this one can.
# --------------------------------------------------------------------------

comptime C_TPB = 32
comptime C_NBINS = 100
comptime C_CLASSES = 2
comptime C_NODES = 4
comptime C_SLOTS = 2


def c_perfect_slot(nid: Int) -> Int:
    return Int(h32(UInt32(nid * 131 + 7)) % 2)


def c_target_bin(nid: Int) -> Int:
    return 10 + Int(h32(UInt32(nid * 977 + 41)) % 80)


def c_column_of(nid: Int, slot: Int) -> Int:
    """Deliberately NOT the identity, so a kernel that indexes the
    histogram by `blockIdx.y` but forgets `column_samples` still lands on
    a legal column and STILL fails the published `colid`."""
    return (slot + nid) % 2


def c_pdf(nid: Int, slot: Int, cls: Int, b: Int) -> UInt32:
    """The planted pdf.

    In the node's PERFECT slot: class 0 occupies bins `[0, tb]` and class 1
    occupies `[tb+1, n_bins-1]`, every bin in its range non-empty and
    hashed. The only bin whose cdf splits the two classes cleanly is `tb`,
    so `tb` is the unique gain maximum and the expected winner is ANALYTIC
    rather than a formula this check would otherwise have to re-derive.

    In the other slot: every bin holds the same count of both classes, so
    every candidate there is as impure as the parent and its gain cannot
    beat a perfect split.
    """
    if slot == c_perfect_slot(nid):
        var tb = c_target_bin(nid)
        if cls == 0:
            if b <= tb:
                return UInt32(1 + Int(h32(UInt32(nid * 31 + b)) % 5))
            return UInt32(0)
        if b > tb:
            return UInt32(1 + Int(h32(UInt32(nid * 57 + b + 900)) % 5))
        return UInt32(0)
    return UInt32(3)


def arm_c_cdf_and_splits[
    sabotage: Int = 0
](ctx: DeviceContext, quiet: Bool = False) raises -> ArmResult:
    """`wrong` counts cdf cells; `bucket_a` counts the ones in the FIRST
    scan chunk (bin < 32), which a broken chunk carry must leave alone;
    `bucket_b` counts nodes whose published split is not the planted
    winner."""
    if not quiet:
        print(
            "arm C: pdf_to_cdf (:259-283) + findBestSplitsKernel"
            " (:353-393), TPB 32, n_bins 100 -> 4 scan chunks"
        )
    var cells = C_NODES * C_SLOTS * C_NBINS * C_CLASSES
    var l_h = List[UInt32]()
    for nid in range(C_NODES):
        for sl in range(C_SLOTS):
            for c in range(C_CLASSES):
                for b in range(C_NBINS):
                    l_h.append(c_pdf(nid, sl, c, b))
    var hists = ctx.enqueue_create_buffer[DType.uint32](cells)
    upload_u32(ctx, hists, l_h)

    # quantiles: strictly increasing, hashed, one array per column
    var l_q = List[Float32]()
    for c in range(2):
        var acc = Float32(0)
        for b in range(C_NBINS):
            acc += 0.5 + Float32(Int(h32(UInt32(c * 13 + b)) % 20)) / 10.0
            l_q.append(acc)
    var q_arr = ctx.enqueue_create_buffer[DT](2 * C_NBINS)
    upload_f32(ctx, q_arr, l_q)
    var l_nb = List[Int32]()
    l_nb.append(Int32(C_NBINS))
    l_nb.append(Int32(C_NBINS))
    var q_nb = ctx.enqueue_create_buffer[DType.int32](2)
    upload_i32(ctx, q_nb, l_nb)

    var l_cs = List[Int32]()
    for nid in range(C_NODES):
        for sl in range(C_SLOTS):
            l_cs.append(Int32(c_column_of(nid, sl)))
    var cs = ctx.enqueue_create_buffer[DType.int32](C_NODES * C_SLOTS)
    upload_i32(ctx, cs, l_cs)

    var dummy = ctx.enqueue_create_buffer[DT](4)
    var dummy_l = ctx.enqueue_create_buffer[LT](4)
    var dummy_i = ctx.enqueue_create_buffer[DType.int32](4)
    dummy.enqueue_fill(0)
    dummy_l.enqueue_fill(0)
    dummy_i.enqueue_fill(0)
    ctx.synchronize()
    var ds = DatasetView[DT, LT](
        dummy.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dummy_l.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dummy.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(4), Int64(2), Int64(1), Int64(4),
        Int32(4), Int32(C_SLOTS),
        dummy_i.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int32(C_CLASSES), False,
    )
    var quantiles = Quantiles[DT](
        q_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        q_nb.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
    )

    var l_sp = List[Split[DT]]()
    for _ in range(C_NODES):
        l_sp.append(Split[DT]())
    var splits = ctx.enqueue_create_buffer[DType.uint8](
        C_NODES * size_of[Split[DT]]()
    )
    upload_structs(ctx, splits, l_sp)
    var l_mx = List[Int32]()
    for _ in range(C_NODES):
        l_mx.append(Int32(0))
    var mutex = ctx.enqueue_create_buffer[DType.int32](C_NODES)
    upload_i32(ctx, mutex, l_mx)

    var obj = ClassificationObjectiveFunction[DT, LT, BinT](
        Int32(C_CLASSES), Int32(1), CRITERION_GINI
    )
    var blob = DeviceArgs[FindBestSplitsArgs[ObjT]](ctx)
    launch_find_best_splits_kernel[TPB=C_TPB, sabotage=sabotage](
        ctx,
        hists.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[BinT](),
        C_NBINS,
        ds,
        quantiles,
        0,
        cs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        mutex.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        splits.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[Split[DT]](),
        obj,
        C_NODES,
        C_SLOTS,
        blob,
    )
    ctx.synchronize()
    # KEEP-ALIVE, and it is not decoration -- see `upload_i32`'s comment.
    # A `DeviceBuffer` whose last use is `.unsafe_ptr()` is destroyed AT
    # THAT LINE, and the next allocation reuses the address: this arm read
    # `n_bins` as -8388609, the bit pattern of `Split::Min()`, because
    # `q_nbins` had been freed and `splits` had been allocated on top of
    # it. Forcing the last use past the synchronize is the fix.
    _ = q_arr^
    _ = q_nb^
    _ = cs^
    _ = dummy^
    _ = dummy_l^
    _ = dummy_i^
    _ = blob^

    # ---- C1: the cdf, per cell, against a host RUNNING TOTAL ----------
    var wrong = 0
    var wrong_first_chunk = 0
    var first = -1
    var got0 = UInt32(0)
    var want0 = UInt32(0)
    with hists.map_to_host() as h:
        for nid in range(C_NODES):
            for sl in range(C_SLOTS):
                for c in range(C_CLASSES):
                    var run = UInt32(0)
                    for b in range(C_NBINS):
                        run += c_pdf(nid, sl, c, b)
                        var idx = (
                            (nid * C_SLOTS + sl) * C_NBINS * C_CLASSES
                            + c * C_NBINS
                            + b
                        )
                        if h[idx] != run:
                            if first < 0:
                                first = b
                                got0 = h[idx]
                                want0 = run
                            wrong += 1
                            if b < C_TPB:
                                wrong_first_chunk += 1
    if wrong != 0:
        if not quiet:
            print(
                "  arm C1 FAILED:", wrong, "of", cells,
                "cdf cells wrong; first at bin", first,
                "got", got0, "want", want0,
            )
    elif not quiet:
        print(
            "  arm C1 OK: all", cells,
            "cdf cells match a host running total, across 4 scan chunks"
            " (bins 0-31, 32-63, 64-95, 96-99)",
        )

    # ---- C2: the published split, per node, ANALYTIC -------------------
    var bad_nodes = 0
    var first_node = -1
    with splits.map_to_host() as h:
        var p = h.unsafe_ptr().unsafe_bitcast[Split[DT]]()
        for nid in range(C_NODES):
            var s = p[unsafe_offset=nid]
            var tb = c_target_bin(nid)
            var want_col = Int32(
                c_column_of(nid, c_perfect_slot(nid))
            )
            var want_nleft = Int64(0)
            for b in range(tb + 1):
                want_nleft += Int64(Int(c_pdf(nid, c_perfect_slot(nid), 0, b)))
            var ok = (
                s.colid == want_col
                and s.split_start == Int32(tb)
                and s.split_end == Int32(tb)
                and s.global_nLeft == want_nleft
                and s.quesval == l_q[Int(want_col) * C_NBINS + tb]
            )
            if not ok:
                bad_nodes += 1
                if first_node < 0:
                    first_node = nid
    if bad_nodes != 0:
        if not quiet:
            print(
                "  arm C2 FAILED:", bad_nodes, "of", C_NODES,
                "nodes published the wrong split; first is node",
                first_node,
            )
    elif not quiet:
        print(
            "  arm C2 OK: all", C_NODES,
            "nodes published the analytically-forced winner -- column,"
            " bin, threshold and global_nLeft, each planted differently"
            " per node, and each requiring the cross-blockIdx.y mutex"
            " merge to pick the perfect column over the noisy one",
        )
    _ = hists^
    _ = splits^
    _ = mutex^
    return ArmResult(wrong, wrong_first_chunk, bad_nodes)


# --------------------------------------------------------------------------
# ARM D -- the node partition. PLACEMENT PER ROW, never a count.
#
# A partition with the right counts and the wrong order passes a count
# check and fails the algorithm, so every position of `row_ids` is compared
# against a host stable partition. Node 5 (two rows) carries an INVALID
# split: their writer skips it (`:198`) and their copy-back returns early
# (`:127`), so its two row ids must come back UNTOUCHED and in their
# original order -- the one thing a check that only inspects nodes that
# did split cannot see.
# --------------------------------------------------------------------------


def d_split_col(nid: Int) -> Int32:
    return Int32(Int(h32(UInt32(nid * 313 + 11)) % N_COLS))


def d_quesval(nid: Int) -> Float32:
    """Hashed per node and inside the data range (0.00 .. 19.99), so every
    node splits somewhere different and no node sends all rows one way."""
    return 4.0 + Float32(Int(h32(UInt32(nid * 727 + 5)) % 1200)) / 100.0


def d_is_invalid(nid: Int) -> Bool:
    return nid == 5


def arm_d_partition[
    sabotage: Int = 0
](
    ctx: DeviceContext, mut fx: Fixture, quiet: Bool = False
) raises -> ArmResult:
    """`wrong` counts row_ids POSITIONS; `bucket_a` counts the wrong
    positions inside the INVALID node's range (node 5), which their writer
    and their copy-back must both leave alone; `bucket_b` counts nodes with
    the wrong `local_nLeft`."""
    if not quiet:
        print(
            "arm D: launchNodeSplitKernel (:144-212), placement per ROW"
        )
    var n = fx.n_rows

    # working copy of row_ids: the copy-back kernel writes through
    # `dataset.row_ids` (`:140`), so the fixture's own array must survive.
    var orig = List[Int32]()
    for i in range(n):
        orig.append(row_id_at(i, n))
    var rid = ctx.enqueue_create_buffer[DType.int32](n)
    upload_i32(ctx, rid, orig)

    var ds = DatasetView[DT, LT](
        fx.data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        fx.labels.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        fx.sample_weight.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ](),
        Int64(n), Int64(N_COLS), Int64(1), Int64(n),
        Int32(n), Int32(N_SAMPLED_COLS),
        rid.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int32(N_CLASSES), False,
    )

    var l_sp = List[Split[DT]]()
    for i in range(N_NODES):
        if d_is_invalid(i):
            l_sp.append(Split[DT]())
        else:
            # `local_nLeft` is planted at 999 so the reset kernel (`:48-53`)
            # has something visible to clear: if it never ran, every node's
            # count is 999 and arm D2 says so.
            l_sp.append(
                Split[DT](
                    d_quesval(i), d_split_col(i), 1.0, Int64(0),
                    Int64(999), Int32(0), Int32(0),
                )
            )
    var splits = ctx.enqueue_create_buffer[DType.uint8](
        N_NODES * size_of[Split[DT]]()
    )
    upload_structs(ctx, splits, l_sp)

    var prid = ctx.enqueue_create_buffer[DType.int32](n)
    prid.enqueue_fill(Int32(-1))
    ctx.synchronize()

    var scratch = NodeSplitScratch[DT, TPB](
        ctx, fx.n_blocks_dimx * TPB, N_NODES
    )
    var blob = DeviceArgs[NodeSplitArgs[DT, LT]](ctx)
    launch_node_split_kernel[TPB=TPB, sabotage=sabotage](
        ctx, ds, fx.wi_ptr(),
        splits.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[Split[DT]](),
        fx.wl_ptr(),
        fx.n_blocks_dimx, N_NODES,
        prid.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        scratch, blob,
    )
    ctx.synchronize()
    # KEEP-ALIVE past the synchronize; see `upload_i32`'s comment.
    _ = prid^
    _ = scratch^
    _ = blob^

    # ---- the host stable partition, spelled as two passes over a list --
    var expected = List[Int32]()
    var want_left = List[Int]()
    for _ in range(n):
        expected.append(Int32(0))
    for nid in range(N_NODES):
        var b = fx.begins[nid]
        var c = fx.counts[nid]
        if d_is_invalid(nid):
            want_left.append(0)
            for i in range(c):
                expected[b + i] = orig[b + i]
            continue
        var col = Int(d_split_col(nid))
        var qv = d_quesval(nid)
        var lefts = List[Int32]()
        var rights = List[Int32]()
        for i in range(c):
            var row = orig[b + i]
            if data_at(Int(row), col) <= qv:
                lefts.append(row)
            else:
                rights.append(row)
        want_left.append(len(lefts))
        for i in range(len(lefts)):
            expected[b + i] = lefts[i]
        for i in range(len(rights)):
            expected[b + len(lefts) + i] = rights[i]

    var wrong = 0
    var wrong_invalid = 0
    var first = -1
    var gv = Int32(0)
    var wv = Int32(0)
    var inv_b = fx.begins[5]
    var inv_e = inv_b + fx.counts[5]
    with rid.map_to_host() as h:
        for i in range(n):
            if h[i] != expected[i]:
                if first < 0:
                    first = i
                    gv = h[i]
                    wv = expected[i]
                wrong += 1
                if i >= inv_b and i < inv_e:
                    wrong_invalid += 1
    if wrong != 0:
        if not quiet:
            print(
                "  arm D1 FAILED:", wrong, "of", n,
                "row_ids positions wrong; first at position", first,
                "got", gv, "want", wv,
            )
    elif not quiet:
        var moved = 0
        for i in range(n):
            if expected[i] != orig[i]:
                moved += 1
        print(
            "  arm D1 OK: all", n,
            "row_ids positions match a host stable partition;",
            moved, "of them actually moved, so the check is not passing"
            " on an identity",
        )

    var bad = 0
    var firstn = -1
    with splits.map_to_host() as h:
        var p = h.unsafe_ptr().unsafe_bitcast[Split[DT]]()
        for nid in range(N_NODES):
            if p[unsafe_offset=nid].local_nLeft != Int64(want_left[nid]):
                bad += 1
                if firstn < 0:
                    firstn = nid
    if bad != 0:
        if not quiet:
            print(
                "  arm D2 FAILED:", bad, "of", N_NODES,
                "nodes have the wrong local_nLeft; first is node", firstn,
            )
    elif not quiet:
        print(
            "  arm D2 OK: local_nLeft matches per node, including 0 for"
            " the invalid node -- so DEVIATION 127's 32-bit shadow plus"
            " publish lands exactly where their 64-bit atomic did",
        )
    _ = rid^
    _ = splits^
    return ArmResult(wrong, wrong_invalid, bad)


# --------------------------------------------------------------------------
# ARM E -- the leaf pass.
# --------------------------------------------------------------------------


def e_is_leaf(nid: Int) -> Bool:
    return nid % 3 != 1


comptime E_POISON = Float32(-7.0)


def arm_e_leaf[
    sabotage: Int = 0
](
    ctx: DeviceContext, mut fx: Fixture, quiet: Bool = False
) raises -> ArmResult:
    """`wrong` counts leaf cells; `bucket_a` counts the wrong cells that
    belong to nodes that are NOT leaves (which must stay poisoned);
    `bucket_b` counts the wrong cells that belong to leaves."""
    if not quiet:
        print("arm E: leafKernel (:213-241), per leaf cell")

    var l_tree = List[SparseTreeNode[DT]]()
    for i in range(N_NODES):
        if e_is_leaf(i):
            l_tree.append(
                SparseTreeNode[DT].CreateLeafNode(Int32(fx.counts[i]))
            )
        else:
            l_tree.append(
                SparseTreeNode[DT].CreateSplitNode(
                    Int32(1), 0.5, 0.25, Int64(2 * i + 1),
                    Int32(fx.counts[i]),
                )
            )
    var tree = ctx.enqueue_create_buffer[DType.uint8](
        N_NODES * size_of[SparseTreeNode[DT]]()
    )
    upload_structs(ctx, tree, l_tree)

    var l_ir = List[InstanceRange]()
    for i in range(N_NODES):
        l_ir.append(InstanceRange(fx.begins[i], fx.counts[i]))
    var ir = ctx.enqueue_create_buffer[DType.uint8](
        N_NODES * size_of[InstanceRange]()
    )
    upload_structs(ctx, ir, l_ir)

    # POISONED, so "untouched" is a value and not an absence.
    var leaves = ctx.enqueue_create_buffer[DT](N_NODES * N_CLASSES)
    var l_poison = List[Float32]()
    for _ in range(N_NODES * N_CLASSES):
        l_poison.append(E_POISON)
    upload_f32(ctx, leaves, l_poison)

    var obj = ClassificationObjectiveFunction[DT, LT, BinT](
        Int32(N_CLASSES), Int32(1), CRITERION_GINI
    )
    var blob = DeviceArgs[LeafArgs[ObjT]](ctx)
    launch_leaf_kernel[TPB=TPB, sabotage=sabotage](
        ctx, obj, fx.dataset(),
        tree.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[SparseTreeNode[DT]](),
        ir.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[InstanceRange](),
        leaves.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        N_NODES, N_CLASSES * size_of[BinT](),
        blob,
    )
    ctx.synchronize()
    # KEEP-ALIVE past the synchronize; see `upload_i32`'s comment.
    _ = tree^
    _ = ir^
    _ = blob^

    var expected = List[Float32]()
    for _ in range(N_NODES * N_CLASSES):
        expected.append(E_POISON)
    for nid in range(N_NODES):
        if not e_is_leaf(nid):
            continue
        var cnt = List[Int]()
        for _ in range(N_CLASSES):
            cnt.append(0)
        for i in range(fx.counts[nid]):
            var row = Int(row_id_at(fx.begins[nid] + i, fx.n_rows))
            cnt[Int(label_at(row))] += 1
        var total = 0
        for c in range(N_CLASSES):
            total += cnt[c]
        for c in range(N_CLASSES):
            expected[nid * N_CLASSES + c] = Float32(cnt[c]) / Float32(total)

    # How many leaf vectors are DISTINCT? A mis-indexed write is only
    # visible if the values differ per node, so the fixture says so out
    # loud rather than hoping.
    var distinct = 0
    for a in range(N_NODES):
        if not e_is_leaf(a):
            continue
        var uniq = True
        for b in range(a):
            if not e_is_leaf(b):
                continue
            var same = True
            for c in range(N_CLASSES):
                if expected[a * N_CLASSES + c] != expected[
                    b * N_CLASSES + c
                ]:
                    same = False
            if same:
                uniq = False
        if uniq:
            distinct += 1

    var wrong = 0
    var wrong_internal = 0
    var wrong_leaf = 0
    var first = -1
    var gv = Float32(0)
    var wv = Float32(0)
    with leaves.map_to_host() as h:
        for i in range(N_NODES * N_CLASSES):
            if h[i] != expected[i]:
                if first < 0:
                    first = i
                    gv = h[i]
                    wv = expected[i]
                wrong += 1
                if e_is_leaf(i // N_CLASSES):
                    wrong_leaf += 1
                else:
                    wrong_internal += 1
    if wrong != 0:
        if not quiet:
            print(
                "  arm E FAILED:", wrong, "of", N_NODES * N_CLASSES,
                "leaf cells wrong; first at flat index", first,
                "got", gv, "want", wv,
            )
    elif not quiet:
        print(
            "  arm E OK: all", N_NODES * N_CLASSES,
            "cells match --", distinct,
            "distinct leaf vectors among the leaves, and the 3 internal"
            " nodes came back still poisoned at", E_POISON,
        )
    _ = leaves^
    return ArmResult(wrong, wrong_internal, wrong_leaf)



# --------------------------------------------------------------------------
# ARM F -- SABOTAGE, one per mechanism, each with its own PREDICTION.
#
# A digest cannot tell a working change from a no-op, and reach is proved
# PER BRANCH, so every mechanism gets its own corruption and every
# corruption gets a prediction that DIFFERS from the others'. An arm that
# "moves under any corruption" is not evidence.
#
# If a prediction turns out wrong, THE MEASUREMENT IS RIGHT AND THE
# PREDICTION WAS WRONG: say so and fix the prediction, never the code.
# --------------------------------------------------------------------------


def verdict(
    label: String, ok: Bool, detail: String, mut failures: Int
) -> None:
    if ok:
        print("  " + label + " MOVED AS PREDICTED: " + detail)
    else:
        failures += 1
        print("  " + label + " PREDICTION WRONG: " + detail)


def arm_f_sabotage(ctx: DeviceContext, mut fx: Fixture) raises -> Int:
    print("arm F: sabotage, one per mechanism, with per-mechanism"
          " predictions")
    var failures = 0

    # -- F1: histogram, `tid` loses `offset_blockid` (:335). ------------
    # PREDICTION: nodes the builder gave ONE block are unaffected (their
    # `offset_blockid` is 0, so dropping it changes nothing); nodes it gave
    # more than one lose their tail and double-count their head. So
    # bucket_a (single-block nodes) must be 0 and bucket_b (nodes 3 and 6,
    # at 300 and 200 instances) must be non-zero.
    var f1 = arm_histogram[1](ctx, fx, True, True)
    verdict(
        "F1 histogram offset_blockid",
        f1.bucket_a == 0 and f1.bucket_b > 0,
        String(f1.wrong) + " cells wrong: " + String(f1.bucket_a)
        + " in single-block nodes (predicted 0) and " + String(f1.bucket_b)
        + " in the two multi-block nodes (predicted > 0)",
        failures,
    )

    # -- F2: histogram, the shared->global flush collapses onto slot 0. --
    # PREDICTION: this is inside `if not USE_GLOBAL_MEMORY_HISTOGRAM`, so
    # the GLOBAL arm must be bit-identical to clean and the SHARED arm must
    # break. That is the per-branch reach proof for DEVIATION 103b's two
    # instantiations: if both arms moved, they would be one path wearing
    # two names.
    var f2g = arm_histogram[2](ctx, fx, True, True)
    var f2s = arm_histogram[2](ctx, fx, False, True)
    verdict(
        "F2 histogram shared flush",
        f2g.wrong == 0 and f2s.wrong > 0,
        "GLOBAL arm " + String(f2g.wrong)
        + " cells wrong (predicted 0 -- the branch is not on that path)"
        + " and SHARED arm " + String(f2s.wrong)
        + " cells wrong (predicted > 0)",
        failures,
    )

    # -- F3: histogram, the bin search is replaced by a constant 0. -----
    # PREDICTION: both arms break, because `lower_bound` is on both paths.
    # Distinguishes a broken SEARCH from a broken FLUSH, which F2 leaves
    # ambiguous.
    var f3g = arm_histogram[3](ctx, fx, True, True)
    var f3s = arm_histogram[3](ctx, fx, False, True)
    verdict(
        "F3 histogram lower_bound",
        f3g.wrong > 0 and f3s.wrong > 0,
        "GLOBAL " + String(f3g.wrong) + " and SHARED " + String(f3s.wrong)
        + " cells wrong (predicted both > 0 -- the search is on both"
        + " paths, unlike F2's flush)",
        failures,
    )

    # -- F4: findBestSplits, global_sample_count sums class 0 alone. ----
    # PREDICTION: the cdf is produced BEFORE the count is used, so the cdf
    # must be untouched and only the published split may move.
    var f4 = arm_c_cdf_and_splits[1](ctx, True)
    verdict(
        "F4 split global_sample_count",
        f4.wrong == 0 and f4.bucket_b > 0,
        String(f4.wrong) + " cdf cells wrong (predicted 0) and "
        + String(f4.bucket_b)
        + " of 4 nodes published a different split (predicted > 0)",
        failures,
    )

    # -- F5: pdf_to_cdf drops the chunk carry (`core/block_scan`, its 2).
    # PREDICTION: the FIRST chunk of every series is a complete scan on its
    # own, so bins 0..31 must stay right and only bins >= 32 may break.
    # This is the bug `n_bins <= TPB` -- cuML's own configuration -- cannot
    # reach, which is why this arm runs at TPB 32 with 100 bins.
    var f5 = arm_c_cdf_and_splits[2](ctx, True)
    verdict(
        "F5 pdf_to_cdf chunk carry",
        f5.bucket_a == 0 and f5.wrong > 0,
        String(f5.wrong) + " cdf cells wrong, of which "
        + String(f5.bucket_a)
        + " are in the first chunk (bins 0-31, predicted 0)",
        failures,
    )

    # -- F6: the partition writer's rank loses their `- 1` (:110). ------
    # PREDICTION: `local_nLeft` is produced by a different kernel and does
    # not move, so the COUNTS stay right and the PLACEMENT breaks. This is
    # exactly the failure a count check cannot see, and it is the reason
    # arm D compares positions. The number of wrong positions is NOT
    # predicted: with the `- 1` gone, ranks -1 and 0 collide on the same
    # `out_idx` and the sabotaged writer races, so it drifts a row or two
    # between runs. Every arm that is NOT sabotaged is bit-stable.
    var f6 = arm_d_partition[4](ctx, fx, True)
    verdict(
        "F6 partition rank off-by-one",
        f6.wrong > 0 and f6.bucket_b == 0,
        String(f6.wrong) + " row positions wrong (predicted > 0; the count"
        + " itself is not predicted -- rank -1 and rank 0 collide, so the"
        + " sabotaged writer races) while "
        + String(f6.bucket_b)
        + " nodes have a wrong local_nLeft (predicted 0 -- the counts are"
        + " still right, the order is not)",
        failures,
    )

    # -- F7: the copy-back drops `split.IsValid()` (:127). --------------
    # PREDICTION: the damage is CONFINED to the one node whose split is
    # invalid -- their comment at `:116-117` says leaf/invalid nodes keep
    # their existing row-id order. So every wrong position must lie inside
    # node 5's two-row range, i.e. bucket_a == wrong.
    var f7 = arm_d_partition[6](ctx, fx, True)
    verdict(
        "F7 partition copy-back IsValid",
        f7.wrong > 0 and f7.bucket_a == f7.wrong,
        String(f7.wrong) + " row positions wrong, of which "
        + String(f7.bucket_a)
        + " lie inside the INVALID node's range (predicted: all of them)",
        failures,
    )

    # -- F8: the scan's inter-block carry is dropped (`scan_by_key` 2). --
    # PREDICTION: the carry is a PLACEMENT mechanism and lives in a
    # different kernel from the counting one, so positions must break and
    # `local_nLeft` must not. The exact position count is NOT predicted and
    # is not stable between runs: with the carry gone, two slots can
    # compute the same `out_idx` and race, which is itself a reminder that
    # a wrong scan does not merely permute, it drops rows.
    var f8 = arm_d_partition[2](ctx, fx, True)
    verdict(
        "F8 partition inter-block carry",
        f8.wrong > 0 and f8.bucket_b == 0,
        String(f8.wrong) + " row positions wrong (predicted > 0; the count"
        + " itself is not predicted, the sabotaged writer races) with "
        + String(f8.bucket_b)
        + " wrong local_nLeft (predicted 0 -- a different kernel)",
        failures,
    )

    # -- F9: the leaf write loses the `num_outputs *` stride (:238-239). -
    # PREDICTION: leaves move and internal nodes are collateral, because
    # `leaves + node_id` for node_id >= 1 lands inside another node's
    # vector. Both buckets non-zero.
    var f9 = arm_e_leaf[1](ctx, fx, True)
    verdict(
        "F9 leaf output stride",
        f9.wrong > 0,
        String(f9.wrong) + " leaf cells wrong (predicted > 0): "
        + String(f9.bucket_b) + " in leaves and " + String(f9.bucket_a)
        + " in internal nodes",
        failures,
    )

    # -- F10: the leaf kernel drops the `IsLeaf()` guard (:227). --------
    # PREDICTION: EXACTLY the three internal nodes' nine cells change, and
    # not one leaf cell. This is the sharpest prediction in the arm -- an
    # exact count -- and it is what proves the poison test has teeth rather
    # than merely observing an absence.
    var f10 = arm_e_leaf[2](ctx, fx, True)
    verdict(
        "F10 leaf IsLeaf guard",
        f10.wrong == 9 and f10.bucket_b == 0,
        String(f10.wrong) + " leaf cells wrong (predicted exactly 9 = 3"
        + " internal nodes x 3 classes) with " + String(f10.bucket_b)
        + " of them in leaves (predicted 0)",
        failures,
    )
    return failures


def main() raises:
    print(
        "builder_kernels_check:"
        " ensemble/decisiontree/batched_levelalgo/kernels/"
        "builder_kernels_impl.mojo"
    )
    print("  mirroring builder_kernels_impl.cuh @ cuml v26.08.00 265b9da6")
    var ctx = DeviceContext()
    var fx = Fixture(ctx)
    var failures = 0

    print("arm A/B: buildHistogramsKernel (:285-352), both arms, per cell")
    failures += 1 if arm_histogram(ctx, fx, True).wrong != 0 else 0
    failures += 1 if arm_histogram(ctx, fx, False).wrong != 0 else 0
    var c = arm_c_cdf_and_splits(ctx)
    failures += 1 if c.wrong != 0 else 0
    failures += 1 if c.bucket_b != 0 else 0
    var d = arm_d_partition(ctx, fx)
    failures += 1 if d.wrong != 0 else 0
    failures += 1 if d.bucket_b != 0 else 0
    failures += 1 if arm_e_leaf(ctx, fx).wrong != 0 else 0
    failures += arm_f_sabotage(ctx, fx)

    if failures == 0:
        print("builder_kernels_check: ALL OK")
    else:
        raise Error(
            "builder_kernels_check: " + String(failures) + " failure(s)"
        )
