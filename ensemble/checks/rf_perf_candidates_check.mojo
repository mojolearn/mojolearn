# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATIONS 2010 / 2011 / 2012 -- the 2026-09-01 RF perf candidates.

    tools/with_build_lock.sh pixi run mojo run -I . \\
        ensemble/checks/rf_perf_candidates_check.mojo

RUN 2026-09-01, ALL ARMS GREEN (G0-G2, H1-H3, S1-S2), Apple M4, serial
niced, at commit `1790aea1`; the gate record is in `archive/plans/ensemble/PLAN.md`. The
command above is the orchestrator's. STILL ONE APPLE M4.

WHAT THIS CHECK IS FOR, and why it cannot lean on the forest. All three
candidates are chosen precisely because the FOREST cannot move under
them (integer/fixed-point accumulation over an unchanged multiset), so
"the fingerprints held" proves identity but can never prove REACH -- a
build where the candidate arm silently compiled out would hold the
fingerprints too. Reach therefore comes from arms that watch the
MECHANISM: the sorted buffer itself (2010), the coarse workload table
actually driving multi-iteration strides (2011), and a flush sabotage
whose predicted movement is UNDERCOUNT-ONLY (2012).

Fixture discipline is `builder_kernels_check.mojo`'s: hashed values per
(row, column), ragged node batch, host expectations computed in a
DIFFERENT SPELLING (linear bin search, plain loops), every comparison
per cell.

THE ARMS

  G0. BASELINE, binned shared, TPB-granular table (cuML's mapping) ==
      host tally per cell. Proves the fixture and the expected array
      before either candidate is asked anything.
  G1. DEVIATION 2011: the SAME kernel under a table built at
      granularity TPB*4 -- nodes now hand each thread up to 4 strided
      elements (the fixture's 700-row node gets 2 coarse blocks where
      the fine table gave 6). Per cell == host tally, and == G0.
      Runs the SHARED and the GLOBAL arm both; an opt-in path is an
      unchecked path.
  G2. `update_workload_info(granularity=512)` on the host: block counts
      and offsets equal a hand-derived map -- the builder-side reach for
      2011 (the flag's only effect is this table).
  H1. DEVIATION 2012: binned shared with SMEM_COPIES=4 == host tally
      per cell (the privatized sums re-associate to the same integers).
  H2. DEVIATION 2012 fallback: SMEM_BIN_SLOTS too small for 4 copies
      (but big enough for 1) still == host tally -- the p_live=1
      dispatch is a named arm, not a hope.
  H3. SABOTAGE (2012 reach): SMEM_COPIES=4 with sabotage=4 (flush sums
      copy 0 alone) must MOVE, and every moved cell must UNDERCOUNT
      (got < want; got > want anywhere fails the arm) -- the predicted
      movement, not "non-zero".
  S1. DEVIATION 2010: hashed keys, device sort, read back ascending AND
      the same multiset (per-value counts).
  S2. SABOTAGE (2010 reach): sabotage=1 drops the top two passes; on a
      fixture spanning those bits the output must NOT be ascending.
"""

from std.math import ceildiv
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer, DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.objectives import (
    CRITERION_GINI,
    ClassificationObjectiveFunction,
)
from ensemble.decisiontree.batched_levelalgo.quantiles import Quantiles
from ensemble.decisiontree.batched_levelalgo.builder import (
    TPB_DEFAULT,
    update_workload_info,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    SharedMemoryConfig,
    WorkloadInfo,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    DeviceArgs,
    HistogramArgs,
    launch_build_histograms_kernel,
)
from ensemble.randomforest import sort_passes_for, sort_selected_rows
from core.segmented_sort import SORT_BLOCK

comptime DT = DType.float32
comptime LT = DType.int32
comptime BinT = ClassificationBin
comptime ObjT = ClassificationObjectiveFunction[DT, LT, BinT]

comptime TPB = 128
comptime N_CLASSES = 3
comptime N_COLS = 4
comptime N_SAMPLED_COLS = 3
comptime MAX_N_BINS = 8
comptime COL_START = 1
comptime GRID_Y = 2
comptime N_NODES = 6
comptime COARSE = TPB * 4  # DEVIATION 2011's candidate granularity


def h32(x_in: UInt32) -> UInt32:
    """FNV-ish avalanche; same role as builder_kernels_check's."""
    var x = x_in
    x ^= x >> 16
    x *= UInt32(0x7FEB352D)
    x ^= x >> 15
    x *= UInt32(0x846CA68B)
    x ^= x >> 16
    return x


def node_counts() -> List[Int]:
    """Ragged ON PURPOSE: 700 spans 6 fine blocks vs 2 coarse ones (the
    2011 stride actually loops); 1 and 2 are sub-warp; 128 is exactly
    one fine block and a PARTIAL coarse one; 300 and 37 are ragged both
    ways."""
    return [1, 700, 128, 300, 37, 2]


def node_begins(counts: List[Int]) -> List[Int]:
    var out = List[Int]()
    var acc = 0
    for i in range(len(counts)):
        out.append(acc)
        acc += counts[i]
    return out^


def build_workload(counts: List[Int], granularity: Int) -> List[WorkloadInfo]:
    """The host map at a given granularity -- the check's OWN spelling
    (a two-pass append), against `update_workload_info`'s in-place
    write."""
    var out = List[WorkloadInfo]()
    for i in range(len(counts)):
        var nb = max(ceildiv(counts[i], granularity), 1)
        for b in range(nb):
            out.append(WorkloadInfo(Int32(i), Int32(b), Int32(nb)))
    return out^


def bins_per_col() -> List[Int]:
    return [5, 8, 6, 7]


def quantile_at(col: Int, b: Int) -> Float32:
    """Ascending per column, hashed spacing."""
    var acc = Float32(0.0)
    for j in range(b + 1):
        var h = h32(UInt32(col * 131 + j * 7 + 3))
        acc += Float32(Int(h % UInt32(97)) + 1) / Float32(1000.0)
    return acc


def data_at(row: Int, col: Int) -> Float32:
    var h = h32(UInt32(row * 2654435761 + col * 40503 + 17))
    return Float32(Int(h % UInt32(1200))) / Float32(1000.0)


def label_at(row: Int) -> Int32:
    return Int32(Int(h32(UInt32(row * 7919 + 5)) % UInt32(N_CLASSES)))


def row_id_at(i: Int, n_rows: Int) -> Int32:
    """A permuted-ish id stream WITH repeats, like a bootstrap draw."""
    return Int32(Int(h32(UInt32(i * 613 + 11)) % UInt32(n_rows)))


def column_sample_at(nid: Int, slot: Int) -> Int32:
    return Int32(Int(h32(UInt32(nid * 31 + slot)) % UInt32(N_COLS)))


def host_lower_bound(
    quantiles_host: List[Float32], col: Int, n_bins: Int, x: Float32
) -> Int:
    """LINEAR, deliberately -- a different spelling from the device's
    binary search and from the precomputed bin matrix's producer, which
    itself is checked against this."""
    for b in range(n_bins - 1):
        if quantiles_host[col * MAX_N_BINS + b] >= x:
            return b
    return n_bins - 1


def upload_i32(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.int32], src: List[Int32]
) raises:
    var h = ctx.enqueue_create_host_buffer[DType.int32](len(src))
    for i in range(len(src)):
        h.unsafe_ptr().unsafe_store(i, src[i])
    ctx.enqueue_copy(dst_buf=dst, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def upload_u8(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.uint8], src: List[UInt8]
) raises:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](len(src))
    for i in range(len(src)):
        h.unsafe_ptr().unsafe_store(i, src[i])
    ctx.enqueue_copy(dst_buf=dst, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def upload_f32(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], src: List[Float32]
) raises:
    var h = ctx.enqueue_create_host_buffer[DType.float32](len(src))
    for i in range(len(src)):
        h.unsafe_ptr().unsafe_store(i, src[i])
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


struct Fixture(Movable):
    """One planted batch, BINNED (DEVIATION 314's shipped path -- the
    path 2011 and 2012 both name). The bin matrix is precomputed on the
    host with the LINEAR search and checked implicitly: if it disagreed
    with the expected tally's own linear search, G0 would fail."""

    var counts: List[Int]
    var begins: List[Int]
    var n_rows: Int
    var quantiles_host: List[Float32]

    var data: DeviceBuffer[DT]
    var bins: DeviceBuffer[DType.uint8]
    var labels: DeviceBuffer[LT]
    var sample_weight: DeviceBuffer[DType.float32]
    var row_ids: DeviceBuffer[DType.int32]
    var q_arr: DeviceBuffer[DT]
    var q_nbins: DeviceBuffer[DType.int32]
    var work_items: DeviceBuffer[DType.uint8]
    var wl_fine: DeviceBuffer[DType.uint8]
    var wl_coarse: DeviceBuffer[DType.uint8]
    var n_fine: Int
    var n_coarse: Int
    var column_samples: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        self.counts = node_counts()
        self.begins = node_begins(self.counts)
        var total = 0
        for i in range(len(self.counts)):
            total += self.counts[i]
        self.n_rows = total
        var nb = bins_per_col()

        # data + bin matrix, column-major (row_stride 1, col_stride n).
        self.quantiles_host = List[Float32]()
        for c in range(N_COLS):
            for b in range(MAX_N_BINS):
                self.quantiles_host.append(quantile_at(c, b))
        var l_data = List[Float32]()
        var l_bins = List[UInt8]()
        for c in range(N_COLS):
            for r in range(total):
                var x = data_at(r, c)
                l_data.append(x)
                l_bins.append(
                    UInt8(host_lower_bound(self.quantiles_host, c, nb[c], x))
                )
        self.data = ctx.enqueue_create_buffer[DT](total * N_COLS)
        upload_f32(ctx, self.data, l_data)
        self.bins = ctx.enqueue_create_buffer[DType.uint8](total * N_COLS)
        upload_u8(ctx, self.bins, l_bins)

        var l_lab = List[Int32]()
        for r in range(total):
            l_lab.append(label_at(r))
        self.labels = ctx.enqueue_create_buffer[LT](total)
        upload_i32(ctx, self.labels, l_lab)

        self.sample_weight = ctx.enqueue_create_buffer[DType.float32](1)
        var l_sw: List[Float32] = [1.0]
        upload_f32(ctx, self.sample_weight, l_sw)

        var l_rid = List[Int32]()
        for i in range(total):
            l_rid.append(row_id_at(i, total))
        self.row_ids = ctx.enqueue_create_buffer[DType.int32](total)
        upload_i32(ctx, self.row_ids, l_rid)

        self.q_arr = ctx.enqueue_create_buffer[DT](N_COLS * MAX_N_BINS)
        upload_f32(ctx, self.q_arr, self.quantiles_host)
        var l_nb = List[Int32]()
        for c in range(N_COLS):
            l_nb.append(Int32(nb[c]))
        self.q_nbins = ctx.enqueue_create_buffer[DType.int32](N_COLS)
        upload_i32(ctx, self.q_nbins, l_nb)

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

        var fine = build_workload(self.counts, TPB)
        var coarse = build_workload(self.counts, COARSE)
        self.n_fine = len(fine)
        self.n_coarse = len(coarse)
        self.wl_fine = ctx.enqueue_create_buffer[DType.uint8](
            self.n_fine * size_of[WorkloadInfo]()
        )
        upload_structs(ctx, self.wl_fine, fine)
        self.wl_coarse = ctx.enqueue_create_buffer[DType.uint8](
            self.n_coarse * size_of[WorkloadInfo]()
        )
        upload_structs(ctx, self.wl_coarse, coarse)

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
            # DEVIATION 314's matrix -- the BINNED path is the subject.
            self.bins.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            True,
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

    def wl_ptr(
        mut self, coarse: Bool
    ) -> MutPointer[WorkloadInfo, MutUntrackedOrigin]:
        if coarse:
            return (
                self.wl_coarse.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
                .unsafe_bitcast[WorkloadInfo]()
            )
        return (
            self.wl_fine.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[WorkloadInfo]()
        )

    def cs_ptr(mut self) -> MutPointer[Int32, MutUntrackedOrigin]:
        return self.column_samples.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()


def expected_histogram(fx: Fixture) -> List[UInt32]:
    """The tally, in the kernel's layout, computed with plain loops."""
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


def run_histogram_arm[
    sabotage: Int = 0, SMEM_COPIES: Int = 1, SMEM_SLOTS: Int = 4096
](
    ctx: DeviceContext,
    mut fx: Fixture,
    coarse_table: Bool,
    use_global: Bool,
) raises -> List[UInt32]:
    """One launch against one table; returns the cells."""
    var cells = N_NODES * GRID_Y * MAX_N_BINS * N_CLASSES
    var hists = ctx.enqueue_create_buffer[DType.uint32](cells)
    var blob = DeviceArgs[HistogramArgs[ObjT]](ctx)
    hists.enqueue_fill(0)
    ctx.synchronize()
    var obj = ClassificationObjectiveFunction[DT, LT, BinT](
        Int32(N_CLASSES), Int32(1), CRITERION_GINI
    )
    var argsp = blob.upload(
        ctx, HistogramArgs[ObjT](fx.dataset(), fx.quantiles(), obj^)
    )
    launch_build_histograms_kernel[
        ObjT,
        TPB=TPB,
        SMEM_BIN_SLOTS=SMEM_SLOTS,
        sabotage=sabotage,
        SMEM_COPIES=SMEM_COPIES,
    ](
        ctx,
        hists.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[BinT](),
        MAX_N_BINS,
        fx.dataset(),
        fx.wi_ptr(),
        COL_START,
        fx.cs_ptr(),
        fx.wl_ptr(coarse_table),
        fx.n_coarse if coarse_table else fx.n_fine,
        GRID_Y,
        SharedMemoryConfig(use_global, 0),
        argsp,
    )
    ctx.synchronize()
    var got = List[UInt32]()
    with hists.map_to_host() as h:
        for i in range(cells):
            got.append(h[i])
    _ = blob^
    _ = hists^
    return got^


def compare_cells(
    name: String, got: List[UInt32], want: List[UInt32]
) raises -> Int:
    var wrong = 0
    var first = -1
    for i in range(len(want)):
        if got[i] != want[i]:
            if first < 0:
                first = i
            wrong += 1
    if wrong == 0:
        print("  arm", name, "OK:", len(want), "cells match per cell")
    else:
        print(
            "  arm", name, "FAILED:", wrong, "of", len(want),
            "cells wrong; first at", first,
            "got", got[first], "want", want[first],
        )
    return wrong


def main() raises:
    var ctx = DeviceContext()
    var fx = Fixture(ctx)
    var want = expected_histogram(fx)
    var failures = 0

    # ---- G0: baseline (fine table, shared + global) ---------------------
    var g0s = run_histogram_arm(ctx, fx, False, False)
    failures += compare_cells("G0-shared-fine", g0s, want)
    var g0g = run_histogram_arm(ctx, fx, False, True)
    failures += compare_cells("G0-global-fine", g0g, want)

    # ---- G1: DEVIATION 2011, coarse table, both arms --------------------
    var g1s = run_histogram_arm(ctx, fx, True, False)
    failures += compare_cells("G1-shared-coarse(2011)", g1s, want)
    var g1g = run_histogram_arm(ctx, fx, True, True)
    failures += compare_cells("G1-global-coarse(2011)", g1g, want)

    # ---- G2: the builder-side table (2011's whole mechanism) ------------
    var items = List[NodeWorkItem]()
    for i in range(N_NODES):
        items.append(
            NodeWorkItem(
                i, Int32(0), InstanceRange(fx.begins[i], fx.counts[i])
            )
        )
    var wl_got = List[WorkloadInfo]()
    for _ in range(fx.n_coarse + 8):
        wl_got.append(WorkloadInfo(Int32(0), Int32(0), Int32(0)))
    var n_got = update_workload_info(
        items, wl_got.unsafe_ptr(), TPB_DEFAULT * 4
    )
    var wl_want = build_workload(fx.counts, COARSE)
    var g2_bad = 0
    if n_got != len(wl_want):
        g2_bad += 1
    else:
        for i in range(n_got):
            if (
                wl_got[i].nodeid != wl_want[i].nodeid
                or wl_got[i].offset_blockid != wl_want[i].offset_blockid
                or wl_got[i].num_blocks != wl_want[i].num_blocks
            ):
                g2_bad += 1
    if g2_bad == 0:
        print(
            "  arm G2-table(2011) OK: update_workload_info(granularity=512)"
            " ==", n_got, "hand-derived entries"
        )
    else:
        print(
            "  arm G2-table(2011) FAILED:", g2_bad,
            "mismatches (n_got", n_got, "want", len(wl_want), ")"
        )
        failures += g2_bad
    _ = wl_got^

    # ---- H1: DEVIATION 2012, four copies, fits --------------------------
    var h1 = run_histogram_arm[SMEM_COPIES=4](ctx, fx, False, False)
    failures += compare_cells("H1-shared-copies4(2012)", h1, want)

    # ---- H2: DEVIATION 2012, copies do NOT fit -> p_live=1 fallback -----
    # histogram_len <= 8*3 = 24 <= 64 slots, but 24*4 = 96 > 64.
    var h2 = run_histogram_arm[SMEM_COPIES=4, SMEM_SLOTS=64](
        ctx, fx, False, False
    )
    failures += compare_cells("H2-shared-copies4-fallback(2012)", h2, want)

    # ---- H3: 2012 sabotage -- flush drops copies 1..3: UNDERCOUNT only --
    var h3 = run_histogram_arm[sabotage=4, SMEM_COPIES=4](
        ctx, fx, False, False
    )
    var moved = 0
    var overcount = 0
    for i in range(len(want)):
        if h3[i] != want[i]:
            moved += 1
            if h3[i] > want[i]:
                overcount += 1
    if moved > 0 and overcount == 0:
        print(
            "  arm H3-sabotage(2012) OK:", moved,
            "cells moved, every one an undercount (reach proven)"
        )
    else:
        print(
            "  arm H3-sabotage(2012) FAILED: moved", moved,
            "overcounts", overcount,
            "(predicted: moved > 0, overcounts == 0)"
        )
        failures += 1

    # ---- S1/S2: DEVIATION 2010's sort ----------------------------------
    comptime SORT_N = 4000
    comptime SORT_BOUND = 5000
    var keys_host = List[Int32]()
    for i in range(SORT_N):
        keys_host.append(Int32(Int(h32(UInt32(i * 977 + 1)) % UInt32(SORT_BOUND))))
    var counts_in = List[Int]()
    for _ in range(SORT_BOUND):
        counts_in.append(0)
    for i in range(SORT_N):
        counts_in[Int(keys_host[i])] += 1

    var rows = ctx.enqueue_create_buffer[DType.int32](SORT_N)
    var scratch_keys = ctx.enqueue_create_buffer[DType.uint32](SORT_N)
    var scratch_off = ctx.enqueue_create_buffer[DType.int32](SORT_N)
    var scratch_bs = ctx.enqueue_create_buffer[DType.int32](
        ceildiv(SORT_N, SORT_BLOCK)
    )

    upload_i32(ctx, rows, keys_host)
    sort_selected_rows(
        ctx, rows, SORT_N, SORT_BOUND, scratch_keys, scratch_off, scratch_bs
    )
    ctx.synchronize()
    var s1_bad = 0
    var counts_out = List[Int]()
    for _ in range(SORT_BOUND):
        counts_out.append(0)
    with rows.map_to_host() as h:
        var prev = Int32(-1)
        for i in range(SORT_N):
            if h[i] < prev:
                s1_bad += 1
            prev = h[i]
            counts_out[Int(h[i])] += 1
    for v in range(SORT_BOUND):
        if counts_in[v] != counts_out[v]:
            s1_bad += 1
    if s1_bad == 0:
        print(
            "  arm S1-sort(2010) OK: ascending and multiset-identical over",
            SORT_N, "hashed keys (", sort_passes_for(SORT_BOUND),
            "passes )"
        )
    else:
        print("  arm S1-sort(2010) FAILED:", s1_bad, "violations")
        failures += s1_bad

    # S2 -- the sabotaged sort must NOT be ascending.
    upload_i32(ctx, rows, keys_host)
    sort_selected_rows[sabotage=1](
        ctx, rows, SORT_N, SORT_BOUND, scratch_keys, scratch_off, scratch_bs
    )
    ctx.synchronize()
    var s2_inversions = 0
    with rows.map_to_host() as h:
        var prev = Int32(-1)
        for i in range(SORT_N):
            if h[i] < prev:
                s2_inversions += 1
            prev = h[i]
    if s2_inversions > 0:
        print(
            "  arm S2-sabotage(2010) OK:", s2_inversions,
            "inversions with the top two passes dropped (reach proven)"
        )
    else:
        print(
            "  arm S2-sabotage(2010) FAILED: sabotaged sort came back"
            " ascending -- the sort is not reached or the fixture does"
            " not span the dropped bits"
        )
        failures += 1

    _ = rows^
    _ = scratch_keys^
    _ = scratch_off^
    _ = scratch_bs^
    _ = fx^

    if failures != 0:
        raise Error(
            "rf_perf_candidates_check: " + String(failures) + " failures"
        )
    print("rf_perf_candidates_check: ALL ARMS GREEN")
    _ = ctx^
