"""Check for `core/block_reduce.mojo`, `core/block_scan.mojo`,
`core/scan_by_key.mojo`.

WHAT THIS CHECK IS BUILT AGAINST, and it is not a digest.

  * EVERY PLANTED VALUE IS HASHED AND SCATTERED. A scan of all ones, or a
    reduce of a constant, verifies the total and NOTHING about placement.
    This repository has the scar: the same kernel read `0 wrong of 512`
    with uniform data and `490 wrong of 512` with hashed data, and two
    earlier checks had called it correct at exactly the failing
    parameters.
  * EVERY COMPARISON IS CELL FOR CELL against a tally computed here on the
    host by a different algorithm (a sequential loop), never against a
    total. For the partition scan the comparison is on the OUTPUT
    PLACEMENT of every row, because a partition with the right counts and
    the wrong order passes a count check and fails the algorithm. The
    writer sabotage below proves that in the strongest available form: it
    leaves the written MULTISET bit-identical and only permutes it, and
    this check still catches it.
  * EVERY MECHANISM HAS A SABOTAGE and the check asserts the comparison
    MOVES. Mechanisms sabotaged: the block reduce's cross-warp fold, the
    block scan's cross-warp prefix, `pdf_to_cdf`'s chunk carry, the
    scan-by-key's segment reset, its inter-block carry, its head-flag
    guard, and the fused writer's placement. Seven.
  * SEGMENT BOUNDARIES ARE DELIBERATELY NOT BLOCK-ALIGNED. That is where
    scan-by-key implementations actually break, and it is a case cuML's
    OWN call site never produces -- their key is
    `workload_info[slot / TPB].nodeid`
    (`builder_kernels_impl.cuh:190-192`), so at their call every key
    change lands exactly on a multiple of TPB. A primitive that only
    worked there would be silently wrong for every other caller, and
    `thrust::inclusive_scan_by_key` is not restricted that way, so the
    fixture puts key changes mid-block, at the very first slot, at the
    very last slot, at an exactly block-aligned slot, and at both ends of
    a segment that spans several blocks.

No dataset, real or otherwise, is used. Fixtures are analytic and hashed.
No timing is taken; this lane does not measure time.
"""

from std.gpu import block_idx, thread_idx
from std.math import ceildiv
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace

from core.block_reduce import block_reduce_sum, block_flush_count_i32
from core.block_scan import BlockScanElement, pdf_to_cdf
from core.scan_by_key import (
    ScanByKeyElement,
    ScanByKeyOps,
    launch_inclusive_scan_by_key,
    upload_device_functor,
)


# ---------------------------------------------------------------------------
# The mixer. SplitMix64's finalizer; used only to make fixtures scattered
# and non-uniform, never as a port of anything.
# ---------------------------------------------------------------------------
def mix(x: Int) -> Int:
    var h = UInt64(x + 1) * UInt64(0x9E3779B97F4A7C15)
    h ^= h >> 30
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 27
    h *= UInt64(0x94D049BB133111EB)
    h ^= h >> 31
    return Int(h & UInt64(0x7FFFFFFF))


comptime RED_TPB = 128
comptime RED_BLOCKS = 7
comptime RED_N = 850  # NOT a multiple of RED_TPB: the last block is short.

comptime PDF_TPB = 128
comptime PDF_BINS = 300  # 2 full chunks + a 44-bin tail.

comptime SCAN_TPB = 32
comptime SCAN_N = 200


# ===========================================================================
# 1. cub::BlockReduce
# ===========================================================================
def reduce_cells_kernel[
    sabotage: Int = 0
](
    vals: MutPointer[Int64, MutAnyOrigin],
    n_in: Int32,
    per_block: MutPointer[Int64, MutAnyOrigin],
):
    """One cell per BLOCK, so the comparison sees placement, not a total."""
    var i = Int(block_idx.x) * RED_TPB + Int(thread_idx.x)
    var v = Int64(0)
    if i < Int(n_in):
        v = vals[unsafe_offset=i]
    var s = block_reduce_sum[DType.int64, RED_TPB, sabotage](v)
    if Int(thread_idx.x) == 0:
        per_block[unsafe_offset = Int(block_idx.x)] = s


def count_local_left_kernel[
    sabotage: Int = 0
](
    node_of_block: MutPointer[Int32, MutAnyOrigin],
    goes_left: MutPointer[UInt8, MutAnyOrigin],
    n_in: Int32,
    splits_local_nleft: MutPointer[Int32, MutAnyOrigin],
):
    """`countLocalLeftKernel`, `builder_kernels_impl.cuh:57-82`, with the
    dataset gather replaced by a planted 0/1 decision so the fixture is
    analytic. The reduce, the `threadIdx.x == 0 && block_count > 0` guard
    and the INDEXED atomic are theirs."""
    var nid = Int(node_of_block[unsafe_offset = Int(block_idx.x)])
    var i = Int(block_idx.x) * RED_TPB + Int(thread_idx.x)
    var thread_count = Int64(0)
    if i < Int(n_in) and goes_left[unsafe_offset=i] != UInt8(0):
        thread_count = Int64(1)
    var block_count = block_reduce_sum[DType.int64, RED_TPB, sabotage](
        thread_count
    )
    block_flush_count_i32(splits_local_nleft, nid, block_count)


def run_block_reduce(ctx: DeviceContext) raises:
    print("-- cub::BlockReduce<int64,128>::Sum")

    var h_vals = ctx.enqueue_create_host_buffer[DType.int64](RED_N)
    for i in range(RED_N):
        # Hashed and SIGNED, so a dropped warp cannot cancel out.
        h_vals[i] = Int64(mix(i) % 2001 - 1000)
    var d_vals = ctx.enqueue_create_buffer[DType.int64](RED_N)
    ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())

    # Independent host tally, per block.
    var want = List[Int64]()
    for b in range(RED_BLOCKS):
        var acc = Int64(0)
        for t in range(RED_TPB):
            var i = b * RED_TPB + t
            if i < RED_N:
                acc += Int64(mix(i) % 2001 - 1000)
        want.append(acc)

    var d_out = ctx.enqueue_create_buffer[DType.int64](RED_BLOCKS)
    var h_out = ctx.enqueue_create_host_buffer[DType.int64](RED_BLOCKS)

    comptime k = reduce_cells_kernel[0]
    d_out.enqueue_fill(0)
    ctx.enqueue_function[k](
        d_vals.unsafe_ptr(),
        Int32(RED_N),
        d_out.unsafe_ptr(),
        grid_dim=RED_BLOCKS,
        block_dim=RED_TPB,
    )
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=d_out)
    ctx.synchronize()
    # Freed-at-enqueue UAF guard (perf-lane find, 2026-08-22): `h_vals`'s
    # last use was the enqueue, and `d_vals` died at its `.unsafe_ptr()`
    # inside the kernel argument list -- the DEVICE form of the same
    # trap. Keep-alives AFTER the sync.
    _ = h_vals^
    _ = d_vals^

    var wrong = 0
    for b in range(RED_BLOCKS):
        if h_out[b] != want[b]:
            wrong += 1
            if wrong <= 3:
                print("   block", b, "got", h_out[b], "want", want[b])
    if wrong != 0:
        raise Error(
            String("block reduce wrong in ") + String(wrong) + " of "
            + String(RED_BLOCKS) + " blocks"
        )
    print("   ", RED_BLOCKS, "block cells match a host tally")

    # SABOTAGE: drop the cross-warp fold.
    comptime ks = reduce_cells_kernel[1]
    d_out.enqueue_fill(0)
    ctx.enqueue_function[ks](
        d_vals.unsafe_ptr(),
        Int32(RED_N),
        d_out.unsafe_ptr(),
        grid_dim=RED_BLOCKS,
        block_dim=RED_TPB,
    )
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=d_out)
    ctx.synchronize()
    var moved = 0
    for b in range(RED_BLOCKS):
        if h_out[b] != want[b]:
            moved += 1
    print("    sabotage(cross-warp fold):", moved, "of", RED_BLOCKS, "cells moved")
    if moved == 0:
        raise Error(
            "sabotaging the cross-warp fold did not move the comparison;"
            " the block-reduce check cannot fail and is not evidence"
        )

    # ---- countLocalLeftKernel: the 32-bit atomic flush, per node ----
    comptime N_NODES = 4
    var h_node = ctx.enqueue_create_host_buffer[DType.int32](RED_BLOCKS)
    var h_left = ctx.enqueue_create_host_buffer[DType.uint8](RED_N)
    for b in range(RED_BLOCKS):
        # Scattered assignment of blocks to nodes; several blocks share a
        # node so the atomic is genuinely contended, and node 3 gets none
        # so an untouched cell is checked too.
        h_node[b] = Int32(mix(b + 7717) % 3)
    for i in range(RED_N):
        h_left[i] = UInt8(1) if (mix(i + 991) % 5) < 2 else UInt8(0)

    var d_node = ctx.enqueue_create_buffer[DType.int32](RED_BLOCKS)
    var d_left = ctx.enqueue_create_buffer[DType.uint8](RED_N)
    ctx.enqueue_copy(dst_buf=d_node, src_ptr=h_node.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_left, src_ptr=h_left.unsafe_ptr())

    var want_n = List[Int32]()
    for _ in range(N_NODES):
        want_n.append(Int32(0))
    for b in range(RED_BLOCKS):
        var nid = Int(mix(b + 7717) % 3)
        var c = Int32(0)
        for t in range(RED_TPB):
            var i = b * RED_TPB + t
            if i < RED_N and (mix(i + 991) % 5) < 2:
                c += Int32(1)
        want_n[nid] = want_n[nid] + c

    var d_nleft = ctx.enqueue_create_buffer[DType.int32](N_NODES)
    var h_nleft = ctx.enqueue_create_host_buffer[DType.int32](N_NODES)
    d_nleft.enqueue_fill(0)
    comptime kc = count_local_left_kernel[0]
    ctx.enqueue_function[kc](
        d_node.unsafe_ptr(),
        d_left.unsafe_ptr(),
        Int32(RED_N),
        d_nleft.unsafe_ptr(),
        grid_dim=RED_BLOCKS,
        block_dim=RED_TPB,
    )
    ctx.enqueue_copy(dst_ptr=h_nleft.unsafe_ptr(), src_buf=d_nleft)
    ctx.synchronize()
    # Freed-at-enqueue UAF guard: keep-alives AFTER the sync (`d_node`
    # is the device form -- dead at its `.unsafe_ptr()` in the kernel
    # argument list).
    _ = h_node^
    _ = h_left^
    _ = d_node^
    var nwrong = 0
    for nd in range(N_NODES):
        if h_nleft[nd] != want_n[nd]:
            nwrong += 1
            print("   node", nd, "got", h_nleft[nd], "want", want_n[nd])
    if nwrong != 0:
        raise Error("countLocalLeft 32-bit flush wrong per node")
    print(
        "    countLocalLeft: ", N_NODES,
        "node cells match (one of them zero, untouched)",
    )


# ===========================================================================
# 2. cub::BlockScan / pdf_to_cdf
# ===========================================================================
@fieldwise_init
struct TestBin(BlockScanElement, TrivialRegisterPassable):
    """`RegressionBin` (`bins.cuh:97-137`) with their `double label_sum`
    carried as Float32, this device having no float64 (DEVIATION 101 in
    `ensemble/decisiontree/batched_levelalgo/bins.mojo`). The planted
    label sums are INTEGER-VALUED and far below 2^24, so a float32 sum is
    exact in any association order and the comparison can demand equality
    rather than a tolerance -- the same argument
    `gbdt/gpu_util/kernel/segmented_reduce.mojo` records."""

    var label_sum: Float32
    var count: Int64

    @staticmethod
    def zero() -> Self:
        return Self(Float32(0.0), Int64(0))

    def plus(self, rhs: Self) -> Self:
        return Self(self.label_sum + rhs.label_sum, self.count + rhs.count)


def pdf_to_cdf_kernel[
    sabotage: Int = 0
](
    histogram: MutPointer[TestBin, MutAnyOrigin],
    n_bins_in: Int32,
    total_out: MutPointer[TestBin, MutAnyOrigin],
):
    var total = pdf_to_cdf[
        TestBin, PDF_TPB, AddressSpace.GENERIC, sabotage
    ](histogram, n_bins_in)
    if Int(thread_idx.x) == 0:
        total_out[unsafe_offset=0] = total


def bin_at(i: Int) -> TestBin:
    return TestBin(Float32(mix(i + 31337) % 97), Int64(mix(i + 5) % 13))


def pdf_load_and_run[
    sab: Int
](
    ctx: DeviceContext,
    mut h_hist: HostBuffer[DType.uint8],
    mut d_hist: DeviceBuffer[DType.uint8],
    mut d_total: DeviceBuffer[DType.uint8],
    mut h_total: HostBuffer[DType.uint8],
    want_sum: List[Float32],
    want_cnt: List[Int64],
) raises -> Int:
    """Plant, run, read back, count the cells that disagree."""
    var hp = h_hist.unsafe_ptr().unsafe_bitcast[TestBin]()
    for i in range(PDF_BINS):
        hp[unsafe_offset=i] = bin_at(i)
    ctx.enqueue_copy(dst_buf=d_hist, src_ptr=h_hist.unsafe_ptr())
    comptime k = pdf_to_cdf_kernel[sab]
    ctx.enqueue_function[k](
        d_hist.unsafe_ptr().unsafe_bitcast[TestBin](),
        Int32(PDF_BINS),
        d_total.unsafe_ptr().unsafe_bitcast[TestBin](),
        grid_dim=1,
        block_dim=PDF_TPB,
    )
    ctx.enqueue_copy(dst_ptr=h_hist.unsafe_ptr(), src_buf=d_hist)
    ctx.enqueue_copy(dst_ptr=h_total.unsafe_ptr(), src_buf=d_total)
    ctx.synchronize()
    var bad = 0
    for i in range(PDF_BINS):
        var g = hp[unsafe_offset=i]
        if g.label_sum != want_sum[i] or g.count != want_cnt[i]:
            bad += 1
    return bad


def run_block_scan(ctx: DeviceContext) raises:
    print("-- cub::BlockScan<BinT,128>::InclusiveSum via pdf_to_cdf")

    var bytes = PDF_BINS * size_of[TestBin]()
    var h_hist = ctx.enqueue_create_host_buffer[DType.uint8](bytes)
    var d_hist = ctx.enqueue_create_buffer[DType.uint8](bytes)
    var d_total = ctx.enqueue_create_buffer[DType.uint8](size_of[TestBin]())
    var h_total = ctx.enqueue_create_host_buffer[DType.uint8](
        size_of[TestBin]()
    )

    var hp = h_hist.unsafe_ptr().unsafe_bitcast[TestBin]()

    # Independent host tally: the inclusive prefix, cell by cell.
    var want_sum = List[Float32]()
    var want_cnt = List[Int64]()
    var acc_s = Float32(0.0)
    var acc_c = Int64(0)
    for i in range(PDF_BINS):
        var b = bin_at(i)
        acc_s += b.label_sum
        acc_c += b.count
        want_sum.append(acc_s)
        want_cnt.append(acc_c)

    var wrong = pdf_load_and_run[0](
        ctx, h_hist, d_hist, d_total, h_total, want_sum, want_cnt
    )
    var tot = h_total.unsafe_ptr().unsafe_bitcast[TestBin]()[unsafe_offset=0]
    if wrong != 0:
        # Re-run to report the first offenders (the buffer now holds the
        # device answer).
        var shown = 0
        for i in range(PDF_BINS):
            var g = hp[unsafe_offset=i]
            if (g.label_sum != want_sum[i] or g.count != want_cnt[i]) and (
                shown < 3
            ):
                shown += 1
                print(
                    "   bin", i, "got", g.label_sum, g.count,
                    "want", want_sum[i], want_cnt[i],
                )
        raise Error(
            String("pdf_to_cdf wrong in ") + String(wrong) + " of "
            + String(PDF_BINS) + " bins"
        )
    if tot.label_sum != want_sum[PDF_BINS - 1] or tot.count != want_cnt[
        PDF_BINS - 1
    ]:
        raise Error("pdf_to_cdf total_aggregate disagrees with the host tally")
    print("   ", PDF_BINS, "bin cells + the returned total match a host tally")

    var m1 = pdf_load_and_run[1](
        ctx, h_hist, d_hist, d_total, h_total, want_sum, want_cnt
    )
    print("    sabotage(cross-warp prefix):", m1, "of", PDF_BINS, "cells moved")
    if m1 == 0:
        raise Error(
            "sabotaging the block scan's cross-warp prefix did not move the"
            " comparison; the pdf_to_cdf check is not evidence"
        )

    var m2 = pdf_load_and_run[2](
        ctx, h_hist, d_hist, d_total, h_total, want_sum, want_cnt
    )
    print("    sabotage(chunk carry):", m2, "of", PDF_BINS, "cells moved")
    if m2 == 0:
        raise Error(
            "sabotaging pdf_to_cdf's chunk carry did not move the comparison"
        )
    if m2 >= PDF_BINS:
        raise Error(
            "the chunk-carry sabotage moved the FIRST chunk too, so the"
            " sabotage is not the mechanism it claims to be"
        )


# ===========================================================================
# 3. thrust::inclusive_scan_by_key -- the fused node partition
# ===========================================================================
@fieldwise_init
struct PartitionState(ScanByKeyElement, TrivialRegisterPassable):
    """`NodeSplitPartitionState`, `builder_kernels_impl.cuh:36-38`, and
    `NodeSplitPartitionScanOp`, `:40-46`."""

    var left_count: Int64
    var valid_row: Bool
    var goes_left: Bool

    @staticmethod
    def zero() -> Self:
        return Self(Int64(0), False, False)

    def combine(self, rhs: Self) -> Self:
        # `{lhs.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left}`
        return Self(
            self.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left
        )


@fieldwise_init
struct PartitionOps[writer_sabotage: Int](
    ScanByKeyOps, TrivialRegisterPassable
):
    """Their three iterators. `key` is `node_key` (`:190-192`), `load` is
    the `partition_state` lambda (`:193-...`), `store` is
    `NodeSplitPartitionWriter::operator()` (`:97-114`).

    The dataset gather and the `dataset.value(row, colid) <= quesval`
    comparison are replaced by a planted per-slot decision so the fixture
    is analytic; the RANK ARITHMETIC and the placement are theirs,
    verbatim, because that is what this check is for.
    """

    var keys: MutPointer[Int32, MutUntrackedOrigin]
    var seg_start: MutPointer[Int32, MutUntrackedOrigin]
    var seg_pos: MutPointer[Int32, MutUntrackedOrigin]
    var seg_count: MutPointer[Int32, MutUntrackedOrigin]
    var seg_left: MutPointer[Int32, MutUntrackedOrigin]
    var goes_left: MutPointer[UInt8, MutUntrackedOrigin]
    var row_ids: MutPointer[Int32, MutUntrackedOrigin]
    var partition_row_ids: MutPointer[Int32, MutUntrackedOrigin]

    comptime Elem = PartitionState

    def key(self, slot: Int) -> Int32:
        return self.keys[unsafe_offset=slot]

    def load(self, slot: Int) -> PartitionState:
        # Their `if (range_pos >= work_item.instances.count) return
        # {0, false, false};` (`:200-202`).
        if (
            self.seg_pos[unsafe_offset=slot]
            >= self.seg_count[unsafe_offset=slot]
        ):
            return PartitionState.zero()
        var gl = self.goes_left[unsafe_offset=slot] != UInt8(0)
        # `{goes_left ? 1 : 0, true, goes_left}` (`:206`).
        return PartitionState(Int64(1) if gl else Int64(0), True, gl)

    def store(self, slot: Int, state: PartitionState):
        # `if (!state.valid_row) { return; }` (`:99`)
        if not state.valid_row:
            return
        var range_start = Int(self.seg_start[unsafe_offset=slot])
        var range_pos = Int(self.seg_pos[unsafe_offset=slot])
        var local_left_count = Int(self.seg_left[unsafe_offset=slot])
        # `rank = goes_left ? left_count - 1 : range_pos - left_count` (`:108`)
        var rank = range_pos - Int(state.left_count)
        if state.goes_left:
            rank = Int(state.left_count) - 1
        # `out_idx = range_start + (goes_left ? rank : local_left_count + rank)`
        var out_idx = range_start + local_left_count + rank
        if state.goes_left:
            out_idx = range_start + rank
        comptime if Self.writer_sabotage == 1:
            # SABOTAGE: write the row at its own slot instead of its
            # partitioned rank. Same positions, same values, PERMUTED.
            out_idx = slot
        # NOT THEIRS. A sabotaged instantiation deliberately produces a
        # wrong `left_count`, and a wrong `left_count` is an out-of-bounds
        # write -- which would corrupt another buffer instead of failing
        # the comparison. The guard is inert on the faithful path and the
        # check proves it: the destination is pre-filled with a sentinel,
        # so a dropped write leaves a cell unequal and is caught.
        if out_idx < 0 or out_idx >= SCAN_N:
            return
        self.partition_row_ids[unsafe_offset=out_idx] = self.row_ids[
            unsafe_offset=slot
        ]


comptime SENTINEL = Int32(-424242)


@fieldwise_init
struct ScanBufs(Copyable, Movable):
    """The fixture's device buffers, bundled only so the runner below can
    be a module-level function; Mojo nested closures need explicit capture
    lists and this repository forbids `@parameter` on them."""

    var keys: DeviceBuffer[DType.int32]
    var seg_start: DeviceBuffer[DType.int32]
    var seg_pos: DeviceBuffer[DType.int32]
    var seg_count: DeviceBuffer[DType.int32]
    var seg_left: DeviceBuffer[DType.int32]
    var goes_left: DeviceBuffer[DType.uint8]
    var row_ids: DeviceBuffer[DType.int32]
    var partition: DeviceBuffer[DType.int32]
    var states: DeviceBuffer[DType.uint8]
    var heads: DeviceBuffer[DType.uint8]
    var block_agg: DeviceBuffer[DType.uint8]
    var block_head: DeviceBuffer[DType.uint8]


def scan_run[
    ws: Int, sab: Int
](
    ctx: DeviceContext,
    mut bufs: ScanBufs,
    mut h_part: HostBuffer[DType.int32],
    mut h_fill: HostBuffer[DType.int32],
) raises:
    """Reset the destination to the sentinel, upload the functor, run the
    three phases, read the destination back."""
    ctx.enqueue_copy(dst_buf=bufs.partition, src_ptr=h_fill.unsafe_ptr())
    var ops = PartitionOps[ws](
        bufs.keys.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        bufs.seg_start.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        bufs.seg_pos.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        bufs.seg_count.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        bufs.seg_left.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        bufs.goes_left.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        bufs.row_ids.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        bufs.partition.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
    )
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](
        size_of[PartitionOps[ws]]()
    )
    var db = ctx.enqueue_create_buffer[DType.uint8](
        size_of[PartitionOps[ws]]()
    )
    var opsp = upload_device_functor[PartitionOps[ws]](ctx, ops, hb, db)
    launch_inclusive_scan_by_key[PartitionOps[ws], SCAN_TPB, sab](
        ctx,
        opsp,
        SCAN_N,
        bufs.states,
        bufs.heads,
        bufs.block_agg,
        bufs.block_head,
    )
    ctx.enqueue_copy(dst_ptr=h_part.unsafe_ptr(), src_buf=bufs.partition)
    ctx.synchronize()


def cells_wrong(
    mut h_part: HostBuffer[DType.int32], want: List[Int32]
) -> Int:
    var bad = 0
    for i in range(SCAN_N):
        if h_part[i] != want[i]:
            bad += 1
    return bad


def multiset_matches(
    mut h_part: HostBuffer[DType.int32], want: List[Int32]
) -> Bool:
    """Sorted-multiset equality: what a count-only or conservation check
    would see. It is asserted to still hold under the writer sabotage,
    which is how this file proves a per-cell comparison was necessary."""
    var a = List[Int]()
    var b = List[Int]()
    for i in range(SCAN_N):
        a.append(Int(h_part[i]))
        b.append(Int(want[i]))
    for i in range(1, len(a)):
        var v = a[i]
        var j = i - 1
        while j >= 0 and a[j] > v:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = v
    for i in range(1, len(b)):
        var v = b[i]
        var j = i - 1
        while j >= 0 and b[j] > v:
            b[j + 1] = b[j]
            j -= 1
        b[j + 1] = v
    for i in range(SCAN_N):
        if a[i] != b[i]:
            return False
    return True


def run_scan_by_key(ctx: DeviceContext) raises:
    print("-- thrust::inclusive_scan_by_key (fused node partition)")

    # ---- the fixture ------------------------------------------------
    # Segment lengths chosen so the boundaries land: at the very first
    # slot (length 1); mid-block; on an exactly block-aligned slot
    # (32, length 1); at the start and the end of segments longer than one
    # block (37 and 40 slots at SCAN_TPB = 32); and at the very last slot
    # (length 1).
    var seg_len = [1, 16, 15, 1, 37, 30, 20, 40, 39, 1]
    var n_segs = len(seg_len)
    var total = 0
    for s in range(n_segs):
        total += seg_len[s]
    if total != SCAN_N:
        raise Error("fixture segment lengths do not sum to SCAN_N")

    var h_keys = ctx.enqueue_create_host_buffer[DType.int32](SCAN_N)
    var h_start = ctx.enqueue_create_host_buffer[DType.int32](SCAN_N)
    var h_pos = ctx.enqueue_create_host_buffer[DType.int32](SCAN_N)
    var h_cnt = ctx.enqueue_create_host_buffer[DType.int32](SCAN_N)
    var h_left = ctx.enqueue_create_host_buffer[DType.int32](SCAN_N)
    var h_gl = ctx.enqueue_create_host_buffer[DType.uint8](SCAN_N)
    var h_rows = ctx.enqueue_create_host_buffer[DType.int32](SCAN_N)

    var starts = List[Int]()
    var counts = List[Int]()
    var lefts = List[Int]()
    var at = 0
    for s in range(n_segs):
        var ln = seg_len[s]
        # Valid rows are the HEAD of the segment and the invalid ones its
        # tail, which is the shape their `range_pos >= count` test makes.
        var cnt = ln - (mix(s + 4242) % 3)
        if cnt < 1:
            cnt = 1
        starts.append(at)
        counts.append(cnt)
        var nl = 0
        for p in range(ln):
            var slot = at + p
            var gl = (mix(slot + 88) % 7) < 3
            h_keys[slot] = Int32(1000 + s * 3)
            h_start[slot] = Int32(at)
            h_pos[slot] = Int32(p)
            h_cnt[slot] = Int32(cnt)
            h_gl[slot] = UInt8(1) if gl else UInt8(0)
            # Distinct by construction (low three digits are the slot) and
            # hashed above that, so a permutation is visible cell by cell.
            h_rows[slot] = Int32((mix(slot) % 1000) * 1000 + slot)
            if p < cnt and gl:
                nl += 1
        lefts.append(nl)
        for p in range(ln):
            h_left[at + p] = Int32(nl)
        at += ln

    # ---- the independent host tally: expected placement, per cell ----
    var want = List[Int32]()
    for _ in range(SCAN_N):
        want.append(SENTINEL)
    for s in range(n_segs):
        var st = starts[s]
        var cnt = counts[s]
        var nl = lefts[s]
        var li = 0
        var ri = 0
        for p in range(cnt):
            var slot = st + p
            var gl = (mix(slot + 88) % 7) < 3
            if gl:
                want[st + li] = h_rows[slot]
                li += 1
            else:
                want[st + nl + ri] = h_rows[slot]
                ri += 1

    # ---- device buffers ---------------------------------------------
    var d_keys = ctx.enqueue_create_buffer[DType.int32](SCAN_N)
    var d_start = ctx.enqueue_create_buffer[DType.int32](SCAN_N)
    var d_pos = ctx.enqueue_create_buffer[DType.int32](SCAN_N)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](SCAN_N)
    var d_left = ctx.enqueue_create_buffer[DType.int32](SCAN_N)
    var d_gl = ctx.enqueue_create_buffer[DType.uint8](SCAN_N)
    var d_rows = ctx.enqueue_create_buffer[DType.int32](SCAN_N)
    var d_part = ctx.enqueue_create_buffer[DType.int32](SCAN_N)
    var h_part = ctx.enqueue_create_host_buffer[DType.int32](SCAN_N)
    var h_fill = ctx.enqueue_create_host_buffer[DType.int32](SCAN_N)
    for i in range(SCAN_N):
        h_fill[i] = SENTINEL

    ctx.enqueue_copy(dst_buf=d_keys, src_ptr=h_keys.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_start, src_ptr=h_start.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_pos, src_ptr=h_pos.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_cnt, src_ptr=h_cnt.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_left, src_ptr=h_left.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_gl, src_ptr=h_gl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_rows, src_ptr=h_rows.unsafe_ptr())

    var n_blocks = ceildiv(SCAN_N, SCAN_TPB)
    var d_states = ctx.enqueue_create_buffer[DType.uint8](
        SCAN_N * size_of[PartitionState]()
    )
    var d_heads = ctx.enqueue_create_buffer[DType.uint8](SCAN_N)
    var d_bagg = ctx.enqueue_create_buffer[DType.uint8](
        n_blocks * size_of[PartitionState]()
    )
    var d_bhead = ctx.enqueue_create_buffer[DType.uint8](n_blocks)

    var bufs = ScanBufs(
        d_keys, d_start, d_pos, d_cnt, d_left, d_gl, d_rows, d_part,
        d_states, d_heads, d_bagg, d_bhead,
    )

    scan_run[0, 0](ctx, bufs, h_part, h_fill)
    # Freed-at-enqueue UAF guard (perf-lane find, 2026-08-22): the seven
    # staging buffers' last uses were the enqueues at the top; scan_run
    # synchronized, so the keep-alives land here, after the copies are
    # actually done.
    _ = h_keys^
    _ = h_start^
    _ = h_pos^
    _ = h_cnt^
    _ = h_left^
    _ = h_gl^
    _ = h_rows^
    var wrong = cells_wrong(h_part, want)
    if wrong != 0:
        var shown = 0
        for i in range(SCAN_N):
            if h_part[i] != want[i] and shown < 5:
                shown += 1
                print("   slot", i, "got", h_part[i], "want", want[i])
        raise Error(
            String("partition placement wrong in ") + String(wrong)
            + " of " + String(SCAN_N) + " cells"
        )
    print(
        "   ", SCAN_N,
        "output cells match a host tally (10 segments, boundaries at"
        " 0/1/17/32/33/70/100/120/160/199; block is 32)",
    )

    scan_run[0, 1](ctx, bufs, h_part, h_fill)
    var m1 = cells_wrong(h_part, want)
    print("    sabotage(segment reset):", m1, "of", SCAN_N, "cells moved")
    if m1 == 0:
        raise Error("sabotaging the segment reset did not move the comparison")

    scan_run[0, 2](ctx, bufs, h_part, h_fill)
    var m2 = cells_wrong(h_part, want)
    print("    sabotage(inter-block carry):", m2, "of", SCAN_N, "cells moved")
    if m2 == 0:
        raise Error(
            "sabotaging the inter-block carry did not move the comparison;"
            " the fixture has no segment spanning a block boundary"
        )

    scan_run[0, 3](ctx, bufs, h_part, h_fill)
    var m3 = cells_wrong(h_part, want)
    print("    sabotage(head-flag guard):", m3, "of", SCAN_N, "cells moved")
    if m3 == 0:
        raise Error(
            "sabotaging the head-flag guard did not move the comparison; the"
            " fixture has no key change strictly inside a block"
        )

    # The writer sabotage, and the point of the whole file: it leaves the
    # written MULTISET identical and only permutes it, so a count check or
    # a conservation check passes and this one must not.
    scan_run[1, 0](ctx, bufs, h_part, h_fill)
    var m4 = cells_wrong(h_part, want)
    var same_multiset = multiset_matches(h_part, want)
    print(
        "    sabotage(fused writer placement):", m4, "of", SCAN_N,
        "cells moved; multiset unchanged:", same_multiset,
    )
    if m4 == 0:
        raise Error(
            "sabotaging the writer's placement did not move the comparison;"
            " the fused writer is not reached or its output is not checked"
        )
    if not same_multiset:
        raise Error(
            "the writer sabotage changed WHICH rows were written, so it does"
            " not demonstrate that a count-only check would have passed"
        )


def main() raises:
    var ctx = DeviceContext()
    run_block_reduce(ctx)
    run_block_scan(ctx)
    run_scan_by_key(ctx)
    print("core primitives check OK")
