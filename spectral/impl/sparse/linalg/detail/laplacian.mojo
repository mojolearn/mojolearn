# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`raft/sparse/linalg/detail/laplacian.cuh`: `compute_graph_laplacian`
(the COO overload, `:119-234` -- the one the 26.08 cuVS path reaches, since
both `create_connectivity_graph` and the precomputed-graph `transform` hand
a COO to `create_laplacian`) and `laplacian_normalized` (`:257-282`).

THE CSR OVERLOAD (`:40-117`) IS NOT THE PATH. cuVS 25.08 converted to CSR
first (`coo_to_csr_matrix`) and reached the CSR kernel; 26.08 deleted that
conversion and the Laplacian is built on the COO. The two overloads agree on
every value but NOT on every rounding: the CSR kernel skips a self-loop in
the degree (`input_value = col_index == row ? 0 : adj_values[...]`), the COO
overload SUMS it into the degree and SUBTRACTS it back on the diagonal
(`degrees[row] - value`), two roundings where the CSR arm had none. We implementation
the COO arm because it is the 26.08 arm; `spectral/NOT_IMPLEMENTED.tsv` names the
other.

The sequence, `:130-231`:
  1. mark which rows already have a diagonal entry (`map_offset` with an
     int `atomicAdd` counter) -- integer work, done on the host here
     (DEVIATION 775's class: a pure function of the index arrays);
  2. append `(idx, idx, 0)` for every row without one, `coo_sort`
     (DEVIATION 775: host total-order sort; repeated keys refused HERE,
     by `refuse_repeated_keys`, not inside the sort -- DEVIATION 777);
  3. `degrees = thrust::reduce_by_key(rows, values)` -- DEVIATION 776 below;
  4. `D - A`: on the diagonal `degrees[row] - value`, elsewhere `-value`
     (`:220-231`), one thread per entry.

============ DEVIATION 776: reduce_by_key -> A PER-ROW ASCENDING FOLD ======
REFERENCE: `thrust::reduce_by_key` over the row-sorted values (`:212-217`), a
segmented reduction whose within-segment combination order is thrust's (a
decoupled look-back scan with warp-width tiles: vendor and launch shaped).
HERE: `degree_kernel`, one thread per row, `acc = ftz(acc + v)` seeded
`+0.0` over the row's entries in sorted (ascending column) order, the order
the COO is in after `coo_sort`. A pure function of the canonical COO, the
same on every vendor, no block shape anywhere in it. For the kNN graph the
values are exactly `0.5` and `1` and every order gives the same sum; for a
weighted precomputed graph the order is the number, and this one is fixed.
MEASURED: `check_spectral_launch_invariance` runs it at two block widths;
the hashed-weight fixture is what separates this fold from a split one.
======================================================================
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from std.math import isfinite
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_sqrt
from core.device_fold import device_exclusive_scan_total
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from spectral.checks.device_io import download_i32, upload_f32, upload_i32
from spectral.impl.sparse.coo import CooGraph
from spectral.spmv_order import IDN_LAP_DEGREE_LANES, SPMV_LANES
from spectral.impl.sparse.matrix.detail.diagonal import (
    coo_diagonal_kernel,
    coo_scale_by_diagonal_symmetric_kernel,
    coo_set_diagonal_kernel,
)
from spectral.impl.sparse.op.coo_ops import (
    coo_sort,
    refuse_repeated_keys,
    sorted_coo_to_csr,
)

#: One thread per entry / per row, the width every elementwise launch in
#: this lane defaults to. Scheduling, not numeric: nothing below folds
#: across threads. The gates vary it.
comptime LAPLACIAN_TPB = 256

#: IDENTICAL, every vendor (fam2-cluster, 2026-10-04): the Laplacian's input
#: is prepared on the device. The bounds walk, the diagonal marking, the
#: `(idx, idx, 0)` insertion, the total-order sort `(row, col, original
#: index)`, the repeated-key test and the row offsets were host loops over
#: `List`s (the host diagonal marking, `coo_sort`, `sorted_coo_to_csr`);
#: here they are one mark launch, one scan, two stable radix sorts
#: (`core/fast_radix_sort`), a gather and a binary-search offsets launch.
#: Integer keys and a stable sort give the host's permutation, so no bit
#: moves and the host column is unchanged. The host walks remain only as
#: the error path (they raise the same messages when a device flag is set).
#: lane cpu2-l9-neighbors (2026-10-04): every mode (FAST too, the same
#: bits) and no _OFF arm (owner rule: the host preparation is not a GPU
#: route), so `MOJOLEARN_IDN_SPECTRAL_LAP_DEVICE_OFF` is retired.
comptime IDN_SPECTRAL_LAP_DEVICE = True


struct DeviceCoo(Movable):
    """A ROW-SORTED COO on the device plus its `n + 1` row offsets, which
    is what the per-row kernels (the degree fold, the matvec) walk. `rows`
    is kept because the per-entry kernels (`D - A`, the diagonal ops) are
    written over entries, as theirs are."""

    var n: Int
    var nnz: Int
    var rows: DeviceBuffer[DType.int32]
    var cols: DeviceBuffer[DType.int32]
    var vals: DeviceBuffer[DType.float32]
    var indptr: DeviceBuffer[DType.int32]

    def __init__(
        out self,
        n: Int,
        nnz: Int,
        var rows: DeviceBuffer[DType.int32],
        var cols: DeviceBuffer[DType.int32],
        var vals: DeviceBuffer[DType.float32],
        var indptr: DeviceBuffer[DType.int32],
    ):
        self.n = n
        self.nnz = nnz
        self.rows = rows^
        self.cols = cols^
        self.vals = vals^
        self.indptr = indptr^


def degree_kernel(
    indptr: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    degrees: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """DEVIATION 776: the row sum, ascending over the sorted segment."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_in):
        return
    var lo = Int(indptr.unsafe_load(r))
    var hi = Int(indptr.unsafe_load(r + 1))
    var acc = Float32(0.0)
    for j in range(lo, hi):
        acc = ftz(acc + vals.unsafe_load(j))
    degrees.unsafe_store(r, acc)


#: `degree_lanes_kernel`: rows a block, and threads a block (one per lane).
comptime LAP_DEG_ROWS = 8
comptime LAP_DEG_TPB = SPMV_LANES * LAP_DEG_ROWS


def degree_lanes_kernel(
    indptr: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    degrees: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`IDN_LAP_DEGREE_LANES` (`spectral/spmv_order.mojo`): one thread per
    lane of a row (SPMV_LANES lanes, LAP_DEG_ROWS rows a block), each a
    strided flushed sum over the row's ascending entries from `+0.0`, then
    the row's lane 0 folds the lane sums in the fixed pairwise tree. A pure
    function of the row's bits: the block shape only schedules."""
    var part = stack_allocation[
        LAP_DEG_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    var slot = t // SPMV_LANES
    var lane = t % SPMV_LANES
    var r = Int(block_idx.x) * LAP_DEG_ROWS + slot
    var live = r < Int(n_in)
    var acc = Float32(0.0)
    if live:
        var hi = Int(indptr.unsafe_load(r + 1))
        var j = Int(indptr.unsafe_load(r)) + lane
        while j < hi:
            acc = ftz(acc + vals.unsafe_load(j))
            j += SPMV_LANES
    part[t] = acc
    barrier()
    if live and lane == 0:
        var base = slot * SPMV_LANES
        var w = SPMV_LANES // 2
        while w >= 1:
            for l in range(w):
                part[base + l] = ftz(part[base + l] + part[base + l + w])
            w = w // 2
        degrees.unsafe_store(r, part[base])


def d_minus_a_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    degrees: MutPointer[Float32, MutAnyOrigin],
    nnz_in: Int32,
):
    """`laplacian.cuh:220-231`: `degrees[row] - value` on the diagonal,
    `-value` off it. One subtraction, flushed; a negation moves no bits."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(nnz_in):
        return
    var r = rows.unsafe_load(idx)
    var v = vals.unsafe_load(idx)
    if r == cols.unsafe_load(idx):
        vals.unsafe_store(idx, ftz(degrees.unsafe_load(Int(r)) - v))
    else:
        vals.unsafe_store(idx, -v)


def sqrt_then_zero_to_one_kernel(
    diag: MutPointer[Float32, MutAnyOrigin], n_in: Int32
):
    """`laplacian.cuh:269-273`: `unary_op(sqrt_op)` then
    `zero_to_one_functor` (`x == 0 ? 1 : x`; `-0.0 == 0` is true, so a
    negative-zero degree also becomes `1`). Two of their launches in one of
    ours: each element is a pure function of itself either way. `sqrt`
    through `identical_sqrt` (row 10: NVIDIA's device sqrt is approximate)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var s = ftz(identical_sqrt(diag.unsafe_load(i)))
    if s == Float32(0.0):
        s = Float32(1.0)
    diag.unsafe_store(i, s)


#: The refusals name their entry from device keys (lane cpu3-neighbors,
#: 2026-10-04): `lapdev_mark_kernel` keeps the smallest out-of-range and the
#: smallest bad-value entry index, `lapdev_repeat_kernel` the smallest sorted
#: slot that repeats its predecessor's `(row, col)`. The host walks that used
#: to find those entries again (`_mark_and_insert_diagonal`,
#: `refuse_out_of_range`, `refuse_bad_values`) are gone: the first offender in
#: entry order is the smallest index, so the messages are the walks' own.
comptime LAP_NO_KEY = Int32(0x7FFFFFFF)


def _refuse_out_of_range_at(g: CooGraph, i: Int) raises:
    """`compute_graph_laplacian`'s index refusal for entry `i`, the first
    out-of-range entry (a device key)."""
    var r = Int(g.rows[i])
    var c = Int(g.cols[i])
    raise Error(
        "connectivity_graph: entry " + String(i) + " has (row, col) = ("
        + String(r) + ", " + String(c) + ") outside [0, " + String(g.n) + ")"
    )


def _refuse_bad_value_at(g: CooGraph, i: Int) raises:
    """The precomputed graph's value refusal (finite, non-negative) for
    entry `i`, the first bad value (a device key)."""
    var v = g.vals[i]
    if not isfinite(v):
        raise Error(
            "spectral: connectivity_graph has a non-finite value at entry "
            + String(i) + " -- refused by name"
        )
    raise Error(
        "spectral: connectivity_graph has a negative value at entry "
        + String(i) + " -- refused by name (sqrt of a negative degree is NaN in theirs)"
    )


def lapdev_mark_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    has_diag: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    first: MutPointer[Int32, MutAnyOrigin],
    nnz_in: Int32,
    n_in: Int32,
):
    """One thread per entry. `flags[0] = 1` on an out-of-range index,
    `flags[1] = 1` on a non-finite or negative value, `has_diag[r] = 1` on
    a diagonal entry. Every racing store writes the same word. `first[0]`
    and `first[1]` keep the smallest such entry index (an integer min,
    order-free), the entry the refusal names."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(nnz_in):
        return
    var n = Int(n_in)
    var v = vals.unsafe_load(i)
    if (bitcast[DType.uint32](v) & UInt32(0x7F800000)) == UInt32(0x7F800000) or v < Float32(0.0):
        flags.unsafe_store(1, Int32(1))
        _ = Atomic.min(first.unsafe_offset(1), Int32(i))
    var r = Int(rows.unsafe_load(i))
    var c = Int(cols.unsafe_load(i))
    if r < 0 or r >= n or c < 0 or c >= n:
        flags.unsafe_store(0, Int32(1))
        _ = Atomic.min(first, Int32(i))
        return
    if r == c:
        has_diag.unsafe_store(r, Int32(1))


def lapdev_missing_kernel(
    has_diag: MutPointer[Int32, MutAnyOrigin],
    scan: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`scan[r] = 1` for a row with no diagonal entry (the scan's input)."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_in):
        return
    scan.unsafe_store(r, Int32(1) - has_diag.unsafe_load(r))


def lapdev_total_kernel(
    scan: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`flags[2]` = the scan's total (rows lacking a diagonal entry)."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        flags.unsafe_store(2, scan.unsafe_load(Int(n_in)))


def lapdev_diag_rows_kernel(
    has_diag: MutPointer[Int32, MutAnyOrigin],
    scan: MutPointer[Int32, MutAnyOrigin],
    diag_rows: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """The appended entries' row list, ascending: `diag_rows[scan[r]] = r`
    for each row with no diagonal entry."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_in):
        return
    if has_diag.unsafe_load(r) == Int32(0):
        diag_rows.unsafe_store(Int(scan.unsafe_load(r)), Int32(r))


def lapdev_key_kernel(
    src: MutPointer[Int32, MutAnyOrigin],
    diag_rows: MutPointer[Int32, MutAnyOrigin],
    perm: MutPointer[UInt32, MutAnyOrigin],
    keys: MutPointer[UInt32, MutAnyOrigin],
    m_in: Int32,
    nnz_in: Int32,
    init_in: Int32,
):
    """The sort key of slot `i` over the input followed by the appended
    diagonal entries: `src[s]` for an input entry `s < nnz`, the row itself
    for an appended one. `init_in != 0` also seeds `perm[i] = i`; otherwise
    `s = perm[i]` (the second pass, over the first pass's order)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(m_in):
        return
    var s = i
    if init_in != Int32(0):
        perm.unsafe_store(i, UInt32(i))
    else:
        s = Int(perm.unsafe_load(i))
    var nnz = Int(nnz_in)
    var k = Int32(0)
    if s < nnz:
        k = src.unsafe_load(s)
    else:
        k = diag_rows.unsafe_load(s - nnz)
    keys.unsafe_store(i, UInt32(Int(k)))


def lapdev_gather_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    diag_rows: MutPointer[Int32, MutAnyOrigin],
    perm: MutPointer[UInt32, MutAnyOrigin],
    o_rows: MutPointer[Int32, MutAnyOrigin],
    o_cols: MutPointer[Int32, MutAnyOrigin],
    o_vals: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    nnz_in: Int32,
):
    """Slot `i` of the sorted COO: entry `perm[i]` of the input followed by
    the appended `(r, r, 0)` entries."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(m_in):
        return
    var s = Int(perm.unsafe_load(i))
    var nnz = Int(nnz_in)
    if s < nnz:
        o_rows.unsafe_store(i, rows.unsafe_load(s))
        o_cols.unsafe_store(i, cols.unsafe_load(s))
        o_vals.unsafe_store(i, vals.unsafe_load(s))
    else:
        var r = diag_rows.unsafe_load(s - nnz)
        o_rows.unsafe_store(i, r)
        o_cols.unsafe_store(i, r)
        o_vals.unsafe_store(i, Float32(0.0))


def lapdev_repeat_kernel(
    o_rows: MutPointer[Int32, MutAnyOrigin],
    o_cols: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    first: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
):
    """`flags[3] = 1` when two adjacent sorted entries share `(row, col)`
    (DEVIATION 777's refusal, decided on the device); `first[2]` keeps the
    smallest such sorted slot, the pair `refuse_repeated_keys` names."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < 1 or i >= Int(m_in):
        return
    if (
        o_rows.unsafe_load(i) == o_rows.unsafe_load(i - 1)
        and o_cols.unsafe_load(i) == o_cols.unsafe_load(i - 1)
    ):
        flags.unsafe_store(3, Int32(1))
        _ = Atomic.min(first.unsafe_offset(2), Int32(i))


def lapdev_indptr_kernel(
    o_rows: MutPointer[Int32, MutAnyOrigin],
    indptr: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
):
    """`indptr[r]` = the first sorted slot whose row is `>= r`, for `r` in
    `[0, n]` (so `indptr[n] = m`): `sorted_coo_to_csr`'s counts-then-scan
    as one binary search per row."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r > Int(n_in):
        return
    var lo = 0
    var hi = Int(m_in)
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(o_rows.unsafe_load(mid)) < r:
            lo = mid + 1
        else:
            hi = mid
    indptr.unsafe_store(r, Int32(lo))


def compute_graph_laplacian_prepared_device(
    ctx: DeviceContext, g: CooGraph, check_values: Bool, tpb: Int
) raises -> DeviceCoo:
    """`compute_graph_laplacian` with the preparation on the device
    (`IDN_SPECTRAL_LAP_DEVICE`). Two scalar readbacks, both for sizing and
    refusal: the number of appended diagonal entries with the input flags,
    then the repeated-key flag."""
    var n = g.n
    var nnz = g.nnz()
    var rows0 = upload_i32(ctx, g.rows)
    var cols0 = upload_i32(ctx, g.cols)
    var vals0 = upload_f32(ctx, g.vals)
    var has_diag = ctx.enqueue_create_buffer[DType.int32](n)
    var scan = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var flags = ctx.enqueue_create_buffer[DType.int32](4)
    var first = ctx.enqueue_create_buffer[DType.int32](3)
    ctx.enqueue_memset(has_diag, Int32(0))
    ctx.enqueue_memset(flags, Int32(0))
    ctx.enqueue_memset(first, LAP_NO_KEY)
    var gn = (n + tpb - 1) // tpb
    if nnz > 0:
        ctx.enqueue_function[lapdev_mark_kernel](
            rows0.unsafe_ptr(), cols0.unsafe_ptr(), vals0.unsafe_ptr(),
            has_diag.unsafe_ptr(), flags.unsafe_ptr(), first.unsafe_ptr(), Int32(nnz), Int32(n),
            grid_dim=((nnz + tpb - 1) // tpb, 1, 1), block_dim=(tpb, 1, 1),
        )
    ctx.enqueue_function[lapdev_missing_kernel](
        has_diag.unsafe_ptr(), scan.unsafe_ptr(), Int32(n),
        grid_dim=(gn, 1, 1), block_dim=(tpb, 1, 1),
    )
    device_exclusive_scan_total(ctx, scan, n)
    ctx.enqueue_function[lapdev_total_kernel](  # small-launch(n: the index of the one slot read): one thread copies one word, no walk
        scan.unsafe_ptr(), flags.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    var f0 = download_i32(ctx, flags, 4)
    if f0[0] != Int32(0) or (check_values and f0[1] != Int32(0)):
        var k0 = download_i32(ctx, first, 3)
        if check_values and f0[1] != Int32(0):
            _refuse_bad_value_at(g, Int(k0[1]))
        _refuse_out_of_range_at(g, Int(k0[0]))
    var n_missing = Int(f0[2])
    var m = nnz + n_missing
    var diag_rows = ctx.enqueue_create_buffer[DType.int32](n_missing if n_missing > 0 else 1)
    if n_missing > 0:
        ctx.enqueue_function[lapdev_diag_rows_kernel](
            has_diag.unsafe_ptr(), scan.unsafe_ptr(), diag_rows.unsafe_ptr(), Int32(n),
            grid_dim=(gn, 1, 1), block_dim=(tpb, 1, 1),
        )
    # m >= n >= 1: every row holds a diagonal entry after the insertion.
    var gm = (m + tpb - 1) // tpb
    var keys = ctx.enqueue_create_buffer[DType.uint32](m)
    var perm = ctx.enqueue_create_buffer[DType.uint32](m)
    var tk = ctx.enqueue_create_buffer[DType.uint32](m)
    var tv = ctx.enqueue_create_buffer[DType.uint32](m)
    var counts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(m))
    # LSD over the pair: columns first, then rows, each pass stable, so the
    # order is (row, col, original index) -- `coo_sort`'s total order.
    ctx.enqueue_function[lapdev_key_kernel](
        cols0.unsafe_ptr(), diag_rows.unsafe_ptr(), perm.unsafe_ptr(), keys.unsafe_ptr(),
        Int32(m), Int32(nnz), Int32(1),
        grid_dim=(gm, 1, 1), block_dim=(tpb, 1, 1),
    )
    fast_radix_sort_pairs_u32(ctx, m, keys, perm, tk, tv, counts)
    ctx.enqueue_function[lapdev_key_kernel](
        rows0.unsafe_ptr(), diag_rows.unsafe_ptr(), perm.unsafe_ptr(), keys.unsafe_ptr(),
        Int32(m), Int32(nnz), Int32(0),
        grid_dim=(gm, 1, 1), block_dim=(tpb, 1, 1),
    )
    fast_radix_sort_pairs_u32(ctx, m, keys, perm, tk, tv, counts)
    var rows = ctx.enqueue_create_buffer[DType.int32](m)
    var cols = ctx.enqueue_create_buffer[DType.int32](m)
    var vals = ctx.enqueue_create_buffer[DType.float32](m)
    var indptr = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.enqueue_function[lapdev_gather_kernel](
        rows0.unsafe_ptr(), cols0.unsafe_ptr(), vals0.unsafe_ptr(), diag_rows.unsafe_ptr(),
        perm.unsafe_ptr(), rows.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(),
        Int32(m), Int32(nnz),
        grid_dim=(gm, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[lapdev_repeat_kernel](
        rows.unsafe_ptr(), cols.unsafe_ptr(), flags.unsafe_ptr(), first.unsafe_ptr(), Int32(m),
        grid_dim=(gm, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[lapdev_indptr_kernel](
        rows.unsafe_ptr(), indptr.unsafe_ptr(), Int32(m), Int32(n),
        grid_dim=((n + tpb) // tpb, 1, 1), block_dim=(tpb, 1, 1),
    )
    var f1 = download_i32(ctx, flags, 4)
    var rep_r = Int32(0)
    var rep_c = Int32(0)
    if f1[3] != Int32(0):
        # the pair the host walk named: two words of the sorted COO
        var at = Int(download_i32(ctx, first, 3)[2])
        rep_r = download_i32(ctx, rows.create_sub_buffer[DType.int32](at, 1), 1)[0]
        rep_c = download_i32(ctx, cols.create_sub_buffer[DType.int32](at, 1), 1)[0]
    # the launches above held raw pointers into these
    _ = rows0^
    _ = cols0^
    _ = vals0^
    _ = has_diag^
    _ = scan^
    _ = flags^
    _ = first^
    _ = diag_rows^
    _ = keys^
    _ = perm^
    _ = tk^
    _ = tv^
    _ = counts^
    if f1[3] != Int32(0):
        raise Error(
            "connectivity_graph: repeated (row, col) pair ("
            + String(rep_r) + ", " + String(rep_c)
            + ") -- refused by name (DEVIATION 775)"
        )
    return laplacian_from_sorted_device(ctx, n, m, rows^, cols^, vals^, indptr^, tpb)


def compute_graph_laplacian(
    ctx: DeviceContext,
    g: CooGraph,
    tpb: Int = LAPLACIAN_TPB,
    check_values: Bool = False,
) raises -> DeviceCoo:
    """`compute_graph_laplacian` (COO, `:119-234`): returns `D - A` on the
    device, row-sorted, one diagonal entry per row. `check_values` adds the
    precomputed graph's value refusal (finite, non-negative), raised before
    the index refusal."""
    if g.n <= 0:
        raise Error("compute_graph_laplacian: n must be positive")
    return compute_graph_laplacian_prepared_device(ctx, g, check_values, tpb)


def laplacian_from_sorted_device(
    ctx: DeviceContext,
    n: Int,
    nnz: Int,
    var rows: DeviceBuffer[DType.int32],
    var cols: DeviceBuffer[DType.int32],
    var vals: DeviceBuffer[DType.float32],
    var indptr: DeviceBuffer[DType.int32],
    tpb: Int = LAPLACIAN_TPB,
) raises -> DeviceCoo:
    """`compute_graph_laplacian`'s device half (the degree fold and `D -
    A`) over a row-sorted COO that already holds one diagonal entry per
    row, as `compute_graph_laplacian_prepared_device` leaves it."""
    var g_n = n
    var degrees = ctx.enqueue_create_buffer[DType.float32](g_n)
    ctx.enqueue_memset(degrees, Float32(0.0))
    comptime if IDN_LAP_DEGREE_LANES:
        ctx.enqueue_function[degree_lanes_kernel](
            indptr.unsafe_ptr(),
            vals.unsafe_ptr(),
            degrees.unsafe_ptr(),
            Int32(g_n),
            grid_dim=((g_n + LAP_DEG_ROWS - 1) // LAP_DEG_ROWS, 1, 1),
            block_dim=(LAP_DEG_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[degree_kernel](
            indptr.unsafe_ptr(),
            vals.unsafe_ptr(),
            degrees.unsafe_ptr(),
            Int32(g_n),
            grid_dim=((g_n + tpb - 1) // tpb, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    ctx.enqueue_function[d_minus_a_kernel](
        rows.unsafe_ptr(),
        cols.unsafe_ptr(),
        vals.unsafe_ptr(),
        degrees.unsafe_ptr(),
        Int32(nnz),
        grid_dim=((nnz + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    _ = degrees^
    return DeviceCoo(g_n, nnz, rows^, cols^, vals^, indptr^)


def laplacian_normalized(
    ctx: DeviceContext,
    g: CooGraph,
    mut diagonal_out: DeviceBuffer[DType.float32],
    tpb: Int = LAPLACIAN_TPB,
    check_values: Bool = False,
) raises -> DeviceCoo:
    """`laplacian_normalized` (`:257-282`): `D^(-1/2) L D^(-1/2)` with the
    diagonal set to `1`, and `diagonal_out = sqrt(degree)` with zeros
    replaced by ones (the vector `compute_eigenpairs` divides the
    eigenvectors by). `diagonal_out` must hold `n` floats."""
    var lap = compute_graph_laplacian(ctx, g, tpb, check_values)
    return laplacian_normalize_device(ctx, lap^, diagonal_out, tpb)


def laplacian_normalize_device(
    ctx: DeviceContext,
    var lap: DeviceCoo,
    mut diagonal_out: DeviceBuffer[DType.float32],
    tpb: Int = LAPLACIAN_TPB,
) raises -> DeviceCoo:
    """`laplacian_normalized`'s scaling of a device Laplacian `D - A`."""
    var n = lap.n
    var nnz = lap.nnz
    ctx.enqueue_memset(diagonal_out, Float32(0.0))
    ctx.enqueue_function[coo_diagonal_kernel](
        lap.rows.unsafe_ptr(),
        lap.cols.unsafe_ptr(),
        lap.vals.unsafe_ptr(),
        diagonal_out.unsafe_ptr(),
        Int32(nnz),
        grid_dim=((nnz + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[sqrt_then_zero_to_one_kernel](
        diagonal_out.unsafe_ptr(),
        Int32(n),
        grid_dim=((n + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[coo_scale_by_diagonal_symmetric_kernel](
        lap.rows.unsafe_ptr(),
        lap.cols.unsafe_ptr(),
        lap.vals.unsafe_ptr(),
        diagonal_out.unsafe_ptr(),
        Int32(nnz),
        grid_dim=((nnz + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[coo_set_diagonal_kernel](
        lap.rows.unsafe_ptr(),
        lap.cols.unsafe_ptr(),
        lap.vals.unsafe_ptr(),
        Int32(nnz),
        Float32(1.0),
        grid_dim=((nnz + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    return lap^
