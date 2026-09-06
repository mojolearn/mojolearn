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
(`degrees[row] - value`), two roundings where the CSR arm had none. We port
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
THEIRS: `thrust::reduce_by_key` over the row-sorted values (`:212-217`), a
segmented reduction whose within-segment combination order is thrust's (a
decoupled look-back scan with warp-width tiles: vendor and launch shaped).
OURS: `degree_kernel`, one thread per row, `acc = ftz(acc + v)` seeded
`+0.0` over the row's entries in sorted (ascending column) order, the order
the COO is in after `coo_sort`. A pure function of the canonical COO, the
same on every vendor, no block shape anywhere in it. For the kNN graph the
values are exactly `0.5` and `1` and every order gives the same sum; for a
weighted precomputed graph the order is the number, and this one is fixed.
MEASURED: `check_spectral_launch_invariance` runs it at two block widths;
the hashed-weight fixture is what separates this fold from a split one.
======================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_sqrt
from spectral.checks.device_io import upload_f32, upload_i32
from spectral.impl.sparse.coo import CooGraph
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


def _mark_and_insert_diagonal(g: CooGraph) raises -> CooGraph:
    """`laplacian.cuh:130-195` on the host: rows lacking a diagonal entry
    get `(idx, idx, 0)` appended, then `coo_sort`, then DEVIATION 777's
    repeated-key refusal -- here and not inside the sort, because
    `coo_symmetrize`'s zero padding is sorted before it is compacted."""
    var n = g.n
    var marked = List[Bool]()
    for _ in range(n):
        marked.append(True)
    for i in range(g.nnz()):
        if g.rows[i] == g.cols[i]:
            marked[Int(g.rows[i])] = False
    var rows = g.rows.copy()
    var cols = g.cols.copy()
    var vals = g.vals.copy()
    for idx in range(n):
        if marked[idx]:
            rows.append(Int32(idx))
            cols.append(Int32(idx))
            vals.append(Float32(0.0))
    var sorted_g = coo_sort(CooGraph(n, rows^, cols^, vals^))
    refuse_repeated_keys(sorted_g)
    return sorted_g^


def compute_graph_laplacian(
    ctx: DeviceContext, g: CooGraph, tpb: Int = LAPLACIAN_TPB
) raises -> DeviceCoo:
    """`compute_graph_laplacian` (COO, `:119-234`): returns `D - A` on the
    device, row-sorted, one diagonal entry per row."""
    if g.n <= 0:
        raise Error("compute_graph_laplacian: n must be positive")
    for i in range(g.nnz()):
        var r = Int(g.rows[i])
        var c = Int(g.cols[i])
        if r < 0 or r >= g.n or c < 0 or c >= g.n:
            raise Error(
                "connectivity_graph: entry " + String(i) + " has (row, col) = ("
                + String(r) + ", " + String(c) + ") outside [0, " + String(g.n) + ")"
            )
    var sorted_g = _mark_and_insert_diagonal(g)
    var nnz = sorted_g.nnz()
    var indptr_h = sorted_coo_to_csr(sorted_g)
    var rows = upload_i32(ctx, sorted_g.rows)
    var cols = upload_i32(ctx, sorted_g.cols)
    var vals = upload_f32(ctx, sorted_g.vals)
    var indptr = upload_i32(ctx, indptr_h)
    var degrees = ctx.enqueue_create_buffer[DType.float32](g.n)
    ctx.enqueue_memset(degrees, Float32(0.0))
    ctx.synchronize()
    ctx.enqueue_function[degree_kernel](
        indptr.unsafe_ptr(),
        vals.unsafe_ptr(),
        degrees.unsafe_ptr(),
        Int32(g.n),
        grid_dim=((g.n + tpb - 1) // tpb, 1, 1),
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
    return DeviceCoo(g.n, nnz, rows^, cols^, vals^, indptr^)


def laplacian_normalized(
    ctx: DeviceContext,
    g: CooGraph,
    mut diagonal_out: DeviceBuffer[DType.float32],
    tpb: Int = LAPLACIAN_TPB,
) raises -> DeviceCoo:
    """`laplacian_normalized` (`:257-282`): `D^(-1/2) L D^(-1/2)` with the
    diagonal set to `1`, and `diagonal_out = sqrt(degree)` with zeros
    replaced by ones (the vector `compute_eigenpairs` divides the
    eigenvectors by). `diagonal_out` must hold `n` floats."""
    var lap = compute_graph_laplacian(ctx, g, tpb)
    var n = lap.n
    var nnz = lap.nnz
    ctx.enqueue_memset(diagonal_out, Float32(0.0))
    ctx.synchronize()
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
