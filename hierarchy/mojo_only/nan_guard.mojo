"""The NaN refusal at the distance seam: no computed NaN reaches a stage.

NOT A PORT. cuVS hands whatever `cuvs::distance` wrote straight to the MST
(`connectivities.cuh:157-175`); a NaN distance there is compared by
`altered_weights` arithmetic that is itself NaN and the solver does what it
does. This file is what the identity claim needs instead.

======================================================================
DEVIATION BLOCK -- DEVIATION 623. A NaN IN THE DISTANCE MATRIX IS REFUSED
BY NAME BEFORE THE MST, THE CARD, OR ANY BITWISE GATE CAN SEE IT.
======================================================================

WHAT THEIRS DOES. Nothing: `pairwise_distances` (`connectivities.cuh:133-
176`) calls the distance, sets the diagonal, and returns. A NaN cell
(a non-finite input row, or two rows whose squared norms overflow Float32
-- `||x||^2 + ||y||^2 - 2 x.y` is `inf - inf` when both `x.x` and `y.y`
are `inf`) flows into `alteration()` (NaN + anything is NaN), into every
`<` (false both ways), into `atomicMin` on a double, and out through
`temp_weights` as the edge weight the dendrogram records.

WHY OURS CANNOT. IDENTITY_PATHS row 39, FACT 2 (measured 2026-08-23 on
Apple M4, NVIDIA H100, AMD MI325X): a COMPUTED NaN carries the VENDOR'S
payload (0x7fc00000 Apple, 0x7fffffff NVIDIA, 0xffc00000 AMD), so its bits
can never sit in a certified stage. `hierarchy/linkage_main.mojo` records
`linkage.dists` and `linkage.mst.weights` as raw Float32 bytes, and
`linkage_check.mojo` compares the device matrix to the host oracle bitwise.
`weight_order_key` already sends every NaN payload to ONE key, so the MST's
DECISIONS are payload-free; the raw weight that `temp_weights` copies out
(`mst_kernels.cuh:148`) is not. Rather than canonicalize (which would put a
NaN of our choosing into a dendrogram distance), the matrix is scanned
once after the self-loop transform and ANY NaN cell raises by name with
the count. `+inf` is NOT refused: its bit pattern is the same on every
vendor and its key orders it below the NaN key and above FLT_MAX.

WHERE IT SITS. `connectivities.mojo::pairwise_distances`, after
`self_loop_max_kernel`, before the function returns -- so before
`linkage_main.mojo` records `linkage.norms` / `linkage.dists`, before
`build_sorted_mst` reads a weight, before any gate copies the matrix back.
A graph handed straight to `build_sorted_mst` (rung 2's k-NN graph, a
caller's precomputed connectivity) is NOT scanned here; that caller owns
its own guard and the README says so.

THE COUNT IS VENDOR-FREE. One thread per cell tests `v != v` and adds an
integer 1 to one cell (`Atomic.fetch_add` on Int32); an integer sum is
order-free, so the count is the same whatever the grid shape or landing
order. MEASURED: `linkage_check.mojo::check_linkage_nan_distances_refused`
plants (a) two rows of 1e20 (norms overflow to +inf, `inf - inf`) and (b)
one NaN cell, and the refusal names `NaN` and the cell count in both
numeric modes; sabotage (the guard skipped) lets the MST complete and the
NaN weight reach `mst.weights` -- recorded in the README.
======================================================================
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext


comptime NAN_GUARD_TPB = 256


def count_nan_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    count: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`count[0] += 1` for every cell `i < n` with `data[i] != data[i]`.
    Integer atomic add: the total is order-free."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var v = data.unsafe_load(i)
        if v != v:
            _ = Atomic.fetch_add(count.unsafe_offset(0), Int32(1))


def count_nan_cells(
    ctx: DeviceContext, mut data: DeviceBuffer[DType.float32], n: Int
) raises -> Int:
    """The number of NaN cells among the first `n` of `data`. Synchronizes."""
    var count = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(count, Int32(0))
    var blocks = (n + NAN_GUARD_TPB - 1) // NAN_GUARD_TPB if n > 0 else 1
    ctx.enqueue_function[count_nan_kernel](
        data.unsafe_ptr(),
        count.unsafe_ptr(),
        Int32(n),
        grid_dim=(blocks, 1, 1),
        block_dim=(NAN_GUARD_TPB, 1, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=count)
    ctx.synchronize()
    var n_nan = Int(h.unsafe_ptr().unsafe_load(0))
    _ = h^
    _ = count^
    return n_nan


def refuse_nan_distances(
    ctx: DeviceContext,
    mut data: DeviceBuffer[DType.float32],
    n: Int,
    where: String,
) raises:
    """DEVIATION 623: raise by name if any of the first `n` cells is NaN."""
    var n_nan = count_nan_cells(ctx, data, n)
    if n_nan != 0:
        raise Error(
            where + ": " + String(n_nan) + " of " + String(n)
            + " distance cells are NaN (a non-finite input row, or two rows"
            " whose squared norms overflow Float32 so the expanded identity"
            " is inf - inf); refused by name (DEVIATION 623, IDENTITY_PATHS"
            " row 39): a computed NaN's payload is the vendor's and cannot"
            " sit in a recorded stage"
        )
