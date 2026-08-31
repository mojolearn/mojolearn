"""The distances a radius query returns, recomputed from the finished CSR.

WHY THIS IS A SEPARATE PASS AND NOT FIVE STORES IN THE SEARCH KERNEL.

The random ball cover's eps query returns a CSR adjacency and no distances.
DBSCAN, its only consumer until now, never reads one. A `RadiusNeighbors`
surface has to return `(distances, indices)`, so the values have to come from
somewhere, and there are exactly two places they can come from.

The obvious one is the search kernel itself, which already holds `d2` in a
register at the moment it decides a point is inside eps. Adding a store there
costs nothing to compute. It was rejected, for the same three reasons
IDENTITY_PATHS row 61 rejected the composite sort key:

  * `neighbors/ported/neighbors/ball_cover/registers.mojo` carries the banner
    "Partial. Do not improve." Its body mirrors cuML's and the value of that
    mirror is that a reader can diff it against the upstream. Five new stores
    at five sites is exactly the kind of local improvement that makes the
    next divergence hunt cost a day.
  * The store would be paid in EVERY mode. A `comptime` cannot remove a write
    from a kernel that FAST also runs, so every DBSCAN fit in the library
    would carry `nnz * 4` extra bytes of traffic to serve a surface it does
    not call. At the shipped DBSCAN size that is not a rounding error.
  * Four entry points and two callers would change signature.

So the distances are RECOMPUTED here, from `(query row, column index)` pairs
the search already committed to. That is one extra `eps_dist_sq` per EDGE, not
per candidate pair, which is the small side of the ball cover's whole point.

WHY THE RECOMPUTED VALUE IS THE SAME BITS THE SEARCH USED.

The membership test is `eps_dist_sq(query, q * n_cols, x_reordered,
(r_start + i) * n_cols, n_cols) <= eps * eps` (`registers.mojo:300`) and the
column it emits is `r_1nn_cols[r_start + i]` (`:307`). `x_reordered` is built
by `rbc_copy_rows_kernel` (`ball_cover.mojo:239`), a plain load/store row
gather, so `x_reordered[p * n_cols + j]` holds THE SAME BITS as
`x[r_1nn_cols[p] * n_cols + j]`. Walking `(q, c = adj_ja[k])` against the
ORIGINAL `x` therefore feeds bit-identical operands to bit-identical code.
This pass calls `eps_dist_sq` itself rather than reimplementing it, so the two
cannot drift apart later.

Under FAST that argument is about the operands only. `identical_mul_add` is a
bare `a * b + c` there and contraction is codegen's choice per compilation
context, so a second kernel may in principle contract differently than the
first did. Under IDENTICAL it is a pinned `fma` and the equality is exact by
construction. FAST promises speed, not bits, so that is the contract and not a
defect; see `fast-is-not-identical`.

WHY `identical_sqrt` AND NOT `sqrt`.

scikit-learn's `radius_neighbors` returns Euclidean distances, so the squared
value has to be rooted. DEVIATION 550 records what a bare device `sqrt` costs:
NVIDIA's approximate PTX sqrt moved ten cells of `knn.out_dist` on an H100,
which is why `registers.mojo` roots its pruning bounds through
`identical_sqrt`. A returned distance is far more visible than a pruning
bound, so it takes the same call. `return_sqrt=False` hands back d^2 with no
root at all, matching `knn_search`'s policy 3.

CANONICAL BY CONSTRUCTION. Every thread writes `out_dist[p]` for the edge at
CSR position `p`. The position is read from `adj_ja`, never from a running
cursor, so this pass has no lane-width dependence of its own and adds none:
whatever order DEVIATION 551 put the columns in, the distances land beside
them. That is the same reason IDENTITY_PATHS row 61 exempts the dense arm.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, thread_idx

from mojo_only.numerics import identical_sqrt
from neighbors.ported.neighbors.ball_cover.common import eps_dist_sq


# A fixed literal, not a device query, for the reason
# `ball_cover_canonical_order.mojo` gives: nothing here depends on the block
# count, so a number that moves with the device would only make the launch
# shape another thing to pin.
comptime RBC_DIST_TPB = 256


def rbc_edge_distance_kernel(
    adj_ia: MutPointer[Int32, MutAnyOrigin],
    adj_ja: MutPointer[Int32, MutAnyOrigin],
    queries: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    out_dist: MutPointer[Float32, MutAnyOrigin],
    n_cols_in: Int32,
    want_sqrt_in: Int32,
):
    """One block per query row; threads stride over that row's edges.

    `out_dist` is indexed by the CSR position `p`, so it is written exactly
    where `adj_ja[p]` lives and no thread needs to know how many edges came
    before it in any other row.
    """
    var n_cols = Int(n_cols_in)
    var q = Int(block_idx.x)
    var start = Int(adj_ia.unsafe_load(q))
    var end = Int(adj_ia.unsafe_load(q + 1))
    var p = start + Int(thread_idx.x)
    while p < end:
        var c = Int(adj_ja.unsafe_load(p))
        # SAME ARGUMENT ORDER AS THE SEARCH (`registers.mojo:300`): query
        # first, dataset row second. The squared difference makes the order
        # arithmetically irrelevant, but the mirror is the point.
        var d2 = eps_dist_sq(queries, q * n_cols, x, c * n_cols, n_cols)
        if want_sqrt_in != Int32(0):
            out_dist.unsafe_store(p, identical_sqrt(d2))
        else:
            out_dist.unsafe_store(p, d2)
        p += RBC_DIST_TPB


def rbc_edge_distances(
    ctx: DeviceContext,
    mut adj_ia: DeviceBuffer[DType.int32],
    mut adj_ja: DeviceBuffer[DType.int32],
    mut queries: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut out_dist: DeviceBuffer[DType.float32],
    n_queries: Int,
    n_cols: Int,
    nnz: Int,
    out_capacity: Int,
    return_sqrt: Bool,
) raises:
    """Fill `out_dist[0, nnz)` with one distance per CSR edge.

    `out_dist` is caller-allocated and may be sized to `adj_ja`'s CAPACITY
    rather than to `nnz`; only the live prefix is written, which is the same
    convention `ball_cover_canonical_order.mojo` had to learn the hard way.
    `out_capacity` is passed rather than asked of the buffer because
    `DeviceBuffer` exposes no length in this Mojo, and a pass that silently
    wrote past the end would be found by a corrupted neighbour list rather
    than by an error.
    """
    if n_queries <= 0 or nnz <= 0:
        return
    if out_capacity < nnz:
        raise Error(
            "rbc_edge_distances: out_dist holds "
            + String(out_capacity)
            + " elements and the CSR has "
            + String(nnz)
            + " edges"
        )
    ctx.enqueue_function[rbc_edge_distance_kernel](
        adj_ia.unsafe_ptr(),
        adj_ja.unsafe_ptr(),
        queries.unsafe_ptr(),
        x.unsafe_ptr(),
        out_dist.unsafe_ptr(),
        Int32(n_cols),
        Int32(1) if return_sqrt else Int32(0),
        grid_dim=(n_queries, 1, 1),
        block_dim=(RBC_DIST_TPB, 1, 1),
    )
    ctx.synchronize()
