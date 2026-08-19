"""Exact brute-force k-nearest-neighbors, tiled.

PORT OF `cuvs/src/neighbors/detail/knn_brute_force.cuh::tiled_brute_force_knn`
at cuVS `2140532c`. Partial. Do not improve.

THE CLAIM THIS EXISTS TO TEST
-----------------------------
Brute force is a matrix multiply plus a top-k. Both are arithmetic-dense and
embarrassingly parallel, which is the shape a GPU wants and the shape a graph
walk like HNSW is not. On a machine where the competition's GPU arms do not
run at all, exact brute force on the GPU can plausibly beat APPROXIMATE k-NN
on the CPU, and that is a strictly stronger result than being faster: it
returns the RIGHT neighbors.

`ROADMAP.md` rules out HNSW permanently for the complementary reason. It is a
graph traversal with a serial dependency per hop, and FAISS ships it CPU-only.

WHAT IS THEIRS AND WHY THE STRUCTURE LOOKS LIKE k-MEANS
-------------------------------------------------------
It looks like k-means because it IS k-means' assignment step with the argmin
replaced by a top-k. Same expanded identity, same precomputed norms, same
tiling. Their own comment at `:107-109` gives the reason the norms are hoisted
out of the tile loop: "this lets us avoid calculating norms repeatedly
per-tile, and just do once for the entire input".

That shared structure is why `core/` now exists. `core/gemm.mojo` and
`core/row_norms.mojo` were written for the k-means port and moved up
unchanged the moment a second consumer appeared, which is the evidence
`PLAN.md` wanted for whether the substrate was real or was quietly
tree-shaped.

SUBSTITUTION AUDIT, 2026-08-19: NOTHING LEFT HERE
--------------------------------------------------
Every vendor-shaped call on this path has already been swapped or has no
counterpart, so this section exists to say the search was done rather than
skipped.

- `cuvs::selection::select_k` -> `nn.topk.top_k`, WIRED, behind
  `use_vendor_topk`, with the ported RAFT radix select kept reachable so the
  vendor answer is checkable. `VENDOR_LIBRARIES.md`.
- `cuvs::distance::pairwise_distance` at `:189` (cuBLAS underneath) ->
  `linalg.matmul` via `core/gemm.mojo::gemm_nt`, WIRED. N-T is the shape it
  supports and the shape every distance wants, so this one costs nothing.
- `raft::linalg::norm` at `:107-149` -> `core/row_norms.mojo`, whose block
  reduction is already `max.gpu.primitives.block.sum`.
- The L2 epilogue at `:207-217` is `raft::linalg::map_offset`, which is
  RAFT's OWN portable elementwise map and not a CUB/cuBLAS call. Rule 1
  applies, not rule 2: port their kernel, which is what
  `core/expand_distances.mojo` is. Not a substitution candidate.
- `raft::matrix::fill` at `:165-172`, their `n < k` short-fill, is not ported
  at all. That is a missing CASE, not a missing primitive; see
  `neighbors/UNPORTED.tsv`.

WHAT IS NOT PORTED
------------------
Their `DistanceEpilogue` template, their precomputed-norm fast paths, the
sparse and the non-expanded metrics, and the multi-index merge. See
`neighbors/UNPORTED.tsv`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from layout import TileTensor
from layout.tile_layout import row_major
from nn.topk import top_k

from neighbors.ported.matrix.detail.select_radix import (
    SELECT_BLOCK,
    radix_topk_one_block_kernel,
)


def compute_norms(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut a_norm: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_features: Int,
    take_sqrt: Bool,
) raises:
    """`knn_brute_force.cuh:110-146`, hoisted out of the tile loop.

    Cosine wants the L2 norm and L2 wants the SQUARED norm, which is their
    comment at `:117-118` and is the same flag `cluster/` carries.
    """
    ctx.enqueue_function[row_norm_kernel](
        a_norm.unsafe_ptr(),
        a.unsafe_ptr(),
        Int32(n_features),
        Int32(1 if take_sqrt else 0),
        grid_dim=(n_rows, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )


def tiled_brute_force_knn(
    ctx: DeviceContext,
    mut queries: DeviceBuffer[DType.float32],
    mut query_norm: DeviceBuffer[DType.float32],
    mut index: DeviceBuffer[DType.float32],
    mut index_norm: DeviceBuffer[DType.float32],
    mut dist_tile: DeviceBuffer[DType.float32],
    mut buf_val: DeviceBuffer[DType.float32],
    mut buf_idx: DeviceBuffer[DType.uint32],
    mut out_dist: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    mut out_idx32: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_index: Int,
    n_features: Int,
    k: Int,
    query_tile: Int,
    buf_len: Int,
    is_sqrt: Bool,
    use_vendor_topk: Bool = False,
) raises:
    """Tile the QUERIES, keep the whole index resident, top-k per query row.

    Their loop tiles both axes. This tiles queries only, because the index
    axis is what the top-k reduces over and splitting it would need a merge
    of partial top-k lists, which is their multi-index path and is not
    ported. So `n_index` columns of one query tile must fit `dist_tile`, and
    `query_tile` is the knob that makes that true.

    That IS a deviation from their shape, and it is the honest one to take
    first: the merge is the part with a correctness trap in it, and this port
    has no number yet.
    """
    var q = 0
    while q < n_queries:
        var rows = min(query_tile, n_queries - q)

        # z = Q_tile . I^T
        # A `create_sub_buffer` window rather than a pointer offset, because
        # MAX's matmul takes a TileTensor over a DeviceBuffer and there is no
        # offset form of that. Same bytes, no copy.
        var q_tile = queries.create_sub_buffer[DType.float32](
            q * n_features, rows * n_features
        )
        gemm_nt(ctx, dist_tile, q_tile, index, rows, n_index, n_features)

        # The epilogue k-means fuses into its reduction has to be its own
        # pass here, because the top-k needs every distance to survive.
        var cells = rows * n_index
        ctx.enqueue_function[expand_distances_kernel](
            dist_tile.unsafe_ptr(),
            query_norm.unsafe_ptr().unsafe_offset(q),
            index_norm.unsafe_ptr(),
            Int32(rows),
            Int32(n_index),
            Int32(1 if is_sqrt else 0),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )

        # THE SELECTION. Two implementations, and which one runs is a
        # parameter rather than a preference.
        #
        # cuVS calls `cuvs::selection::select_k`, a VENDOR primitive, so by
        # the standing rule the faithful port calls OURS: `nn.topk.top_k`,
        # which has a GPU form (`ctx: DeviceContext`, `target`). The ported
        # RAFT radix select stays reachable because it is the only way to
        # check the vendor call's answer against something, and because a
        # backend without a tuned top-k needs it.
        if use_vendor_topk:
            var dv = dist_tile.create_sub_buffer[DType.float32](
                0, rows * n_index
            )
            var ov = out_dist.create_sub_buffer[DType.float32](q * k, rows * k)
            var oi = out_idx32.create_sub_buffer[DType.int32](q * k, rows * k)
            top_k[largest=False, target="gpu"](
                TileTensor(dv, row_major(rows, n_index)),
                k,
                1,
                TileTensor(ov, row_major(rows, k)),
                TileTensor(oi, row_major(rows, k)),
                False,
                ctx,
            )
        else:
            ctx.enqueue_function[radix_topk_one_block_kernel](
                dist_tile.unsafe_ptr(),
                out_dist.unsafe_ptr().unsafe_offset(q * k),
                out_idx.unsafe_ptr().unsafe_offset(q * k),
                buf_val.unsafe_ptr(),
                buf_idx.unsafe_ptr(),
                Int32(n_index),
                Int32(k),
                Int32(buf_len),
                Int32(1),
                grid_dim=(rows, 1, 1),
                block_dim=(SELECT_BLOCK, 1, 1),
            )
        q += rows
    ctx.synchronize()
