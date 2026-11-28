# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fitted outputs of a UMAP fit besides the embedding (lane sc-ann-1-umap,
2026-11-28): the fuzzy simplicial set as CSR (`UMAP.graph_`, umap-learn's
`graph_`) and the curve's (a, b) the optimizer used (`a_`, `b_`).

The fit already holds the finished CSR on the host (the graph build reads it
back once into `SparseFuzzySimplicialGraph`, umap/sparse_graph.mojo; the CPU
column builds it on the host): publishing it is one memcpy per array into the
caller's arrays, no device read and no pass over the edges. The caller sizes
the arrays by `umap_graph_capacity` (the stored-edge bound below) before the
fit, since the edge count is known only after it; the door trims them to the
count `indptr[n]` names. A leaf with no device and no Python imports, so the
device and the CPU columns and the door read the same bound.

THE BOUND. The directed graph keeps at most k - 1 edges per row (self is
never stored); the fuzzy union of it and its transpose holds at most twice
that, 2 n (k - 1). A categorical target rescales those edges and the reset of
the local connectivity unions a symmetric pattern with its transpose (itself):
no new edge. A continuous target intersects with its own graph of
`target_k` neighbors, whose union pattern adds at most 2 n (target_k - 1).
"""
from std.memory import unsafe_memcpy
from umap.sparse_graph_cells import SparseFuzzySimplicialGraph

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def umap_graph_capacity(n: Int, k: Int, target_kind: Int, target_k: Int) -> Int:
    """The most edges the fitted graph can store (module docstring)."""
    var cap = 2 * n * (k - 1)
    if target_kind == 2:
        var tk = target_k if target_k > 0 else k
        cap += 2 * n * (tk - 1)
    return cap if cap > 1 else 1


@fieldwise_init
struct UmapFitOut(Copyable, Movable):
    """The caller's output addresses (0: not asked): `indptr` n + 1 int64,
    `indices` capacity int32, `data` capacity float32, `ab` 2 float32."""
    var indptr: Int
    var indices: Int
    var data: Int
    var ab: Int

    @staticmethod
    def none() -> UmapFitOut:
        return UmapFitOut(0, 0, 0, 0)


def umap_publish_graph(graph: SparseFuzzySimplicialGraph, cap: Int, dst: UmapFitOut) raises:
    """The fitted CSR into the caller's arrays: three memcpys (the offsets
    are 64-bit Int words, the columns 32-bit words below 2^31)."""
    if dst.indptr == 0:
        return
    var n = graph.n_samples
    var nnz = len(graph.indices)
    if len(graph.offsets) != n + 1 or len(graph.values) != nnz:
        raise Error("RuntimeError: UMAP fitted graph is malformed")
    if nnz > cap:
        raise Error("RuntimeError: UMAP fitted graph exceeds its output capacity")
    unsafe_memcpy(dest=_I64P(unsafe_from_address=dst.indptr),
                  src=_I64P(unsafe_from_address=Int(graph.offsets.unsafe_ptr())), count=n + 1)
    if nnz > 0:
        unsafe_memcpy(dest=_U32P(unsafe_from_address=dst.indices),
                      src=_U32P(unsafe_from_address=Int(graph.indices.unsafe_ptr())), count=nnz)
        unsafe_memcpy(dest=_F32P(unsafe_from_address=dst.data),
                      src=_F32P(unsafe_from_address=Int(graph.values.unsafe_ptr())), count=nnz)
    _ = len(graph.offsets)  # the graph's lists outlive the copies above


def umap_publish_curve(a: Float32, b: Float32, dst: UmapFitOut):
    """The optimizer's (a, b) into the caller's two float32 words."""
    if dst.ab == 0:
        return
    var p = _F32P(unsafe_from_address=dst.ab)
    p[0] = a
    p[1] = b


@always_inline
def umap_strip_row(knn_d: MutPointer[Float32, MutAnyOrigin], knn_i: MutPointer[UInt32, MutAnyOrigin],
                   indptr: MutPointer[Int64, MutAnyOrigin], dst_i: MutPointer[Int32, MutAnyOrigin],
                   dst_d: MutPointer[Float32, MutAnyOrigin], r: Int, n: Int, k: Int):
    """`mojolearn.neighbors_graph`'s distances CSR, row `r` of the self-first
    n x k graph (lane sc-leftovers-1: the one body of the device's
    `umap_knn_strip_kernel` thread and the host's strip task): indptr[r] =
    r (k - 1) for r <= n, and row r < n keeps slots 1..k-1 (self, slot 0,
    is dropped: Scanpy's `distances` holds no self edge) in their order."""
    var kk = k - 1
    if r <= n:
        indptr.unsafe_store(r, Int64(r * kk))
    if r < n:
        for j in range(kk):  # one row's k - 1 kept slots
            dst_i.unsafe_store(r * kk + j, Int32(knn_i.unsafe_load(r * k + 1 + j)))
            dst_d.unsafe_store(r * kk + j, knn_d.unsafe_load(r * k + 1 + j))
