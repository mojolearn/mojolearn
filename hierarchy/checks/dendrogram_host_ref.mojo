# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`build_dendrogram_host` (`agglomerative.cuh:104-155`) and its `UnionFind`
(`:41-80`), kept as the CHECK-SIDE reference for
`hierarchy/impl/cluster/detail/dendrogram_device.mojo::build_dendrogram_device`,
which every fit (single linkage, HDBSCAN) now runs. Nothing on a fit's path
imports this file; `linkage_check.mojo` compares the device dendrogram to it.

WHY THE DENDROGRAM IS DETERMINISTIC GIVEN THE SORTED MST. The walk takes the
sorted edge list in order and, per edge, merges the two ROOTS of its
endpoints and records them; no float is compared, no order is chosen. So
`children`, `out_delta`, `out_size` are a pure function of the sorted edge
list (order AND orientation: `children[2i] = find(src)`,
`children[2i+1] = find(dst)`).
"""

from max.gpu.host import DeviceBuffer, DeviceContext


struct UnionFind(Movable):
    """`agglomerative.cuh:41-80`.

    ======================================================================
    DEVIATION BLOCK -- DEVIATION 622. `find`'s PATH COMPRESSION IS THE
    TEXTBOOK ONE; THE REFERENCE READS `parent[-1]` AND WRITES `parent[n_indices-1]`.
    ======================================================================
    WHAT THE REFERENCE DOES (`:56-70`):

        while (parent[n] != -1) n = parent[n];
        while (parent[p] != n) {
          p                                   = parent[p == -1 ? n_indices - 1 : p];
          parent[p == -1 ? n_indices - 1 : p] = n;
        }

    Trace `find(a)` with `a` a ROOT (every leaf is one until merged, so
    this is the FIRST thing `build_dendrogram_host` does on every edge):
    `n = a`; `p = a`; `parent[a] == -1 != a` so enter: `p = parent[a] = -1`;
    then `parent[n_indices - 1] = a`; the loop test reads `parent[-1]`,
    one `int` BEFORE the vector's allocation (undefined behavior; on glibc
    it is the top half of the chunk size, usually 0). If that is not `a`:
    `p = parent[n_indices - 1] = a`, `parent[a] = a`, test `parent[a] == a`,
    exit. Net effect per root-find: a self-loop `parent[a] = a` (harmless
    only because `perform_union` overwrites `parent[aa]` before anyone
    calls `find(a)` again) and a stray write into `parent[2N - 2]`, the
    final merge's slot, which nothing ever reads. The return value `n` is
    correct throughout. For a non-root start the loop compresses every
    node on the path EXCEPT the starting one.

    WHAT THIS IMPLEMENTATION DOES: find the root, then point every node on the path at
    it. Same root returned for every call, so `children`, `out_size`,
    `out_delta` and therefore the labels are IDENTICAL to the reference's; what
    differs is that this one performs no out-of-bounds access. Mojo's `List`
    would have trapped on `parent[-1]`, which is how this was found.
    MEASUREMENT: `check_linkage_union_find_matches_a_naive_one` in
    `linkage_check.mojo` runs this `find`/`perform_union` against a
    compression-free union-find over every fixture's sorted MST and
    requires identical `children` rows.
    ======================================================================
    """

    var next_label: Int
    var parent: List[Int]
    var size: List[Int]
    var n_indices: Int

    def __init__(out self, N_: Int):
        """`:50-54`: `2N - 1` slots, parents `-1`, sizes 1 for the leaves
        and 0 for the not-yet-made internal nodes, `next_label = N`."""
        self.n_indices = 2 * N_ - 1
        self.next_label = N_
        self.parent = List[Int](capacity=self.n_indices)
        self.size = List[Int](capacity=self.n_indices)
        for i in range(self.n_indices):
            self.parent.append(-1)
            self.size.append(1 if i < N_ else 0)

    def find(mut self, n_in: Int) -> Int:
        var n = n_in
        while self.parent[n] != -1:
            n = self.parent[n]
        # path compression (DEVIATION 622: the textbook one)
        var p = n_in
        while p != n and self.parent[p] != n:
            var nxt = self.parent[p]
            self.parent[p] = n
            p = nxt
        return n

    def perform_union(mut self, m: Int, n: Int):
        """`:72-79`."""
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.next_label += 1


def build_dendrogram_host(
    ctx: DeviceContext,
    mut rows: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32],
    mut data: DeviceBuffer[DType.float32],
    nnz: Int,
    mut children: DeviceBuffer[DType.int32],
    mut out_delta: DeviceBuffer[DType.float32],
    mut out_size: DeviceBuffer[DType.int32],
) raises:
    """`agglomerative.cuh:104-155`. Edges to the host, union-find, the
    three outputs back to the device."""
    var n_edges = nnz
    var mst_src_h = ctx.enqueue_create_host_buffer[DType.int32](n_edges)
    var mst_dst_h = ctx.enqueue_create_host_buffer[DType.int32](n_edges)
    var mst_weights_h = ctx.enqueue_create_host_buffer[DType.float32](n_edges)
    ctx.synchronize()
    var v_rows = rows.create_sub_buffer[DType.int32](0, n_edges)
    var v_cols = cols.create_sub_buffer[DType.int32](0, n_edges)
    var v_data = data.create_sub_buffer[DType.float32](0, n_edges)
    ctx.enqueue_copy(dst_ptr=mst_src_h.unsafe_ptr(), src_buf=v_rows)
    ctx.enqueue_copy(dst_ptr=mst_dst_h.unsafe_ptr(), src_buf=v_cols)
    ctx.enqueue_copy(dst_ptr=mst_weights_h.unsafe_ptr(), src_buf=v_data)
    ctx.synchronize()

    var children_h = ctx.enqueue_create_host_buffer[DType.int32](n_edges * 2)
    var out_size_h = ctx.enqueue_create_host_buffer[DType.int32](n_edges)
    var out_delta_h = ctx.enqueue_create_host_buffer[DType.float32](n_edges)
    ctx.synchronize()

    var U = UnionFind(nnz + 1)

    for i in range(nnz):
        var a = Int(mst_src_h.unsafe_ptr().unsafe_load(i))
        var b = Int(mst_dst_h.unsafe_ptr().unsafe_load(i))
        var delta = mst_weights_h.unsafe_ptr().unsafe_load(i)

        var aa = U.find(a)
        var bb = U.find(b)

        var children_idx = i * 2
        children_h.unsafe_ptr().unsafe_store(children_idx, Int32(aa))
        children_h.unsafe_ptr().unsafe_store(children_idx + 1, Int32(bb))
        out_delta_h.unsafe_ptr().unsafe_store(i, delta)
        out_size_h.unsafe_ptr().unsafe_store(i, Int32(U.size[aa] + U.size[bb]))

        U.perform_union(aa, bb)

    var v_children = children.create_sub_buffer[DType.int32](0, n_edges * 2)
    var v_size = out_size.create_sub_buffer[DType.int32](0, n_edges)
    var v_delta = out_delta.create_sub_buffer[DType.float32](0, n_edges)
    ctx.enqueue_copy(dst_buf=v_children, src_ptr=children_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=v_size, src_ptr=out_size_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=v_delta, src_ptr=out_delta_h.unsafe_ptr())
    ctx.synchronize()
    _ = mst_src_h^
    _ = mst_dst_h^
    _ = mst_weights_h^
    _ = children_h^
    _ = out_size_h^
    _ = out_delta_h^
    _ = v_rows^
    _ = v_cols^
    _ = v_data^
    _ = v_children^
    _ = v_size^
    _ = v_delta^
