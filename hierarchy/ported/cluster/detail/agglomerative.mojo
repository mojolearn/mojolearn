"""The dendrogram on the host, the flat labels on the device.

PORT OF `cuvs/cpp/src/cluster/detail/agglomerative.cuh`, cuVS `94c2819`:
`UnionFind` (`:41-80`), `build_dendrogram_host` (`:104-155`),
`write_levels_kernel` (`:157-166`), `inherit_labels` (`:179-210`),
`init_label_roots` (`:212-224`) and `extract_flattened_clusters`
(`:238-326`). Transliterated, with ONE declared departure (DEVIATION 622,
below) whose output is identical to theirs. Do not improve.

WHY THE DENDROGRAM IS DETERMINISTIC GIVEN THE SORTED MST. `build_dendrogram
_host` walks the sorted edge list in order and, per edge, merges the two
ROOTS of its endpoints and records them; no float is compared, no order is
chosen. So `children`, `out_delta`, `out_size` are a pure function of the
sorted edge list (order AND orientation: `children[2i] = find(src)`,
`children[2i+1] = find(dst)`). DEVIATION 621 makes that list a pure function
of the edge set, and DEVIATION 620 makes the edge set a pure function of the
weights; the orientation is Boruvka's (`temp_src[tid] = tid`, the vertex
that added the edge, `mst_kernels.cuh:146`), which is itself a function of
the colors and therefore of the input.

WHY THE LABELS ARE TOO. `extract_flattened_clusters` cuts at
`cut_level = (n_leaves - 1) - (n_clusters - 1)` and labels the `n_clusters`
cluster roots 0.. in DESCENDING root-index order (`:293-310`: the last
`2(n_clusters-1)` children sorted descending, the `n_clusters` smallest of
them are the roots, label `j` to the `j`-th from the top of that tail).
`inherit_labels` then walks every node at or below the cut up to its root.
Reads of `labels[...]` race with writes of the SAME value (a node is
labelled by whichever thread reaches it, with the root's label either
way), so the output does not depend on thread order. Nothing here reads
`children`'s orientation: `write_levels_kernel` maps each CHILD to its
merge row, and the label roots are a SET.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from hierarchy.ported.sparse.op.sort import merge_sort_u64_with_index


struct UnionFind(Movable):
    """`agglomerative.cuh:41-80`.

    ======================================================================
    DEVIATION BLOCK -- DEVIATION 622. `find`'s PATH COMPRESSION IS THE
    TEXTBOOK ONE; THEIRS READS `parent[-1]` AND WRITES `parent[n_indices-1]`.
    ======================================================================
    WHAT THEIRS DOES (`:56-70`):

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

    WHAT OURS DOES: find the root, then point every node on the path at
    it. Same root returned for every call, so `children`, `out_size`,
    `out_delta` and therefore the labels are IDENTICAL to theirs; what
    differs is that ours performs no out-of-bounds access. Mojo's `List`
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


def write_levels_kernel(
    children: MutPointer[Int32, MutAnyOrigin],
    parents: MutPointer[Int32, MutAnyOrigin],
    n_vertices_in: Int32,
):
    """`agglomerative.cuh:157-166`. `parents[child] = merge row`."""
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < Int(n_vertices_in):
        var level = tid // 2
        var child = children.unsafe_load(tid)
        parents.unsafe_store(Int(child), Int32(level))


def inherit_labels(
    children: MutPointer[Int32, MutAnyOrigin],
    levels: MutPointer[Int32, MutAnyOrigin],
    n_leaves_in: Int32,
    labels: MutPointer[Int32, MutAnyOrigin],
    cut_level: Int32,
    n_vertices_in: Int32,
):
    """`agglomerative.cuh:179-210`."""
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < Int(n_vertices_in):
        var node = children.unsafe_load(tid)
        var cur_level = Int32(tid // 2)
        # Any roots above the cut level should be ignored.
        # Any leaves at the cut level should already be labeled
        if cur_level > cut_level:
            return
        var cur_parent = node
        var label = labels.unsafe_load(Int(cur_parent))
        while label == Int32(-1):
            cur_parent = cur_level + n_leaves_in
            cur_level = levels.unsafe_load(Int(cur_parent))
            label = labels.unsafe_load(Int(cur_parent))
        labels.unsafe_store(Int(node), label)


def fill_labels_kernel(
    labels: MutPointer[Int32, MutAnyOrigin], value: Int32, n_in: Int32
):
    """`thrust::fill` over the labels (`:250`, `:301`)."""
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        labels.unsafe_store(tid, value)


def init_label_roots_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    roots: MutPointer[Int32, MutAnyOrigin],
    n_clusters_in: Int32,
):
    """`init_label_roots` under `thrust::for_each` (`:212-224`, `:304-310`):
    `labels[roots[j]] = j`."""
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < Int(n_clusters_in):
        labels.unsafe_store(Int(roots.unsafe_load(tid)), Int32(tid))


comptime EXTRACT_TPB = 256
"""Their `int tpb = 256` template default (`:238`)."""


def extract_flattened_clusters(
    ctx: DeviceContext,
    mut labels: DeviceBuffer[DType.int32],
    mut children: DeviceBuffer[DType.int32],
    n_clusters: Int,
    n_leaves: Int,
    tpb: Int = EXTRACT_TPB,
) raises:
    """`agglomerative.cuh:238-326`. `tpb` is their template parameter,
    exposed so the check can launch at two block sizes."""
    if n_clusters == 1:
        ctx.enqueue_function[fill_labels_kernel](
            labels.unsafe_ptr(),
            Int32(0),
            Int32(n_leaves),
            grid_dim=((n_leaves + tpb - 1) // tpb, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        return

    var n_edges = (n_leaves - 1) * 2

    # `:263-264` n_vertices = max(children) + 1, a thrust::max_element; the
    # children are on the device, and `m - 1` rows is a host read here.
    var h_children = ctx.enqueue_create_host_buffer[DType.int32](n_edges)
    ctx.synchronize()
    var v_children = children.create_sub_buffer[DType.int32](0, n_edges)
    ctx.enqueue_copy(dst_ptr=h_children.unsafe_ptr(), src_buf=v_children)
    ctx.synchronize()
    var max_child = Int32(-1)
    for i in range(n_edges):
        var c = h_children.unsafe_ptr().unsafe_load(i)
        if c > max_child:
            max_child = c
    var n_vertices = Int(max_child) + 1

    # `:266-272`
    if n_leaves <= 0:
        raise Error("hierarchy.extract_flattened_clusters: n_leaves must be positive")
    if n_vertices != (n_leaves - 1) * 2:
        raise Error(
            "hierarchy.extract_flattened_clusters: Multiple components found"
            " in MST or MST is invalid. Cannot find single-linkage solution."
            " (n_vertices=" + String(n_vertices) + ", expected "
            + String((n_leaves - 1) * 2) + ")"
        )

    var levels = ctx.enqueue_create_buffer[DType.int32](n_vertices)
    var n_blocks = (n_vertices + tpb - 1) // tpb
    ctx.enqueue_function[write_levels_kernel](
        children.unsafe_ptr(),
        levels.unsafe_ptr(),
        Int32(n_vertices),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )

    # `:279-296` Step 1: label roots = the last (n_clusters - 1) * 2 children,
    # sorted DESCENDING (thrust::sort with thrust::greater); the n_clusters
    # at the TAIL of that order are the cluster roots.
    var child_size = (n_clusters - 1) * 2
    var children_cpy_start = n_edges - child_size
    var keys = List[UInt64](capacity=child_size)
    var idx = List[Int](capacity=child_size)
    for j in range(child_size):
        var c = Int(h_children.unsafe_ptr().unsafe_load(children_cpy_start + j))
        # descending: sort on the complement
        keys.append(UInt64(0x7FFFFFFF - c))
        idx.append(j)
    merge_sort_u64_with_index(keys, idx)
    var h_roots = ctx.enqueue_create_host_buffer[DType.int32](n_clusters)
    ctx.synchronize()
    for j in range(n_clusters):
        var pos = child_size - n_clusters + j
        var c = Int32(0x7FFFFFFF - Int(keys[pos]))
        h_roots.unsafe_ptr().unsafe_store(j, c)
    var d_roots = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_copy(dst_buf=d_roots, src_ptr=h_roots.unsafe_ptr())

    # `:298-310` tmp_labels = -1; labels for the roots
    var tmp_labels = ctx.enqueue_create_buffer[DType.int32](n_vertices)
    ctx.enqueue_function[fill_labels_kernel](
        tmp_labels.unsafe_ptr(),
        Int32(-1),
        Int32(n_vertices),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[init_label_roots_kernel](
        tmp_labels.unsafe_ptr(),
        d_roots.unsafe_ptr(),
        Int32(n_clusters),
        grid_dim=((n_clusters + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )

    # `:312-321` Step 2: propagate
    var cut_level = (n_edges // 2) - (n_clusters - 1)
    ctx.enqueue_function[inherit_labels](
        children.unsafe_ptr(),
        levels.unsafe_ptr(),
        Int32(n_leaves),
        tmp_labels.unsafe_ptr(),
        Int32(cut_level),
        Int32(n_vertices),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )

    # `:323-324` copy tmp labels to actual labels
    var v_tmp = tmp_labels.create_sub_buffer[DType.int32](0, n_leaves)
    var v_labels = labels.create_sub_buffer[DType.int32](0, n_leaves)
    ctx.enqueue_copy(dst_buf=v_labels, src_buf=v_tmp)
    ctx.synchronize()
    _ = h_children^
    _ = v_children^
    _ = levels^
    _ = h_roots^
    _ = d_roots^
    _ = tmp_labels^
    _ = v_tmp^
    _ = v_labels^
