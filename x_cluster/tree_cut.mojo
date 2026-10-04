# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AGGLOMERATIVE TREE LABELS IN MOJO (lane apple-fast-py2mojo-cluster,
2026-10-03). `_hierarchy_impl.py` computed them in Python after the fit:
`_hc_cut` (scikit-learn's heap, then a stack walk per cluster), `_heads` (a
parent walk per leaf, then a sorted rank) and the distance-threshold count.
Here the parent table, the count and the leaf walk are `ClusterOps`
primitives (one thread per merge or leaf on the GPU column); only
scikit-learn's heap, O(n_clusters log n_clusters) on the merge ids, stays a
host loop, because the label numbering is that heap's array order.

Switch: `PY2MOJO_CLUSTER` (default on; `-D MOJOLEARN_PY2MOJO_cluster_OFF`
restores the Python path: the binding reports 0 through
`x_cluster_py2mojo` and the Python side takes its old code)."""
from std.memory import bitcast
from std.sys.compile import is_defined

from x_cluster.ops import ClusterOps

#: M3 A/B (lane/apple-fast-py2mojo-cluster 2b6f3bfd4, 2026-10-04, 1 run per
#: arm): ivf-refine taxi 1196.0 -> 1198.9 ms with -D MOJOLEARN_PY2MOJO_cluster_OFF,
#: NEUTRAL; default unchanged (on).
comptime PY2MOJO_CLUSTER = not is_defined["MOJOLEARN_PY2MOJO_cluster_OFF"]()


def f32_at_least(thr: Float64) -> Float32:
    """The smallest float32 t with `Float64(t) >= thr`, so `v >= t` in
    float32 is `Float64(v) >= thr` for every float32 v (Python compares the
    widened value with the float64 threshold)."""
    var t = Float32(thr)
    if Float64(t) < thr:
        if t == Float32(0):
            return bitcast[DType.float32](UInt32(1))
        var b = bitcast[DType.uint32](t)
        if t > Float32(0):
            return bitcast[DType.float32](b + 1)
        return bitcast[DType.float32](b - 1)
    return t


def _sift_down(mut heap: List[Int], startpos: Int, pos_in: Int):
    """CPython `heapq._siftdown`."""
    var pos = pos_in
    var newitem = heap[pos]
    while pos > startpos:
        var parentpos = (pos - 1) >> 1
        var parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _sift_up(mut heap: List[Int], pos_in: Int):
    """CPython `heapq._siftup`."""
    var pos = pos_in
    var endpos = len(heap)
    var startpos = pos
    var newitem = heap[pos]
    var childpos = 2 * pos + 1
    while childpos < endpos:
        var rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _sift_down(heap, startpos, pos)


def heap_cut_nodes(children: List[Int32], n_leaves: Int, n_merges: Int, n_clusters: Int) raises -> List[Int32]:
    """scikit-learn `_hc_cut`'s heap: the n_clusters cluster roots, in the
    heap's array order (which numbers the labels)."""
    if n_clusters > n_leaves:
        raise Error(
            "Cannot extract more clusters than samples: " + String(n_clusters)
            + " clusters were given for a tree with " + String(n_leaves) + " leaves."
        )
    var heap = List[Int]()
    var last = n_merges - 1
    heap.append(-(max(Int(children[2 * last]), Int(children[2 * last + 1])) + 1))
    for _ in range(n_clusters - 1):
        var idx = -heap[0] - n_leaves
        if idx < 0 or idx >= n_merges:
            raise Error("AgglomerativeClustering: the tree cut reached a leaf")
        var a = Int(children[2 * idx])
        var b = Int(children[2 * idx + 1])
        # heappush(nodes, -a)
        heap.append(-a)
        _sift_down(heap, 0, len(heap) - 1)
        # heappushpop(nodes, -b)
        var item = -b
        if heap[0] < item:
            heap[0] = item
            _sift_up(heap, 0)
    var out = List[Int32](capacity=len(heap))
    for v in heap:
        out.append(Int32(-v))
    return out^


def tree_labels[O: ClusterOps](
    mut ops: O, children: List[Int32], dist: List[Float32], n: Int, n_merges: Int, cut: Int,
    thr: Float64, use_thr: Bool, mut labels: List[Int32],
) raises -> Int:
    """The labels of the agglomerative tree `children` (n_merges x 2 over n
    leaves) into `labels`; returns n_clusters. cut >= 1 (a full tree):
    `_hc_cut` with n_clusters = cut, or with `use_thr` the number of merge
    distances >= thr plus one; cut == 0 (a partial tree): each leaf's root,
    numbered by ascending root id (`_heads`)."""
    var total = n + n_merges
    var cs = ops.put_i(children)
    var parent = ops.zeros_i(total)
    ops.tree_parent(cs, n, n_merges, parent)
    var rank1 = ops.zeros_i(total)
    var k = cut
    if cut <= 0:
        k = ops.tree_roots(parent, total, rank1)
    else:
        if use_thr:
            var ds = ops.put(dist)
            k = ops.count_ge(ds, n_merges, f32_at_least(thr)) + 1
        var nodes = heap_cut_nodes(children, n, n_merges, k)
        var ns = ops.put_i(nodes)
        ops.tree_scatter(ns, k, rank1)
    var ls = ops.zeros_i(n)
    ops.tree_leaf_label(parent, rank1, n, ls)
    labels = ops.get_i(ls, n)
    return k
