# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S TREE-CUT AND CONNECTIVITY KERNELS (lane
apple-fast-py2mojo-cluster, 2026-10-03): the AgglomerativeClustering steps
that ran in Python after the device fit (`_hierarchy_impl._connectivity_edges`,
`_hc_cut`'s leaf walk, `_heads`, the distance-threshold count). One thread per
cell, leaf or merge; every count an atomic integer. `DeviceOps` enqueues them;
the host column runs the same steps in loops (`x_cluster/host/host_ops.mojo`).
Only the GPU binding imports this file."""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx

from x_cluster.bodies import FPtr, IPtr


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def agc_edges_mode_kernel(src: FPtr, cnt: Int32, n: Int32, mode: Int32, adj: IPtr, bad: IPtr):
    """mode 1: `src` is the dense n x n connectivity matrix (cnt = n * n),
    cell t = (t / n, t % n); mode 2: `src` is the COO rows, columns and values
    (cnt each, concatenated, the indices as exact floats). A nonzero entry
    off the diagonal is an edge, both ways (sklearn `_fix_connectivity`)."""
    var t = _tid()
    if t >= Int(cnt):
        return
    var N = Int(n)
    var r: Int
    var c: Int
    var v: Float32
    if mode == 1:
        r = t // N
        c = t - r * N
        v = src[t]
    else:
        r = Int(src[t])
        c = Int(src[Int(cnt) + t])
        v = src[2 * Int(cnt) + t]
    if v == Float32(0) or r == c:
        return
    if r < 0 or r >= N or c < 0 or c >= N:
        bad[0] = 1
        return
    adj[r * N + c] = 1
    adj[c * N + r] = 1


def tree_parent_init_kernel(total: Int32, parent: IPtr):
    var j = _tid()
    if j < Int(total):
        parent[j] = Int32(j)


def tree_parent_kernel(children: IPtr, n: Int32, m: Int32, parent: IPtr):
    """Merge t joins children[2t] and children[2t+1] into node n + t."""
    var t = _tid()
    if t < Int(m):
        parent[Int(children[2 * t])] = n + Int32(t)
        parent[Int(children[2 * t + 1])] = n + Int32(t)


def tree_root_flag_kernel(parent: IPtr, total: Int32, flags: IPtr):
    var j = _tid()
    if j < Int(total):
        flags[j] = 1 if Int(parent[j]) == j else 0


def tree_root_rank_kernel(flags: IPtr, scan: IPtr, total: Int32, rank1: IPtr):
    """rank1[j] = 1 + the number of roots below j, for a root; 0 otherwise."""
    var j = _tid()
    if j < Int(total):
        rank1[j] = scan[j] + 1 if flags[j] != 0 else 0


def tree_scatter_kernel(nodes: IPtr, c: Int32, rank1: IPtr):
    var i = _tid()
    if i < Int(c):
        rank1[Int(nodes[i])] = Int32(i + 1)


def tree_leaf_kernel(parent: IPtr, rank1: IPtr, n: Int32, labels: IPtr):
    """Leaf t walks up to its first marked ancestor (rank1 != 0); its label
    is that mark minus one (-1 if it reaches an unmarked root)."""
    var t = _tid()
    if t >= Int(n):
        return
    var v = t
    while rank1[v] == 0:
        var p = Int(parent[v])
        if p == v:
            labels[t] = -1
            return
        v = p
    labels[t] = rank1[v] - 1


def count_ge_kernel(x: FPtr, n: Int32, thr: Float32, cnt: IPtr):
    var t = _tid()
    if t < Int(n) and x[t] >= thr:
        _ = Atomic.fetch_add(cnt, Int32(1))
