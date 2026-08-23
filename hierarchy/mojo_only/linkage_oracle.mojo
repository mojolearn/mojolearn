"""The host oracle for single linkage, and the fixtures it is gated on.

NOT A PORT. cuVS ships one GPU backend and checks it against scikit-learn
in Python; this tree checks a port on one Apple GPU against a host
computation that is BIT-FOR-BIT predictable, which is what the identity
claim needs. Three things live here:

1. THE PINNED DISTANCES, ON THE HOST. The same arithmetic the device's
   IDENTICAL arm performs, in the same order, through the same helpers:
   `core/row_norms.row_norm_kernel`'s strided-partials-then-halving-tree
   fold at `NORM_TPB` for the norms, then
   `neighbors/mojo_only/pinned_distance_tile.mojo`'s ascending single-
   thread `identical_mul_add`/`ftz` contraction, the `-2 * dot + (n_i +
   n_j)` epilogue, the clamp, `identical_sqrt`. Under IDENTICAL the device
   matrix must equal this one byte for byte; under FAST the device arm is
   MAX's matmul plus a library block fold and no host can replicate it, so
   the same comparison is a REPORT.

2. KRUSKAL, SERIAL, on ALL `m(m-1)/2` pairs under the total order
   `(weight_order_key(w), min(u,v), max(u,v))` -- a DIFFERENT MST
   algorithm from the device's Boruvka, sharing only the order. Under a
   total order the MST is unique, so the two must return the SAME edge
   set; an oracle that ran Boruvka serially could agree with the device
   for a shared wrong reason, and Kruskal cannot.

3. THE DENDROGRAM AND LABELS, serially, from that sorted edge list: a
   compression-free union-find for the `children` rows (DEVIATION 622's
   control) and a serial transliteration of `extract_flattened_clusters`'
   cut-and-inherit for the labels; plus a Float64 reference MST
   (`sqrt(sum (x_i - x_j)^2)` in double, direct form) for tolerance sanity
   on the MST's total weight.

THE FIXTURES are hashed and non-uniform (COMMON_BRIEF rule 6), and each
exists to separate one thing:

    FIX_BLOBS      three well-separated blobs, hashed jitter: the
                   "it clusters" case, tie-free in practice
    FIX_DUPS       a 6 x 6 integer lattice in 2-D plus 12 exact duplicates
                   of lattice points: weight-0 ties and exact equal
                   distances (1, sqrt 2, 2, ...) by the hundred. DEVIATION
                   620's and 621's measurement lives here
    FIX_CHAIN      points on a line at hashed gaps: the MST is the path
                   and the dendrogram is the gap order
    FIX_HASHED     203 rows (odd, no block divides it) x 8 hashed features:
                   the launch-invariance and batch-composition fixture
    FIX_BLOBS_DUPS the blobs with a few exact duplicates: the card's
                   fixture, one input that exercises both regimes
"""

from std.math import fma, sqrt
from max.gpu.host import DeviceContext, HostBuffer

from core.row_norms import NORM_TPB
from hierarchy.mojo_only.edge_order import (
    edge_hi,
    edge_lo,
    pack_edge_key,
    unpack_edge_hi,
    unpack_edge_lo,
    weight_order_key,
)
from hierarchy.ported.cluster.detail.connectivities import FLOAT32_MAX
from hierarchy.ported.sparse.op.sort import merge_sort_u64_with_index
from mojo_only.numerics import ftz, identical_mul_add, identical_sqrt


# ======================================================================
# FIXTURES
# ======================================================================

comptime FIX_BLOBS = 0
comptime FIX_DUPS = 1
comptime FIX_CHAIN = 2
comptime FIX_HASHED = 3
comptime FIX_BLOBS_DUPS = 4
comptime FIX_COUNT = 5


def fixture_name(fix: Int) -> String:
    if fix == FIX_BLOBS:
        return String("blobs")
    if fix == FIX_DUPS:
        return String("dups_lattice")
    if fix == FIX_CHAIN:
        return String("chain")
    if fix == FIX_HASHED:
        return String("hashed203")
    return String("blobs_dups")


def fixture_n(fix: Int) -> Int:
    if fix == FIX_BLOBS:
        return 96
    if fix == FIX_DUPS:
        return 48
    if fix == FIX_CHAIN:
        return 64
    if fix == FIX_HASHED:
        return 203
    return 102


def fixture_d(fix: Int) -> Int:
    if fix == FIX_BLOBS:
        return 5
    if fix == FIX_DUPS:
        return 2
    if fix == FIX_CHAIN:
        return 3
    if fix == FIX_HASHED:
        return 8
    return 5


def fixture_n_clusters(fix: Int) -> Int:
    if fix == FIX_BLOBS or fix == FIX_BLOBS_DUPS:
        return 3
    if fix == FIX_DUPS:
        return 4
    if fix == FIX_CHAIN:
        return 5
    return 7


def _hash_unit(i: Int, f: Int, salt: Int) -> Float32:
    """A hashed value in [0, 1) that is NOT uniform across cells: the low 16
    bits of a splitmix-style mix of (i, f, salt). Distinct per cell, so a
    permutation of cells is visible."""
    var h = UInt64(i + 3) * UInt64(0x9E3779B97F4A7C15) + UInt64(
        f + 11
    ) * UInt64(0xBF58476D1CE4E5B9) + UInt64(salt + 7) * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> UInt64(32))
    return Float32(Int(h & UInt64(0xFFFF))) / Float32(65536.0)


def fixture_value(fix: Int, i: Int, f: Int) -> Float32:
    """The value at cell (i, f) of fixture `fix`, a pure function."""
    if fix == FIX_BLOBS or fix == FIX_BLOBS_DUPS:
        var n = fixture_n(fix)
        var src = i
        if fix == FIX_BLOBS_DUPS and i >= 96:
            # six exact duplicates: rows 96..101 copy rows 0, 1, 33, 0, 65, 1
            var dup_of = [0, 1, 33, 0, 65, 1]
            src = dup_of[i - 96]
        var blob = src % 3
        var center = Float32(0.0)
        if blob == 1:
            center = Float32(10.0)
        elif blob == 2:
            center = Float32(-10.0)
        var v = center + (_hash_unit(src, f, 1) - Float32(0.5)) * Float32(1.5)
        if f == 1:
            v = v + Float32(blob) * Float32(0.25)
        _ = n
        return v
    if fix == FIX_DUPS:
        # 36 lattice points (0..5 x 0..5) then 12 duplicates of lattice points
        var src = i
        if i >= 36:
            var dup_of = [0, 7, 14, 21, 28, 35, 0, 7, 5, 30, 17, 17]
            src = dup_of[i - 36]
        if f == 0:
            return Float32(src % 6)
        return Float32(src // 6)
    if fix == FIX_CHAIN:
        if f == 0:
            # cumulative hashed gaps, so the line position is a prefix sum
            var pos = Float32(0.0)
            for k in range(i):
                pos = pos + Float32(0.1) + _hash_unit(k, 0, 2) * Float32(2.0)
            return pos
        return Float32(0.0)
    # FIX_HASHED: a heavy-tailed spread so magnitudes vary by cell
    var u = _hash_unit(i, f, 3)
    var sign = Float32(1.0) if (i + f) % 3 != 0 else Float32(-1.0)
    var scale = Float32(1.0) + Float32(f) * Float32(0.37)
    return sign * (u * u * Float32(4.0) + u * Float32(0.25)) * scale


def build_fixture(ctx: DeviceContext, fix: Int) raises -> HostBuffer[DType.float32]:
    var n = fixture_n(fix)
    var d = fixture_d(fix)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    ctx.synchronize()
    for i in range(n):
        for f in range(d):
            host.unsafe_ptr().unsafe_store(i * d + f, fixture_value(fix, i, f))
    return host^


def fixture_as_list(fix: Int) -> List[Float32]:
    var n = fixture_n(fix)
    var d = fixture_d(fix)
    var out = List[Float32](capacity=n * d)
    for i in range(n):
        for f in range(d):
            out.append(fixture_value(fix, i, f))
    return out^


# ======================================================================
# 1. THE PINNED DISTANCES, ON THE HOST
# ======================================================================


def host_row_norms_pinned(x: List[Float32], m: Int, d: Int) -> List[Float32]:
    """`core/row_norms.row_norm_kernel` with `take_sqrt = 0`, replicated:
    thread `t` folds columns `t, t + NORM_TPB, ...` ascending through
    `ftz(identical_mul_add(v, v, acc))`, then `pinned_block_sum[NORM_TPB]`'s
    halving tree over the NORM_TPB partials (`red[t] += red[t + step]`,
    `step = NORM_TPB/2 .. 1`), then `ftz` of the total. Under FAST the
    device fold is the library's and this replica does not describe it."""
    var out = List[Float32](capacity=m)
    var red = List[Float32](capacity=NORM_TPB)
    for _ in range(NORM_TPB):
        red.append(Float32(0.0))
    for row in range(m):
        for t in range(NORM_TPB):
            var acc = Float32(0.0)
            var col = t
            while col < d:
                var v = ftz(x[row * d + col])
                acc = ftz(identical_mul_add(v, v, acc))
                col += NORM_TPB
            red[t] = acc
        var step = NORM_TPB // 2
        while step > 0:
            for t in range(step):
                red[t] = red[t] + red[t + step]
            step //= 2
        out.append(ftz(red[0]))
    return out^


def host_pinned_distance(
    x: List[Float32], norms: List[Float32], d: Int, i: Int, j: Int, is_sqrt: Bool
) -> Float32:
    """One cell of `pinned_distance_tile_kernel`, on the host."""
    var acc = Float32(0.0)
    for f in range(d):
        var qv = ftz(x[i * d + f])
        var yv = ftz(x[j * d + f])
        acc = ftz(identical_mul_add(qv, yv, acc))
    var dist = ftz(
        identical_mul_add(
            Float32(-2.0), acc, ftz(ftz(norms[i]) + ftz(norms[j]))
        )
    )
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt:
        dist = ftz(identical_sqrt(dist))
    return dist


def host_pinned_distance_matrix(
    x: List[Float32], m: Int, d: Int, is_sqrt: Bool
) -> List[Float32]:
    """The full `m x m` matrix the device arm writes, self-loops FLT_MAX
    (`connectivities.cuh:162-175`)."""
    var norms = host_row_norms_pinned(x, m, d)
    var out = List[Float32](capacity=m * m)
    for i in range(m):
        for j in range(m):
            if i == j:
                out.append(FLOAT32_MAX)
            else:
                out.append(host_pinned_distance(x, norms, d, i, j, is_sqrt))
    return out^


# ======================================================================
# 2. KRUSKAL
# ======================================================================


struct NaiveUnionFind(Movable):
    """No path compression, no union by rank: `find` walks parents until
    `-1`. Deliberately the SLOWEST correct union-find, so it shares no
    line with `agglomerative.mojo::UnionFind` (DEVIATION 622's control)."""

    var parent: List[Int]
    var size: List[Int]
    var next_label: Int

    def __init__(out self, n_leaves: Int):
        self.parent = List[Int](capacity=2 * n_leaves - 1)
        self.size = List[Int](capacity=2 * n_leaves - 1)
        for i in range(2 * n_leaves - 1):
            self.parent.append(-1)
            self.size.append(1 if i < n_leaves else 0)
        self.next_label = n_leaves

    def find(self, n_in: Int) -> Int:
        var n = n_in
        while self.parent[n] != -1:
            n = self.parent[n]
        return n

    def union(mut self, a: Int, b: Int):
        self.size[self.next_label] = self.size[a] + self.size[b]
        self.parent[a] = self.next_label
        self.parent[b] = self.next_label
        self.next_label += 1


@fieldwise_init
struct OracleLinkage(Movable):
    """Everything the oracle computes for one fixture."""

    var m: Int
    var dists: List[Float32]
    """`m x m`, the pinned matrix, FLT_MAX on the diagonal."""
    var mst_lo: List[Int32]
    var mst_hi: List[Int32]
    var mst_w: List[Float32]
    """The MST in the total order, `m - 1` edges, `lo < hi`."""
    var children: List[Int32]
    """`(m - 1) * 2`, rows `(find(lo), find(hi))` in sorted-edge order."""
    var labels: List[Int32]
    """`m`, the serial `extract_flattened_clusters`."""
    var partition: List[Int32]
    """`m`, cluster id = the root's smallest leaf, from a cut that does not
    know about label numbering (an independent partition)."""
    var total_weight_f64_of_f32_mst: Float64
    var total_weight_f64_reference: Float64
    """Sum of the MST weights (Float32 MST summed in double) against the
    Float64 reference MST's sum."""


def host_kruskal(
    dists: List[Float32], m: Int
) -> Tuple[List[Int32], List[Int32], List[Float32]]:
    """The MST of the complete graph on `dists` under `(weight key, lo, hi)`,
    ascending. Returns the `m - 1` edges IN THAT ORDER."""
    var n_pairs = m * (m - 1) // 2
    var keys = List[UInt64](capacity=n_pairs)
    var idx = List[Int](capacity=n_pairs)
    var k = 0
    for i in range(m):
        for j in range(i + 1, m):
            keys.append(
                pack_edge_key(weight_order_key(dists[i * m + j]), Int32(i), Int32(j))
            )
            idx.append(k)
            k += 1
    merge_sort_u64_with_index(keys, idx)
    var uf = NaiveUnionFind(m)
    var lo = List[Int32](capacity=m - 1)
    var hi = List[Int32](capacity=m - 1)
    var w = List[Float32](capacity=m - 1)
    for t in range(n_pairs):
        var u = Int(unpack_edge_lo(keys[t]))
        var v = Int(unpack_edge_hi(keys[t]))
        var ru = uf.find(u)
        var rv = uf.find(v)
        if ru != rv:
            uf.union(ru, rv)
            lo.append(Int32(u))
            hi.append(Int32(v))
            w.append(dists[u * m + v])
            if len(lo) == m - 1:
                break
    return (lo^, hi^, w^)


def merge_sort_f64_with_index(mut keys: List[Float64], mut idx: List[Int]):
    """Stable bottom-up merge sort on Float64 keys carrying `idx`. Stable,
    so equal distances keep generation order, which is (i, j) ascending:
    the same tie-break the Float32 oracle uses."""
    var n = len(keys)
    if n < 2:
        return
    var tk = List[Float64](capacity=n)
    var ti = List[Int](capacity=n)
    for _ in range(n):
        tk.append(Float64(0.0))
        ti.append(0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if keys[j] < keys[i]:
                    tk[k] = keys[j]
                    ti[k] = idx[j]
                    j += 1
                else:
                    tk[k] = keys[i]
                    ti[k] = idx[i]
                    i += 1
                k += 1
            while i < mid:
                tk[k] = keys[i]
                ti[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                tk[k] = keys[j]
                ti[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(n):
            keys[t] = tk[t]
            idx[t] = ti[t]
        width *= 2


def host_kruskal_f64(x: List[Float32], m: Int, d: Int) -> Float64:
    """The Float64 reference: direct-form distances in double,
    `sqrt(sum (x_i - x_j)^2)`, sorted ascending with ties in (i, j) order,
    Kruskal, the MST's total weight."""
    var n_pairs = m * (m - 1) // 2
    var keys = List[Float64](capacity=n_pairs)
    var idx = List[Int](capacity=n_pairs)
    var pi = List[Int](capacity=n_pairs)
    var pj = List[Int](capacity=n_pairs)
    var k = 0
    for i in range(m):
        for j in range(i + 1, m):
            var s = Float64(0.0)
            for f in range(d):
                var diff = Float64(x[i * d + f]) - Float64(x[j * d + f])
                s = s + diff * diff
            keys.append(sqrt(s))
            idx.append(k)
            pi.append(i)
            pj.append(j)
            k += 1
    merge_sort_f64_with_index(keys, idx)
    var uf = NaiveUnionFind(m)
    var total = Float64(0.0)
    var taken = 0
    for t in range(n_pairs):
        var u = pi[idx[t]]
        var v = pj[idx[t]]
        var ru = uf.find(u)
        var rv = uf.find(v)
        if ru != rv:
            uf.union(ru, rv)
            total = total + keys[t]
            taken += 1
            if taken == m - 1:
                break
    return total


# ======================================================================
# 3. THE DENDROGRAM AND THE LABELS, SERIALLY
# ======================================================================


def host_dendrogram(
    lo: List[Int32], hi: List[Int32], m: Int
) -> List[Int32]:
    """`build_dendrogram_host`'s rows over a NaiveUnionFind."""
    var uf = NaiveUnionFind(m)
    var children = List[Int32](capacity=(m - 1) * 2)
    for i in range(m - 1):
        var aa = uf.find(Int(lo[i]))
        var bb = uf.find(Int(hi[i]))
        children.append(Int32(aa))
        children.append(Int32(bb))
        uf.union(aa, bb)
    return children^


def host_extract_flattened_clusters(
    children: List[Int32], n_clusters: Int, n_leaves: Int
) -> List[Int32]:
    """`extract_flattened_clusters` (`agglomerative.cuh:238-326`), serial:
    levels, the descending-sorted tail of children, the `n_clusters`
    smallest as roots labelled by position, then every child at or below
    the cut walks up to a labelled ancestor."""
    var labels = List[Int32](capacity=n_leaves)
    if n_clusters == 1:
        for _ in range(n_leaves):
            labels.append(Int32(0))
        return labels^
    var n_edges = (n_leaves - 1) * 2
    var n_vertices = 0
    for i in range(n_edges):
        if Int(children[i]) + 1 > n_vertices:
            n_vertices = Int(children[i]) + 1
    var levels = List[Int32](capacity=n_vertices)
    for _ in range(n_vertices):
        levels.append(Int32(0))
    for tid in range(n_vertices):
        levels[Int(children[tid])] = Int32(tid // 2)
    var child_size = (n_clusters - 1) * 2
    var start = n_edges - child_size
    var keys = List[UInt64](capacity=child_size)
    var idx = List[Int](capacity=child_size)
    for j in range(child_size):
        keys.append(UInt64(0x7FFFFFFF - Int(children[start + j])))
        idx.append(j)
    merge_sort_u64_with_index(keys, idx)
    var tmp = List[Int32](capacity=n_vertices)
    for _ in range(n_vertices):
        tmp.append(Int32(-1))
    for j in range(n_clusters):
        var root = 0x7FFFFFFF - Int(keys[child_size - n_clusters + j])
        tmp[root] = Int32(j)
    var cut_level = (n_edges // 2) - (n_clusters - 1)
    for tid in range(n_vertices):
        var node = Int(children[tid])
        var cur_level = tid // 2
        if cur_level > cut_level:
            continue
        var cur_parent = node
        var label = tmp[cur_parent]
        while label == Int32(-1):
            cur_parent = cur_level + n_leaves
            cur_level = Int(levels[cur_parent])
            label = tmp[cur_parent]
        tmp[node] = label
    for i in range(n_leaves):
        labels.append(tmp[i])
    return labels^


def host_partition(
    lo: List[Int32], hi: List[Int32], m: Int, n_clusters: Int
) -> List[Int32]:
    """The partition after the first `m - n_clusters` merges of the sorted
    MST, each point labelled by the SMALLEST leaf index in its cluster. No
    dendrogram, no numbering convention: the independent control for the
    labels gate (two labelings agree iff they induce the same partition)."""
    var uf = NaiveUnionFind(m)
    for i in range(m - n_clusters):
        var aa = uf.find(Int(lo[i]))
        var bb = uf.find(Int(hi[i]))
        uf.union(aa, bb)
    var root_min = List[Int](capacity=2 * m - 1)
    for _ in range(2 * m - 1):
        root_min.append(0x7FFFFFFF)
    for i in range(m):
        var r = uf.find(i)
        if i < root_min[r]:
            root_min[r] = i
    var out = List[Int32](capacity=m)
    for i in range(m):
        out.append(Int32(root_min[uf.find(i)]))
    return out^


def partitions_agree(a: List[Int32], b: List[Int32]) -> Bool:
    """Two labelings induce the same partition iff `a[i] == a[j]` exactly
    when `b[i] == b[j]`. Checked through a label-to-label map each way."""
    var n = len(a)
    var seen_a = List[Int]()
    var seen_b = List[Int]()
    var map_ab = List[Int]()
    var map_ba = List[Int]()
    for i in range(n):
        var la = Int(a[i])
        var lb = Int(b[i])
        var found = False
        for k in range(len(seen_a)):
            if seen_a[k] == la:
                found = True
                if map_ab[k] != lb:
                    return False
        if not found:
            seen_a.append(la)
            map_ab.append(lb)
        var found_b = False
        for k in range(len(seen_b)):
            if seen_b[k] == lb:
                found_b = True
                if map_ba[k] != la:
                    return False
        if not found_b:
            seen_b.append(lb)
            map_ba.append(la)
    return True


def oracle_linkage(fix: Int, n_clusters: Int) -> OracleLinkage:
    var m = fixture_n(fix)
    var d = fixture_d(fix)
    var x = fixture_as_list(fix)
    var dists = host_pinned_distance_matrix(x, m, d, True)
    var mst = host_kruskal(dists, m)
    var children = host_dendrogram(mst[0], mst[1], m)
    var labels = host_extract_flattened_clusters(children, n_clusters, m)
    var partition = host_partition(mst[0], mst[1], m, n_clusters)
    var total32 = Float64(0.0)
    for i in range(m - 1):
        total32 = total32 + Float64(mst[2][i])
    var total64 = host_kruskal_f64(x, m, d)
    return OracleLinkage(
        m, dists^, mst[0].copy(), mst[1].copy(), mst[2].copy(), children^,
        labels^, partition^, total32, total64,
    )
