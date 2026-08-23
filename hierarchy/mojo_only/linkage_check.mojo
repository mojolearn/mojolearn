"""The single-linkage gates.

    tools/with_build_lock.sh     pixi run mojo run -I . hierarchy/mojo_only/linkage_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . hierarchy/mojo_only/linkage_check.mojo

Every check prints the mode this binary COMPILED in. Under IDENTICAL the
distance gates are ASSERTIONS (the device tile must equal the host pinned
arithmetic byte for byte); under FAST the same comparisons are REPORTS,
because the FAST arm is MAX's matmul plus a library block fold and no host
can replicate that. The MST / dendrogram / label gates hold in BOTH modes,
because they are stated over whatever weights the device produced: given
these bytes, the MST is the lexicographic one, the dendrogram follows from
it, the labels follow from that.

CHECKS
  check_linkage_distances_match_host_pinned   per cell, every fixture,
                                              symmetric on the device
  check_linkage_mst_matches_kruskal           device Boruvka edge SET ==
                                              host Kruskal, as sorted
                                              (lo, hi, w) triples, w bitwise
  check_linkage_dendrogram_and_labels         children rows (as unordered
                                              pairs) == oracle; labels
                                              BITWISE == serial extract;
                                              partition == cut
  check_linkage_entry_matches_stages          cuML's `single_linkage` entry
                                              returns the staged run's bytes
  check_linkage_union_find_matches_a_naive_one   DEVIATION 622
  check_linkage_launch_invariance             THE HEADLINE: children, labels,
                                              MST bytes do not move across
                                              two tile block sizes, two MST
                                              block sizes, two extract block
                                              sizes, two paddings / poisons
  check_linkage_batch_composition             the distance cell (i, j) of the
                                              first 37 rows computed alone ==
                                              the same cell inside the 203
  check_linkage_float64_reference             MST total weight within 1e-4
                                              of a Float64 direct-form MST
  check_linkage_refusals                      use_knn, L1, n_clusters > m,
                                              n_rows > 46340 raise BY NAME
  check_linkage_sabotages                     the four arms of edge_order.mojo
  check_linkage_card_is_stable                two cards from two launch
                                              shapes are record-identical

SABOTAGE TABLE (results are copied into hierarchy/README.md):
  LINK_SAB_RANDOM_ALTERATION   FIX_DUPS        MUST FAIL the MST gate
  LINK_SAB_ROTATE_CONTRACTION  FIX_HASHED      MUST FAIL the distance gate
                                               (IDENTICAL; FAST reports)
  LINK_SAB_STD_SQRT            FIX_HASHED      REPORT (Apple: both sqrts are
                                               correctly rounded; expected
                                               0 cells moved)
  LINK_SAB_SORT_WEIGHT_ONLY    FIX_DUPS        MUST FAIL the dendrogram gate
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import IdentityTrace, first_divergence
from hierarchy.mojo_only.edge_order import (
    LINK_SAB_NONE,
    LINK_SAB_RANDOM_ALTERATION,
    LINK_SAB_ROTATE_CONTRACTION,
    LINK_SAB_SORT_WEIGHT_ONLY,
    LINK_SAB_STD_SQRT,
    edge_hi,
    edge_lo,
    pack_edge_key,
    weight_order_key,
)
from hierarchy.mojo_only.linkage_oracle import (
    FIX_BLOBS,
    FIX_BLOBS_DUPS,
    FIX_CHAIN,
    FIX_COUNT,
    FIX_DUPS,
    FIX_HASHED,
    NaiveUnionFind,
    build_fixture,
    fixture_as_list,
    fixture_d,
    fixture_n,
    fixture_n_clusters,
    fixture_name,
    host_dendrogram,
    host_extract_flattened_clusters,
    host_kruskal,
    host_kruskal_f64,
    host_partition,
    host_pinned_distance_matrix,
    partitions_agree,
)
from hierarchy.ported.cluster.detail.agglomerative import (
    UnionFind,
    build_dendrogram_host,
    extract_flattened_clusters,
)
from hierarchy.ported.cluster.detail.connectivities import (
    DISTANCE_L1,
    DISTANCE_L2_SQRT_EXPANDED,
    FLOAT32_MAX,
    pairwise_distances,
)
from hierarchy.ported.cluster.detail.mst import build_sorted_mst
from hierarchy.ported.hierarchy.linkage import single_linkage
from hierarchy.ported.sparse.op.sort import merge_sort_u64_with_index
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def _mode_name() -> String:
    comptime if IDENTICAL_BUILD:
        return String("IDENTICAL")
    return String("FAST")


def _hex32(v: Float32) -> String:
    var b = rebind[UInt32](v.to_bits())
    var digits = String("0123456789abcdef")
    var out = String("0x")
    var shift = 28
    while shift >= 0:
        var nib = Int((b >> UInt32(shift)) & UInt32(0xF))
        out += String(digits[byte=nib])
        shift -= 4
    return out


# ======================================================================
# DEVICE HELPERS: each stage exposed, with the scheduling knobs
# ======================================================================


@fieldwise_init
struct StagedRun(Movable):
    var dists: List[Float32]
    var mst_src: List[Int32]
    var mst_dst: List[Int32]
    var mst_w: List[Float32]
    var children: List[Int32]
    var labels: List[Int32]
    var rounds: Int


def _upload(
    ctx: DeviceContext, fix: Int, pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    """The fixture on the device with `pad` extra floats of `poison` after
    it (a buffer longer than its use, which a kernel reading past `m * d`
    would reveal)."""
    var m = fixture_n(fix)
    var d = fixture_d(fix)
    var host = ctx.enqueue_create_host_buffer[DType.float32](m * d + pad)
    ctx.synchronize()
    var vals = fixture_as_list(fix)
    for i in range(m * d):
        host.unsafe_ptr().unsafe_store(i, vals[i])
    for i in range(pad):
        host.unsafe_ptr().unsafe_store(m * d + i, poison)
    var dev = ctx.enqueue_create_buffer[DType.float32](m * d + pad)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _copy_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var v = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = v^
    return out^


def _copy_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.synchronize()
    var v = buf.create_sub_buffer[DType.int32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = List[Int32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = v^
    return out^


def _staged_run(
    ctx: DeviceContext,
    fix: Int,
    n_clusters: Int,
    tile_tpb: Int = 256,
    mst_tpb: Int = 256,
    extract_tpb: Int = 256,
    pad: Int = 0,
    poison: Float32 = 0.0,
    out_pad: Int = 0,
    out_poison: Int32 = 0,
    sabotage: Int32 = LINK_SAB_NONE,
) raises -> StagedRun:
    """`build_dist_linkage` + `extract_flattened_clusters`, stage by stage,
    every intermediate copied back. `out_pad` / `out_poison` pad and poison
    the OUTPUT buffers (children, labels, mst) past their used length."""
    var m = fixture_n(fix)
    var d = fixture_d(fix)
    var x = _upload(ctx, fix, pad, poison)
    var nnz = m * m
    var indptr = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var indices = ctx.enqueue_create_buffer[DType.int32](nnz)
    var dists = ctx.enqueue_create_buffer[DType.float32](nnz + out_pad)
    var norms = ctx.enqueue_create_buffer[DType.float32](m)
    if out_pad > 0:
        ctx.enqueue_memset(dists, bitcast[DType.float32](out_poison))
    pairwise_distances(
        ctx, x, m, d, DISTANCE_L2_SQRT_EXPANDED, indptr, indices, dists, norms,
        tile_tpb, sabotage,
    )
    var h_dists = _copy_f32(ctx, dists, nnz)

    var n_edges = m - 1
    var mst_src = ctx.enqueue_create_buffer[DType.int32](n_edges + out_pad)
    var mst_dst = ctx.enqueue_create_buffer[DType.int32](n_edges + out_pad)
    var mst_w = ctx.enqueue_create_buffer[DType.float32](n_edges + out_pad)
    var color = ctx.enqueue_create_buffer[DType.int32](m)
    if out_pad > 0:
        ctx.enqueue_memset(mst_src, out_poison)
        ctx.enqueue_memset(mst_dst, out_poison)
        ctx.enqueue_memset(mst_w, bitcast[DType.float32](out_poison))
    var rounds = build_sorted_mst(
        ctx, indptr, indices, dists, m, d, mst_src, mst_dst, mst_w, color, nnz,
        10, mst_tpb, sabotage,
    )
    var h_src = _copy_i32(ctx, mst_src, n_edges)
    var h_dst = _copy_i32(ctx, mst_dst, n_edges)
    var h_w = _copy_f32(ctx, mst_w, n_edges)

    var children = ctx.enqueue_create_buffer[DType.int32](n_edges * 2 + out_pad)
    var out_delta = ctx.enqueue_create_buffer[DType.float32](n_edges)
    var out_sizes = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var labels = ctx.enqueue_create_buffer[DType.int32](m + out_pad)
    if out_pad > 0:
        ctx.enqueue_memset(children, out_poison)
        ctx.enqueue_memset(labels, out_poison)
    build_dendrogram_host(
        ctx, mst_src, mst_dst, mst_w, n_edges, children, out_delta, out_sizes
    )
    extract_flattened_clusters(ctx, labels, children, n_clusters, m, extract_tpb)
    var h_children = _copy_i32(ctx, children, n_edges * 2)
    var h_labels = _copy_i32(ctx, labels, m)

    _ = x^
    _ = indptr^
    _ = indices^
    _ = dists^
    _ = norms^
    _ = mst_src^
    _ = mst_dst^
    _ = mst_w^
    _ = color^
    _ = children^
    _ = out_delta^
    _ = out_sizes^
    _ = labels^
    return StagedRun(h_dists^, h_src^, h_dst^, h_w^, h_children^, h_labels^, rounds)


def _canonical_mst(
    src: List[Int32], dst: List[Int32], w: List[Float32]
) -> Tuple[List[Int32], List[Int32], List[Float32]]:
    """The device's (src, dst, w) list as (lo, hi, w) sorted in the total
    order, for a SET comparison that ignores orientation and order."""
    var n = len(src)
    var keys = List[UInt64](capacity=n)
    var idx = List[Int](capacity=n)
    for i in range(n):
        keys.append(
            pack_edge_key(weight_order_key(w[i]), edge_lo(src[i], dst[i]), edge_hi(src[i], dst[i]))
        )
        idx.append(i)
    merge_sort_u64_with_index(keys, idx)
    var lo = List[Int32](capacity=n)
    var hi = List[Int32](capacity=n)
    var ww = List[Float32](capacity=n)
    for k in range(n):
        var i = idx[k]
        lo.append(edge_lo(src[i], dst[i]))
        hi.append(edge_hi(src[i], dst[i]))
        ww.append(w[i])
    return (lo^, hi^, ww^)


def _count_dist_mismatches(a: List[Float32], b: List[Float32]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i].to_bits() != b[i].to_bits():
            n += 1
    return n


def _mst_set_equal(
    a_lo: List[Int32], a_hi: List[Int32], a_w: List[Float32],
    b_lo: List[Int32], b_hi: List[Int32], b_w: List[Float32],
) -> Bool:
    if len(a_lo) != len(b_lo):
        return False
    for i in range(len(a_lo)):
        if a_lo[i] != b_lo[i] or a_hi[i] != b_hi[i] or a_w[i].to_bits() != b_w[i].to_bits():
            return False
    return True


def _children_pairs_equal(a: List[Int32], b: List[Int32]) -> Bool:
    if len(a) != len(b):
        return False
    var rows = len(a) // 2
    for i in range(rows):
        var a0 = a[2 * i]
        var a1 = a[2 * i + 1]
        var b0 = b[2 * i]
        var b1 = b[2 * i + 1]
        if not ((a0 == b0 and a1 == b1) or (a0 == b1 and a1 == b0)):
            return False
    return True


def _i32_equal(a: List[Int32], b: List[Int32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _f32_bits_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i].to_bits() != b[i].to_bits():
            return False
    return True


# ======================================================================
# THE CHECKS
# ======================================================================


def check_linkage_distances_match_host_pinned() raises:
    """Per cell, every fixture. IDENTICAL asserts; FAST reports. Also: the
    device matrix is exactly symmetric and FLT_MAX on the diagonal, which
    DEVIATION 620's symmetric triple relies on."""
    var ctx = DeviceContext()
    for fix in range(FIX_COUNT):
        var m = fixture_n(fix)
        var d = fixture_d(fix)
        var run = _staged_run(ctx, fix, fixture_n_clusters(fix))
        var asym = 0
        var diag_bad = 0
        for i in range(m):
            if run.dists[i * m + i].to_bits() != FLOAT32_MAX.to_bits():
                diag_bad += 1
            for j in range(i + 1, m):
                if run.dists[i * m + j].to_bits() != run.dists[j * m + i].to_bits():
                    asym += 1
        if diag_bad != 0:
            raise Error(
                "check_linkage_distances_match_host_pinned [" + _mode_name() + "] "
                + fixture_name(fix) + ": " + String(diag_bad)
                + " diagonal cells are not FLT_MAX"
            )
        var oracle = host_pinned_distance_matrix(fixture_as_list(fix), m, d, True)
        var mism = _count_dist_mismatches(run.dists, oracle)
        comptime if IDENTICAL_BUILD:
            if asym != 0:
                raise Error(
                    "check_linkage_distances_match_host_pinned [IDENTICAL] "
                    + fixture_name(fix) + ": " + String(asym)
                    + " (i, j) cells differ from (j, i)"
                )
            if mism != 0:
                var first = -1
                for c in range(m * m):
                    if run.dists[c].to_bits() != oracle[c].to_bits():
                        first = c
                        break
                raise Error(
                    "check_linkage_distances_match_host_pinned [IDENTICAL] "
                    + fixture_name(fix) + ": " + String(mism) + " of "
                    + String(m * m) + " cells differ from the host pinned"
                    " arithmetic; first at cell " + String(first) + " device "
                    + _hex32(run.dists[first]) + " host " + _hex32(oracle[first])
                )
            print(
                "check_linkage_distances_match_host_pinned [IDENTICAL] OK: "
                + fixture_name(fix) + " " + String(m * m)
                + " cells bitwise equal to the host pinned arithmetic,"
                " symmetric, FLT_MAX diagonal"
            )
        else:
            print(
                "check_linkage_distances_match_host_pinned [FAST] REPORT: "
                + fixture_name(fix) + " " + String(mism) + " of "
                + String(m * m) + " cells differ from the host pinned"
                " arithmetic (the FAST arm is MAX's matmul; no assertion),"
                " asymmetric cells " + String(asym)
            )


def check_linkage_mst_matches_kruskal() raises:
    """The device Boruvka's edge SET, as sorted (lo, hi, w) triples with w
    bitwise, equals a host Kruskal run on THE DEVICE'S weights, every
    fixture, both modes. Also the edge count and that the device list IS
    already in the total order (DEVIATION 621's sort)."""
    var ctx = DeviceContext()
    for fix in range(FIX_COUNT):
        var m = fixture_n(fix)
        var run = _staged_run(ctx, fix, fixture_n_clusters(fix))
        var canon = _canonical_mst(run.mst_src, run.mst_dst, run.mst_w)
        var kr = host_kruskal(run.dists, m)
        if len(run.mst_src) != m - 1:
            raise Error(
                "check_linkage_mst_matches_kruskal " + fixture_name(fix)
                + ": device MST has " + String(len(run.mst_src)) + " edges, want "
                + String(m - 1)
            )
        if not _mst_set_equal(canon[0], canon[1], canon[2], kr[0], kr[1], kr[2]):
            var first = -1
            for i in range(m - 1):
                if canon[0][i] != kr[0][i] or canon[1][i] != kr[1][i] or canon[2][i].to_bits() != kr[2][i].to_bits():
                    first = i
                    break
            raise Error(
                "check_linkage_mst_matches_kruskal [" + _mode_name() + "] "
                + fixture_name(fix) + ": device Boruvka edge set != host"
                " Kruskal; first differing sorted position " + String(first)
                + " device (" + String(canon[0][first]) + "," + String(canon[1][first])
                + "," + _hex32(canon[2][first]) + ") kruskal (" + String(kr[0][first])
                + "," + String(kr[1][first]) + "," + _hex32(kr[2][first]) + ")"
            )
        # the device list is already sorted in the total order
        for i in range(m - 1):
            var lo = edge_lo(run.mst_src[i], run.mst_dst[i])
            var hi = edge_hi(run.mst_src[i], run.mst_dst[i])
            if lo != kr[0][i] or hi != kr[1][i]:
                raise Error(
                    "check_linkage_mst_matches_kruskal " + fixture_name(fix)
                    + ": the device MST list is not in the total order at "
                    + String(i)
                )
        print(
            "check_linkage_mst_matches_kruskal [" + _mode_name() + "] OK: "
            + fixture_name(fix) + " " + String(m - 1)
            + " edges, set and order equal to host Kruskal, weights bitwise,"
            " boruvka rounds " + String(run.rounds)
        )


def check_linkage_dendrogram_and_labels() raises:
    """The children rows as unordered pairs == the oracle's rows over the same
    sorted MST; labels BITWISE == the serial extract over the oracle's
    children; the label partition == the independent cut partition."""
    var ctx = DeviceContext()
    for fix in range(FIX_COUNT):
        var m = fixture_n(fix)
        var k = fixture_n_clusters(fix)
        var run = _staged_run(ctx, fix, k)
        var kr = host_kruskal(run.dists, m)
        var o_children = host_dendrogram(kr[0], kr[1], m)
        var o_labels = host_extract_flattened_clusters(o_children, k, m)
        var o_part = host_partition(kr[0], kr[1], m, k)
        if not _children_pairs_equal(run.children, o_children):
            var first = -1
            for i in range(m - 1):
                if not (
                    (run.children[2 * i] == o_children[2 * i] and run.children[2 * i + 1] == o_children[2 * i + 1])
                    or (run.children[2 * i] == o_children[2 * i + 1] and run.children[2 * i + 1] == o_children[2 * i])
                ):
                    first = i
                    break
            raise Error(
                "check_linkage_dendrogram_and_labels [" + _mode_name() + "] "
                + fixture_name(fix) + ": children row " + String(first)
                + " device (" + String(run.children[2 * first]) + ","
                + String(run.children[2 * first + 1]) + ") oracle ("
                + String(o_children[2 * first]) + "," + String(o_children[2 * first + 1]) + ")"
            )
        if not _i32_equal(run.labels, o_labels):
            var first = -1
            for i in range(m):
                if run.labels[i] != o_labels[i]:
                    first = i
                    break
            raise Error(
                "check_linkage_dendrogram_and_labels [" + _mode_name() + "] "
                + fixture_name(fix) + ": labels differ from the serial extract"
                " at row " + String(first) + " device " + String(run.labels[first])
                + " oracle " + String(o_labels[first])
            )
        if not partitions_agree(run.labels, o_part):
            raise Error(
                "check_linkage_dendrogram_and_labels [" + _mode_name() + "] "
                + fixture_name(fix) + ": the label partition differs from the"
                " cut partition"
            )
        print(
            "check_linkage_dendrogram_and_labels [" + _mode_name() + "] OK: "
            + fixture_name(fix) + " " + String(m - 1) + " children rows, "
            + String(m) + " labels bitwise, partition at n_clusters="
            + String(k) + " agrees"
        )


def check_linkage_entry_matches_stages() raises:
    """The cuML `single_linkage` entry (`hierarchy/ported/hierarchy/linkage.
    mojo`) returns the same children and labels bytes as the staged run, on
    every fixture: REACH of the real entry, not a reconstruction of it."""
    var ctx = DeviceContext()
    for fix in range(FIX_COUNT):
        var m = fixture_n(fix)
        var d = fixture_d(fix)
        var k = fixture_n_clusters(fix)
        var run = _staged_run(ctx, fix, k)
        var x = _upload(ctx, fix, 0, 0.0)
        var children = ctx.enqueue_create_buffer[DType.int32]((m - 1) * 2)
        var labels = ctx.enqueue_create_buffer[DType.int32](m)
        var out = single_linkage(
            ctx, x, m, d, k, DISTANCE_L2_SQRT_EXPANDED, children, labels
        )
        var h_children = _copy_i32(ctx, children, (m - 1) * 2)
        var h_labels = _copy_i32(ctx, labels, m)
        if not _i32_equal(h_children, run.children) or not _i32_equal(h_labels, run.labels):
            raise Error(
                "check_linkage_entry_matches_stages [" + _mode_name() + "] "
                + fixture_name(fix) + ": the entry's children/labels differ"
                " from the staged run"
            )
        if out.n_connected_components != 1 or out.n_leaves != m or out.n_clusters != k:
            raise Error("check_linkage_entry_matches_stages: output struct fields")
        print(
            "check_linkage_entry_matches_stages [" + _mode_name() + "] OK: "
            + fixture_name(fix) + " entry == stages, rounds "
            + String(out.n_boruvka_rounds)
        )
        _ = x^
        _ = children^
        _ = labels^


def check_linkage_union_find_matches_a_naive_one() raises:
    """DEVIATION 622. The ported `UnionFind` (textbook compression) and the
    compression-free `NaiveUnionFind` return the same roots on every
    fixture's sorted MST, so the `children` rows are identical. Host only."""
    for fix in range(FIX_COUNT):
        var m = fixture_n(fix)
        var d = fixture_d(fix)
        var dists = host_pinned_distance_matrix(fixture_as_list(fix), m, d, True)
        var kr = host_kruskal(dists, m)
        var U = UnionFind(m)
        var N = NaiveUnionFind(m)
        for i in range(m - 1):
            var a = Int(kr[0][i])
            var b = Int(kr[1][i])
            var ua = U.find(a)
            var ub = U.find(b)
            var na = N.find(a)
            var nb = N.find(b)
            if ua != na or ub != nb:
                raise Error(
                    "check_linkage_union_find_matches_a_naive_one "
                    + fixture_name(fix) + ": row " + String(i) + " ported ("
                    + String(ua) + "," + String(ub) + ") naive (" + String(na)
                    + "," + String(nb) + ")"
                )
            if U.size[ua] + U.size[ub] != N.size[na] + N.size[nb]:
                raise Error("check_linkage_union_find_matches_a_naive_one: sizes")
            U.perform_union(ua, ub)
            N.union(na, nb)
        # and no slot outside [0, 2m-1) was ever touched: the List would
        # have trapped; the ported find never indexes -1 or 2m-2 by design
        print(
            "check_linkage_union_find_matches_a_naive_one OK: "
            + fixture_name(fix) + " " + String(m - 1)
            + " unions, identical roots and sizes"
        )


def check_linkage_launch_invariance() raises:
    """THE HEADLINE: the output bytes do not move across launch shapes.

    On FIX_HASHED (203 rows, no block divides it) and FIX_DUPS (the tie
    fixture): the distance bytes, the MST (src, dst, w) bytes, the children
    bytes and the label bytes do not move across

        tile block   256 -> 64 -> 32
        MST block    256 -> 128 -> 1024
        extract block 256 -> 64 -> 512
        input padding  0 -> 37 floats of 1e30 -> 1024 floats of NaN
        output padding 0 -> 33 cells of 0x7fffffff -> 9 cells of 0xdeadbeef

    Under IDENTICAL every stage must agree; under FAST the distance and
    MST stages are reported (a MAX matmul's bytes are not pinned) and the
    dendrogram/labels are still asserted GIVEN the MST of run A -- i.e.
    the check is that nothing downstream of the weights moved."""
    var ctx = DeviceContext()
    var fixes = [FIX_HASHED, FIX_DUPS]
    for t in range(len(fixes)):
        var fix = fixes[t]
        var k = fixture_n_clusters(fix)
        var a = _staged_run(ctx, fix, k, 256, 256, 256, 0, 0.0, 0, 0)
        var b = _staged_run(
            ctx, fix, k, 64, 128, 64, 37, Float32(1e30), 33, Int32(0x7FFFFFFF)
        )
        var nan = bitcast[DType.float32](UInt32(0x7FC00000))
        var c = _staged_run(
            ctx, fix, k, 32, 1024, 512, 1024, nan, 9, Int32(0xDEADBEEF)
        )
        var dists_ok = _f32_bits_equal(a.dists, b.dists) and _f32_bits_equal(a.dists, c.dists)
        var mst_ok = (
            _i32_equal(a.mst_src, b.mst_src) and _i32_equal(a.mst_dst, b.mst_dst)
            and _f32_bits_equal(a.mst_w, b.mst_w)
            and _i32_equal(a.mst_src, c.mst_src) and _i32_equal(a.mst_dst, c.mst_dst)
            and _f32_bits_equal(a.mst_w, c.mst_w)
        )
        var dendro_ok = _i32_equal(a.children, b.children) and _i32_equal(a.children, c.children)
        var labels_ok = _i32_equal(a.labels, b.labels) and _i32_equal(a.labels, c.labels)
        var rounds_ok = a.rounds == b.rounds and a.rounds == c.rounds
        comptime if IDENTICAL_BUILD:
            if not dists_ok:
                raise Error(
                    "check_linkage_launch_invariance [IDENTICAL] " + fixture_name(fix)
                    + ": the DISTANCE bytes moved across launch shapes ("
                    + String(_count_dist_mismatches(a.dists, b.dists)) + " / "
                    + String(_count_dist_mismatches(a.dists, c.dists)) + " cells)"
                )
            if not mst_ok:
                raise Error(
                    "check_linkage_launch_invariance [IDENTICAL] " + fixture_name(fix)
                    + ": the MST bytes moved across launch shapes"
                )
        else:
            print(
                "check_linkage_launch_invariance [FAST] REPORT: " + fixture_name(fix)
                + " distances " + ("stable" if dists_ok else "MOVED")
                + ", MST " + ("stable" if mst_ok else "MOVED")
                + " across launch shapes (FAST distance bytes are not pinned)"
            )
        if IDENTICAL_BUILD or mst_ok:
            if not dendro_ok or not labels_ok or not rounds_ok:
                raise Error(
                    "check_linkage_launch_invariance [" + _mode_name() + "] "
                    + fixture_name(fix) + ": children/labels/rounds moved across"
                    " launch shapes with the same MST (children "
                    + String(dendro_ok) + " labels " + String(labels_ok)
                    + " rounds " + String(rounds_ok) + ")"
                )
        print(
            "check_linkage_launch_invariance [" + _mode_name() + "] OK: "
            + fixture_name(fix) + " children, labels and round count (and under"
            " IDENTICAL the distance and MST bytes) identical across tile 256/64/32,"
            " MST 256/128/1024, extract 256/64/512, two input paddings/poisons,"
            " two output paddings/poisons"
        )


def check_linkage_batch_composition() raises:
    """The distance cell (i, j) for the first 37 rows of FIX_HASHED, computed
    in a 37 x 37 launch, equals the same cell inside the 203 x 203 launch,
    bitwise, per cell. IDENTICAL asserts; FAST reports (the matmul's tile
    schedule may legitimately differ with the shape)."""
    var ctx = DeviceContext()
    var fix = FIX_HASHED
    var m = fixture_n(fix)
    var d = fixture_d(fix)
    var sub = 37
    var full = _staged_run(ctx, fix, fixture_n_clusters(fix))
    var x = _upload(ctx, fix, 0, 0.0)
    var indptr = ctx.enqueue_create_buffer[DType.int32](sub + 1)
    var indices = ctx.enqueue_create_buffer[DType.int32](sub * sub)
    var dists = ctx.enqueue_create_buffer[DType.float32](sub * sub)
    var norms = ctx.enqueue_create_buffer[DType.float32](sub)
    pairwise_distances(
        ctx, x, sub, d, DISTANCE_L2_SQRT_EXPANDED, indptr, indices, dists, norms
    )
    var small = _copy_f32(ctx, dists, sub * sub)
    var mism = 0
    for i in range(sub):
        for j in range(sub):
            if small[i * sub + j].to_bits() != full.dists[i * m + j].to_bits():
                mism += 1
    comptime if IDENTICAL_BUILD:
        if mism != 0:
            raise Error(
                "check_linkage_batch_composition [IDENTICAL]: " + String(mism)
                + " of " + String(sub * sub) + " cells differ between the 37-row"
                " launch and the 203-row launch"
            )
        print(
            "check_linkage_batch_composition [IDENTICAL] OK: all " + String(sub * sub)
            + " cells of the 37-row launch equal the same cells of the 203-row launch"
        )
    else:
        print(
            "check_linkage_batch_composition [FAST] REPORT: " + String(mism) + " of "
            + String(sub * sub) + " cells differ between the 37-row and 203-row"
            " launches (no assertion under FAST)"
        )
    _ = x^
    _ = indptr^
    _ = indices^
    _ = dists^
    _ = norms^


def check_linkage_float64_reference() raises:
    """Tolerance sanity: the device MST's total weight (summed in double)
    is within 1e-4 relative of a Float64 direct-form Kruskal's, every
    fixture. The partition agreement with the Float64 MST is REPORTED (a
    near-tie can legitimately flip under a different rounding)."""
    var ctx = DeviceContext()
    for fix in range(FIX_COUNT):
        var m = fixture_n(fix)
        var d = fixture_d(fix)
        var run = _staged_run(ctx, fix, fixture_n_clusters(fix))
        var total32 = Float64(0.0)
        for i in range(m - 1):
            total32 = total32 + Float64(run.mst_w[i])
        var total64 = host_kruskal_f64(fixture_as_list(fix), m, d)
        var denom = total64 if total64 > 0 else Float64(1.0)
        var rel = abs(total32 - total64) / denom
        if rel > 1e-4:
            raise Error(
                "check_linkage_float64_reference " + fixture_name(fix)
                + ": MST total " + String(total32) + " vs Float64 " + String(total64)
                + " (rel " + String(rel) + ")"
            )
        print(
            "check_linkage_float64_reference OK: " + fixture_name(fix)
            + " MST total " + String(total32) + " Float64 reference "
            + String(total64) + " rel " + String(rel)
        )


def check_linkage_refusals() raises:
    """Every refusal names its parameter."""
    var ctx = DeviceContext()
    var fix = FIX_BLOBS
    var m = fixture_n(fix)
    var d = fixture_d(fix)
    var x = _upload(ctx, fix, 0, 0.0)
    var children = ctx.enqueue_create_buffer[DType.int32]((m - 1) * 2)
    var labels = ctx.enqueue_create_buffer[DType.int32](m)

    var got_knn = False
    try:
        _ = single_linkage(
            ctx, x, m, d, 3, DISTANCE_L2_SQRT_EXPANDED, children, labels, use_knn=True
        )
    except e:
        got_knn = String(e).find("KNN_GRAPH") >= 0
    if not got_knn:
        raise Error("check_linkage_refusals: use_knn=True did not raise by name")

    var got_l1 = False
    try:
        _ = single_linkage(ctx, x, m, d, 3, DISTANCE_L1, children, labels)
    except e:
        got_l1 = String(e).find("metric=3") >= 0
    if not got_l1:
        raise Error("check_linkage_refusals: metric=L1 did not raise by name")

    var got_k = False
    try:
        _ = single_linkage(ctx, x, m, d, m + 1, DISTANCE_L2_SQRT_EXPANDED, children, labels)
    except e:
        got_k = String(e).find("n_clusters") >= 0
    if not got_k:
        raise Error("check_linkage_refusals: n_clusters > n_rows did not raise by name")

    var got_big = False
    try:
        var ip = ctx.enqueue_create_buffer[DType.int32](4)
        var ii = ctx.enqueue_create_buffer[DType.int32](4)
        var dd = ctx.enqueue_create_buffer[DType.float32](4)
        var nn = ctx.enqueue_create_buffer[DType.float32](4)
        pairwise_distances(ctx, x, 46341, d, DISTANCE_L2_SQRT_EXPANDED, ip, ii, dd, nn)
        _ = ip^
        _ = ii^
        _ = dd^
        _ = nn^
    except e:
        got_big = String(e).find("46340") >= 0
    if not got_big:
        raise Error("check_linkage_refusals: n_rows > 46340 did not raise by name")
    print(
        "check_linkage_refusals OK: use_knn=True, metric=L1, n_clusters > n_rows,"
        " n_rows > 46340 each raise by name"
    )
    _ = x^
    _ = children^
    _ = labels^


def check_linkage_sabotages() raises:
    """Break each pin and watch the gate move. The table in the header."""
    var ctx = DeviceContext()

    # 1. RANDOM ALTERATION on the tie fixture: the MST edge set must differ
    #    from Kruskal's (or the MST must fail to be a tree).
    var m = fixture_n(FIX_DUPS)
    var failed_1 = False
    var why_1 = String("")
    try:
        var run = _staged_run(ctx, FIX_DUPS, fixture_n_clusters(FIX_DUPS), 256, 256, 256, 0, 0.0, 0, 0, LINK_SAB_RANDOM_ALTERATION)
        var canon = _canonical_mst(run.mst_src, run.mst_dst, run.mst_w)
        var kr = host_kruskal(run.dists, m)
        if not _mst_set_equal(canon[0], canon[1], canon[2], kr[0], kr[1], kr[2]):
            failed_1 = True
            var n_diff = 0
            for i in range(m - 1):
                if canon[0][i] != kr[0][i] or canon[1][i] != kr[1][i]:
                    n_diff += 1
            why_1 = String(n_diff) + " of " + String(m - 1) + " sorted edge slots differ from Kruskal"
        var o_children = host_dendrogram(kr[0], kr[1], m)
        if not _children_pairs_equal(run.children, o_children):
            why_1 += "; children differ"
    except e:
        failed_1 = True
        why_1 = String("raised: ") + String(e)
    if not failed_1:
        raise Error(
            "check_linkage_sabotages: LINK_SAB_RANDOM_ALTERATION did NOT move"
            " the MST on the duplicate/equal-distance fixture; the tie-break"
            " pin is not reached"
        )
    print(
        "check_linkage_sabotages OK: LINK_SAB_RANDOM_ALTERATION on "
        + fixture_name(FIX_DUPS) + " FAILED the MST gate as required: " + why_1
    )
    # ... and on the tie-free blobs it is allowed to pass; report.
    var run_b = _staged_run(ctx, FIX_BLOBS, 3, 256, 256, 256, 0, 0.0, 0, 0, LINK_SAB_RANDOM_ALTERATION)
    var canon_b = _canonical_mst(run_b.mst_src, run_b.mst_dst, run_b.mst_w)
    var kr_b = host_kruskal(run_b.dists, fixture_n(FIX_BLOBS))
    print(
        "check_linkage_sabotages REPORT: LINK_SAB_RANDOM_ALTERATION on "
        + fixture_name(FIX_BLOBS) + " (tie-free) MST "
        + ("equal" if _mst_set_equal(canon_b[0], canon_b[1], canon_b[2], kr_b[0], kr_b[1], kr_b[2]) else "DIFFERS")
        + " -- a random tie-break only moves ties"
    )

    # 2. ROTATE the contraction start by block: distance bytes must move.
    var mh = fixture_n(FIX_HASHED)
    var dh = fixture_d(FIX_HASHED)
    var run_r = _staged_run(ctx, FIX_HASHED, 7, 256, 256, 256, 0, 0.0, 0, 0, LINK_SAB_ROTATE_CONTRACTION)
    var oracle = host_pinned_distance_matrix(fixture_as_list(FIX_HASHED), mh, dh, True)
    var mism_r = _count_dist_mismatches(run_r.dists, oracle)
    comptime if IDENTICAL_BUILD:
        if mism_r == 0:
            raise Error(
                "check_linkage_sabotages [IDENTICAL]: LINK_SAB_ROTATE_CONTRACTION"
                " moved NO distance cell; the contraction-order pin is not reached"
            )
        print(
            "check_linkage_sabotages [IDENTICAL] OK: LINK_SAB_ROTATE_CONTRACTION on "
            + fixture_name(FIX_HASHED) + " FAILED the distance gate as required: "
            + String(mism_r) + " of " + String(mh * mh) + " cells moved"
        )
    else:
        print(
            "check_linkage_sabotages [FAST] REPORT: LINK_SAB_ROTATE_CONTRACTION on "
            + fixture_name(FIX_HASHED) + ": " + String(mism_r) + " of "
            + String(mh * mh) + " cells differ from the host pinned arithmetic"
            " (no assertion under FAST)"
        )

    # 3. std sqrt at the seam: REPORT.
    var run_s = _staged_run(ctx, FIX_HASHED, 7, 256, 256, 256, 0, 0.0, 0, 0, LINK_SAB_STD_SQRT)
    var mism_s = _count_dist_mismatches(run_s.dists, oracle)
    print(
        "check_linkage_sabotages REPORT: LINK_SAB_STD_SQRT on "
        + fixture_name(FIX_HASHED) + ": " + String(mism_s) + " of "
        + String(mh * mh) + " cells differ from identical_sqrt's (Apple's"
        " sqrt is correctly rounded, so 0 is the expected Apple result; this"
        " arm would move cells on NVIDIA's approximate sqrt, DEVIATION 258)"
    )

    # 4. Weight-only unstable sort: the dendrogram must differ on ties.
    var run_w = _staged_run(ctx, FIX_DUPS, fixture_n_clusters(FIX_DUPS), 256, 256, 256, 0, 0.0, 0, 0, LINK_SAB_SORT_WEIGHT_ONLY)
    var kr_w = host_kruskal(run_w.dists, m)
    var o_children_w = host_dendrogram(kr_w[0], kr_w[1], m)
    if _children_pairs_equal(run_w.children, o_children_w):
        raise Error(
            "check_linkage_sabotages: LINK_SAB_SORT_WEIGHT_ONLY did NOT move the"
            " dendrogram on the tie fixture; the sort-order pin is not reached"
        )
    var n_rows_moved = 0
    for i in range(m - 1):
        if not (
            (run_w.children[2 * i] == o_children_w[2 * i] and run_w.children[2 * i + 1] == o_children_w[2 * i + 1])
            or (run_w.children[2 * i] == o_children_w[2 * i + 1] and run_w.children[2 * i + 1] == o_children_w[2 * i])
        ):
            n_rows_moved += 1
    var o_labels_w = host_extract_flattened_clusters(o_children_w, fixture_n_clusters(FIX_DUPS), m)
    print(
        "check_linkage_sabotages OK: LINK_SAB_SORT_WEIGHT_ONLY on "
        + fixture_name(FIX_DUPS) + " FAILED the dendrogram gate as required: "
        + String(n_rows_moved) + " of " + String(m - 1) + " children rows moved;"
        " labels " + ("unchanged" if _i32_equal(run_w.labels, o_labels_w) else "MOVED")
    )


def check_linkage_card_is_stable() raises:
    """Two identity cards from two launch shapes, record-identical. Under
    FAST the distance/MST records may legitimately differ; under IDENTICAL
    the whole card must."""
    var ctx = DeviceContext()
    var fix = FIX_BLOBS_DUPS
    var m = fixture_n(fix)
    var k = fixture_n_clusters(fix)
    var base = String("/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/c57d661c-4399-4b46-b361-5e5abdab7fe0/scratchpad/linkage_card_")
    var paths = [base + "a.trace", base + "b.trace"]
    var tiles = [256, 64]
    var msts = [256, 1024]
    for t in range(2):
        var tr = IdentityTrace.to_path(paths[t])
        var run = _staged_run(ctx, fix, k, tiles[t], msts[t], 256, 0, 0.0, 0, 0)
        tr.header("linkage_check card " + String(t) + " mode=" + _mode_name())
        tr.record_list_f32("linkage.dists", run.dists)
        var rounds = List[Int32]()
        rounds.append(Int32(run.rounds))
        tr.record_list_i32("linkage.mst.rounds", rounds)
        var edges = List[Int32]()
        for i in range(m - 1):
            edges.append(run.mst_src[i])
            edges.append(run.mst_dst[i])
        tr.record_list_i32("linkage.mst.edges", edges)
        tr.record_list_f32("linkage.mst.weights", run.mst_w)
        tr.record_list_i32("linkage.children", run.children)
        tr.record_list_i32("linkage.labels", run.labels)
    var div = first_divergence(paths[0], paths[1])
    comptime if IDENTICAL_BUILD:
        if div != "":
            raise Error(
                "check_linkage_card_is_stable [IDENTICAL]: the two cards differ: " + div
            )
        print("check_linkage_card_is_stable [IDENTICAL] OK: two launch shapes, 6 records, identical")
    else:
        print(
            "check_linkage_card_is_stable [FAST] REPORT: "
            + ("identical" if div == "" else ("first divergence " + div))
        )


def main() raises:
    print("linkage_check mode=" + _mode_name())
    check_linkage_distances_match_host_pinned()
    check_linkage_mst_matches_kruskal()
    check_linkage_dendrogram_and_labels()
    check_linkage_entry_matches_stages()
    check_linkage_union_find_matches_a_naive_one()
    check_linkage_launch_invariance()
    check_linkage_batch_composition()
    check_linkage_float64_reference()
    check_linkage_refusals()
    check_linkage_sabotages()
    check_linkage_card_is_stable()
    print("linkage_check mode=" + _mode_name() + " ALL OK")
