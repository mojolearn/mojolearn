"""Single-linkage driver: one fit, one identity card.

    tools/with_build_lock.sh     pixi run mojo run -I . hierarchy/linkage_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . hierarchy/linkage_main.mojo

    MOJOLEARN_IDENTITY_TRACE=/tmp/linkage.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . hierarchy/linkage_main.mojo
    python3 tools/identity_trace_diff.py /tmp/linkage.apple.card /tmp/linkage.other.card

Runs `hierarchy/ported/hierarchy/linkage.mojo::single_linkage` (cuML's
entry, PAIRWISE connectivity, L2SqrtExpanded) on the check's hashed
"three blobs plus duplicates" fixture and records every stage through
`core/identity_trace.mojo`:

    linkage.x              the input bytes (so a differing card is read
                           from a differing input, not a differing kernel)
    linkage.norms          the row norms the expanded identity uses
    linkage.dists          the m x m distance matrix, self-loops FLT_MAX
    linkage.mst.rounds     Boruvka round count (integer; a card that first
                           differs here disagreed about how much work)
    linkage.mst.edges      the sorted MST as (src, dst) pairs, Int32
    linkage.mst.weights    the sorted MST weights, Float32 bytes
    linkage.children       the dendrogram, (m - 1) x 2
    linkage.labels         the flat labels at n_clusters

Every line prints the mode this binary COMPILED in. No timing is measured
or printed here (COMMON_BRIEF: no performance numbers from this lane).
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import IdentityTrace
from core.row_norms import NORM_TPB, row_norm_kernel
from hierarchy.mojo_only.edge_order import LINK_SAB_NONE
from hierarchy.mojo_only.linkage_oracle import (
    FIX_BLOBS_DUPS,
    fixture_n,
    fixture_d,
    fixture_name,
    build_fixture,
)
from hierarchy.ported.cluster.detail.connectivities import (
    DISTANCE_L2_SQRT_EXPANDED,
    pairwise_distances,
)
from hierarchy.ported.cluster.detail.mst import build_sorted_mst
from hierarchy.ported.hierarchy.linkage import single_linkage
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _mode_name() -> String:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def main() raises:
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    var fix = FIX_BLOBS_DUPS
    var m = fixture_n(fix)
    var d = fixture_d(fix)
    var n_clusters = 3

    var hx = build_fixture(ctx, fix)
    var x = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    trace.header(
        "hierarchy/linkage_main.mojo mode=" + _mode_name() + " fixture="
        + fixture_name(fix) + " n_rows=" + String(m) + " n_cols=" + String(d)
        + " n_clusters=" + String(n_clusters) + " metric=L2SqrtExpanded"
        + " connectivity=pairwise"
    )
    trace.record_device[DType.float32](ctx, "linkage.x", x)

    # The two intermediate stages the entry does not hand back: recomputed
    # here through the SAME ported kernels, so the card carries the distance
    # bytes a cross-vendor diff needs to localize a divergence.
    var indptr = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var indices = ctx.enqueue_create_buffer[DType.int32](m * m)
    var dists = ctx.enqueue_create_buffer[DType.float32](m * m)
    var norms = ctx.enqueue_create_buffer[DType.float32](m)
    pairwise_distances(
        ctx, x, m, d, DISTANCE_L2_SQRT_EXPANDED, indptr, indices, dists, norms
    )
    trace.record_device[DType.float32](ctx, "linkage.norms", norms)
    trace.record_device[DType.float32](ctx, "linkage.dists", dists)

    var children = ctx.enqueue_create_buffer[DType.int32]((m - 1) * 2)
    var labels = ctx.enqueue_create_buffer[DType.int32](m)
    var out = single_linkage(
        ctx, x, m, d, n_clusters, DISTANCE_L2_SQRT_EXPANDED, children, labels
    )
    var rounds = List[Int32]()
    rounds.append(Int32(out.n_boruvka_rounds))
    trace.record_list_i32("linkage.mst.rounds", rounds)

    # The sorted MST is internal to `build_dist_linkage`; the dendrogram's
    # rows ARE the sorted edges' roots and `out_delta` its weights, but the
    # card wants the edges themselves, so run the MST stage once more
    # through the same entry points and record.
    var mst_src = ctx.enqueue_create_buffer[DType.int32](m - 1)
    var mst_dst = ctx.enqueue_create_buffer[DType.int32](m - 1)
    var mst_w = ctx.enqueue_create_buffer[DType.float32](m - 1)
    var color = ctx.enqueue_create_buffer[DType.int32](m)
    _ = build_sorted_mst(
        ctx, indptr, indices, dists, m, d, mst_src, mst_dst, mst_w, color, m * m
    )
    var h_src = ctx.enqueue_create_host_buffer[DType.int32](m - 1)
    var h_dst = ctx.enqueue_create_host_buffer[DType.int32](m - 1)
    ctx.enqueue_copy(dst_ptr=h_src.unsafe_ptr(), src_buf=mst_src)
    ctx.enqueue_copy(dst_ptr=h_dst.unsafe_ptr(), src_buf=mst_dst)
    ctx.synchronize()
    var edges = List[Int32]()
    for i in range(m - 1):
        edges.append(h_src.unsafe_ptr().unsafe_load(i))
        edges.append(h_dst.unsafe_ptr().unsafe_load(i))
    trace.record_list_i32("linkage.mst.edges", edges)
    trace.record_device[DType.float32](ctx, "linkage.mst.weights", mst_w)
    trace.record_device[DType.int32](ctx, "linkage.children", children)
    trace.record_device[DType.int32](ctx, "linkage.labels", labels)

    var h_labels = ctx.enqueue_create_host_buffer[DType.int32](m)
    ctx.enqueue_copy(dst_ptr=h_labels.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    var counts = List[Int]()
    for _ in range(n_clusters):
        counts.append(0)
    for i in range(m):
        var l = Int(h_labels.unsafe_ptr().unsafe_load(i))
        if l >= 0 and l < n_clusters:
            counts[l] += 1
    print(
        "linkage_main mode=" + _mode_name() + " fixture=" + fixture_name(fix)
        + " n_rows=" + String(m) + " n_cols=" + String(d)
        + " n_clusters=" + String(n_clusters)
        + " boruvka_rounds=" + String(out.n_boruvka_rounds)
        + " n_connected_components=" + String(out.n_connected_components)
    )
    var sizes = String("cluster sizes:")
    for k in range(n_clusters):
        sizes += " " + String(counts[k])
    print(sizes)
    if trace.enabled:
        print("identity card written to " + trace.path)
    else:
        print("set MOJOLEARN_IDENTITY_TRACE=<path> to write the identity card")
    _ = hx^
    _ = h_src^
    _ = h_dst^
    _ = h_labels^
    _ = x^
    _ = indptr^
    _ = indices^
    _ = dists^
    _ = norms^
    _ = children^
    _ = labels^
    _ = mst_src^
    _ = mst_dst^
    _ = mst_w^
    _ = color^
