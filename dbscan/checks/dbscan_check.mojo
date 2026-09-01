# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Launch DBSCAN against a fixture whose clustering is unambiguous.

NO CUML COUNTERPART. Same discipline as the other sections.

THE FIXTURE
-----------
Three blobs of 200 points, centred 10 apart in feature 0, each point jittered
by at most 0.3 in every feature. With `eps = 2`:

- the largest WITHIN-blob squared distance is `4 * 0.6^2 = 1.44 < 4`, so
  every blob is fully connected and every point is core,
- the smallest BETWEEN-blob squared distance is about `9.4^2 = 88 >> 4`, so
  no edge crosses.

Then 12 noise points spread 40 apart, far from the blobs and from each other,
so each has degree 1 (itself) and cannot be core at `min_pts = 5`.

The expected answer is therefore exact and needs no tolerance: three
clusters, each blob whole, twelve points labelled noise.

**Coordinates are kept small on purpose.** An earlier fixture in the k-NN
section put points at a spacing of 100 and the expanded identity's float32
cancellation destroyed it: norms about 1e10 against distances about 1e3.
Here the largest coordinate is 540, so the largest norm is about 3e5 and the
float32 ulp there is 0.03, against an `eps^2` of 4. See `PORTING.md 21`.

WHAT IS COMPARED
----------------
The PARTITION, not the label values. Cluster ids from `weak_cc` are
`min(vertex index) + 1` over each component and the monotonic relabelling
that would renumber them `0..k-1` is not ported. Comparing numbers instead of
the partition would be testing a renumbering convention, and it is the same
reason the k-means check compares centroids as a permutation.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from dbscan.impl.dbscan.adjgraph.algo import (
    exclusive_scan,
    scan_blocks_needed,
)
from dbscan.impl.dbscan.dbscan import (
    compute_batch_size,
    dbscan_fit_impl,
    dbscan_fit_impl_weighted,
)
from dbscan.impl.dbscan.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC
from dbscan.impl.dbscan.runner import dbscan_fit, rbc_take_one_pass
from dbscan.impl.dbscan.vertexdeg.algo import (
    VD_TPB,
    WVD_TPB,
    eps_neighborhood_kernel,
    vertex_deg_dispatch,
    weighted_vertex_deg_csr,
    weighted_vertex_deg_dense,
)
from dbscan.impl.neighbors.epsilon_neighborhood import (
    DBSCAN_METRIC_L1,
    DBSCAN_METRIC_L2,
    EPS_MBLK,
    EPS_NBLK,
    EPS_THREADS,
    dbscan_metric_threshold,
    eps_unexp_neigh_kernel,
)
from dbscan.impl.sparse.detail.csr import MAX_LABEL
from checks.kernel_matrix import (
    COLUMN_BIT_IDENTICAL,
    K_LIB_WEIGHTED_VERTEX_DEG,
    TARGET_COLUMN,
    lib_block_bounds_a_float_fold,
    lib_block_size_for,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz


comptime BLOBS = 3
comptime PER_BLOB = 200
comptime NOISE = 12
comptime DB_ROWS = BLOBS * PER_BLOB + NOISE
comptime DB_FEATURES = 4
comptime DB_EPS = 2.0
comptime DB_MIN_PTS = 5


def _jitter(row: Int, feature: Int) -> Float64:
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(feature + 1) * 0xBF58476D1CE4E5B9
        + 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return (Float64(z >> 11) * (1.0 / 9007199254740992.0) - 0.5) * 0.6


def _coord(row: Int, feature: Int) -> Float32:
    if row < BLOBS * PER_BLOB:
        var blob = row // PER_BLOB
        var base = 10.0 * Float64(blob) if feature == 0 else 0.0
        return Float32(base + _jitter(row, feature))
    var k = row - BLOBS * PER_BLOB
    var base2 = 100.0 + 40.0 * Float64(k) if feature == 0 else 0.0
    return Float32(base2)


def _no_weights(ctx: DeviceContext) raises -> DeviceBuffer[DType.float32]:
    """A one-element placeholder standing in for `sample_weight == nullptr`.

    Mojo has no null `DeviceBuffer`, so `dbscan_fit` carries a buffer beside
    the `has_weights` Bool that decides whether it is read. An unweighted
    call passes this and leaves `has_weights` at its `False` default.
    """
    var b = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    return b


def _load_fixture(ctx: DeviceContext, n: Int, d: Int) raises -> DeviceBuffer[
    DType.float32
]:
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _coord(i, f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    return x


def check_dbscan() raises:
    var ctx = DeviceContext()
    var n = DB_ROWS
    var d = DB_FEATURES

    var x = _load_fixture(ctx, n, d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()

    var passes = dbscan_fit_impl(ctx, x, labels, n, d, DB_EPS, DB_MIN_PTS)

    var hl = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    # --- each blob is ONE cluster ----------------------------------------
    var blob_label = List[Int32]()
    for b in range(BLOBS):
        var first = hl.unsafe_ptr().unsafe_load(b * PER_BLOB)
        blob_label.append(first)
        for k in range(PER_BLOB):
            var got = hl.unsafe_ptr().unsafe_load(b * PER_BLOB + k)
            if got != first:
                raise Error(
                    "blob " + String(b) + " split: point " + String(k)
                    + " has label " + String(got) + " against "
                    + String(first)
                )

    # --- the blobs are DIFFERENT clusters --------------------------------
    for a in range(BLOBS):
        for b in range(a + 1, BLOBS):
            if blob_label[a] == blob_label[b]:
                raise Error(
                    "blobs " + String(a) + " and " + String(b)
                    + " were merged, which no edge in the graph permits"
                )

    # --- noise is -1, and the cluster ids are EXACTLY 0..k-1 -------------
    # This is what `final_relabel` + `relabelForSkl` (`runner.cuh:410-416`)
    # buy, and it is the half that could not be checked before they were
    # ported: the old port compared the PARTITION because its label VALUES
    # were `min(vertex index) + 1` and matched neither cuML nor sklearn.
    for i in range(BLOBS * PER_BLOB, n):
        if hl.unsafe_ptr().unsafe_load(i) != Int32(-1):
            raise Error(
                "noise point " + String(i) + " has label "
                + String(hl.unsafe_ptr().unsafe_load(i))
                + ", and scikit-learn's noise label is -1"
            )
    var seen = List[Int]()
    for _c in range(BLOBS):
        seen.append(0)
    for b in range(BLOBS):
        var v = Int(blob_label[b])
        if v < 0 or v >= BLOBS:
            raise Error(
                "cluster label " + String(v) + " is outside 0.."
                + String(BLOBS - 1) + "; final_relabel did not run"
            )
        seen[v] = seen[v] + 1
    for c in range(BLOBS):
        if seen[c] != 1:
            raise Error(
                "cluster id " + String(c) + " was used " + String(seen[c])
                + " times; the relabelling is not a bijection onto 0.."
                + String(BLOBS - 1)
            )

    print(
        "check_dbscan OK: 3/3 blobs each one whole cluster with ids exactly"
        " {0, 1, 2}, 0 merges, "
        + String(NOISE)
        + "/"
        + String(NOISE)
        + " isolated points labelled -1 as scikit-learn does, converged in "
        + String(passes)
        + " propagation passes"
    )


def check_dbscan_eps_sensitivity() raises:
    """The reach test, and it is an INVARIANT rather than a corruption.

    Raise `eps` past the blob separation and the three clusters MUST merge
    into one. There is no way to produce one cluster at `eps = 12` and three
    at `eps = 2` without the radius test actually running on real distances:
    a no-op neighbourhood gives the same answer at both.
    """
    var ctx = DeviceContext()
    var n = DB_ROWS
    var d = DB_FEATURES

    var x = _load_fixture(ctx, n, d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()

    _ = dbscan_fit_impl(ctx, x, labels, n, d, 12.0, DB_MIN_PTS)
    var hl = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    var first = hl.unsafe_ptr().unsafe_load(0)
    for i in range(BLOBS * PER_BLOB):
        if hl.unsafe_ptr().unsafe_load(i) != first:
            raise Error(
                "at eps = 12 the three blobs did NOT merge; point "
                + String(i) + " has a different label. The radius test is"
                " not reading real distances."
            )
    print(
        "check_dbscan_eps_sensitivity OK: eps=2 gives 3 clusters and eps=12"
        " gives 1, which is impossible without the neighbourhood kernel"
        " running on real distances"
    )


def check_exclusive_scan_beyond_the_old_cap() raises:
    """Regression for a SILENT correctness bug, plus the multi-block scan.

    Two caps have lived in this kernel. The first gave each of `SCAN_TPB`
    threads a FIXED 64 rows, capping it at 16,384: past that it stopped
    scanning, returned a wrong `nnz` and a truncated CSR, and raised nothing.
    The second was the launch: `grid_dim = (1, 1, 1)`, one threadgroup for
    the whole array, which was correct and serial.

    So this runs the DEVICE-WIDE scan at 2,000,000 entries -- 977 blocks of
    the first pass, which is the only size at which pass 2 and pass 3 can be
    wrong -- and diffs every entry against a host scan.

    The lesson generalizes past this kernel: a launch geometry that encodes a
    maximum size needs a test at that size, or the maximum is a trapdoor.
    """
    var ctx = DeviceContext()
    var n = 2000000

    var vd = ctx.enqueue_create_buffer[DType.int32](n)
    var ex = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](
        scan_blocks_needed(n) + 1
    )
    ctx.synchronize()

    var hv = ctx.enqueue_create_host_buffer[DType.int32](n)
    for i in range(n):
        hv.unsafe_ptr().unsafe_store(i, Int32((i * 7 + 3) % 13))
    ctx.enqueue_copy(dst_buf=vd, src_ptr=hv.unsafe_ptr())
    ctx.synchronize()

    exclusive_scan(ctx, ex, vd, block_sums, n)
    ctx.synchronize()

    var he = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    ctx.enqueue_copy(dst_ptr=he.unsafe_ptr(), src_buf=ex)
    ctx.synchronize()

    var running = 0
    var wrong = 0
    for i in range(n):
        if Int(he.unsafe_ptr().unsafe_load(i)) != running:
            wrong += 1
        running += Int(hv.unsafe_ptr().unsafe_load(i))
    var total = Int(he.unsafe_ptr().unsafe_load(n))

    if wrong != 0:
        raise Error(
            String(wrong) + " of " + String(n)
            + " exclusive-scan entries are wrong at n = " + String(n)
            + ", across " + String(scan_blocks_needed(n)) + " blocks"
        )
    if total != running:
        raise Error(
            "the scan total is " + String(total) + " against " + String(running)
        )
    print(
        "check_exclusive_scan_beyond_the_old_cap OK: "
        + String(n)
        + " entries exact across "
        + String(scan_blocks_needed(n))
        + " blocks, total "
        + String(total)
    )


def check_dbscan_batching_agrees() raises:
    """Batched and unbatched must return the SAME labels, and the batched run
    must fit in a buffer the unbatched run could not.

    Both halves matter. Equality alone would pass if `batch_size` were
    ignored, so this also allocates `adj` at `batch x N` instead of `N x N`.
    A run that secretly ignored the batching would write off the end of that
    buffer, not quietly agree.

    Since `weak_cc` now runs PER BATCH and `merge_labels` folds the results
    (`runner.cuh:374-400`), this is also the only check on the merge: with
    128-row batches over 612 points, four merges happen and every blob spans
    at least two batches, so a broken merge splits a blob.
    """
    var ctx = DeviceContext()
    var n = DB_ROWS
    var d = DB_FEATURES
    var batch = 128

    var x = _load_fixture(ctx, n, d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    var labels_temp = ctx.enqueue_create_buffer[DType.int32](n)
    var work_buffer = ctx.enqueue_create_buffer[DType.int32](n)
    var core = ctx.enqueue_create_buffer[DType.uint8](n)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](
        scan_blocks_needed(n) + 1
    )
    # FULL-SIZE buffers for the reference run.
    var adj_full = ctx.enqueue_create_buffer[DType.uint8](n * n)
    var vd_full = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var ex_full = ctx.enqueue_create_buffer[DType.int32](n + 1)
    # BATCH-SIZE buffers for the batched run. Deliberately too small for a
    # run that ignored `batch_size`.
    var adj_b = ctx.enqueue_create_buffer[DType.uint8](batch * n)
    var vd_b = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var ex_b = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    ctx.synchronize()

    # `dbscan_fit` takes `sample_weight` and `wght_sum` since 2026-09-01.
    # `_no_weights` is their `sample_weight == nullptr`: a one-element
    # placeholder that the `has_weights = False` default leaves unread.
    var nw0 = _no_weights(ctx)
    var ws0 = _no_weights(ctx)
    _ = dbscan_fit(
        ctx, x, adj_full, vd_full, core, ex_full, labels, labels_temp,
        work_buffer, block_sums, nw0, ws0, n, d, DB_EPS, DB_MIN_PTS, 0,
    )
    var baseline = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=baseline.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    var nw1 = _no_weights(ctx)
    var ws1 = _no_weights(ctx)
    _ = dbscan_fit(
        ctx, x, adj_b, vd_b, core, ex_b, labels, labels_temp,
        work_buffer, block_sums, nw1, ws1, n, d, DB_EPS, DB_MIN_PTS, batch,
    )
    var got = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    var differ = 0
    for i in range(n):
        if baseline.unsafe_ptr().unsafe_load(i) != got.unsafe_ptr().unsafe_load(
            i
        ):
            differ += 1
    if differ != 0:
        raise Error(
            String(differ) + " of " + String(n)
            + " labels differ between one batch and "
            + String((n + batch - 1) // batch)
            + " batches. Batching changes the memory, not the answer."
        )
    print(
        "check_dbscan_batching_agrees OK: "
        + String((n + batch - 1) // batch)
        + " batches with "
        + String((n + batch - 1) // batch - 1)
        + " merge_labels folds give labels identical to one batch, in an adj"
        " buffer "
        + String(n // batch)
        + "x smaller than the unbatched run needs"
    )


def check_fused_eps_agrees_with_materialized() raises:
    """The FUSED neighborhood against the materialized one, CELL BY CELL.

    `dbscan/gbdt/neighbors/epsilon_neighborhood.mojo` replaced a three-
    kernel path (`gemm_nt` -> `expand_distances_kernel` ->
    `eps_neighborhood_kernel`) with one kernel that never writes a float. The
    old path stays reachable precisely so this check can exist. That is what a
    second implementation is FOR: diffing a fused kernel against a
    materialized one. It is not the same as keeping an unfused kernel beside
    an unfused library call, which is what `core/gemm.mojo` was doing and no
    longer does.

    **This compares every cell, not a count.** A check whose expected value is
    the same everywhere verifies the total and nothing about placement, and
    that failure mode has already passed a wrong-reduction bug in this
    repository twice. So the fixture is hashed coordinates with a radius
    chosen to make the adjacency roughly half full and IRREGULAR, and the
    comparison is per `(i, j)`, plus the degrees, plus the total.

    The shape is deliberately not a multiple of the 64x64 tile in either
    axis, so the boundary guards in both kernels are exercised.
    """
    var ctx = DeviceContext()
    var m = 150
    var n = 213
    var d = 9
    var eps_sq = Float32(2.0)

    var xb = ctx.enqueue_create_buffer[DType.float32](m * d)
    var yb = ctx.enqueue_create_buffer[DType.float32](n * d)
    var yb2 = ctx.enqueue_create_buffer[DType.float32](n * d)
    var xn = ctx.enqueue_create_buffer[DType.float32](m)
    var yn = ctx.enqueue_create_buffer[DType.float32](n)
    var dist = ctx.enqueue_create_buffer[DType.float32](m * n)
    var adj_ref = ctx.enqueue_create_buffer[DType.uint8](m * n)
    var adj_new = ctx.enqueue_create_buffer[DType.uint8](m * n)
    var vd_ref = ctx.enqueue_create_buffer[DType.int32](m)
    var vd_new = ctx.enqueue_create_buffer[DType.int32](m + 1)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](m * d)
    for i in range(m):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, Float32(_jitter(i, f) * 2.0))
    ctx.enqueue_copy(dst_buf=xb, src_ptr=hx.unsafe_ptr())
    var hy = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hy.unsafe_ptr().unsafe_store(
                i * d + f, Float32(_jitter(i + 5000, f) * 2.0)
            )
    ctx.enqueue_copy(dst_buf=yb, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=yb2, src_ptr=hy.unsafe_ptr())
    ctx.synchronize()

    # --- the OLD path: norms, GEMM, expand, threshold -------------------
    ctx.enqueue_function[row_norm_kernel](
        xn.unsafe_ptr(), xb.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(m, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.enqueue_function[row_norm_kernel](
        yn.unsafe_ptr(), yb.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.synchronize()
    gemm_nt(ctx, dist, xb, yb2, m, n, d)
    ctx.enqueue_function[expand_distances_kernel](
        dist.unsafe_ptr(), xn.unsafe_ptr(), yn.unsafe_ptr(),
        Int32(m), Int32(n), Int32(0),
        grid_dim=((m * n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[eps_neighborhood_kernel](
        adj_ref.unsafe_ptr(), vd_ref.unsafe_ptr(), dist.unsafe_ptr(),
        Int32(n), eps_sq,
        grid_dim=(m, 1, 1), block_dim=(VD_TPB, 1, 1),
    )
    ctx.synchronize()

    # --- the FUSED path -------------------------------------------------
    ctx.enqueue_memset(vd_new, Int32(0))
    ctx.synchronize()
    ctx.enqueue_function[eps_unexp_neigh_kernel[DBSCAN_METRIC_L2]](
        adj_new.unsafe_ptr(), vd_new.unsafe_ptr(),
        xb.unsafe_ptr(), yb.unsafe_ptr(),
        Int32(m), Int32(n), Int32(d), eps_sq,
        grid_dim=(
            (m + EPS_MBLK - 1) // EPS_MBLK,
            (n + EPS_NBLK - 1) // EPS_NBLK,
            1,
        ),
        block_dim=(EPS_THREADS, 1, 1),
    )
    ctx.synchronize()

    var ha = ctx.enqueue_create_host_buffer[DType.uint8](m * n)
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](m * n)
    var hvr = ctx.enqueue_create_host_buffer[DType.int32](m)
    var hvn = ctx.enqueue_create_host_buffer[DType.int32](m + 1)
    ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=adj_ref)
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=adj_new)
    ctx.enqueue_copy(dst_ptr=hvr.unsafe_ptr(), src_buf=vd_ref)
    ctx.enqueue_copy(dst_ptr=hvn.unsafe_ptr(), src_buf=vd_new)
    ctx.synchronize()

    var cells = 0
    var ones = 0
    for i in range(m):
        for j in range(n):
            var a = Int(ha.unsafe_ptr().unsafe_load(i * n + j))
            var b = Int(hb.unsafe_ptr().unsafe_load(i * n + j))
            if a != 0:
                ones += 1
            if a != b:
                if cells < 4:
                    print(
                        "  first mismatches: (" + String(i) + ", " + String(j)
                        + ") materialized " + String(a) + " fused " + String(b)
                    )
                cells += 1
    if cells != 0:
        raise Error(
            String(cells) + " of " + String(m * n)
            + " adjacency cells disagree between the fused kernel and the"
            " materialized path"
        )
    if ones == 0 or ones == m * n:
        raise Error(
            "the fixture is degenerate: " + String(ones) + " of "
            + String(m * n) + " cells are neighbours, so agreeing everywhere"
            " proves nothing about placement"
        )

    # --- AND against a HOST oracle, which depends on nothing of ours ----
    # The materialized arm above shares `core/gemm.mojo` and
    # `core/expand_distances.mojo` with the fused kernel's ancestry, so the
    # two could in principle be wrong together. This one is float64 on the
    # host, straight from the coordinates, and agrees with neither by
    # construction. It is an ORACLE, not a CPU path: nothing ships through
    # it.
    var host_wrong = 0
    for i in range(m):
        for j in range(n):
            var acc = Float64(0.0)
            for f in range(d):
                var df = Float64(
                    hx.unsafe_ptr().unsafe_load(i * d + f)
                ) - Float64(hy.unsafe_ptr().unsafe_load(j * d + f))
                acc += df * df
            var want = 1 if acc <= Float64(eps_sq) else 0
            if want != Int(hb.unsafe_ptr().unsafe_load(i * n + j)):
                host_wrong += 1
    if host_wrong != 0:
        raise Error(
            String(host_wrong) + " of " + String(m * n)
            + " fused adjacency cells disagree with a float64 host oracle"
        )

    var total = 0
    for i in range(m):
        var vr = Int(hvr.unsafe_ptr().unsafe_load(i))
        var vn = Int(hvn.unsafe_ptr().unsafe_load(i))
        if vr != vn:
            raise Error(
                "degree of row " + String(i) + " is " + String(vr)
                + " materialized against " + String(vn) + " fused"
            )
        total += vr
    if Int(hvn.unsafe_ptr().unsafe_load(m)) != total:
        raise Error(
            "vd[m] is " + String(Int(hvn.unsafe_ptr().unsafe_load(m)))
            + " against a row-degree total of " + String(total)
            + "; runner.cuh:281 reads that element to size the CSR"
        )

    print(
        "check_fused_eps_agrees_with_materialized OK: " + String(m * n)
        + " cells identical to the materialized path AND to a float64 host"
        " oracle (" + String(ones) + " of them neighbours, so the pattern is"
        " irregular), " + String(m) + " degrees identical, vd[m] = "
        + String(total)
    )


def check_dbscan_rbc_matches_brute() raises:
    """The INDEX and the brute force must label identically. No tolerance.

    `cuml/cpp/src/dbscan/vertexdeg/algo.cuh:226-231` is one `if`: the
    ball-cover index on one side, `epsUnexpL2SqNeighborhood` on the other.
    Both are supposed to answer the same question, so the only honest test of
    the index is that swapping the branch changes nothing about the ANSWER.

    Two clusterings agreeing is a strong assertion here because `final_relabel`
    + `relabelForSkl` (`runner.cuh:410-416`) make the ids canonical: labels are
    `0..k-1` in first-appearance order and noise is `-1`, so equality is
    literal and not up to permutation.

    THE FAILURE THIS EXISTS TO CATCH is an index that drops a real neighbor.
    That is silent: the point still gets a label, just possibly its own
    cluster instead of its neighbor's. It cannot be seen by counting clusters
    or by eyeballing inertia, only by comparing the partition point by point.
    """
    var ctx = DeviceContext()
    var n = DB_ROWS
    var d = DB_FEATURES

    var x_b = _load_fixture(ctx, n, d)
    var lab_b = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    _ = dbscan_fit_impl(
        ctx, x_b, lab_b, n, d, DB_EPS, DB_MIN_PTS, 0, 200,
        EPS_NN_BRUTE_FORCE,
    )
    var hb = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=lab_b)
    ctx.synchronize()

    var x_r = _load_fixture(ctx, n, d)
    var lab_r = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    _ = dbscan_fit_impl(
        ctx, x_r, lab_r, n, d, DB_EPS, DB_MIN_PTS, 0, 200, EPS_NN_RBC
    )
    var hr = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=lab_r)
    ctx.synchronize()

    var wrong = 0
    var first_bad = -1
    for i in range(n):
        if hb.unsafe_ptr().unsafe_load(i) != hr.unsafe_ptr().unsafe_load(i):
            wrong += 1
            if first_bad < 0:
                first_bad = i
    if wrong != 0:
        raise Error(
            "check_dbscan_rbc_matches_brute: the ball-cover index and the"
            " brute force disagree on "
            + String(wrong)
            + " of "
            + String(n)
            + " points, first at "
            + String(first_bad)
            + " (brute "
            + String(hb.unsafe_ptr().unsafe_load(first_bad))
            + " vs rbc "
            + String(hr.unsafe_ptr().unsafe_load(first_bad))
            + "). The index is dropping or inventing neighbours."
        )

    # The comparison is only worth something if it saw real structure. An
    # all-noise or one-cluster labelling would match trivially.
    var seen = 0
    for i in range(n):
        if hb.unsafe_ptr().unsafe_load(i) > Int32(seen - 1):
            seen = Int(hb.unsafe_ptr().unsafe_load(i)) + 1
    if seen < 2:
        raise Error(
            "check_dbscan_rbc_matches_brute: only "
            + String(seen)
            + " cluster(s) in the brute-force labelling, so agreement proves"
            " nothing. Pick a fixture and eps with real structure."
        )

    print(
        "check_dbscan_rbc_matches_brute OK:",
        n,
        "points labelled identically by the ball-cover index and by brute"
        " force,",
        seen,
        "clusters",
    )


comptime WIDE_BLOBS = 6
comptime WIDE_PER_BLOB = 320
comptime WIDE_NOISE = 12
comptime WIDE_ROWS = WIDE_BLOBS * WIDE_PER_BLOB + WIDE_NOISE


def _wide_coord(row: Int, feature: Int) -> Float32:
    """Same construction as `_coord`, six blobs instead of three.

    Hashed jitter per (row, feature) -- scattered, not uniform -- so no two
    rows have the same neighbour set and a placement bug cannot cancel. Blob
    centres 10 apart in feature 0 against eps = 2; noise from 100 on, 40
    apart, so blob 5 (feature 0 near 50) is 50 away from the first noise
    point.
    """
    if row < WIDE_BLOBS * WIDE_PER_BLOB:
        var blob = row // WIDE_PER_BLOB
        var base = 10.0 * Float64(blob) if feature == 0 else 0.0
        return Float32(base + _jitter(row, feature))
    var k = row - WIDE_BLOBS * WIDE_PER_BLOB
    var base2 = 100.0 + 40.0 * Float64(k) if feature == 0 else 0.0
    return Float32(base2)


def check_dbscan_max_mbytes_moves_the_batch() raises:
    """REACH: `max_mbytes_per_batch` must change the computed batch, and
    `eps_nn_method` must change the clamp. Host arithmetic only --
    `compute_batch_size` allocates nothing, exactly like theirs.

    Two assertions, each on a branch the suite did not previously run:

    1. Two budgets at one n give DIFFERENT batch counts. A signature that
       ignored its budget would return the same batch twice.
    2. `dbscan.cuh:71`: at an n where the int32 clamp binds, RBC and
       BRUTE_FORCE must DISAGREE -- RBC keeps the budget's answer, brute is
       clamped to `MAX_LABEL / n_rows` -- and at an n where it cannot bind
       they must AGREE. A gate stuck on either side fails one of the two.
    """
    var n = WIDE_ROWS

    # (1) the budget is load-bearing.
    var b_tiny = compute_batch_size(n, n, EPS_NN_RBC, 1)
    var b_big = compute_batch_size(n, n, EPS_NN_RBC, 1000)
    var nb_tiny = (n + b_tiny - 1) // b_tiny
    var nb_big = (n + b_big - 1) // b_big
    if nb_big != 1:
        raise Error(
            "a 1000 MB budget for " + String(n) + " rows should be one"
            " batch, got " + String(nb_big)
        )
    if nb_tiny <= 4:
        raise Error(
            "a 1 MB budget for " + String(n) + " rows should force many"
            " batches, got " + String(nb_tiny)
        )

    # (2) the `eps_nn_method != RBC` gate (`dbscan.cuh:71`).
    var big_n = 50000
    var rbc_batch = compute_batch_size(big_n, big_n, EPS_NN_RBC, 100000)
    var brute_batch = compute_batch_size(
        big_n, big_n, EPS_NN_BRUTE_FORCE, 100000
    )
    var clamp = 2147483647 // big_n
    if rbc_batch != big_n:
        raise Error(
            "RBC with a 100 GB budget should take the whole " + String(big_n)
            + " rows in one batch, got " + String(rbc_batch)
        )
    if brute_batch != clamp:
        raise Error(
            "BRUTE_FORCE at n = " + String(big_n) + " must clamp to"
            " MAX_LABEL / n = " + String(clamp) + ", got "
            + String(brute_batch)
        )
    # Below the clamp the two methods must agree, or the gate leaks into the
    # estimate itself.
    var rbc_small = compute_batch_size(n, n, EPS_NN_RBC, 3)
    var brute_small = compute_batch_size(n, n, EPS_NN_BRUTE_FORCE, 3)
    if rbc_small != brute_small:
        raise Error(
            "the eps_nn_method gate changed an UNCLAMPED batch: rbc "
            + String(rbc_small) + " vs brute " + String(brute_small)
        )

    print(
        "check_dbscan_max_mbytes_moves_the_batch OK: 1 MB -> "
        + String(nb_tiny) + " batches and 1000 MB -> 1 batch at n = "
        + String(n) + "; at n = 50000 the dbscan.cuh:71 gate keeps rbc at "
        + String(rbc_batch) + " rows where brute clamps to " + String(clamp)
        + ", and both agree at " + String(rbc_small)
        + " rows when the clamp cannot bind"
    )


def check_dbscan_tiny_budget_agrees() raises:
    """A user-forced tiny `max_mbytes_per_batch` must change the batching and
    nothing else, END TO END through `dbscan_fit_impl` -- the plumbing check
    for the public knob, on BOTH sides of the `algo.cuh:226` switch.

    The fixture is scattered (hashed jitter, six blobs each spanning several
    tiny batches, twelve isolated noise points), so a merge that misplaces
    labels cannot cancel. `final_relabel` + `relabelForSkl` make ids
    canonical, so the comparison is literal equality.

    Reach is asserted twice over: the expected batch counts differ (host
    arithmetic above), and the returned PASS COUNT must be strictly larger
    under many batches, because each batch runs its own `weak_cc_batched`
    and each run is at least one pass. Identical labels with identical pass
    counts would mean the budget never reached the loop.
    """
    var ctx = DeviceContext()
    var n = WIDE_ROWS
    var d = DB_FEATURES

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _wide_coord(i, f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    var nb_tiny = (
        n + compute_batch_size(n, n, EPS_NN_RBC, 1) - 1
    ) // compute_batch_size(n, n, EPS_NN_RBC, 1)

    var labels_one = ctx.enqueue_create_buffer[DType.int32](n)
    var labels_many = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()

    var methods = [EPS_NN_RBC, EPS_NN_BRUTE_FORCE]
    var method_names = [String("rbc"), String("brute")]
    for mi in range(len(methods)):
        var method = methods[mi]
        var passes_one = dbscan_fit_impl(
            ctx, x, labels_one, n, d, DB_EPS, DB_MIN_PTS, 1000, 200, method
        )
        var passes_many = dbscan_fit_impl(
            ctx, x, labels_many, n, d, DB_EPS, DB_MIN_PTS, 1, 200, method
        )

        var h_one = ctx.enqueue_create_host_buffer[DType.int32](n)
        var h_many = ctx.enqueue_create_host_buffer[DType.int32](n)
        ctx.enqueue_copy(dst_ptr=h_one.unsafe_ptr(), src_buf=labels_one)
        ctx.enqueue_copy(dst_ptr=h_many.unsafe_ptr(), src_buf=labels_many)
        ctx.synchronize()

        if passes_many <= passes_one:
            raise Error(
                method_names[mi] + ": " + String(nb_tiny)
                + " batches returned " + String(passes_many)
                + " propagation passes against " + String(passes_one)
                + " for one batch; the budget never reached the batch loop"
            )

        var differ = 0
        for i in range(n):
            if h_one.unsafe_ptr().unsafe_load(i) != h_many.unsafe_ptr().unsafe_load(i):
                differ += 1
        if differ != 0:
            raise Error(
                method_names[mi] + ": " + String(differ) + " of " + String(n)
                + " labels differ between 1 batch and " + String(nb_tiny)
                + " budget-forced batches"
            )

        # The agreement must have seen real structure: six clusters as ids
        # exactly 0..5, noise as -1, no blob split.
        for b in range(WIDE_BLOBS):
            var first = h_many.unsafe_ptr().unsafe_load(b * WIDE_PER_BLOB)
            if Int(first) != b:
                raise Error(
                    method_names[mi] + ": blob " + String(b)
                    + " has canonical label " + String(first)
                    + "; final_relabel should give first-appearance ids 0.."
                    + String(WIDE_BLOBS - 1)
                )
            for k in range(WIDE_PER_BLOB):
                if h_many.unsafe_ptr().unsafe_load(b * WIDE_PER_BLOB + k) != first:
                    raise Error(
                        method_names[mi] + ": blob " + String(b)
                        + " split at point " + String(k)
                    )
        for i in range(WIDE_BLOBS * WIDE_PER_BLOB, n):
            if h_many.unsafe_ptr().unsafe_load(i) != Int32(-1):
                raise Error(
                    method_names[mi] + ": noise point " + String(i)
                    + " has label "
                    + String(h_many.unsafe_ptr().unsafe_load(i))
                )
        print(
            "check_dbscan_tiny_budget_agrees OK (" + method_names[mi]
            + "): max_mbytes_per_batch = 1 forced " + String(nb_tiny)
            + " batches (" + String(passes_many) + " passes vs "
            + String(passes_one) + " for one batch) and all " + String(n)
            + " labels match one batch: " + String(WIDE_BLOBS)
            + " blobs whole with ids 0.." + String(WIDE_BLOBS - 1) + ", "
            + String(WIDE_NOISE) + " noise points at -1"
        )


def _tl_coord(row: Int, feature: Int, dense: Bool) -> Float32:
    """Two blob layouts for `check_dbscan_rbc_two_loop_arms`, hashed jitter.

    `dense = False`: blobs of 250 / 200 / 150 rows at 0 / 10 / 20 in
    feature 0. `dense = True`: blobs of 500 / 100 at 0 / 10. Same `_jitter`
    as `_coord` (at most 0.3 per feature), so within-blob squared distances
    stay under 1.44 against eps^2 = 4 and between-blob ones start near 88 --
    every within-blob pair is a neighbour, no cross-blob pair is, and the
    degrees below are exact by construction rather than by luck.
    """
    var blob: Int
    if dense:
        blob = 0 if row < 500 else 1
    else:
        if row < 250:
            blob = 0
        elif row < 450:
            blob = 1
        else:
            blob = 2
    var base = 10.0 * Float64(blob) if feature == 0 else 0.0
    return Float32(base + _jitter(row, feature))


def _run_two_loop_arm(
    dense: Bool, expect_one_pass: Bool, name: String
) raises -> Int:
    """One fixture, pinned to ONE arm of `algo.cuh:119-122`, labels against
    brute force. Returns the cluster count. See the caller for what each
    fixture pins and why.
    """
    var ctx = DeviceContext()
    var n = 600
    var d = DB_FEATURES
    var batch = 200
    var n_batches = (n + batch - 1) // batch
    var eps = 2.0

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _tl_coord(i, f, dense))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    # Host degrees with the device's own float32 arithmetic, then the
    # ROUTING assertion: every batch after 0 must take the EXPECTED arm of
    # `algo.cuh:119-122`, decided by `rbc_take_one_pass` -- the identical
    # function the runner calls, fed the identical numbers the runner will
    # derive (`maxadjlen` is the largest batch's edge count, `max_k` the
    # batch's longest row). A fixture that stopped pinning its arm fails
    # HERE, not by silently testing the other branch.
    var eps2 = Float32(eps) * Float32(eps)
    var deg = List[Int]()
    for i in range(n):
        var c = 0
        for j in range(n):
            var sd = Float32(0.0)
            for f in range(d):
                var diff = hx.unsafe_ptr().unsafe_load(
                    i * d + f
                ) - hx.unsafe_ptr().unsafe_load(j * d + f)
                sd += diff * diff
            if sd <= eps2:
                c += 1
        deg.append(c)

    var maxadjlen = 1
    for b in range(n_batches):
        var np_b = min(n - b * batch, batch)
        var nnz_b = 0
        for i2 in range(np_b):
            nnz_b += deg[b * batch + i2]
        if nnz_b > maxadjlen:
            maxadjlen = nnz_b
    for b in range(1, n_batches):
        var np_b = min(n - b * batch, batch)
        var mk = 0
        for i2 in range(np_b):
            if deg[b * batch + i2] > mk:
                mk = deg[b * batch + i2]
        var one_pass = rbc_take_one_pass(batch, n, maxadjlen, np_b, mk)
        if one_pass != expect_one_pass:
            raise Error(
                "two-loop " + name + ": batch " + String(b)
                + " routes one_pass=" + String(one_pass) + " (max_k "
                + String(mk) + ", maxadjlen " + String(maxadjlen)
                + "), so the fixture does not pin the arm it claims to"
            )

    # RBC, batched -> loop 2 takes the arm just pinned. Brute force,
    # unbatched, is the reference labelling; `final_relabel` +
    # `relabelForSkl` make the ids canonical, so equality is literal.
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    var labels_temp = ctx.enqueue_create_buffer[DType.int32](n)
    var work_buffer = ctx.enqueue_create_buffer[DType.int32](n)
    var core = ctx.enqueue_create_buffer[DType.uint8](n)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](
        scan_blocks_needed(n) + 1
    )
    var adj_b = ctx.enqueue_create_buffer[DType.uint8](batch * n)
    var vd_b = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var ex_b = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    ctx.synchronize()
    var nw2 = _no_weights(ctx)
    var ws2 = _no_weights(ctx)
    _ = dbscan_fit(
        ctx, x, adj_b, vd_b, core, ex_b, labels, labels_temp, work_buffer,
        block_sums, nw2, ws2, n, d, eps, DB_MIN_PTS, batch, 200, EPS_NN_RBC,
    )
    var hr = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    var adj_f = ctx.enqueue_create_buffer[DType.uint8](n * n)
    var vd_f = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var ex_f = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.synchronize()
    var nw3 = _no_weights(ctx)
    var ws3 = _no_weights(ctx)
    _ = dbscan_fit(
        ctx, x, adj_f, vd_f, core, ex_f, labels, labels_temp, work_buffer,
        block_sums, nw3, ws3, n, d, eps, DB_MIN_PTS, 0, 200,
        EPS_NN_BRUTE_FORCE,
    )
    var hb = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    var wrong = 0
    for i in range(n):
        if hr.unsafe_ptr().unsafe_load(i) != hb.unsafe_ptr().unsafe_load(i):
            wrong += 1
    if wrong != 0:
        raise Error(
            "two-loop " + name + ": " + String(wrong) + " of " + String(n)
            + " labels differ between batched RBC and unbatched brute force"
        )
    var seen = 0
    for i in range(n):
        if hb.unsafe_ptr().unsafe_load(i) > Int32(seen - 1):
            seen = Int(hb.unsafe_ptr().unsafe_load(i)) + 1
    if seen < 2:
        raise Error(
            "two-loop " + name + ": only " + String(seen)
            + " cluster(s), so agreement proves nothing"
        )
    return seen


def check_dbscan_rbc_two_loop_arms() raises:
    """BOTH upstream arms of loop 2's RBC dispatch, one named fixture each.

    PORTING_RULES 8: a switch is exercised on both sides or one side is
    unchecked, and the RBC arm's loop 2 now has a switch --
    `algo.cuh:119-122` sends a batch down the ONE-PASS `max_k` form when
    loop 1's bound fits the spare room, and down the two-pass count + fill
    otherwise. The DBSCAN runner currently disables the one-pass composition
    on Metal because its separately compiled distance loop disagreed at an
    epsilon boundary; the standalone max-k kernel stays checked. The guard is
    data-derived, not a parameter, so each fixture PINS the arm upstream would
    select and the check proves the pin with the runner's own
    `rbc_take_one_pass` on host-recomputed degrees before trusting the
    labels.

    - one-pass: blobs of 250 / 200 / 150 rows, batch 200 -- batches straddle
      blob boundaries (so `merge_labels` works across the new arm's CSR) and
      the longest row (250) is far under the spare room (350).
    - fallback: blobs of 500 / 100, batch 200 -- the 500-blob makes
      `maxadjlen` 100,000 of the 120,000 budgeted, the spare room collapses
      to 100, and the 500-long rows cannot take the one-pass arm.

    Both fits must label every point exactly as unbatched BRUTE_FORCE does.
    """
    var c1 = _run_two_loop_arm(False, True, "one-pass")
    var c2 = _run_two_loop_arm(True, False, "fallback")
    print(
        "check_dbscan_rbc_two_loop_arms OK: the one-pass arm ("
        + String(c1) + " clusters) and the two-pass fallback (" + String(c2)
        + " clusters) both label identically to unbatched brute force, and"
        " algo.cuh:119's guard provably selects each upstream arm"
    )


# ===========================================================================
# THE MANHATTAN ARM (DEVIATION 27) AND THE WEIGHTED CORE TEST (DEVIATION 28)
# ===========================================================================
#
# Added 2026-09-01 with the two features. Every gate below has a SABOTAGE ARM
# in the same function, because a gate never shown capable of failing does not
# count here: each one first proves that the thing it is about to assert could
# have come out differently, and raises "degenerate" if it could not.


# 1.2, NOT 1.0, AND TWO GATES PIN IT THERE.
#
# At eps = 1.0 the squared threshold IS the threshold, so SABOTAGE A, which
# runs the L1 kernel against a squared threshold, could not move a single
# adjacency cell and the gate refused itself as vacuous on 2026-09-01. That
# refusal was correct: with eps = 1 an implementation that squared the L1
# radius by mistake would have passed every cell of this fixture.
#
# 1.2 squares to 1.44, so the two radii admit visibly different sets over a
# fixture spread across roughly [-1.5, 1.5] in two dimensions, and the arm
# has something to detect.
#
# It cannot be just any non-unit value. check_dbscan_manhattan_changes_the_
# labels plants a diagonal bridge at L2 0.98995 and L1 1.4, and needs that
# bridge INSIDE under L2 and OUTSIDE under L1 so the metric is shown to be
# load bearing. A first attempt at 1.4 put the bridge exactly ON the L1
# radius and broke that gate. 1.2 clears both: 1.2 != 1.44, 0.98995 < 1.2,
# and 1.4 > 1.2.
comptime L1_EPS = Float64(1.2)
"""The planted fixture's radius, in the units `_l1_coord` lays out.

Spelled `Float64` rather than left a literal because it crosses into
`dbscan_metric_threshold(metric, eps: Float64)` and into `dbscan_fit_impl`,
and a fixture whose radius changed type between the two would be comparing
two thresholds."""


def _l1_coord(row: Int, feature: Int) -> Float32:
    """THE PLANTED AXIS-ALIGNED FIXTURE, WHERE L1 AND L2 PROVABLY DISAGREE.

    Two dimensions, six points, `eps = 1`:

        group A, on the x axis   (0.0, 0)  (0.4, 0)  (0.8, 0)
        group B, offset (0.7, 0.7) from A's last point
                                 (1.5, 0.7)  (1.9, 0.7)  (2.3, 0.7)

    WITHIN a group consecutive points are 0.4 apart along ONE axis, so their
    L1 and their L2 distances are both 0.4 and both metrics connect them.

    THE BRIDGE is the one DIAGONAL step, `(0.8, 0)` to `(1.5, 0.7)`:

        dx = 0.7, dy = 0.7
        L2 = sqrt(0.49 + 0.49) = 0.98995  <= 1   -> a neighbour
        L1 = 0.7 + 0.7         = 1.4      >  1   -> NOT a neighbour

    so under `euclidean` the six points are ONE cluster and under
    `manhattan` they are TWO. The margins are 0.01 and 0.4 against a
    threshold of 1, which is enormous next to a float32 ulp at this
    magnitude, so no rounding decides this fixture -- the METRIC does.

    The next-nearest cross pair is `(0.8, 0)` to `(1.9, 0.7)`, at L2 1.3, so
    the bridge is the only edge either metric could disagree about.

    `min_samples = 2` makes every point core under both metrics (each has
    itself and at least one in-group neighbour), so nothing here is noise
    and the difference the check reads is the PARTITION and not a
    core-point accident.
    """
    var g = row // 3
    var k = row - g * 3
    var base = 0.4 * Float64(k)
    if g == 0:
        return Float32(base) if feature == 0 else Float32(0.0)
    return Float32(1.5 + base) if feature == 0 else Float32(0.7)


def _l1_host_dist(
    mut hx: HostBuffer[DType.float32], i: Int, j: Int, d: Int, metric: Int
) -> Float64:
    """The float64 host oracle for one pair. Depends on nothing of ours.

    Ascending over the features in both arms, which is the same k order the
    kernel's `Policy4x4` walk takes, so a disagreement is arithmetic and not
    a summation order.
    """
    var acc = Float64(0.0)
    for f in range(d):
        var df = Float64(hx.unsafe_ptr().unsafe_load(i * d + f)) - Float64(
            hx.unsafe_ptr().unsafe_load(j * d + f)
        )
        if metric == DBSCAN_METRIC_L1:
            acc += abs(df)
        else:
            acc += df * df
    return acc


def check_dbscan_manhattan_neighborhood() raises:
    """The L1 eps neighborhood, CELL BY CELL, against a float64 host oracle,
    plus the two sabotages that make it mean something.

    THE FIXTURE is the six planted points of `_l1_coord` followed by 144
    hashed points, so the adjacency is both PROVABLE where it matters and
    IRREGULAR everywhere else. A check whose expected adjacency is uniform
    verifies a total and nothing about placement, and that failure mode has
    passed a wrong-reduction bug in this repository twice
    (`check_fused_eps_agrees_with_materialized` says so at length).

    WHAT IS ASSERTED
      1. every cell of the L1 adjacency equals the float64 host oracle's,
      2. the L1 and L2 adjacencies DIFFER, and differ at the planted bridge
         in the predicted direction (L2 yes, L1 no), so the `metric`
         parameter reaches the arithmetic rather than being accepted and
         dropped,
      3. SABOTAGE A -- the L1 kernel run against a SQUARED threshold, which
         is the exact mistake DEVIATION 27 exists to prevent, produces a
         DIFFERENT adjacency. If it did not, `dbscan_metric_threshold`'s
         branch would be untested and an L1 fit could silently carry an
         eps^2 radius,
      4. SABOTAGE B -- the L2 kernel run against the L1 threshold likewise
         differs, so the two arms are not accidentally the same kernel.
    """
    var ctx = DeviceContext()
    var planted = 6
    var hashed = 144
    var m = planted + hashed
    var d = 2

    var hx = ctx.enqueue_create_host_buffer[DType.float32](m * d)
    for i in range(planted):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _l1_coord(i, f))
    for i in range(planted, m):
        for f in range(d):
            # Spread over roughly [-1.5, 1.5] so an eps of 1 admits some
            # pairs under L1 and more under L2, and neither all nor none.
            hx.unsafe_ptr().unsafe_store(
                i * d + f, Float32(_jitter(i * 7 + f, f) * 5.0)
            )
    var x = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    var thresh_l1 = dbscan_metric_threshold(DBSCAN_METRIC_L1, L1_EPS)
    var thresh_l2 = dbscan_metric_threshold(DBSCAN_METRIC_L2, L1_EPS)
    if thresh_l1 != Float32(L1_EPS):
        raise Error(
            "dbscan_metric_threshold squared the L1 threshold: got "
            + String(thresh_l1) + " for eps " + String(L1_EPS)
        )
    if thresh_l2 != Float32(L1_EPS * L1_EPS):
        raise Error(
            "dbscan_metric_threshold did not square the L2 threshold: got "
            + String(thresh_l2)
        )

    var a_l1 = _run_eps[DBSCAN_METRIC_L1](ctx, x, m, d, thresh_l1)
    var a_l2 = _run_eps[DBSCAN_METRIC_L2](ctx, x, m, d, thresh_l2)
    var a_sabA = _run_eps[DBSCAN_METRIC_L1](ctx, x, m, d, thresh_l2)
    var a_sabB = _run_eps[DBSCAN_METRIC_L2](ctx, x, m, d, thresh_l1)

    # --- 1. cell by cell against the float64 host oracle -----------------
    var wrong = 0
    var ones = 0
    for i in range(m):
        for j in range(m):
            var want = 1 if _l1_host_dist(
                hx, i, j, d, DBSCAN_METRIC_L1
            ) <= Float64(L1_EPS) else 0
            var got = Int(a_l1.unsafe_ptr().unsafe_load(i * m + j))
            if want != 0:
                ones += 1
            if want != got:
                if wrong < 4:
                    print(
                        "  first L1 mismatches: (" + String(i) + ", "
                        + String(j) + ") oracle " + String(want) + " kernel "
                        + String(got)
                    )
                wrong += 1
    if wrong != 0:
        raise Error(
            String(wrong) + " of " + String(m * m)
            + " L1 adjacency cells disagree with a float64 host oracle"
        )
    if ones <= m or ones == m * m:
        raise Error(
            "the L1 fixture is degenerate: " + String(ones) + " of "
            + String(m * m) + " cells are neighbours (the diagonal alone is "
            + String(m) + "), so agreeing everywhere proves nothing"
        )

    # --- 2. the metric is LOAD BEARING, and at the planted bridge --------
    var metric_moves = 0
    for e in range(m * m):
        if a_l1.unsafe_ptr().unsafe_load(e) != a_l2.unsafe_ptr().unsafe_load(
            e
        ):
            metric_moves += 1
    if metric_moves == 0:
        raise Error(
            "the L1 and L2 adjacencies are identical on this fixture, so"
            " nothing here shows the metric parameter reaches the kernel"
        )
    # rows 2 and 3 are `(0.8, 0)` and `(1.5, 0.7)`, the bridge.
    if a_l2.unsafe_ptr().unsafe_load(2 * m + 3) != UInt8(1):
        raise Error(
            "the planted bridge (0.8, 0) -- (1.5, 0.7) is at L2 0.98995 and"
            " must be inside eps = 1 under euclidean, and is not"
        )
    if a_l1.unsafe_ptr().unsafe_load(2 * m + 3) != UInt8(0):
        raise Error(
            "the planted bridge is at L1 1.4 and must be OUTSIDE eps = 1"
            " under manhattan, and is not: the L1 arm is computing"
            " something else"
        )

    # --- 3/4. the two sabotages --------------------------------------------
    var sab_a = 0
    var sab_b = 0
    for e in range(m * m):
        if a_l1.unsafe_ptr().unsafe_load(e) != a_sabA.unsafe_ptr().unsafe_load(
            e
        ):
            sab_a += 1
        if a_l2.unsafe_ptr().unsafe_load(e) != a_sabB.unsafe_ptr().unsafe_load(
            e
        ):
            sab_b += 1
    if sab_a == 0:
        raise Error(
            "SABOTAGE A DID NOT MOVE: the L1 kernel gave the same adjacency"
            " with a SQUARED threshold as with eps itself, so nothing in"
            " this check would notice DEVIATION 27's mistake"
        )
    if sab_b == 0:
        raise Error(
            "SABOTAGE B DID NOT MOVE: the L2 kernel gave the same adjacency"
            " with the L1 threshold, so the two thresholds are not"
            " distinguishable on this fixture"
        )

    print(
        "check_dbscan_manhattan_neighborhood OK: " + String(m * m)
        + " L1 cells identical to a float64 host oracle (" + String(ones)
        + " neighbours), L1 vs L2 differ in " + String(metric_moves)
        + " cells including the planted bridge, sabotage A moved "
        + String(sab_a) + " cells and sabotage B moved " + String(sab_b)
    )


def _run_eps[
    metric: Int
](
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    d: Int,
    thresh: Float32,
) raises -> HostBuffer[DType.uint8]:
    """One full `m x m` eps-neighborhood, read back. Query rows ARE the
    dataset, which is the shape a one-batch DBSCAN fit runs."""
    var adj = ctx.enqueue_create_buffer[DType.uint8](m * m)
    var vd = ctx.enqueue_create_buffer[DType.int32](m + 1)
    ctx.synchronize()
    ctx.enqueue_memset(vd, Int32(0))
    ctx.synchronize()
    # x is BOTH operands here (a self-join). Taking `unsafe_ptr()` twice in
    # one argument list is two mutable borrows of one value, which Mojo
    # refuses as aliasing. Bind the raw pointer once and pass it twice; the
    # kernel only reads it.
    # A SELF-JOIN NEEDS TWO HANDLES, not one pointer passed twice. The
    # production launcher takes xb and x as separate buffers
    # (epsilon_neighborhood.mojo:545); this check joins x with itself, and
    # `x.unsafe_ptr()` twice in one argument list is two mutable borrows of
    # one value, which Mojo refuses as aliasing. A zero-offset sub-buffer is
    # a distinct handle over the same bytes, which is exactly the shape the
    # kernel expects and costs nothing.
    var _xb = x.create_sub_buffer[DType.float32](0, m * d)
    ctx.enqueue_function[eps_unexp_neigh_kernel[metric]](
        adj.unsafe_ptr(), vd.unsafe_ptr(), _xb.unsafe_ptr(), x.unsafe_ptr(),
        Int32(m), Int32(m), Int32(d), thresh,
        grid_dim=(
            (m + EPS_MBLK - 1) // EPS_MBLK,
            (m + EPS_NBLK - 1) // EPS_NBLK,
            1,
        ),
        block_dim=(EPS_THREADS, 1, 1),
    )
    ctx.synchronize()
    var h = ctx.enqueue_create_host_buffer[DType.uint8](m * m)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=adj)
    ctx.synchronize()
    return h


def check_dbscan_manhattan_changes_the_labels() raises:
    """END TO END: the same six points, two metrics, two different answers.

    The neighborhood check above proves the KERNEL sees the metric. This
    proves the metric survives `vertex_deg_dispatch`, the runner, the CSR,
    the propagation and the relabel, which is the only way a user's `metric='manhattan'`
    is worth anything.

    `_l1_coord`'s six points are ONE cluster under euclidean (the diagonal
    bridge is inside eps) and TWO under manhattan (it is not). Both answers
    are derived in that function's docstring from distances with margins of
    0.01 and 0.4 against a threshold of 1, so neither is a measurement.

    THE SABOTAGE IS THE EUCLIDEAN ARM ITSELF. If `metric` never reached the
    runner, both fits would return the same partition and the euclidean
    assertion below would fail against the manhattan one. There is no
    corruption to plant: the check is two arms that must disagree, and each
    arm's expected value is stated independently.
    """
    var ctx = DeviceContext()
    var n = 6
    var d = 2
    var eps = L1_EPS
    var min_pts = 2

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _l1_coord(i, f))

    var lab_l2 = _fit_metric(ctx, hx, n, d, eps, min_pts, DBSCAN_METRIC_L2)
    var lab_l1 = _fit_metric(ctx, hx, n, d, eps, min_pts, DBSCAN_METRIC_L1)

    # --- euclidean: ONE cluster over all six -----------------------------
    for i in range(n):
        if lab_l2.unsafe_ptr().unsafe_load(i) != Int32(0):
            raise Error(
                "euclidean: point " + String(i) + " has label "
                + String(lab_l2.unsafe_ptr().unsafe_load(i))
                + "; the diagonal bridge is at L2 0.98995 < eps 1 so all six"
                " points are one cluster and final_relabel numbers it 0"
            )
    # --- manhattan: TWO clusters, split exactly at the bridge ------------
    var a = lab_l1.unsafe_ptr().unsafe_load(0)
    var b = lab_l1.unsafe_ptr().unsafe_load(3)
    if a < Int32(0) or b < Int32(0):
        raise Error(
            "manhattan: a point came back as noise at min_samples = 2, but"
            " every point has itself and an in-group neighbour 0.4 away"
        )
    if a == b:
        raise Error(
            "manhattan: groups A and B share label " + String(a)
            + "; the bridge is at L1 1.4 > eps 1 and must not connect them."
            " The metric did not reach the runner."
        )
    for i in range(3):
        if lab_l1.unsafe_ptr().unsafe_load(i) != a:
            raise Error(
                "manhattan: group A split at point " + String(i)
            )
        if lab_l1.unsafe_ptr().unsafe_load(3 + i) != b:
            raise Error(
                "manhattan: group B split at point " + String(3 + i)
            )

    print(
        "check_dbscan_manhattan_changes_the_labels OK: euclidean gives 1"
        " cluster over 6 points, manhattan gives 2 split at the planted"
        " diagonal bridge (L2 0.98995 in, L1 1.4 out, eps 1)"
    )


def _fit_metric(
    ctx: DeviceContext,
    mut hx: HostBuffer[DType.float32],
    n: Int,
    d: Int,
    eps: Float64,
    min_pts: Int,
    metric: Int,
) raises -> HostBuffer[DType.int32]:
    """One unweighted fit on the BRUTE arm, labels read back.

    BRUTE and not RBC: the ball cover is Euclidean-only by construction
    (`neighbors/impl/neighbors/ball_cover/`), and `dbscan_fit` REFUSES
    `metric != L2` on that arm rather than downgrading. Pinning the arm here
    keeps this check about the metric.
    """
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    _ = dbscan_fit_impl(
        ctx, x, labels, n, d, eps, min_pts, 0, 200, EPS_NN_BRUTE_FORCE,
        False, metric,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    return h


def check_dbscan_manhattan_refused_on_the_ball_cover() raises:
    """`metric='manhattan', algorithm='rbc'` RAISES and does not downgrade.

    `runner.cuh:152-156` downgrades an unsupported metric to L2Sqrt and logs.
    Copying that would answer a Manhattan query with Euclidean neighborhoods,
    which is the "accepted and ignored" failure this repository refuses by
    house rule, and the answer would be WRONG rather than slow -- the fixture
    above is one cluster under L2 and two under L1.

    The refusal is about SCOPE and not about the algorithm: the ball cover's
    pruning rests on the triangle inequality, which L1 satisfies. What is
    Euclidean is `neighbors/impl/neighbors/ball_cover/`'s implementation of
    the landmark radii and the three bounds, and that is another lane's file.

    The sabotage is the same fixture on the BRUTE arm, which must NOT raise:
    a check that only shows a raise cannot tell "refuses manhattan" from
    "refuses everything".
    """
    var ctx = DeviceContext()
    var n = 6
    var d = 2
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _l1_coord(i, f))
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    var raised = False
    try:
        _ = dbscan_fit_impl(
            ctx, x, labels, n, d, L1_EPS, 2, 0, 200, EPS_NN_RBC, False,
            DBSCAN_METRIC_L1,
        )
    except:
        raised = True
    if not raised:
        raise Error(
            "metric='manhattan' on the ball-cover arm returned a labelling."
            " The ball cover computes Euclidean distances, so that answer is"
            " a Euclidean clustering wearing a Manhattan label."
        )

    # THE SABOTAGE: the same call on the brute arm must SUCCEED.
    _ = dbscan_fit_impl(
        ctx, x, labels, n, d, L1_EPS, 2, 0, 200, EPS_NN_BRUTE_FORCE, False,
        DBSCAN_METRIC_L1,
    )
    print(
        "check_dbscan_manhattan_refused_on_the_ball_cover OK: rbc raises,"
        " brute serves the same call"
    )


# ---------------------------------------------------------------------------
# THE WEIGHTED CORE-POINT TEST (DEVIATION 28)
#
# THESE FOUR GATES HAVE NEVER RUN. Read this before quoting any of them.
#
# Building them used to take the whole dbscan lane down with an LLVM pass
# assertion, `DeadArgumentElimination surveyUse failed`. Bisected 2026-09-01:
# with all seven new gates disabled the lane builds, the three manhattan
# gates build, the weighted family does not. Commenting out a CALL was not
# enough -- the entry point's IMPORT is what pulls the function into
# codegen, so `dbscan_main.mojo` had to comment out both.
#
# CURED 2026-09-01 BY THE BUILD'S OPTIMIZATION LEVEL, AND BY NOTHING IN
# THIS FILE. `pixi run check-dbscan` passes `-O1`. Measured on an Apple M4,
# one variable at a time: -O3 asserts, -O2 asserts, -O1 builds, -O0 builds,
# and `DeadArgumentElimination` is an -O2-and-above pass. All seven gates
# then run and pass, the four weighted ones for the first time ever.
#
# FOUR SOURCE REWRITES WERE TRIED FIRST AND ALL FOUR FAILED. They are kept
# where they are still improvements and are labelled at their sites as
# attempts rather than cures, because a reader who finds an unexplained
# restructuring assumes it is load bearing:
#
#   1  called `vertex_deg_dispatch` with a compile-time constant metric,
#      replaced by the comptime instantiation. REVERTED -- it took the
#      dispatcher out from under the gate for no benefit.
#   2  `_fit_weighted` carried a `weights` list read only under a
#      `has_weights` Bool that every call site passed as a literal. Split
#      into `_fit_unweighted` and `_fit_weighted`. KEPT, as a simplification.
#   3  `_host_pinned_fold` padded with a ternary whose live arm was a
#      reference into its argument. Now a load into a local. KEPT.
#   4  `rows[i]`, an element reference into a `List[List[Int]]`, handed
#      straight into a `def`. Bound to a local copy first. KEPT.
#
# THE BISECT IS THE REUSABLE PART. An import and its call must BOTH be
# commented to disable a gate, since the import is what pulls it into
# codegen. Enabling them one at a time found TWO INDEPENDENT TRIGGERS, not
# one: the fold gate alone asserts, and the three fit gates alone assert.
# A single-candidate build that still crashed would therefore have retired
# a good fix, which is why nothing was retired on one build.
#
# NOT A WEAKENED GATE, and this is measured. With the weighted four disabled
# so that -O3 can build at all, the -O1 and -O3 binaries print BYTE-IDENTICAL
# output across all 13 remaining gates, the float-heavy ones included.
#
# THE CURE IS UNVERIFIED. Nobody has compiled this file since the
# restructuring, so every claim below is a claim about code that has not
# been executed. A green from these four gates counts only after the run in
# `dbscan/README.md`'s gate list has actually happened, and until then
# `sample_weight` stays IMPLEMENTED AND UNGATED. Do not quote a
# sample_weight result from this lane.
# ---------------------------------------------------------------------------


def _host_pinned_fold(partials: List[Float32], width: Int) -> Float32:
    """`pinned_block_sum`'s halving tree, on the host, at a stated width.

    `red[t] += red[t + step]` for `step = width/2 ... 1`, which is
    `core/pinned_reduce.mojo`'s IDENTICAL arm verbatim. It is spelled again
    here rather than imported because an oracle that calls the code it is
    checking is not an oracle; and it takes `width` as an ARGUMENT precisely
    so this file can fold the same partials two ways and show that the width
    is load bearing.

    CANDIDATE 3 (attempted 2026-09-01, NOT the cure; kept as a tidy-up --
    the cure was building at -O1). The pad loop
    used to read `red.append(partials[t] if t < len(partials) else
    Float32(0.0))`. A ternary whose live arm is a REFERENCE into `partials`
    and whose dead arm is a temporary makes the compiler merge two origins
    into one value before handing it to `append`, and a reference threaded
    into a call it does not reach into is one of the shapes
    `DeadArgumentElimination surveyUse failed` fires on. The statement form
    below loads first and appends a plain value. It is the same padding, slot
    for slot: `width` entries, `partials[t]` where one exists and 0.0 past
    the end.
    """
    var red = List[Float32]()
    for t in range(width):
        var v = Float32(0.0)
        if t < len(partials):
            v = partials[t]
        red.append(v)
    var step = width // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return red[0]


def _host_weighted_degree_strided(
    cols: List[Int], w: List[Float32], width: Int
) -> Float32:
    """The dense/CSR kernels' fold, on the host, at a stated width.

    `cols` is the list of positions this row's threads walk IN THE ORDER THE
    KERNEL WALKS THEM -- for the dense kernel that is every column index
    `0..n-1` and the guard skips non-neighbours, for the CSR kernel it is the
    row's neighbour list. Thread `t` takes `cols[t], cols[t + width], ...`
    ascending, exactly as the `while j < n_cols: ... j += WVD_TPB` loop does,
    and the partials go into the halving tree.
    """
    var partials = List[Float32]()
    for t in range(width):
        var acc = Float32(0.0)
        var k = t
        while k < len(cols):
            acc = ftz(acc + ftz(w[cols[k]]))
            k += width
        partials.append(acc)
    return _host_pinned_fold(partials, width)


def _host_weighted_degree_dense(
    mask: List[Bool], w: List[Float32], n_cols: Int, width: Int
) -> Float32:
    """`weighted_vertex_deg_dense_kernel`'s fold, on the host.

    THE DENSE ARM AND THE CSR ARM DO NOT TAKE THE SAME PARTIALS, and this
    function beside `_host_weighted_degree_strided` is where that becomes
    visible. The dense kernel's thread `t` walks COLUMN INDICES
    `t, t + width, ...` over the whole dataset and skips the non-neighbours,
    so a row whose neighbours are `{5, 200}` puts `w[5]` on thread 5 and
    `w[200]` on thread `200 % width`. The CSR kernel's thread `t` walks the
    row's `t`-th, `(t + width)`-th NEIGHBOUR, so the same row puts `w[5]` on
    thread 0 and `w[200]` on thread 1.

    Same multiset, two assignments, two float sums that agree only where the
    arithmetic is exact. DEVIATION 28 states that in those words; this
    oracle is what would catch it if either kernel drifted onto the other's
    assignment.
    """
    var partials = List[Float32]()
    for t in range(width):
        var acc = Float32(0.0)
        var j = t
        while j < n_cols:
            if mask[j]:
                acc = ftz(acc + ftz(w[j]))
            j += width
        partials.append(acc)
    return _host_pinned_fold(partials, width)


def _host_weighted_degree_sequential(
    cols: List[Int], w: List[Float32]
) -> Float32:
    """The SAME multiset, summed left to right in float32. A DIFFERENT fold.

    Used only as the sabotage: if this and `_host_weighted_degree_strided`
    agree on a fixture, that fixture cannot tell a right fold from a wrong
    one and the check says so instead of passing.
    """
    var acc = Float32(0.0)
    for k in range(len(cols)):
        acc = ftz(acc + ftz(w[cols[k]]))
    return acc


def _host_weighted_degree_f64(cols: List[Int], w: List[Float32]) -> Float64:
    """float64, left to right. The magnitude oracle: it cannot certify the
    last bits and is not asked to, it certifies that the kernel summed the
    WEIGHTS of the RIGHT NEIGHBOURS."""
    var acc = Float64(0.0)
    for k in range(len(cols)):
        acc += Float64(w[cols[k]])
    return acc


def check_dbscan_weighted_degree_matches_host_oracle() raises:
    """Both weighted-degree kernels against host oracles, and the fold
    sabotage that proves the fold width is load bearing.

    THE TWO KERNELS ARE THE TWO ARMS OF `launcher`'s sample-weight tail:
    `weighted_vertex_deg_dense` is their `coalescedReduction` over `adj`
    (`algo.cuh:243-254`) and `weighted_vertex_deg_csr` is their
    `accumulateWeights` over the CSR (`algo.cuh:62-91`). Both must produce
    the same weighted degree for the same neighbourhood, and each must fold
    it in the pinned order.

    WHAT IS ASSERTED
      0. THE FIXTURE IS DISCRIMINATING. Two host folds of the SAME multiset
         -- the kernel's strided-then-halving order and a plain left-to-right
         float32 sum -- must DISAGREE on at least one row. If they agree
         everywhere, the weights are all exactly representable and this
         check cannot tell a right fold from a wrong one; it raises rather
         than passing. That is the sabotage arm, and it is planted rather
         than hoped for: row 0's weights are `1.0` and a run of `2^-24`,
         whose sum is 1.0 left to right and strictly greater strided.
      1. every row of the DENSE kernel equals `_host_weighted_degree_strided`
         at `WVD_TPB`, EXACTLY, under IDENTICAL (where `pinned_block_sum` is
         the halving tree the oracle spells); within 1e-4 relative in every
         mode, since under FAST the primitive is the library fold and its
         shape is the codegen's business,
      2. the same for the CSR kernel over the same neighbourhoods,
      3. every row within 1e-4 relative of a float64 sum, which is what
         catches a kernel that folded beautifully over the wrong columns,
      4. the two ARMS agree EXACTLY on a second, INTEGER weight vector.
         That is not a tolerance being hidden: integer weights with a total
         below 2^24 are exactly representable, so float addition is exact
         and the order stops mattering. For general weights the two arms sum
         the same multiset in two different assignments and are NOT promised
         to agree; DEVIATION 28 says so in those words.
    """
    var ctx = DeviceContext()
    var m = 37
    var n = 200
    var d = 3

    # --- a neighbourhood with irregular row lengths ----------------------
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f, Float32(_jitter(i * 13 + f, f) * 3.0)
            )
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    var adj = ctx.enqueue_create_buffer[DType.uint8](m * n)
    var vd = ctx.enqueue_create_buffer[DType.int32](m + 1)
    ctx.synchronize()
    # CANDIDATE 1 WAS TRIED HERE AND REVERTED, 2026-09-01. The hypothesis
    # was that this is the ONLY call in the tree handing
    # `vertex_deg_dispatch` a COMPILE-TIME CONSTANT metric -- the runner's
    # two calls both pass a runtime `metric` threaded from the fit's params
    # -- so a constant would make `if metric == DBSCAN_METRIC_L1` a dead
    # branch and `metric` a dead argument of the specialized clone, which is
    # the shape `DeadArgumentElimination surveyUse failed` fires on. It was
    # a good hypothesis. It was WRONG: replacing this with the comptime
    # instantiation still asserted at -O3, and the real cure is the -O1
    # build level. The dispatcher is deliberately back, because routing
    # around it would leave `vertex_deg_dispatch` untested here for no gain.
    vertex_deg_dispatch(ctx, adj, vd, x, 0, m, n, d, 0.9, DBSCAN_METRIC_L2)
    ctx.synchronize()
    var hadj = ctx.enqueue_create_host_buffer[DType.uint8](m * n)
    ctx.enqueue_copy(dst_ptr=hadj.unsafe_ptr(), src_buf=adj)
    ctx.synchronize()

    # --- the weights. Order-sensitive by construction: a run of values at
    # half an ulp of 1.0, which vanish in one summation order and survive in
    # another. See the sabotage block below for the derivation.
    var tiny = Float32(1.0) / Float32(16777216.0)  # 2^-24
    var w = List[Float32]()
    for j in range(n):
        if j % 8 == 0:
            w.append(Float32(1.0))
        else:
            w.append(tiny)
    var wint = List[Float32]()
    for j in range(n):
        wint.append(Float32(1 + (j % 4)))

    # --- the per-row neighbour lists, host side --------------------------
    var rows = List[List[Int]]()
    var total_edges = 0
    var minlen = n + 1
    var maxlen = 0
    for i in range(m):
        var r = List[Int]()
        for j in range(n):
            if hadj.unsafe_ptr().unsafe_load(i * n + j) != UInt8(0):
                r.append(j)
        total_edges += len(r)
        if len(r) < minlen:
            minlen = len(r)
        if len(r) > maxlen:
            maxlen = len(r)
        rows.append(r.copy())
    if minlen == maxlen or maxlen < 4:
        raise Error(
            "the weighted-degree fixture is degenerate: every row has "
            + String(minlen) + " neighbours, so nothing here exercises a"
            " ragged row or a partial strided pass"
        )

    # --- 0. THE SABOTAGE, PLANTED RATHER THAN HOPED FOR ------------------
    # The pattern is `1.0` followed by `2 * WVD_TPB - 1` copies of 2^-24,
    # and its two folds are DERIVED, not measured:
    #   LEFT TO RIGHT the running sum is 1.0 and every addend is exactly
    #     half an ulp of 1.0, so round-to-nearest-even returns 1.0 every
    #     time. The answer is EXACTLY 1.0 and 2*WVD_TPB - 1 addends vanish.
    #   STRIDED THEN HALVED the tiny values meet each other before they meet
    #     the 1.0: each thread pairs two of them into 2^-23, the tree keeps
    #     doubling, and what finally reaches slot 0 is a value large enough
    #     to move 1.0. The answer is strictly greater than 1.0.
    # So a fold that lost its shape or its width cannot pass the assertions
    # below, and this check says so before making them.
    var planted_cols = List[Int]()
    var planted_w = List[Float32]()
    for k in range(2 * WVD_TPB):
        planted_cols.append(k)
        planted_w.append(Float32(1.0) if k == 0 else tiny)
    var pinned_fold = _host_weighted_degree_strided(
        planted_cols, planted_w, WVD_TPB
    )
    var flat_fold = _host_weighted_degree_sequential(planted_cols, planted_w)
    if pinned_fold == flat_fold:
        raise Error(
            "SABOTAGE DID NOT MOVE: 1.0 followed by "
            + String(2 * WVD_TPB - 1) + " copies of 2^-24 folded to "
            + String(pinned_fold) + " both left to right and through the"
            " pinned strided halving tree, so nothing below can tell a"
            " right fold from a wrong one"
        )
    if flat_fold != Float32(1.0):
        raise Error(
            "the planted pattern's left-to-right sum is " + String(flat_fold)
            + " and must be exactly 1.0; 2^-24 is half an ulp of 1.0 and"
            " round-to-nearest-even must absorb every one of them"
        )

    # --- run both kernels on the same neighbourhood ----------------------
    var wbuf = ctx.enqueue_create_buffer[DType.float32](n)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    for j in range(n):
        hw.unsafe_ptr().unsafe_store(j, w[j])
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=wbuf, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var ws_dense = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.synchronize()
    weighted_vertex_deg_dense(ctx, ws_dense, adj, wbuf, m, n)
    ctx.synchronize()
    var h_dense = ctx.enqueue_create_host_buffer[DType.float32](m)
    ctx.enqueue_copy(dst_ptr=h_dense.unsafe_ptr(), src_buf=ws_dense)
    ctx.synchronize()

    # The CSR the ball cover would hand us, built here in ASCENDING COLUMN
    # ORDER, which is what DEVIATION 551 canonicalizes the real one to under
    # IDENTICAL. Building it from the dense adjacency keeps this check about
    # the reduction and not about the ball cover.
    var ia = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var ja = ctx.enqueue_create_buffer[DType.int32](
        total_edges if total_edges > 0 else 1
    )
    var hia = ctx.enqueue_create_host_buffer[DType.int32](m + 1)
    var hja = ctx.enqueue_create_host_buffer[DType.int32](
        total_edges if total_edges > 0 else 1
    )
    ctx.synchronize()
    var pos = 0
    for i in range(m):
        hia.unsafe_ptr().unsafe_store(i, Int32(pos))
        for k in range(len(rows[i])):
            hja.unsafe_ptr().unsafe_store(pos, Int32(rows[i][k]))
            pos += 1
    hia.unsafe_ptr().unsafe_store(m, Int32(pos))
    ctx.enqueue_copy(dst_buf=ia, src_ptr=hia.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ja, src_ptr=hja.unsafe_ptr())
    ctx.synchronize()

    var ws_csr = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.synchronize()
    weighted_vertex_deg_csr(ctx, ws_csr, ia, ja, wbuf, m)
    ctx.synchronize()
    var h_csr = ctx.enqueue_create_host_buffer[DType.float32](m)
    ctx.enqueue_copy(dst_ptr=h_csr.unsafe_ptr(), src_buf=ws_csr)
    ctx.synchronize()

    # --- 1/2/3. against the oracles --------------------------------------
    comptime exact = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var bad_exact = 0
    var bad_mag = 0
    var arms_differ = 0
    for i in range(m):
        var mask = List[Bool]()
        for j in range(n):
            mask.append(hadj.unsafe_ptr().unsafe_load(i * n + j) != UInt8(0))
        var want_dense = _host_weighted_degree_dense(mask, w, n, WVD_TPB)
        # CANDIDATE 4 (2026-09-01): `rows[i]` is an element reference into a
        # `List[List[Int]]` handed straight into a `def`. Bound to a local
        # copy first -- the reference is the shape the DeadArgumentElimination
        # assertion is suspected to survey badly, and the degree gate is the
        # one that crashes ALONE.
        var row_csr = rows[i].copy()
        var want_csr = _host_weighted_degree_strided(row_csr, w, WVD_TPB)
        if want_dense != want_csr:
            arms_differ += 1
        var got_dense = h_dense.unsafe_ptr().unsafe_load(i)
        var got_csr = h_csr.unsafe_ptr().unsafe_load(i)
        comptime if exact:
            if got_dense != want_dense or got_csr != want_csr:
                if bad_exact < 4:
                    print(
                        "  row " + String(i) + ": dense " + String(got_dense)
                        + " want " + String(want_dense) + ", csr "
                        + String(got_csr) + " want " + String(want_csr)
                    )
                bad_exact += 1
        var row_f64 = rows[i].copy()
        var want64 = _host_weighted_degree_f64(row_f64, w)
        var scale = want64 if want64 > 1.0 else 1.0
        if abs(Float64(got_dense) - want64) > 1.0e-4 * scale:
            bad_mag += 1
        if abs(Float64(got_csr) - want64) > 1.0e-4 * scale:
            bad_mag += 1
    if bad_exact != 0:
        raise Error(
            String(bad_exact) + " of " + String(m) + " weighted degrees do"
            " not match the pinned host fold BIT FOR BIT under IDENTICAL."
            " The fold shape is the answer here: DEVIATION 28."
        )
    if bad_mag != 0:
        raise Error(
            String(bad_mag) + " weighted degrees are more than 1e-4 relative"
            " from a float64 sum of the same neighbours' weights, so the"
            " kernel is summing the wrong columns"
        )
    # THE TWO ORACLES ARE DIFFERENT FUNCTIONS, and the assertions above only
    # distinguish the dense kernel's column walk from the CSR kernel's
    # neighbour walk on inputs where the two folds actually separate. The
    # separation is PLANTED rather than left to the fixture: `4 * WVD_TPB`
    # columns of which every SECOND one is a neighbour, weight 1.0 on column
    # 0 and 2^-24 everywhere else.
    #   DENSE  thread `t` walks columns `t, t+W, t+2W, t+3W`, all of one
    #     parity, so the odd threads get nothing and only `W/2` partials are
    #     non-zero -- `(W/2 - 1) * 4` copies of 2^-24 survive their local
    #     sums.
    #   CSR    thread `t` walks the `t`-th and `(t+W)`-th NEIGHBOUR, columns
    #     `2t` and `2t + 2W`, so all `W` threads are busy and `(W - 1) * 2`
    #     copies survive.
    # Two different counts of a value at half an ulp of 1.0 reach the tree,
    # so the two answers differ. If they did not, either kernel could be
    # running the other's assignment and nothing above would notice.
    var pm_mask = List[Bool]()
    var pm_w = List[Float32]()
    var pm_cols = List[Int]()
    for k in range(4 * WVD_TPB):
        pm_mask.append(k % 2 == 0)
        pm_w.append(Float32(1.0) if k == 0 else tiny)
        if k % 2 == 0:
            pm_cols.append(k)
    var pm_dense = _host_weighted_degree_dense(
        pm_mask, pm_w, 4 * WVD_TPB, WVD_TPB
    )
    var pm_csr = _host_weighted_degree_strided(pm_cols, pm_w, WVD_TPB)
    if pm_dense == pm_csr:
        raise Error(
            "the dense and CSR host folds gave the same float ("
            + String(pm_dense) + ") on the planted separating mask, so"
            " nothing above distinguishes the dense kernel's column walk"
            " from the CSR kernel's neighbour walk and either could be"
            " running the other's assignment"
        )

    # --- 4. the two arms agree EXACTLY on integer weights ----------------
    for j in range(n):
        hw.unsafe_ptr().unsafe_store(j, wint[j])
    ctx.enqueue_copy(dst_buf=wbuf, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    weighted_vertex_deg_dense(ctx, ws_dense, adj, wbuf, m, n)
    weighted_vertex_deg_csr(ctx, ws_csr, ia, ja, wbuf, m)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h_dense.unsafe_ptr(), src_buf=ws_dense)
    ctx.enqueue_copy(dst_ptr=h_csr.unsafe_ptr(), src_buf=ws_csr)
    ctx.synchronize()
    for i in range(m):
        var gd = h_dense.unsafe_ptr().unsafe_load(i)
        var gc = h_csr.unsafe_ptr().unsafe_load(i)
        if gd != gc:
            raise Error(
                "arms disagree on INTEGER weights at row " + String(i)
                + ": dense " + String(gd) + " against csr " + String(gc)
                + ". Integer weights below 2^24 are exactly representable,"
                " so the two folds must give identical bits"
            )
        var row_int = rows[i].copy()
        var want = Float32(_host_weighted_degree_f64(row_int, wint))
        if gd != want:
            raise Error(
                "row " + String(i) + " integer weighted degree is "
                + String(gd) + " against an exact " + String(want)
            )

    print(
        "check_dbscan_weighted_degree_matches_host_oracle OK: " + String(m)
        + " rows (" + String(minlen) + "-" + String(maxlen)
        + " neighbours each, " + String(total_edges) + " edges), both arms"
        " match their own pinned host folds (which the planted mask separates,"
        " " + String(pm_dense) + " against " + String(pm_csr)
        + ", and which disagreed on " + String(arms_differ) + " of "
        + String(m) + " fixture rows), the planted fold sabotage separates"
        " the two orders (" + String(pinned_fold) + " against "
        + String(flat_fold) + "), and the two arms are bit-identical on"
        " integer weights"
    )


def check_dbscan_weighted_fold_is_pinned() raises:
    """`K_LIB_WEIGHTED_VERTEX_DEG` is a NUMERIC row, not a scheduling one.

    The consumer-side gate DEVIATION 524 asks for. `WVD_TPB` is at once the
    fold's width and the stride the per-thread partials are taken at, so two
    vendor columns carrying two values would build two different multisets of
    partials and then fold each of them correctly -- and every point whose
    weight sum sits on `min_pts` would change core status between them.

    Bit-inert today: `lib_block_size` returns 128 in every column, so this
    moves nothing on any machine that exists. What it buys is the NEXT
    measurement, which the matrix's own docstring invites to land "WITHOUT
    touching a kernel".

    THE SABOTAGE IS IN THE ASSERTION ITSELF: the same partial vector folded at
    `WVD_TPB` and at `WVD_TPB // 2` must give DIFFERENT floats. If a width
    could not change the answer, the pin would be decoration and this check
    would be `[[reached-but-inert]]`.
    """
    comptime listed = lib_block_bounds_a_float_fold[
        K_LIB_WEIGHTED_VERTEX_DEG
    ]()
    if not listed:
        raise Error(
            "K_LIB_WEIGHTED_VERTEX_DEG is not in"
            " lib_block_bounds_a_float_fold, so under IDENTICAL its block"
            " size follows the vendor column and the weighted core-point"
            " test is not cross-vendor"
        )
    if WVD_TPB < 2 or (WVD_TPB & (WVD_TPB - 1)) != 0:
        raise Error(
            "WVD_TPB is " + String(WVD_TPB) + ", which is not a power of"
            " two; pinned_block_sum's halving tree is exact only at a power"
            " of two"
        )
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        comptime floor_w = lib_block_size_for[
            K_LIB_WEIGHTED_VERTEX_DEG, COLUMN_BIT_IDENTICAL
        ]()
        if WVD_TPB != floor_w:
            raise Error(
                "under IDENTICAL the weighted fold resolved to "
                + String(WVD_TPB) + " on column "
                + String(TARGET_COLUMN) + " against the identity floor's "
                + String(floor_w) + "; the two vendors would fold differently"
            )

    # THE SABOTAGE: A WIDTH THAT CANNOT MOVE THE ANSWER IS NOT WORTH PINNING.
    # `2 * WVD_TPB` values -- 1.0 and a run of 2^-24 -- summed at this width
    # and at HALF it. The two must differ, or the whole pin is decoration.
    # A NOTE ON WHAT THIS IS NOT: an earlier draft folded the partials once
    # by hand and then ran the tree at half width, which is the SAME
    # computation the full tree performs in its first step and could never
    # have failed. The two arms below re-STRIDE the values, which is what a
    # different width actually does to a kernel.
    var half = WVD_TPB // 2
    var cols = List[Int]()
    var vals = List[Float32]()
    var tiny = Float32(1.0) / Float32(16777216.0)
    for k in range(2 * WVD_TPB):
        cols.append(k)
        vals.append(Float32(1.0) if k == 0 else tiny)
    var wide = _host_weighted_degree_strided(cols, vals, WVD_TPB)
    var narrow = _host_weighted_degree_strided(cols, vals, half)
    var flat = _host_weighted_degree_sequential(cols, vals)
    if wide == narrow:
        raise Error(
            "SABOTAGE DID NOT MOVE: 1.0 and " + String(2 * WVD_TPB - 1)
            + " copies of 2^-24 folded to " + String(wide)
            + " at width " + String(WVD_TPB) + " and at width "
            + String(half) + ", so nothing here shows the width is load"
            " bearing and pinning it buys nothing"
        )
    if wide == flat:
        raise Error(
            "SABOTAGE DID NOT MOVE: the pinned fold and a plain left-to-"
            "right sum agree on the planted pattern, so the fold SHAPE is"
            " not load bearing on it either"
        )

    print(
        "check_dbscan_weighted_fold_is_pinned OK: WVD_TPB = "
        + String(WVD_TPB) + ", listed as a float fold, a power of two, and"
        " the width demonstrably moves the sum -- " + String(wide)
        + " at " + String(WVD_TPB) + ", " + String(narrow) + " at "
        + String(half) + ", " + String(flat) + " left to right"
    )


def _fit_unweighted(
    ctx: DeviceContext,
    mut hx: HostBuffer[DType.float32],
    n: Int,
    d: Int,
    eps: Float64,
    min_pts: Int,
    method: Int,
) raises -> HostBuffer[DType.int32]:
    """One UNWEIGHTED fit through the weighted entry point. Labels read back.

    `dbscan_fit_impl_weighted` with `has_weights = False` and the one-element
    placeholder that stands in for cuML's `sample_weight == nullptr`
    (`_no_weights`'s docstring has the reason Mojo needs a buffer at all).
    That is the arm the equality below compares against, so it has to be the
    weighted entry point and not `dbscan_fit_impl`: the claim is that the
    same driver with weights of 1.0 lands on the same labels, not that two
    different drivers do.

    CANDIDATE 2 (attempted toolchain workaround, 2026-09-01, AND IT WAS NOT
    THE CURE -- kept because the split is a real simplification, not because
    it fixed anything). The assertion is cleared by BUILDING AT -O1, which
    `pixi run check-dbscan` now does; measured on an Apple M4, -O3 and -O2
    assert and -O1 and -O0 build, DeadArgumentElimination being an
    -O2-and-above pass. This candidate, and three others, still asserted at
    -O3. The reasoning below was a good hypothesis and it was wrong. This and
    `_fit_weighted` used to be ONE function carrying `weights:
    List[Float32]` beside a `has_weights: Bool`, read only inside
    `if has_weights:`, and every call site passed a LITERAL `True` or
    `False`. Constant-propagating that literal makes the branch dead and
    `weights` a dead argument of the clone, which is the shape
    `DeadArgumentElimination surveyUse failed` fires on; one of the two call
    sites additionally passed an EMPTY list it never read. Splitting removes
    the flag and the conditionally-dead argument without moving a single
    launch: the buffer sizes, the copies, the synchronize points and the
    arguments handed to `dbscan_fit_impl_weighted` are what the merged
    function did on each branch.
    """
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    _ = dbscan_fit_impl_weighted(
        ctx, x, labels, w, n, d, eps, min_pts, 0, 200, method, False,
        DBSCAN_METRIC_L2, False,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    return h


def _fit_weighted(
    ctx: DeviceContext,
    mut hx: HostBuffer[DType.float32],
    n: Int,
    d: Int,
    eps: Float64,
    min_pts: Int,
    weights: List[Float32],
    method: Int,
) raises -> HostBuffer[DType.int32]:
    """One WEIGHTED fit on a stated arm. Labels read back.

    The weighted half of the split described in `_fit_unweighted`. `weights`
    is read unconditionally here, so it is not a dead argument on any clone.
    """
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        hw.unsafe_ptr().unsafe_store(i, weights[i])
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    _ = hw^
    ctx.synchronize()
    _ = dbscan_fit_impl_weighted(
        ctx, x, labels, w, n, d, eps, min_pts, 0, 200, method, False,
        DBSCAN_METRIC_L2, True,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    return h


def check_dbscan_uniform_weight_matches_unweighted() raises:
    """WEIGHTS OF 1.0 REPRODUCE THE UNWEIGHTED LABELS EXACTLY, ON BOTH ARMS.

    The strongest cheap statement about a weighted path, and it is NOT a
    tautology here: the weighted fit runs a different core-point kernel over
    a different buffer produced by a different reduction. The only thing that
    makes the two agree is that the reduction is correct -- a sum of `k` ones
    is exactly `k` in float32 for `k < 2^24`, in ANY order, so this holds
    without appealing to the fold at all and would break for any kernel that
    summed the wrong columns, dropped a strided tail, or mis-indexed `vd` by
    batch against `core` by dataset.

    BOTH ARMS, because they are two different producers: the dense
    `coalescedReduction` port on `brute` and the `accumulateWeights` port on
    `rbc`, and the `rbc` arm additionally exercises DEVIATION 29's
    `need_ja_compute` fill in loop 1 which the unweighted path never runs.

    THE SABOTAGE ARM IS A WEIGHT OF `min_samples` ON THE NOISE POINTS, which
    is scikit-learn's own documented sentence: "a sample with a weight of at
    least min_samples is by itself a core sample" (`_dbscan.py:412-415`). The
    fixture's twelve isolated points have degree 1, so at weight 1 they are
    noise and at weight `DB_MIN_PTS` each becomes a core point and therefore
    its own singleton cluster. If that arm did NOT move, the weighted path
    would be ignoring the weights and the equality above would be passing for
    the wrong reason.
    """
    var ctx = DeviceContext()
    var n = DB_ROWS
    var d = DB_FEATURES
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _coord(i, f))

    var ones = List[Float32]()
    var heavy_noise = List[Float32]()
    for i in range(n):
        ones.append(Float32(1.0))
        if i < BLOBS * PER_BLOB:
            heavy_noise.append(Float32(1.0))
        else:
            heavy_noise.append(Float32(DB_MIN_PTS))

    var methods = [EPS_NN_BRUTE_FORCE, EPS_NN_RBC]
    var names = [String("brute"), String("rbc")]
    for mi in range(len(methods)):
        var method = methods[mi]
        var plain = _fit_unweighted(
            ctx, hx, n, d, DB_EPS, DB_MIN_PTS, method
        )
        var uniform = _fit_weighted(
            ctx, hx, n, d, DB_EPS, DB_MIN_PTS, ones, method
        )
        var differ = 0
        var first_bad = -1
        for i in range(n):
            if plain.unsafe_ptr().unsafe_load(i) != uniform.unsafe_ptr(
            ).unsafe_load(i):
                if first_bad < 0:
                    first_bad = i
                differ += 1
        if differ != 0:
            raise Error(
                names[mi] + ": " + String(differ) + " of " + String(n)
                + " labels changed when every sample_weight is 1.0, first at"
                " point " + String(first_bad) + " (" 
                + String(plain.unsafe_ptr().unsafe_load(first_bad))
                + " -> "
                + String(uniform.unsafe_ptr().unsafe_load(first_bad))
                + "). A sum of k ones is exactly k in float32, so the"
                " weighted degree is not the degree."
            )

        # --- THE SABOTAGE ARM -------------------------------------------
        var heavy = _fit_weighted(
            ctx, hx, n, d, DB_EPS, DB_MIN_PTS, heavy_noise, method
        )
        var moved = 0
        for i in range(BLOBS * PER_BLOB, n):
            if heavy.unsafe_ptr().unsafe_load(i) == Int32(-1):
                raise Error(
                    names[mi] + ": isolated point " + String(i)
                    + " is still noise at sample_weight = "
                    + String(DB_MIN_PTS)
                    + ", but a sample whose own weight reaches min_samples"
                    " is by itself a core sample. The weights are not"
                    " reaching the core-point test."
                )
            moved += 1
        for b in range(BLOBS):
            for k in range(PER_BLOB):
                if heavy.unsafe_ptr().unsafe_load(
                    b * PER_BLOB + k
                ) != heavy.unsafe_ptr().unsafe_load(b * PER_BLOB):
                    raise Error(
                        names[mi] + ": blob " + String(b) + " split when only"
                        " the isolated points were weighted; the blobs are"
                        " 200-point cliques and nothing there changed"
                    )
        if moved != NOISE:
            raise Error(
                names[mi] + ": " + String(moved) + " isolated points changed"
                " where " + String(NOISE) + " were expected"
            )

    print(
        "check_dbscan_uniform_weight_matches_unweighted OK on brute and rbc:"
        " all " + String(n) + " labels identical at weight 1.0, and the"
        " sabotage arm turned all " + String(NOISE) + " noise points core at"
        " weight " + String(DB_MIN_PTS)
    )


comptime DUP_ROWS = 8
comptime DUP_EPS = 1.0
comptime DUP_MIN_PTS = 3


def _dup_coord(row: Int, feature: Int) -> Float32:
    """AN ISOLATED PAIR PLUS A FAR CLIQUE, so a duplicate DECIDES the answer.

    Two dimensions:

        rows 0, 1   the PAIR, at (0, 0) and (0.5, 0)
        rows 2..7   a clique at x = 50 + 0.3 k, y = 0

    At `eps = 1` and `min_samples = 3`:
      - each of the pair has degree 2 (itself and the other), so UNWEIGHTED
        both are noise and there is no cluster there at all,
      - give row 0 a weight of 2 and both weighted degrees become 3, so both
        are core and the pair is one cluster,
      - DUPLICATE row 0 instead and the three coincident/near points give
        every member degree 3, so the pair is one cluster again.

    That is the equivalence this fixture exists to test, and each side of it
    is derived rather than measured. The far clique is there so the answer
    contains a cluster either way and the check is comparing labellings
    rather than comparing "all noise" with "all noise".
    """
    if row < 2:
        return Float32(0.5 * Float64(row)) if feature == 0 else Float32(0.0)
    var k = row - 2
    return Float32(50.0 + 0.3 * Float64(k)) if feature == 0 else Float32(0.0)


def check_dbscan_duplicate_equals_weight_two() raises:
    """DUPLICATING A POINT EQUALS GIVING IT WEIGHT 2.

    scikit-learn's own reason for having `sample_weight` at all -- "remove
    (near-)duplicate points and use sample_weight instead"
    (`_dbscan.py:157-158`) -- so this is the property the parameter is FOR,
    and it does not share our spelling with anything in the port.

    The equivalence is exact rather than approximate: a duplicate at distance
    0 has the same neighbourhood as its original, so every neighbour's degree
    gains exactly 1 where a weight of 2 adds exactly 1 to the same sum, and
    the duplicate's own degree equals the original's. Small integers, so no
    tolerance.

    THREE ARMS, and the third is the sabotage:
      A  UNWEIGHTED on the 9-row duplicated set,
      B  WEIGHTED   on the 8-row set with `w[0] = 2`,
      C  WEIGHTED   on the 8-row set with `w[0] = 1.5`  <- must DIFFER.
    A and B must agree on the 8 shared points. C must not: 1.5 + 1 = 2.5 is
    below `min_samples = 3`, so the pair stays noise, which is exactly the
    line the check is claiming to sit on. Without C, a weighted path that
    ignored the weights entirely would pass A == B whenever the unweighted
    answer happened to match.
    """
    var ctx = DeviceContext()
    var d = 2
    var n = DUP_ROWS

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _dup_coord(i, f))

    # A: the duplicated set, UNWEIGHTED. Row `n` is a copy of row 0.
    var hxd = ctx.enqueue_create_host_buffer[DType.float32]((n + 1) * d)
    for i in range(n):
        for f in range(d):
            hxd.unsafe_ptr().unsafe_store(i * d + f, _dup_coord(i, f))
    for f in range(d):
        hxd.unsafe_ptr().unsafe_store(n * d + f, _dup_coord(0, f))

    var w2 = List[Float32]()
    var w15 = List[Float32]()
    for i in range(n):
        w2.append(Float32(2.0) if i == 0 else Float32(1.0))
        w15.append(Float32(1.5) if i == 0 else Float32(1.0))

    var methods = [EPS_NN_BRUTE_FORCE, EPS_NN_RBC]
    var names = [String("brute"), String("rbc")]
    for mi in range(len(methods)):
        var method = methods[mi]
        var dup = _fit_unweighted(
            ctx, hxd, n + 1, d, DUP_EPS, DUP_MIN_PTS, method
        )
        var wt2 = _fit_weighted(
            ctx, hx, n, d, DUP_EPS, DUP_MIN_PTS, w2, method
        )
        var wt15 = _fit_weighted(
            ctx, hx, n, d, DUP_EPS, DUP_MIN_PTS, w15, method
        )

        # The pair must be a real cluster on both A and B, or the fixture is
        # asserting "noise equals noise".
        if dup.unsafe_ptr().unsafe_load(0) < Int32(0):
            raise Error(
                names[mi] + ": the duplicated pair is still noise at"
                " min_samples = 3, so this fixture cannot show the"
                " equivalence"
            )
        for i in range(n):
            if dup.unsafe_ptr().unsafe_load(i) != wt2.unsafe_ptr().unsafe_load(
                i
            ):
                raise Error(
                    names[mi] + ": point " + String(i) + " is labelled "
                    + String(dup.unsafe_ptr().unsafe_load(i))
                    + " when point 0 is DUPLICATED and "
                    + String(wt2.unsafe_ptr().unsafe_load(i))
                    + " when point 0 has weight 2; those are the same"
                    " density and must be the same clustering"
                )
        if dup.unsafe_ptr().unsafe_load(n) != dup.unsafe_ptr().unsafe_load(0):
            raise Error(
                names[mi] + ": the duplicate itself landed in a different"
                " cluster from the point it copies"
            )

        # --- THE SABOTAGE ARM -------------------------------------------
        var same = True
        for i in range(n):
            if wt15.unsafe_ptr().unsafe_load(i) != wt2.unsafe_ptr(
            ).unsafe_load(i):
                same = False
        if same:
            raise Error(
                names[mi] + ": SABOTAGE DID NOT MOVE -- weight 1.5 on point 0"
                " gave the same labelling as weight 2.0, but 1.5 + 1 = 2.5 is"
                " below min_samples = 3 and the pair must fall back to noise."
                " The weighted core test is not reading the weight."
            )
        if wt15.unsafe_ptr().unsafe_load(0) != Int32(-1):
            raise Error(
                names[mi] + ": at weight 1.5 point 0 has label "
                + String(wt15.unsafe_ptr().unsafe_load(0))
                + " and must be noise (weighted degree 2.5 < 3)"
            )

    print(
        "check_dbscan_duplicate_equals_weight_two OK on brute and rbc:"
        " duplicating point 0 and giving it weight 2 give the same"
        " labelling on all " + String(n) + " shared points, and the"
        " weight-1.5 sabotage falls back to noise as it must"
    )
