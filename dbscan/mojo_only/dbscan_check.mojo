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

from max.gpu.host import DeviceBuffer, DeviceContext

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from dbscan.ported.dbscan.adjgraph.algo import (
    exclusive_scan,
    scan_blocks_needed,
)
from dbscan.ported.dbscan.dbscan import compute_batch_size, dbscan_fit_impl
from dbscan.ported.dbscan.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC
from dbscan.ported.dbscan.runner import dbscan_fit, rbc_take_one_pass
from dbscan.ported.dbscan.vertexdeg.algo import (
    VD_TPB,
    eps_neighborhood_kernel,
)
from dbscan.ported.neighbors.epsilon_neighborhood import (
    EPS_MBLK,
    EPS_NBLK,
    EPS_THREADS,
    eps_unexp_l2_sq_neigh_kernel,
)
from dbscan.ported.sparse.detail.csr import MAX_LABEL


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

    _ = dbscan_fit(
        ctx, x, adj_full, vd_full, core, ex_full, labels, labels_temp,
        work_buffer, block_sums, n, d, DB_EPS, DB_MIN_PTS, 0,
    )
    var baseline = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=baseline.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    _ = dbscan_fit(
        ctx, x, adj_b, vd_b, core, ex_b, labels, labels_temp,
        work_buffer, block_sums, n, d, DB_EPS, DB_MIN_PTS, batch,
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
    ctx.enqueue_function[eps_unexp_l2_sq_neigh_kernel](
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
    _ = dbscan_fit(
        ctx, x, adj_b, vd_b, core, ex_b, labels, labels_temp, work_buffer,
        block_sums, n, d, eps, DB_MIN_PTS, batch, 200, EPS_NN_RBC,
    )
    var hr = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    var adj_f = ctx.enqueue_create_buffer[DType.uint8](n * n)
    var vd_f = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var ex_f = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.synchronize()
    _ = dbscan_fit(
        ctx, x, adj_f, vd_f, core, ex_f, labels, labels_temp, work_buffer,
        block_sums, n, d, eps, DB_MIN_PTS, 0, 200, EPS_NN_BRUTE_FORCE,
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
    """BOTH arms of loop 2's RBC dispatch, one named fixture each.

    PORTING_RULES 8: a switch is exercised on both sides or one side is
    unchecked, and the RBC arm's loop 2 now has a switch --
    `algo.cuh:119-122` sends a batch down the ONE-PASS `max_k` form when
    loop 1's bound fits the spare room, and down the two-pass count + fill
    otherwise. The guard is data-derived, not a parameter, so each fixture
    PINS its arm and the check proves the pin with the runner's own
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
        " algo.cuh:119's guard provably routes each fixture to its arm"
    )
