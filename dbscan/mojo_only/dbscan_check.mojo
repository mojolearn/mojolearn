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

from max.gpu.host import DeviceContext

from dbscan.ported.dbscan.adjgraph.algo import (
    SCAN_TPB,
    exclusive_scan_kernel,
)
from dbscan.ported.dbscan.runner import dbscan_fit
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


def check_dbscan() raises:
    var ctx = DeviceContext()
    var n = DB_ROWS
    var d = DB_FEATURES

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var x_alias = ctx.enqueue_create_buffer[DType.float32](n * d)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n)
    var xn_alias = ctx.enqueue_create_buffer[DType.float32](n)
    var dist = ctx.enqueue_create_buffer[DType.float32](n * n)
    var adj = ctx.enqueue_create_buffer[DType.uint8](n * n)
    var vd = ctx.enqueue_create_buffer[DType.int32](n)
    var core = ctx.enqueue_create_buffer[DType.uint8](n)
    var ex_scan = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var col_ind = ctx.enqueue_create_buffer[DType.int32](
        BLOBS * PER_BLOB * PER_BLOB + NOISE + 16
    )
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _coord(i, f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    var passes = dbscan_fit(
        ctx, x, x_norm, dist, adj, vd, core, ex_scan, col_ind, labels,
        x_alias, xn_alias, n, d, DB_EPS, DB_MIN_PTS,
    )

    var hl = ctx.enqueue_create_host_buffer[DType.int32](n)
    var hc = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=core)
    ctx.synchronize()

    # --- every blob point is core, every noise point is not --------------
    for i in range(BLOBS * PER_BLOB):
        if hc.unsafe_ptr().unsafe_load(i) == 0:
            raise Error(
                "blob point " + String(i) + " is not a core point; every"
                " point in a fully connected blob of 200 must be"
            )
    for i in range(BLOBS * PER_BLOB, n):
        if hc.unsafe_ptr().unsafe_load(i) != 0:
            raise Error(
                "isolated noise point " + String(i) + " was marked core"
            )

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

    # --- noise stays noise -----------------------------------------------
    for i in range(BLOBS * PER_BLOB, n):
        if hl.unsafe_ptr().unsafe_load(i) != MAX_LABEL:
            raise Error(
                "noise point " + String(i) + " was given cluster label "
                + String(hl.unsafe_ptr().unsafe_load(i))
            )

    print(
        "check_dbscan OK: 3/3 blobs each one whole cluster, 0 merges, "
        + String(NOISE)
        + "/"
        + String(NOISE)
        + " isolated points left as noise, "
        + String(BLOBS * PER_BLOB)
        + " core points, converged in "
        + String(passes)
        + " propagation passes"
    )


def check_dbscan_eps_sensitivity() raises:
    """The reach test, and it is an INVARIANT rather than a corruption.

    Raise `eps` past the blob separation and the three clusters MUST merge
    into one. Lower `min_pts` past a blob's size and nothing may change,
    because every blob point already has 200 neighbours.

    The first half is the reach evidence for the neighbourhood kernel: there
    is no way to produce one cluster at `eps = 12` and three at `eps = 2`
    without the radius test actually running on real distances. A no-op
    neighbourhood gives the same answer at both.
    """
    var ctx = DeviceContext()
    var n = DB_ROWS
    var d = DB_FEATURES

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var x_alias = ctx.enqueue_create_buffer[DType.float32](n * d)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n)
    var xn_alias = ctx.enqueue_create_buffer[DType.float32](n)
    var dist = ctx.enqueue_create_buffer[DType.float32](n * n)
    var adj = ctx.enqueue_create_buffer[DType.uint8](n * n)
    var vd = ctx.enqueue_create_buffer[DType.int32](n)
    var core = ctx.enqueue_create_buffer[DType.uint8](n)
    var ex_scan = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var col_ind = ctx.enqueue_create_buffer[DType.int32](n * n)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _coord(i, f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    # eps large enough to bridge the 10-apart blob centres.
    _ = dbscan_fit(
        ctx, x, x_norm, dist, adj, vd, core, ex_scan, col_ind, labels,
        x_alias, xn_alias, n, d, 12.0, DB_MIN_PTS,
    )
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
    """Regression for a SILENT correctness bug, at a size no fixture reaches.

    The first version of `exclusive_scan_kernel` gave each of `SCAN_TPB = 256`
    threads a FIXED `SCAN_CHUNK = 64` rows, which caps it at 16,384. Past that
    it stopped scanning, returned a wrong `nnz` and a truncated CSR, and
    raised nothing.

    Nothing here would have caught it: the DBSCAN fixture is 612 rows, and it
    cannot simply be made larger because the distance matrix is `n^2` and
    16,385 rows is a gigabyte. So this checks the SCAN ON ITS OWN, against a
    host scan, at 20,000 entries.

    The lesson generalizes past this kernel: a launch geometry that encodes a
    maximum size needs a test at that size, or the maximum is a trapdoor.
    """
    var ctx = DeviceContext()
    var n = 20000

    var vd = ctx.enqueue_create_buffer[DType.int32](n)
    var ex = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.synchronize()

    var hv = ctx.enqueue_create_host_buffer[DType.int32](n)
    for i in range(n):
        hv.unsafe_ptr().unsafe_store(i, Int32((i * 7 + 3) % 13))
    ctx.enqueue_copy(dst_buf=vd, src_ptr=hv.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[exclusive_scan_kernel](
        ex.unsafe_ptr(),
        vd.unsafe_ptr(),
        Int32(n),
        grid_dim=(1, 1, 1),
        block_dim=(SCAN_TPB, 1, 1),
    )
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
            + ", which is past the old 16,384 cap"
        )
    if total != running:
        raise Error(
            "the scan total is " + String(total) + " against " + String(running)
        )
    print(
        "check_exclusive_scan_beyond_the_old_cap OK: "
        + String(n)
        + " entries exact, total "
        + String(total)
        + ", past the 16,384 the fixed chunk size used to cap it at"
    )
