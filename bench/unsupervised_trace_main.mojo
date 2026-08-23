"""ONE traced unsupervised fit per process: the E1 payload for rows 19-26.

`tools/e1_traced_fit.py` is the GBDT/ET/RF driver and it goes through the
Python bindings. There is no binding for `cluster/`, `neighbors/` or
`dbscan/`, so the unsupervised half of the ledger had no way to produce a
card at all -- which is why `UNSUPERVISED_IDENTITY.md`'s first owed item
("a second vendor") could not even be attempted. This main is that way.

ONE ARM PER PROCESS, AND THAT IS THE DIFFER'S CONTRACT, NOT A CONVENIENCE.
`core/identity_trace.mojo`'s records carry a sequence number and an
algorithm-position tag; two fits in one process repeat the tags and
`tools/identity_trace_diff.py` refuses a file whose sequence numbers
restart. `MOJOLEARN_UNSUP_ARM` selects `kmeans`, `knn` or `dbscan` and the
process ends.

THE FIXTURE IS AN INTEGER-EXACT FUNCTION OF A CONSTANT SEED, for the same
reason E1's is (`E1_RESULTS.md`, "Inputs are a pure function of a fixed seed
with integer-exact target construction"): a cross-vendor claim about a fit
is worth nothing until both machines are proven to have fitted the SAME
BYTES. Every coordinate here is `<small integer> / <power of two>`, which
is exact in Float32 on any backend, and the driver prints a hash of the raw
bits before it fits. Compare THAT first; a card diff against different
inputs measures nothing.

WHAT EACH ARM IS CHOSEN TO REACH

    kmeans    the Lloyd loop with `INIT_ARRAY` -- centroids passed in, so
              the fit is deterministic without depending on k-means++'s
              float scan. Reaches the norm fold (row 19), the assignment
              contraction (503), `reduce_by_key`'s inertia and shift folds
              (row 21 via 504/508) and the convergence test that decides
              the ITERATION COUNT, which is the stage where a last-bit
              difference stops being a last-bit difference.
    knn       `knn_search` at the DEFAULT arm, which under IDENTICAL is the
              tiled one on every column (DEVIATION 509). Reaches the norms,
              `pinned_distance_tile` (505) and the composite-key selector
              (500/501). k = 10 with `n_index` well above it, and duplicate
              points PLANTED so the tie class is non-empty -- a k-NN card
              off a fixture with no ties would agree across vendors while
              proving nothing about the thing row 11 was refused for.
    dbscan    `dbscan_fit` at the default memory budget, so the batch count
              is whatever the device's free memory gives it. That is
              deliberate: the recorded stages are the three that do NOT
              depend on the batch count, and a card that agreed only
              because both machines happened to pick the same number of
              batches would be evidence of nothing. Reaches the eps
              accumulators (506) and the propagation (507).

USAGE (the shell driver `tools/e1_unsupervised.sh` does all of this):

    : > /tmp/kmeans.card
    MOJOLEARN_IDENTITY_TRACE=/tmp/kmeans.card \\
    MOJOLEARN_UNSUP_ARM=kmeans \\
        pixi run mojo run -I . bench/unsupervised_trace_main.mojo

The trace path is read by the FITS, through the environment, exactly as a
user would set it. This main never touches `MOJOLEARN_IDENTITY_TRACE`.
"""

from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import bitcast

from cluster.ported.cluster.detail.kmeans import kmeans_fit_main
from cluster.ported.cluster.kmeans_params import INIT_ARRAY, KMeansParams
from dbscan.estimator import dbscan_fit
from dbscan.ported.dbscan.runner import EPS_NN_BRUTE_FORCE
from neighbors.estimator import knn_search
from mojo_only.kernel_matrix import TARGET_COLUMN, lib_lane_width_for
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: The three fixtures. Sizes are deliberately small: this is a CERTIFICATE
#: run, not a benchmark, and a rented box is not a machine anything is timed
#: on (`tools/remote_gpu.sh`'s own rule).
comptime KM_N = 4096
comptime KM_D = 8
comptime KM_K = 8
comptime KM_ITERS = 10

comptime KNN_N = 4096
comptime KNN_D = 8
comptime KNN_Q = 512
comptime KNN_K = 10
#: Every 97th index is overwritten with row 0's coordinates, so the tie class
#: has 43 members and `k = 10` has to choose inside it.
comptime KNN_TIE_STRIDE = 97

comptime DB_N = 2048
comptime DB_D = 4
comptime DB_EPS = 1.05
comptime DB_MIN_PTS = 8


def _mode_name() -> String:
    comptime if IDENTICAL_BUILD:
        return String("IDENTICAL")
    return String("FAST")


def _mix(i: Int, f: Int, salt: Int) -> UInt64:
    """The fixture generator. Not a port, and not a hash anything depends on
    beyond reproducibility: integer arithmetic, so every backend agrees."""
    var h = UInt64(i + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(
        f + salt
    ) * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _coord(i: Int, f: Int, salt: Int) -> Float32:
    """`k / 256` for an integer `k` in [0, 256), which is EXACT in Float32.

    The division is by a power of two and the numerator is below 2^24, so
    the value has an exact binary representation and no backend can round it
    differently. That is the whole requirement on a cross-vendor fixture.
    """
    return Float32(Int(_mix(i, f, salt) & UInt64(0xFFFF))) / Float32(256.0)


def _bits_hash(ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int) -> UInt64:
    """FNV-1a over the RAW BITS of the fixture.

    Over the bits and not the values: two backends that printed the same
    decimals could still hold different floats, and it is the floats that
    get fitted. `String(Float32)` does not round-trip in this stdlib
    (`[[mojo-string-float-roundtrip]]`), so a decimal comparison would be
    unsound as well as weaker.
    """
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n):
        var b = bitcast[DType.uint32](ptr.unsafe_load(i))
        for byte in range(4):
            var v = UInt64((b >> UInt32(byte * 8)) & UInt32(0xFF))
            h = (h ^ v) * UInt64(0x100000001B3)
    return h


def _run_kmeans(ctx: DeviceContext) raises:
    var hx = ctx.enqueue_create_host_buffer[DType.float32](KM_N * KM_D)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](KM_N)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](KM_K * KM_D)
    ctx.synchronize()
    for i in range(KM_N):
        for f in range(KM_D):
            hx.unsafe_ptr().unsafe_store(i * KM_D + f, _coord(i, f, 7))
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0))
    # The initial centroids are ROWS of X, which is `INIT_ARRAY`'s contract
    # and makes the start of the fit exact rather than sampled.
    for c in range(KM_K):
        for f in range(KM_D):
            hc.unsafe_ptr().unsafe_store(
                c * KM_D + f, _coord(c * 37, f, 7)
            )
    print("input.x", _bits_hash(hx.unsafe_ptr(), KM_N * KM_D))
    print("input.centroids", _bits_hash(hc.unsafe_ptr(), KM_K * KM_D))

    var x = ctx.enqueue_create_buffer[DType.float32](KM_N * KM_D)
    var w = ctx.enqueue_create_buffer[DType.float32](KM_N)
    var cent = ctx.enqueue_create_buffer[DType.float32](KM_K * KM_D)
    var labels = ctx.enqueue_create_buffer[DType.uint32](KM_N)
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=cent, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    var params = KMeansParams.default()
    params.n_clusters = KM_K
    params.init = INIT_ARRAY
    params.max_iter = KM_ITERS
    params.n_init = 1
    params.seed = 7
    # `sum_scale` / `weight_scale` are the caller's bound on the data
    # (`cluster/mojo_only/reduce_by_key.mojo`); the fixture's coordinates are
    # below 256 and its weights are 1, so 4096 covers both with room.
    _ = kmeans_fit_main(
        ctx, x, w, cent, labels, params, KM_N, KM_D,
        Float32(4096.0), Float32(4096.0),
    )

    var out_c = ctx.enqueue_create_host_buffer[DType.float32](KM_K * KM_D)
    var out_l = ctx.enqueue_create_host_buffer[DType.uint32](KM_N)
    ctx.enqueue_copy(dst_ptr=out_c.unsafe_ptr(), src_buf=cent)
    ctx.enqueue_copy(dst_ptr=out_l.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    print("output.centroids", _bits_hash(out_c.unsafe_ptr(), KM_K * KM_D))
    var lh = UInt64(0xCBF29CE484222325)
    for i in range(KM_N):
        var v = UInt64(out_l.unsafe_ptr().unsafe_load(i))
        for byte in range(4):
            lh = (lh ^ ((v >> UInt64(byte * 8)) & UInt64(0xFF))) * UInt64(
                0x100000001B3
            )
    print("output.labels", lh)
    _ = hx^
    _ = hw^
    _ = hc^
    _ = out_c^
    _ = out_l^


def _run_knn(ctx: DeviceContext) raises:
    var hi = ctx.enqueue_create_host_buffer[DType.float32](KNN_N * KNN_D)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](KNN_Q * KNN_D)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](KNN_Q * KNN_K)
    var hidx = ctx.enqueue_create_host_buffer[DType.uint32](KNN_Q * KNN_K)
    ctx.synchronize()
    for i in range(KNN_N):
        for f in range(KNN_D):
            hi.unsafe_ptr().unsafe_store(i * KNN_D + f, _coord(i, f, 13))
    # THE PLANTED TIE CLASS. Overwritten AFTER the cloud so the copies are
    # exact rather than nearly equal; a fixture whose "ties" are 1 ULP apart
    # tests the comparator, not the tie rule.
    var t = KNN_TIE_STRIDE
    while t < KNN_N:
        for f in range(KNN_D):
            hi.unsafe_ptr().unsafe_store(
                t * KNN_D + f, hi.unsafe_ptr().unsafe_load(f)
            )
        t += KNN_TIE_STRIDE
    for i in range(KNN_Q):
        for f in range(KNN_D):
            hq.unsafe_ptr().unsafe_store(
                i * KNN_D + f, _coord(i * 5 + 1, f, 13)
            )
    # Half the queries sit exactly ON the planted point, so their whole
    # neighbour list comes out of the tie class.
    for i in range(0, KNN_Q, 2):
        for f in range(KNN_D):
            hq.unsafe_ptr().unsafe_store(
                i * KNN_D + f, hi.unsafe_ptr().unsafe_load(f)
            )
    print("input.index", _bits_hash(hi.unsafe_ptr(), KNN_N * KNN_D))
    print("input.queries", _bits_hash(hq.unsafe_ptr(), KNN_Q * KNN_D))

    var tile = knn_search(
        ctx,
        hi.unsafe_ptr(),
        KNN_N,
        hq.unsafe_ptr(),
        KNN_Q,
        KNN_D,
        KNN_K,
        hd.unsafe_ptr(),
        hidx.unsafe_ptr(),
        True,
    )
    print("query_tile", tile)
    print("output.distances", _bits_hash(hd.unsafe_ptr(), KNN_Q * KNN_K))
    var ih = UInt64(0xCBF29CE484222325)
    for i in range(KNN_Q * KNN_K):
        var v = UInt64(hidx.unsafe_ptr().unsafe_load(i))
        for byte in range(4):
            ih = (ih ^ ((v >> UInt64(byte * 8)) & UInt64(0xFF))) * UInt64(
                0x100000001B3
            )
    # SEPARATED FROM THE DISTANCES ON PURPOSE. Two runs that agree on the
    # distances and disagree here have diverged in the SELECTOR and nowhere
    # else, which is a one-line diagnosis the combined hash cannot give.
    print("output.indices", ih)
    _ = hi^
    _ = hq^
    _ = hd^
    _ = hidx^


def _run_dbscan(ctx: DeviceContext) raises:
    var hx = ctx.enqueue_create_host_buffer[DType.float32](DB_N * DB_D)
    var hl = ctx.enqueue_create_host_buffer[DType.int32](DB_N)
    ctx.synchronize()
    # Two blobs and a sparse bridge between them, the shape
    # `dbscan_identity_check` established: the bridge points are the ones
    # whose label is a CHOICE, and a fixture of separated blobs has none.
    var bridge_from = DB_N - 64
    for i in range(DB_N):
        if i < bridge_from:
            var center = Float32(0.0) if (i % 2) == 0 else Float32(10.0)
            for f in range(DB_D):
                hx.unsafe_ptr().unsafe_store(
                    i * DB_D + f,
                    center
                    + Float32(Int(_mix(i, f, 17) & UInt64(0xFF)))
                    / Float32(512.0),
                )
        else:
            var step = Float32(i - bridge_from) / Float32(64.0)
            for f in range(DB_D):
                var v = step * Float32(10.0)
                if f > 0:
                    v = Float32(
                        Int(_mix(i, f, 17) & UInt64(0xFF))
                    ) / Float32(2048.0)
                hx.unsafe_ptr().unsafe_store(i * DB_D + f, v)
    print("input.x", _bits_hash(hx.unsafe_ptr(), DB_N * DB_D))

    # `max_mbytes = 0` is cuML's default and means "ask the device": the
    # batch count is therefore NOT the same number on two machines, which is
    # why the trace records only batch-count-independent stages.
    var batches = dbscan_fit(
        ctx,
        hx.unsafe_ptr(),
        DB_N,
        DB_D,
        Float64(DB_EPS),
        DB_MIN_PTS,
        hl.unsafe_ptr(),
        # max_mbytes_per_batch = 0 is cuML's default, "ask the device".
        0,
        # max_iterations: THE DEFAULT, 200. Passing 0 here was this
        # driver's own bug on its first run and the refusal caught it --
        # under IDENTICAL a propagation that stops before its fixed point
        # raises (DEVIATION 507) instead of returning a snapshot of the
        # atomic order, which is exactly what a truncated run would be.
        200,
        # eps_nn_method: brute force, so the card does not depend on the
        # ball-cover index's own tuning. `dbscan_identity_check` gates that
        # the two arms agree.
        EPS_NN_BRUTE_FORCE,
    )
    print("batches", batches, "(device-dependent, NOT part of the claim)")
    var lh = UInt64(0xCBF29CE484222325)
    for i in range(DB_N):
        var v = UInt64(UInt32(hl.unsafe_ptr().unsafe_load(i)))
        for byte in range(4):
            lh = (lh ^ ((v >> UInt64(byte * 8)) & UInt64(0xFF))) * UInt64(
                0x100000001B3
            )
    print("output.labels", lh)
    _ = hx^
    _ = hl^


def main() raises:
    var arm = getenv("MOJOLEARN_UNSUP_ARM")
    print("arm", arm)
    print("mode", _mode_name())
    print("column", TARGET_COLUMN, "lane_width", lib_lane_width_for[
        TARGET_COLUMN
    ]())
    var trace = getenv("MOJOLEARN_IDENTITY_TRACE")
    if trace.byte_length() == 0:
        # NOT a warning that is easy to miss. A run with no trace path
        # produces no card, and a card that is missing is indistinguishable
        # from a card that agreed when someone later fetches a directory.
        raise Error(
            "MOJOLEARN_IDENTITY_TRACE is unset. This main exists to write a"
            " card; running it without one produces console hashes and"
            " nothing the differ can read. Use tools/e1_unsupervised.sh."
        )

    var ctx = DeviceContext()
    if arm == "kmeans":
        _run_kmeans(ctx)
    elif arm == "knn":
        _run_knn(ctx)
    elif arm == "dbscan":
        _run_dbscan(ctx)
    else:
        raise Error(
            "MOJOLEARN_UNSUP_ARM must be kmeans, knn or dbscan; got '"
            + arm
            + "'. One arm per process: the differ refuses a card whose"
            " sequence numbers restart."
        )
    print("done", arm)
