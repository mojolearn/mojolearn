# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The int32 CSR edge bound: exact counts, and a split that moves no label.

Lane dbscan-int64, 2026-09-29. The NVIDIA bench board (L40S, mojolearn
0.8.25) ran classical/dbscan on taxi (about 4.1M rows x 16, eps 3,
min_samples 2) and the IDENTICAL fit raised "the ball-cover neighbourhood
has -1799116104 edges in one batch". Two defects, one root: the count pass
returned the int32 exclusive scan's tail, which had WRAPPED (about 2.5e9
edges), and the runner refused on that wrapped number. A count past 2^32
wraps back POSITIVE and would have passed the check with a garbage CSR.

The fix, and what this driver gates:

1. `rbc_exact_edge_total_host` (`neighbors/impl/ball_cover/scan.mojo`) recovers
   the exact 64-bit count from a wrapped int32 scan. Gated on synthetic
   scans whose totals pass 2^31 and 2^32, where the tail reads negative and
   small-positive respectively.

2. The runner splits a ball-cover batch whose exact count passes `edge_cap`
   (MAX_LABEL in production) and re-counts the halves. Gated on a real fit:
   the labels at a forced tiny `edge_cap` (many uneven batches), at a forced
   small uniform batch, at both, and on the brute-force arm must equal the
   one-batch fit BIT FOR BIT, and the forced cap must actually have split.

THE FIXTURE, AND WHY ITS TAIL IS A CHAIN. Rows 0..599 are six dense blobs
(about 100 neighbours per row, so a small cap cuts them into many batches);
rows 600..629 are isolated noise; rows 630..999 are a chain at spacing 0.4
with eps 1.05, so each interior point has exactly 5 neighbours (itself and
two each side) and is core at min_samples 5, and the two ends are BORDER
points. A chain is what makes the last batch matter: its labels travel one
hop per propagation pass, so a batch whose edges are dropped leaves every
core point more than two hops from the previous batch unlabelled. That is
the sabotage arm `dbscan/checks/sabotage/dbscan_split_drops_last_batch.patch`,
which skips the LAST batch's labelling whenever there is more than one batch:
the one-batch baseline is untouched and every split fit disagrees with it.
The second arm, `dbscan_exact_total_reads_wrapped_tail.patch`, makes the
exact total return the wrapped tail and must fail part 1.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from dbscan.impl.adjgraph.algo import scan_blocks_needed
from dbscan.impl.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC, dbscan_fit
from dbscan.impl.sparse.detail.csr import MAX_LABEL
from neighbors.impl.ball_cover.scan import rbc_exact_edge_total_host


comptime ES_BLOBS = 6
comptime ES_PER_BLOB = 100
comptime ES_NOISE = 30
comptime ES_CHAIN = 370
comptime ES_N = ES_BLOBS * ES_PER_BLOB + ES_NOISE + ES_CHAIN
comptime ES_D = 4
comptime ES_EPS = 1.05
comptime ES_MIN_PTS = 5
comptime ES_CHAIN_FROM = ES_BLOBS * ES_PER_BLOB + ES_NOISE
#: Small enough that a blob row's ~100 neighbours force batches of a few
#: dozen rows, large enough that one chain row (5 neighbours) always fits.
comptime ES_EDGE_CAP = 1500


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
    if row < ES_BLOBS * ES_PER_BLOB:
        var blob = row // ES_PER_BLOB
        var base = 10.0 * Float64(blob) if feature == 0 else 0.0
        return Float32(base + _jitter(row, feature))
    if row < ES_CHAIN_FROM:
        var k = row - ES_BLOBS * ES_PER_BLOB
        return Float32(300.0 + 10.0 * Float64(k)) if feature == 0 else Float32(0.0)
    var c = row - ES_CHAIN_FROM
    if feature == 0:
        return Float32(150.0)
    if feature == 1:
        return Float32(0.4 * Float64(c))
    return Float32(0.0)


def _fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    batch: Int,
    method: Int,
    edge_cap: Int,
) raises -> Tuple[List[Int32], Int]:
    """One fit with fresh workspace; returns (labels, batches loop 1 used)."""
    var n = ES_N
    var b = batch if batch > 0 else n
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    var labels_temp = ctx.enqueue_create_buffer[DType.int32](n)
    var work_buffer = ctx.enqueue_create_buffer[DType.int32](n)
    var core = ctx.enqueue_create_buffer[DType.uint8](n)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](
        scan_blocks_needed(n) + 1
    )
    var adj = ctx.enqueue_create_buffer[DType.uint8](b * n)
    var vd = ctx.enqueue_create_buffer[DType.int32](b + 1)
    var ex = ctx.enqueue_create_buffer[DType.int32](b + 1)
    var nw = ctx.enqueue_create_buffer[DType.float32](1)
    var ws = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    var n_batches = List[Int](length=1, fill=0)
    _ = dbscan_fit(
        ctx, x, adj, vd, core, ex, labels, labels_temp, work_buffer,
        block_sums, nw, ws, n, ES_D, ES_EPS, ES_MIN_PTS, batch,
        n + 1, method,
        edge_cap=edge_cap,
        n_batches_out_addr=Int(n_batches.unsafe_ptr()),
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    var out = List[Int32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    return (out^, n_batches[0])


def _differ(a: List[Int32], b: List[Int32]) -> Int:
    var d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
    return d


def check_exact_total_past_the_wrap(ctx: DeviceContext) raises:
    """Part 1: a wrapped int32 scan still yields the exact count."""
    comptime ROWS = 4
    var h = ctx.enqueue_create_host_buffer[DType.int32](ROWS + 1)
    ctx.synchronize()
    # Degrees per row, each below 2^31 as any real degree is.
    var cases = List[List[Int]]()
    cases.append([1_250_000_000, 1_250_000_000, 7, 0])            # 2.5e9: tail < 0
    cases.append([2_000_000_000, 2_000_000_000, 1_000_000_000, 5])  # 5e9: tail > 0
    cases.append([3, 0, 11, 2])                                   # no wrap
    var tails = List[Int]()
    for c in range(len(cases)):
        var want = 0
        var run = Int32(0)
        h.unsafe_ptr().unsafe_store(0, run)
        for r in range(ROWS):
            want += cases[c][r]
            run = run + Int32(cases[c][r])   # the device scan's int32 wrap
            h.unsafe_ptr().unsafe_store(r + 1, run)
        tails.append(Int(h.unsafe_ptr().unsafe_load(ROWS)))
        var got = rbc_exact_edge_total_host(h, ROWS)
        if got != want:
            raise Error(
                "rbc_exact_edge_total_host case " + String(c) + ": got "
                + String(got) + ", the exact count is " + String(want)
                + " (the wrapped int32 tail reads "
                + String(Int(h.unsafe_ptr().unsafe_load(ROWS))) + ")"
            )
    # The fixture must hold the two hazards it claims: case 0's tail reads
    # negative (the L40S message), case 1's reads POSITIVE and below
    # MAX_LABEL (the wrap the old `nnz1 < 0 or nnz1 > MAX_LABEL` passed).
    if tails[0] >= 0:
        raise Error("fixture: case 0's wrapped tail should read negative")
    var tail1 = tails[1]
    if tail1 <= 0 or tail1 > Int(MAX_LABEL):
        raise Error("fixture: case 1's wrapped tail should read small positive")
    print(
        "check_exact_total_past_the_wrap OK: exact totals 2500000007 and"
        " 5000000005 recovered from int32 tails that read negative and "
        + String(tail1) + " (the silent-pass wrap)"
    )


def check_split_moves_no_label(ctx: DeviceContext) raises:
    """Part 2: forced splits and forced small batches, bit for bit."""
    var n = ES_N
    var x = ctx.enqueue_create_buffer[DType.float32](n * ES_D)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * ES_D)
    ctx.synchronize()
    for i in range(n):
        for f in range(ES_D):
            hx.unsafe_ptr().unsafe_store(i * ES_D + f, _coord(i, f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    var base = _fit(ctx, x, 0, EPS_NN_RBC, Int(MAX_LABEL))
    var ref_labels = base[0].copy()
    if base[1] != 1:
        raise Error("the default fit used " + String(base[1]) + " batches, not 1")

    # The fixture must be the one described: clusters, noise, and border
    # points at the chain's ends that carry a cluster label.
    var n_noise = 0
    var max_label = Int32(-1)
    for i in range(n):
        if ref_labels[i] < 0:
            n_noise += 1
        if ref_labels[i] > max_label:
            max_label = ref_labels[i]
    if n_noise != ES_NOISE or Int(max_label) + 1 != ES_BLOBS + 1:
        raise Error(
            "fixture: expected " + String(ES_NOISE) + " noise rows and "
            + String(ES_BLOBS + 1) + " clusters, got " + String(n_noise)
            + " and " + String(Int(max_label) + 1)
        )
    var end_a = ref_labels[ES_CHAIN_FROM]
    var end_b = ref_labels[n - 1]
    if end_a < 0 or end_a != end_b:
        raise Error("fixture: the chain's border ends are not one cluster")

    var arms = List[String]()
    var got = List[List[Int32]]()
    var nb = List[Int]()

    var a1 = _fit(ctx, x, 0, EPS_NN_RBC, ES_EDGE_CAP)
    arms.append("rbc, edge_cap " + String(ES_EDGE_CAP))
    got.append(a1[0].copy())
    nb.append(a1[1])
    var a2 = _fit(ctx, x, 100, EPS_NN_RBC, Int(MAX_LABEL))
    arms.append("rbc, batch 100")
    got.append(a2[0].copy())
    nb.append(a2[1])
    var a3 = _fit(ctx, x, 333, EPS_NN_RBC, ES_EDGE_CAP)
    arms.append("rbc, batch 333 + edge_cap " + String(ES_EDGE_CAP))
    got.append(a3[0].copy())
    nb.append(a3[1])
    var a4 = _fit(ctx, x, 0, EPS_NN_BRUTE_FORCE, Int(MAX_LABEL))
    arms.append("brute, one batch")
    got.append(a4[0].copy())
    nb.append(a4[1])

    # The forced cap must actually have split, into more batches than the
    # uniform arithmetic alone would give.
    if nb[0] < 8:
        raise Error(
            "edge_cap " + String(ES_EDGE_CAP) + " split the fit into only "
            + String(nb[0]) + " batches; the split path was not exercised"
        )
    if nb[2] <= (n + 332) // 333:
        raise Error("batch 333 + edge_cap did not split below the uniform batches")

    for k in range(len(arms)):
        var d = _differ(ref_labels, got[k])
        if d != 0:
            raise Error(
                String(d) + " of " + String(n) + " labels differ between the"
                " one-batch fit and '" + arms[k] + "' (" + String(nb[k])
                + " batches). Splitting a batch changes memory, not labels."
            )
        print(
            "  AGREE one batch vs " + arms[k] + ": " + String(nb[k])
            + " batches, " + String(n) + "/" + String(n) + " labels equal"
        )
    print(
        "check_split_moves_no_label OK: " + String(Int(max_label) + 1)
        + " clusters, " + String(n_noise) + " noise, chain borders labelled;"
        " every split and batch arm bitwise equal to one batch"
    )


def main() raises:
    # One context for both checks, matching the production estimator binding's
    # process_ctx lifetime. On RTX 4090, destroying the count-only context and
    # then allocating in a new context can deadlock before the first DBSCAN
    # kernel (the existing runtime hazard documented in core/neural_context,
    # DEVIATION 2513). This does not fix arbitrary context teardown/recreation.
    var ctx = DeviceContext()
    check_exact_total_past_the_wrap(ctx)
    check_split_moves_no_label(ctx)
    print("dbscan_edge_split_check: PASS")
