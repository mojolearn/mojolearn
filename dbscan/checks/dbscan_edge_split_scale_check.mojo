# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The int32 CSR edge bound AT SCALE: 5e9 edges in one batch, on a real GPU.

Lane dbscan-int64, 2026-09-29. The seam driver
`dbscan_edge_split_check.mojo` proves the split on a small fixture with a
forced `edge_cap`. This one reaches the production bound itself, which the
small fixture cannot: two tight cliques of 50,000 rows each, 4 features, so
every row is within eps of its whole clique and one batch over all 100,000
rows holds 2 x 50,000^2 = 5,000,000,000 edges.

That number is chosen to be the SILENT case. Mod 2^32 it is 705,032,704:
positive and under MAX_LABEL, so the old `nnz1 < 0 or nnz1 > MAX_LABEL`
test PASSED it and sized a 705M-entry CSR for 5e9 columns. Now the exact
count is 5e9, the batch splits (100,000 -> 50,000 rows at 2.5e9 edges, still
over -> 25,000 rows at 1.25e9 edges each, 4 batches), and the answer must be
exactly two clusters: label 0 on rows 0..49,999, label 1 on the rest.

Memory, on the ball-cover arm with `batch = n_rows`: the dense `adj` the
NVIDIA path allocates is n x n = 1e10 bytes, the largest batch's columns
1.25e9 x 4 = 5e9 bytes; the whole graph (5e9 x 4 = 2e10 bytes) is never
resident. About 16 GB of device memory; heavy, so it is not in the seam
listing. Run it on a 48 GB GPU:

    sh tools/with_identical_mode.sh pixi run mojo run -I . \
        dbscan/checks/dbscan_edge_split_scale_check.mojo
"""

from max.gpu.host import DeviceContext

from dbscan.impl.adjgraph.algo import scan_blocks_needed
from dbscan.impl.runner import EPS_NN_RBC, dbscan_fit
from dbscan.impl.sparse.detail.csr import MAX_LABEL


comptime SC_HALF = 50_000
comptime SC_N = 2 * SC_HALF
comptime SC_D = 4


def _coord(row: Int, feature: Int) -> Float32:
    # A clique: every row within 0.01 * 2 of every other row of its half.
    var z = UInt64(row) * 0x9E3779B97F4A7C15 + UInt64(feature + 1) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 31)) * 0x94D049BB133111EB
    var j = (Float64(z >> 11) * (1.0 / 9007199254740992.0) - 0.5) * 0.02
    var base = 100.0 if (row >= SC_HALF and feature == 0) else 0.0
    return Float32(base + j)


def main() raises:
    var ctx = DeviceContext()
    var n = SC_N
    var x = ctx.enqueue_create_buffer[DType.float32](n * SC_D)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * SC_D)
    ctx.synchronize()
    for i in range(n):
        for f in range(SC_D):
            hx.unsafe_ptr().unsafe_store(i * SC_D + f, _coord(i, f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    var labels_temp = ctx.enqueue_create_buffer[DType.int32](n)
    var work_buffer = ctx.enqueue_create_buffer[DType.int32](n)
    var core = ctx.enqueue_create_buffer[DType.uint8](n)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](
        scan_blocks_needed(n) + 1
    )
    var adj = ctx.enqueue_create_buffer[DType.uint8](n * n)
    var vd = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var ex = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var nw = ctx.enqueue_create_buffer[DType.float32](1)
    var ws = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()

    var wrapped = truncate_to_32(2 * SC_HALF * SC_HALF)
    print(
        "scale: " + String(n) + " rows, one batch of "
        + String(2 * SC_HALF * SC_HALF) + " edges; the int32 tail reads "
        + String(wrapped) + " (MAX_LABEL " + String(Int(MAX_LABEL)) + ")"
    )
    # lane dbscan-taxi-speed: arm 0 pins the split route (edge_free=False);
    # arm 1 lets the fit take the edge-free route (IDN_DBSCAN_EDGE_FREE,
    # where compiled in) and must give the same two cliques.
    for arm in range(2):
        var allow_ef = arm == 1
        var n_batches = List[Int](length=1, fill=0)
        _ = dbscan_fit(
            ctx, x, adj, vd, core, ex, labels, labels_temp, work_buffer,
            block_sums, nw, ws, n, SC_D, 1.0, 2, 0, n + 1, EPS_NN_RBC,
            phase_timing=True,
            n_batches_out_addr=Int(n_batches.unsafe_ptr()),
            edge_free=allow_ef,
        )
        var h = ctx.enqueue_create_host_buffer[DType.int32](n)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=labels)
        ctx.synchronize()
        var bad = 0
        for i in range(n):
            var want = Int32(0) if i < SC_HALF else Int32(1)
            if h.unsafe_ptr().unsafe_load(i) != want:
                bad += 1
        if bad != 0:
            raise Error(
                String(bad) + " of " + String(n) + " rows are not in their"
                " clique's cluster after " + String(n_batches[0])
                + " batches (edge_free " + String(allow_ef) + ")"
            )
        if not allow_ef and n_batches[0] < 4:
            raise Error(
                "expected the 5e9-edge batch to split into at least 4 batches,"
                " got " + String(n_batches[0])
            )
        print(
            "dbscan_edge_split_scale_check: arm edge_free " + String(allow_ef)
            + ": 5000000000 edges, " + String(n_batches[0])
            + " batches/ranges, both cliques whole, labels 0 and 1"
        )
    print("dbscan_edge_split_scale_check: PASS")


def truncate_to_32(v: Int) -> Int:
    """`v` mod 2^32 as a signed int32 would read it."""
    var m = v % 4294967296
    return m - 4294967296 if m >= 2147483648 else m
