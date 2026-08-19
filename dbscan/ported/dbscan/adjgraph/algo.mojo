"""Boolean adjacency to CSR.

PORT OF `cuml/cpp/src/dbscan/adjgraph/algo.cuh` at cuML `7e29955c`.
Partial. Do not improve.

Theirs is an exclusive scan of the vertex degrees followed by a compaction,
and both are copied. The reason it exists at all is memory: the boolean
adjacency is `batch_size x N` bytes and the CSR is `nnz` indices, and for the
sparse graphs DBSCAN actually produces the second is far smaller.

**THE CSR IS BUILT FROM EVERY ROW, CORE OR NOT.** I first wrote this with a
core-point filter in both the scan and the compaction, which was an
invention: their `thrust::exclusive_scan` runs over the whole `vd` array with
no mask, and `adj_to_csr` emits every non-zero.

The core restriction is real but it lives one step later, in `weak_cc`'s
`filter_op` (`dbscan/runner.cuh:365`), which decides which vertices may
PROPAGATE a label. Putting it here instead gives a graph missing the
border-point edges that the labeler still needs to see in order to attach
borders to clusters. Same idea, wrong place, and the fix was to read their
file instead of describing it.

**Their compaction is deliberately UNORDERED**, and their own docstring says
so: "High performance comes at the cost of non-deterministic output: the
column indices are not guaranteed to be stored in order"
(`raft/sparse/convert/detail/adj_to_csr.cuh`). Multiple blocks cooperate on
one row through an atomic counter. That is the THIRD documented source of
run-to-run nondeterminism found in these upstreams, after CatBoost's float
atomics and RAFT's radix-select tie handling. It does not change DBSCAN's
answer, because label propagation converges to the same fixed point whatever
order the edges are visited in.

Ours emits in ascending order, one thread per row. That is a DEVIATION and it
is recorded as one rather than sold as an improvement: it is the simple
correct thing, it makes the CSR diffable against a host reference, and it
gives up their multi-block cooperation on a single wide row. If a row is wide
enough for that to matter, this is the first thing to revisit.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import prefix_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime SCAN_TPB = 256
comptime SCAN_CHUNK = 64


def exclusive_scan_kernel(
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """`thrust::exclusive_scan` over EVERY vertex degree. One block.

    No core mask: see the module docstring for why that was wrong.
    `ex_scan[n_rows]` ends as the total `nnz`.

    **THE CHUNK SIZE IS NOW DYNAMIC, AND THE FIXED ONE WAS A SILENT
    CORRECTNESS BUG.** The first version used `SCAN_CHUNK = 64` rows per
    thread against `SCAN_TPB = 256` threads, which caps the kernel at 16,384
    rows. Past that it simply stopped scanning, produced a wrong `nnz` and a
    truncated CSR, and raised nothing. Found by audit, not by a test: every
    fixture in this repository is smaller than the cap.

    The per-thread block scan is `max.gpu.primitives.block.prefix_sum` with
    `exclusive=True`, which is `cub::BlockScan::ExclusiveSum`'s counterpart
    and is what the hand-written Hillis-Steele here replaced.
    """
    var n_rows = Int(n_rows_in)
    var tid = Int(thread_idx.x)

    var chunk = (n_rows + SCAN_TPB - 1) // SCAN_TPB
    var begin = tid * chunk
    var end = min(begin + chunk, n_rows)
    if begin > n_rows:
        begin = n_rows
    if end < begin:
        end = begin

    var sum = Int32(0)
    var i = begin
    while i < end:
        sum += vd.unsafe_load(i)
        i += 1

    var offset = prefix_sum[block_size=SCAN_TPB, exclusive=True](sum)

    var running = offset
    i = begin
    while i < end:
        ex_scan.unsafe_store(i, running)
        running += vd.unsafe_load(i)
        i += 1

    if tid == SCAN_TPB - 1:
        ex_scan.unsafe_store(n_rows, offset + sum)


def compact_adjacency_kernel(
    col_ind: MutPointer[Int32, MutAnyOrigin],
    adj: MutPointer[UInt8, MutAnyOrigin],
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`adj_to_csr`: write every row's neighbor indices into its CSR slot.

    **ALL THREADS NOW WORK.** The first version opened with
    `if thread_idx.x != 0: return` and let ONE thread walk an entire row of
    `n_cols`, which was the largest raw-parallelism loss in the port. Theirs
    (`raft/sparse/convert/detail/adj_to_csr.cuh`) runs 512 threads per row
    with vectorized `chunk_bool` loads and a per-row atomic counter.

    Ours uses `block.prefix_sum` instead of their atomic counter, which is
    the one deliberate difference and it BUYS something: theirs is
    documented non-deterministic in column order, ours comes out ascending.
    Same set, and a diffable CSR is worth more here than their last increment
    of throughput.
    """
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    if row >= Int(n_rows_in):
        return
    var tid = Int(thread_idx.x)
    var base = Int(ex_scan.unsafe_load(row))

    var written = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    if tid == 0:
        written[0] = Int32(0)
    barrier()

    var c0 = 0
    while c0 < n_cols:
        var c = c0 + tid
        var hit = Int32(0)
        if c < n_cols and adj.unsafe_load(row * n_cols + c) != 0:
            hit = Int32(1)
        var pos = prefix_sum[block_size=SCAN_TPB, exclusive=True](hit)
        if hit != 0:
            col_ind.unsafe_store(
                base + Int(written[0]) + Int(pos), Int32(c)
            )
        # BARRIER BEFORE THE CARRY, not only after it. Without this the last
        # thread can advance `written[0]` while other threads are still
        # reading it for their own store, and the CSR comes out with rows
        # overlapping. That merged two DBSCAN blobs on the first run.
        barrier()
        if tid == SCAN_TPB - 1:
            written[0] = written[0] + pos + hit
        barrier()
        c0 += SCAN_TPB
