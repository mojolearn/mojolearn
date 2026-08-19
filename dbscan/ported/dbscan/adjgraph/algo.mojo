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

    Single block with `SCAN_CHUNK` rows per thread: a serial pass per thread,
    a block scan over the thread totals, then a serial pass to write out.
    Standard, and the same shape as the scan in `select_radix.mojo`.
    """
    var n_rows = Int(n_rows_in)
    var tid = Int(thread_idx.x)

    var totals = stack_allocation[
        SCAN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var begin = tid * SCAN_CHUNK
    var end = min(begin + SCAN_CHUNK, n_rows)

    var sum = Int32(0)
    var i = begin
    while i < end:
        sum += vd.unsafe_load(i)
        i += 1
    totals[tid] = sum
    barrier()

    var offset = 1
    while offset < SCAN_TPB:
        var addend = Int32(0)
        if tid >= offset:
            addend = totals[tid - offset]
        barrier()
        totals[tid] = totals[tid] + addend
        barrier()
        offset *= 2

    var running = totals[tid] - sum  # exclusive prefix for this thread
    i = begin
    while i < end:
        ex_scan.unsafe_store(i, running)
        running += vd.unsafe_load(i)
        i += 1

    if tid == SCAN_TPB - 1:
        ex_scan.unsafe_store(n_rows, totals[SCAN_TPB - 1])


def compact_adjacency_kernel(
    col_ind: MutPointer[Int32, MutAnyOrigin],
    adj: MutPointer[UInt8, MutAnyOrigin],
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`adj_to_csr`: write every row's neighbor indices into its CSR slot.

    Every row, not only core rows. One thread per row, so the indices come
    out ascending where theirs come out in atomic order. See the module
    docstring: that is a recorded deviation, not a fix.
    """
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    if row >= Int(n_rows_in):
        return
    if Int(thread_idx.x) != 0:
        return

    var pos = Int(ex_scan.unsafe_load(row))
    for j in range(n_cols):
        if adj.unsafe_load(row * n_cols + j) != 0:
            col_ind.unsafe_store(pos, Int32(j))
            pos += 1
