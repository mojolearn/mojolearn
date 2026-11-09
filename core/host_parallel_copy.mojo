# SPDX-License-Identifier: Apache-2.0
"""THE ONE PARALLEL HOST BYTE COPY (lane fix-c3-c5-no-copies, 2026-11-08).

`host_parallel_copy_bytes(dst, src, nbytes)` copies a contiguous span in
contiguous byte blocks, one `memcpy` per task, split over the host's cores
through `core/host_parallel.mojo` (the one host thread split) with the
grain floor of `core/host_predict_threads.mojo`. The copied bytes do not
depend on the split: every task writes only its own disjoint block, and a
copy has no arithmetic, so no bit can move.

Why (docs/plans/REGRESSIONS_0866_VS_0855.md, cause C3): the owned copies
of a caller's X at kNN fit (`bindings/family_common.mojo`
`family_owned_snapshot`, 176 MB on istella, knn fit 8.5x) and IsolationForest
fit (`bindings/trees_public_iforest.mojo`, 1.8 GB, iforest fit 39.6x with C1)
were one serial `memcpy` on one host thread. Spans must not overlap; both
allocations stay alive for the whole call; no pointer is retained.
"""
from std.memory import memcpy
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

#: One block is the unit the tasks share out: 1 MiB, large enough that a
#: `memcpy` per block runs at streaming bandwidth, small enough that a
#: 64-core host still gets an even share of a 100 MB span. Cost reasoning,
#: not a shape: no rule here keys on a dataset dimension.
comptime HOST_COPY_BLOCK_BYTES = 1 << 20


def host_parallel_copy_bytes(dst_addr: Int, src_addr: Int, nbytes: Int):
    """Copy `nbytes` bytes from `src_addr` to `dst_addr` (disjoint spans),
    split into contiguous blocks over the host's cores. A span below the
    grain floor (2 * HOST_MIN_TASK_WORK four-byte words) is one `memcpy` on
    the calling thread."""
    if nbytes <= 0 or dst_addr == src_addr:
        return
    var blocks = (nbytes + HOST_COPY_BLOCK_BYTES - 1) // HOST_COPY_BLOCK_BYTES
    # Work in four-byte words: one load and one store per word, the unit
    # the grain floor of host_predict_threads is stated in.
    var tasks = host_predict_task_count_work(blocks, (nbytes + 3) // 4)
    if tasks <= 1:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst_addr),
               src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=src_addr), count=nbytes)
        return
    var chunk = host_predict_chunk(blocks, tasks) * HOST_COPY_BLOCK_BYTES

    def _span(t: Int) {imm dst_addr, imm src_addr, imm nbytes, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, nbytes)
        if lo < hi:
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst_addr + lo),
                   src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=src_addr + lo), count=hi - lo)

    host_parallelize(_span, tasks)
