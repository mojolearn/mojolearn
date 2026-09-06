# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one kernel k-means++ needs that is a fusion of two RAFT primitives.

NOT A PORT, but it is a direct translation of two consecutive RAFT calls in
`detail/kmeans.cuh:189-217`:

    matrix_vector_op<ALONG_ROWS>(pwd, minClusterDistance, minDistBuf, min_op)
    reduce<ALONG_ROWS>(minDistBuf, costPerCandidate, 0)

which is "for each candidate, what would the clustering cost become if this
candidate joined the centroid set". Their two passes write and then re-read
an `n_trials x n_samples` matrix; fusing them removes that buffer entirely,
and cannot change the answer because the reduction consumes each element
exactly once immediately after it is formed.

**That fusion is a DEVIATION and it is recorded as one** (archive/reference/PORTING.md 16),
even though it is arithmetically identical, because it changes the summation
ORDER over samples and therefore the last bits of `costPerCandidate`. When
two candidates tie to the last bit, a different order picks a different
centroid and the whole fit diverges. Ties at that precision are not expected
and are not impossible.
"""

from checks.kernel_matrix import (
    K_LIB_PLUS_PLUS,
    TARGET_COLUMN,
    lib_block_size_for,
)


from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from core.pinned_reduce import pinned_block_sum
from checks.numerics import ftz, identical_mul_add
from max.gpu.primitives.block import prefix_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


# READ FROM THE MATRIX, not restated here. `checks/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime PLUS_PLUS_TPB = lib_block_size_for[K_LIB_PLUS_PLUS, TARGET_COLUMN]()


def candidate_cost_kernel(
    out_cost: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    cand_norm: MutPointer[Float32, MutAnyOrigin],
    current_min: MutPointer[Float32, MutAnyOrigin],
    n_samples_in: Int32,
    n_trials_in: Int32,
):
    """One block per candidate; the block strides the samples.

    `z` is `[n_samples x n_trials]`, the product this tree's GEMM produces
    when the candidates play the role of centroids. Their `pwd` is the
    transpose of that, `[n_trials x n_samples]`, because their
    `pairwise_distance_kmeans` call passes the candidates first
    (`detail/kmeans.cuh:190`, `raft::matrix::gather`). Same numbers, and
    reading down a column here
    keeps consecutive threads on consecutive samples.
    """
    var n_samples = Int(n_samples_in)
    var n_trials = Int(n_trials_in)
    var trial = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var cn = cand_norm.unsafe_load(trial)
    var acc = Float32(0.0)

    var i = tid
    while i < n_samples:
        # ONE pinned multiply-add (IDENTITY_PATHS row 9) with the seams
        # flushed (row 10). Under FAST it is bit-for-bit the subtraction it
        # replaces.
        var d = ftz(
            identical_mul_add(
                Float32(-2.0),
                ftz(z.unsafe_load(i * n_trials + trial)),
                ftz(ftz(x_norm.unsafe_load(i)) + ftz(cn)),
            )
        )
        if d <= Float32(0.0):
            d = Float32(0.0)
        var cur = current_min.unsafe_load(i)
        acc = ftz(acc + (d if d < cur else cur))
        i += PLUS_PLUS_TPB

    # `cub::BlockReduce`'s counterpart from
    # `max.gpu.primitives.block`. The hand-written shared-memory tree
    # reduction this replaced is gone: same arithmetic, one call, and
    # the reduction shape is Modular's to tune rather than ours to
    # guess. See archive/reference/VENDOR_LIBRARIES.md.
    var s0 = pinned_block_sum[PLUS_PLUS_TPB](acc)
    if tid == 0:
        out_cost.unsafe_store(trial, s0)


def adopt_candidate_min_kernel(
    current_min: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    cand_norm: MutPointer[Float32, MutAnyOrigin],
    n_samples_in: Int32,
    n_trials_in: Int32,
    chosen_in: Int32,
):
    """`detail/kmeans.cuh:249-257`, the two copies that follow the argmin.

    Theirs copies row `bestCandidateIdx` of the already-materialized
    `minDistBuf` into `minClusterDistance`. Having fused that buffer away,
    the winning column is recomputed instead, which is one extra pass over
    `n_samples` per accepted centroid and removes an `n_trials x n_samples`
    allocation for the whole init.
    """
    var n_samples = Int(n_samples_in)
    var n_trials = Int(n_trials_in)
    var trial = Int(chosen_in)
    var cn = cand_norm.unsafe_load(trial)

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n_samples:
        var d = ftz(
            identical_mul_add(
                Float32(-2.0),
                ftz(z.unsafe_load(i * n_trials + trial)),
                ftz(ftz(x_norm.unsafe_load(i)) + ftz(cn)),
            )
        )
        if d <= Float32(0.0):
            d = Float32(0.0)
        var cur = current_min.unsafe_load(i)
        if d < cur:
            current_min.unsafe_store(i, d)


# ---------------------------------------------------------------------------
# Device-side weighted sampling: what `raft::random::discrete` does for them.
#
# `detail/kmeans.cuh:189` (`raft::random::discrete`) draws the k-means++
# candidates ON DEVICE. The first
# version of this port copied all `n_samples` distances to the host and drew
# there, once per accepted centroid, which is O(rows) host traffic and breaks
# `archive/reference/HOST_AND_DEVICE.md`'s first rule outright.
#
# The replacement is a two-level search and moves NOTHING that scales with
# rows across the bus:
#
#   1. chunk sums          contiguous blocks, so a prefix over them is real
#   2. pick the chunk      one thread per trial over at most 256 chunk totals
#   3. pick inside it      one block per trial, scanning only its own chunk
#
# The host still draws the uniforms and still takes the greedy argmin over
# `n_trials` costs, and BOTH of those are correct to keep: they are
# O(candidates), which is what the rule permits, and cuVS also brings
# `bestCandidateIdx` to the host every accepted centroid
# (`detail/kmeans.cuh:242-244`). Host DECIDING was never the problem. Host WAITING
# on a row-sized transfer was.
# ---------------------------------------------------------------------------


def chunk_sums_kernel(
    out_partial: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    chunk_in: Int32,
):
    """Sum of one CONTIGUOUS chunk per block. Stage 1 of the device scan."""
    var n = Int(n_in)
    var chunk = Int(chunk_in)
    var tid = Int(thread_idx.x)
    var begin = Int(block_idx.x) * chunk
    var end = min(begin + chunk, n)

    var acc = Float32(0.0)
    var i = begin + tid
    while i < end:
        acc += a.unsafe_load(i)
        i += PLUS_PLUS_TPB

    var s0 = pinned_block_sum[PLUS_PLUS_TPB](acc)
    if tid == 0:
        out_partial.unsafe_store(Int(block_idx.x), s0)


def scan_chunk_offsets_kernel(
    offsets: MutPointer[Float32, MutAnyOrigin],
    totals: MutPointer[Float32, MutAnyOrigin],
    n_chunks_in: Int32,
):
    """Exclusive scan of the chunk totals. Stage 2. One block, any length.

    The per-thread slice is derived from `n_chunks` rather than fixed, which
    is the same lesson `dbscan/.../adjgraph/algo.mojo` paid for: a fixed
    chunk size silently caps the kernel at `threads * chunk` inputs.
    """
    var n_chunks = Int(n_chunks_in)
    var tid = Int(thread_idx.x)
    var per = (n_chunks + PLUS_PLUS_TPB - 1) // PLUS_PLUS_TPB
    var begin = min(tid * per, n_chunks)
    var end = min(begin + per, n_chunks)

    var sum = Float32(0.0)
    var i = begin
    while i < end:
        sum += totals.unsafe_load(i)
        i += 1

    var offset = prefix_sum[block_size=PLUS_PLUS_TPB, exclusive=True](sum)

    var running = offset
    i = begin
    while i < end:
        offsets.unsafe_store(i, running)
        running += totals.unsafe_load(i)
        i += 1


def write_inclusive_scan_kernel(
    csum: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    offsets: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    chunk_in: Int32,
):
    """Stage 3: `csum[i]` = running total of `a` up to and including `i`.

    One block per chunk. `block.prefix_sum` INCLUSIVE within the chunk, plus
    that chunk's exclusive offset from stage 2. Together the three stages are
    `cub::DeviceScan::InclusiveSum`, which has no shipped GPU counterpart:
    `nn.cumsum` is CPU-only. This is the genuine gap `archive/reference/VENDOR_LIBRARIES.md`
    records, built out of the block primitive that DOES ship.
    """
    var n = Int(n_in)
    var chunk = Int(chunk_in)
    var tid = Int(thread_idx.x)
    var begin = Int(block_idx.x) * chunk
    var base = offsets.unsafe_load(Int(block_idx.x))

    var carry = stack_allocation[
        1, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    if tid == 0:
        carry[0] = Float32(0.0)
    barrier()

    var c0 = 0
    while c0 < chunk:
        var i = begin + c0 + tid
        var v = Float32(0.0)
        if i < n and c0 + tid < chunk:
            v = a.unsafe_load(i)
        var inc = prefix_sum[block_size=PLUS_PLUS_TPB](v)
        if i < n and c0 + tid < chunk:
            csum.unsafe_store(i, base + carry[0] + inc)
        barrier()
        if tid == PLUS_PLUS_TPB - 1:
            carry[0] = carry[0] + inc
        barrier()
        c0 += PLUS_PLUS_TPB


def binary_search_kernel(
    out_index: MutPointer[UInt32, MutAnyOrigin],
    csum: MutPointer[Float32, MutAnyOrigin],
    u01: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    n_trials_in: Int32,
):
    """`sample_with_replacement_kernel`, transliterated.

    PORT OF `raft/random/detail/rng_device.cuh:697-727`, which is what
    `raft::random::discrete` reaches at `cuvs/.../kmeans.cuh:189`.

    **This replaces a DIFFERENT DECOMPOSITION of the same draw**, and the
    difference was the point. Theirs ranks per ELEMENT with a real prefix sum
    and then binary-searches it:

        IdxT idx_start = 0; IdxT idx_end = len;
        while (idx_end > idx_start) {
          IdxT idx_middle = (idx_start + idx_end) / 2;
          ...
        }

    Ours counted zeros per 256-element chunk and then LINEAR-WALKED the
    winning chunk on a single thread, with 127 of 128 threads returning
    immediately. Same distribution, `O(n / 256)` serial against their
    `O(log n)`, and ours had never been diffed against a reference draw.

    Their own comment there reads `// todo(lsugy): warp-collaborative binary
    search`, so even they consider this the unoptimized version. Copied as it
    is, not as they wish it were.
    """
    var n = Int(n_in)
    var trial = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if trial >= Int(n_trials_in):
        return

    var total = csum.unsafe_load(n - 1)
    var target = u01.unsafe_load(trial) * total

    var lo = 0
    var hi = n
    while hi > lo:
        var mid = (lo + hi) // 2
        if csum.unsafe_load(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    if lo >= n:
        lo = n - 1
    out_index.unsafe_store(trial, UInt32(lo))


def gather_rows_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    n_features_in: Int32,
):
    """`raft::matrix::gather` for the candidate rows.

    Grid x is the trial, so the whole candidate block is gathered in ONE
    launch. The version this replaces issued one `copy_f32_kernel` per trial
    from a host-side index, which needed the index on the host to exist at
    all.
    """
    var n_features = Int(n_features_in)
    var trial = Int(block_idx.x)
    var row = Int(indices.unsafe_load(trial))

    var f = Int(thread_idx.x)
    while f < n_features:
        dst.unsafe_store(
            trial * n_features + f, src.unsafe_load(row * n_features + f)
        )
        f += PLUS_PLUS_TPB
