"""The one kernel k-means++ needs that is a fusion of two RAFT primitives.

NOT A PORT, but it is a direct translation of two consecutive RAFT calls in
`detail/kmeans.cuh:196-215`:

    matrix_vector_op<ALONG_ROWS>(pwd, minClusterDistance, minDistBuf, min_op)
    reduce<ALONG_ROWS>(minDistBuf, costPerCandidate, 0)

which is "for each candidate, what would the clustering cost become if this
candidate joined the centroid set". Their two passes write and then re-read
an `n_trials x n_samples` matrix; fusing them removes that buffer entirely,
and cannot change the answer because the reduction consumes each element
exactly once immediately after it is formed.

**That fusion is a DEVIATION and it is recorded as one** (PORTING.md 16),
even though it is arithmetically identical, because it changes the summation
ORDER over samples and therefore the last bits of `costPerCandidate`. When
two candidates tie to the last bit, a different order picks a different
centroid and the whole fit diverges. Ties at that precision are not expected
and are not impossible.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime PLUS_PLUS_TPB = 128


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
    (`detail/kmeans.cuh:190`). Same numbers, and reading down a column here
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
        var d = (
            x_norm.unsafe_load(i)
            + cn
            - Float32(2.0) * z.unsafe_load(i * n_trials + trial)
        )
        if d <= Float32(0.0):
            d = Float32(0.0)
        var cur = current_min.unsafe_load(i)
        acc += d if d < cur else cur
        i += PLUS_PLUS_TPB

    var s = stack_allocation[
        PLUS_PLUS_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    s[tid] = acc
    barrier()

    var half = PLUS_PLUS_TPB // 2
    while half > 0:
        if tid < half:
            s[tid] = s[tid] + s[tid + half]
        barrier()
        half //= 2

    if tid == 0:
        out_cost.unsafe_store(trial, s[0])


def adopt_candidate_min_kernel(
    current_min: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    cand_norm: MutPointer[Float32, MutAnyOrigin],
    n_samples_in: Int32,
    n_trials_in: Int32,
    chosen_in: Int32,
):
    """`detail/kmeans.cuh:230-234`, the copy that follows the argmin.

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
        var d = (
            x_norm.unsafe_load(i)
            + cn
            - Float32(2.0) * z.unsafe_load(i * n_trials + trial)
        )
        if d <= Float32(0.0):
            d = Float32(0.0)
        var cur = current_min.unsafe_load(i)
        if d < cur:
            current_min.unsafe_store(i, d)


# ---------------------------------------------------------------------------
# Device-side weighted sampling: what `raft::random::discrete` does for them.
#
# `detail/kmeans.cuh:187` draws the k-means++ candidates ON DEVICE. The first
# version of this port copied all `n_samples` distances to the host and drew
# there, once per accepted centroid, which is O(rows) host traffic and breaks
# `HOST_AND_DEVICE.md`'s first rule outright.
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
# (`detail/kmeans.cuh:224`). Host DECIDING was never the problem. Host WAITING
# on a row-sized transfer was.
# ---------------------------------------------------------------------------


def chunk_sums_kernel(
    out_partial: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    chunk_in: Int32,
):
    """Sum of one CONTIGUOUS chunk per block.

    Deliberately not the grid-stride `sum_partials_kernel`: that one is
    correct for a total and useless here, because a weighted draw needs the
    partials to correspond to contiguous ranges or the prefix over them means
    nothing.
    """
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

    var s = stack_allocation[
        PLUS_PLUS_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    s[tid] = acc
    barrier()

    var half = PLUS_PLUS_TPB // 2
    while half > 0:
        if tid < half:
            s[tid] = s[tid] + s[tid + half]
        barrier()
        half //= 2

    if tid == 0:
        out_partial.unsafe_store(Int(block_idx.x), s[0])


def select_chunk_kernel(
    out_chunk: MutPointer[Int32, MutAnyOrigin],
    out_residual: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    u01: MutPointer[Float32, MutAnyOrigin],
    n_chunks_in: Int32,
    n_trials_in: Int32,
):
    """One thread per trial over at most 256 chunk totals.

    `total` is recomputed here from the partials rather than passed in, so
    the target and the walk are consistent to the last bit. Passing a total
    summed by a different reduction tree would let a target exceed the walk's
    final accumulator and fall off the end.
    """
    var n_chunks = Int(n_chunks_in)
    var trial = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if trial >= Int(n_trials_in):
        return

    var total = Float32(0.0)
    for c in range(n_chunks):
        total += partials.unsafe_load(c)

    var target = u01.unsafe_load(trial) * total
    var acc = Float32(0.0)
    var chosen = n_chunks - 1
    var before = Float32(0.0)
    for c in range(n_chunks):
        before = acc
        acc += partials.unsafe_load(c)
        if acc >= target:
            chosen = c
            break

    out_chunk.unsafe_store(trial, Int32(chosen))
    out_residual.unsafe_store(trial, target - before)


def select_index_in_chunk_kernel(
    out_index: MutPointer[UInt32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    chunks: MutPointer[Int32, MutAnyOrigin],
    residuals: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    chunk_in: Int32,
):
    """One block per trial, scanning only its own chunk.

    Serial inside the chunk, which is the honest shape for a prefix search
    and is why the chunk count is capped: the serial walk is `n / n_chunks`
    steps, not `n`.
    """
    if Int(thread_idx.x) != 0:
        return
    var n = Int(n_in)
    var chunk = Int(chunk_in)
    var trial = Int(block_idx.x)

    var begin = Int(chunks.unsafe_load(trial)) * chunk
    var end = min(begin + chunk, n)
    var target = residuals.unsafe_load(trial)

    var acc = Float32(0.0)
    var chosen = end - 1
    var i = begin
    while i < end:
        acc += a.unsafe_load(i)
        if acc >= target:
            chosen = i
            break
        i += 1
    if chosen < 0:
        chosen = 0
    out_index.unsafe_store(trial, UInt32(chosen))


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
