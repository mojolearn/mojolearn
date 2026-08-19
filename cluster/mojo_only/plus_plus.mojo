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
