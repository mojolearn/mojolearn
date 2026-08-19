"""Weakly connected components by label propagation.

PORT OF `raft/sparse/detail/csr.cuh::weak_cc_label_device` and
`weak_cc_init_all_kernel` at RAFT `9aa17e5`. Transliterated. Do not improve.

This is the step that turns DBSCAN's neighbor graph into clusters, and it is
the only part of DBSCAN that is not distance arithmetic.

THE `filter_op` IS THE WHOLE OF DBSCAN'S SEMANTICS
--------------------------------------------------
`weak_cc` is a general primitive; DBSCAN passes it a lambda that returns
whether a vertex is a CORE point (`cuml/cpp/src/dbscan/runner.cuh:365`). That
one predicate is what makes this DBSCAN rather than plain connected
components:

- a core point may PROPAGATE its label to its neighbors,
- a border point may RECEIVE a label but never pass one on,
- so two clusters touching a common border point stay separate,

which is the single most commonly got-wrong part of DBSCAN. Their kernel
encodes it as `ci_allow_prop` and `cj_allow_prop`, and both guards are
copied exactly.

Initialization is theirs too: a core point starts labelled `i + 1` and a
non-core point starts at `MAX_LABEL`, so the min-propagation naturally leaves
untouched non-core points at `MAX_LABEL` and those are the noise.

DETERMINISM
-----------
The propagation uses `atomicMin`, so the ORDER of updates varies run to run,
but the fixed point does not: a minimum over a set is the same whatever order
it is taken in. This is one of the rare places in these ports where an
atomic costs nothing in reproducibility, the same situation as the k-means
assignment argmin and unlike CatBoost's float histogram flush.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer


comptime WEAK_CC_TPB = 256
comptime MAX_LABEL = Int32(2147483647)


def weak_cc_init_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    n_in: Int32,
):
    """`weak_cc_init_all_kernel`, copied.

        if (filter_op(tid)) labels[tid] = tid + 1;
        else                labels[tid] = MAX_LABEL;
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        if core.unsafe_load(tid) != 0:
            labels.unsafe_store(tid, Int32(tid + 1))
        else:
            labels.unsafe_store(tid, MAX_LABEL)


def weak_cc_label_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    row_ind: MutPointer[Int32, MutAnyOrigin],
    col_ind: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    changed: MutPointer[Int32, MutAnyOrigin],
    start_vertex_id_in: Int32,
    batch_size_in: Int32,
    n_in: Int32,
):
    """`weak_cc_label_device`, copied including both propagation guards.

    One thread per vertex OF THE BATCH. `tid` indexes the batch's CSR rows and
    `global_id = tid + start_vertex_id` indexes `labels` and the core mask,
    which is theirs (`csr.cuh:61-63`) and is what makes the CSR a
    `batch_size x N` slice rather than an `N x N` graph. The first version of
    this port used `tid` for both and only worked because it was handed one
    global CSR.

    Each pass pushes a smaller label to every neighbor it may propagate to,
    and pulls the smallest neighbor label it may accept. `changed` is their
    `*m`, and the host repeats until it stays zero.

    The asymmetry between the two arms is theirs and is load-bearing:
    pushing requires `ci_allow_prop` (the SOURCE must be core) while pulling
    requires `cj_allow_prop` (the NEIGHBOR must be core). A border point can
    therefore end up labelled by a core neighbor and still never join two
    clusters together.

    `end` comes from `row_ind[tid + 1]` rather than their
    `get_stop_idx(tid, batch_size, nnz, row_ind)`. That is the extra element
    their own TODO one line above asks for (`csr.cuh:72`, "add one element to
    row_ind and avoid get_stop_idx"), and the scan in
    `adjgraph/algo.mojo` writes it.
    """
    var n = Int(n_in)
    var batch_size = Int(batch_size_in)
    var start_vertex_id = Int(start_vertex_id_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var global_id = tid + start_vertex_id
    if tid >= batch_size or global_id >= n:
        return

    var start = Int(row_ind.unsafe_load(tid))
    var end = Int(row_ind.unsafe_load(tid + 1))

    var ci = labels.unsafe_load(global_id)
    var ci_mod = False
    var ci_allow_prop = core.unsafe_load(global_id) != 0

    for j in range(start, end):
        var j_ind = Int(col_ind.unsafe_load(j))
        var cj = labels.unsafe_load(j_ind)
        var cj_allow_prop = core.unsafe_load(j_ind) != 0
        if ci < cj and ci_allow_prop:
            _ = Atomic.min(labels.unsafe_offset(j_ind), ci)
            if cj_allow_prop:
                changed.unsafe_store(0, Int32(1))
        elif ci > cj and cj_allow_prop:
            ci = cj
            ci_mod = True

    if ci_mod:
        _ = Atomic.min(labels.unsafe_offset(global_id), ci)
        if ci_allow_prop:
            changed.unsafe_store(0, Int32(1))


def weak_cc_batched(
    ctx: DeviceContext,
    mut labels: DeviceBuffer[DType.int32],
    mut row_ind: DeviceBuffer[DType.int32],
    mut col_ind: DeviceBuffer[DType.int32],
    mut core: DeviceBuffer[DType.uint8],
    mut d_changed: DeviceBuffer[DType.int32],
    mut h_changed: HostBuffer[DType.int32],
    n_rows: Int,
    start_vertex_id: Int,
    batch_size: Int,
    max_iterations: Int,
) raises -> Int:
    """`weak_cc_batched` (`csr.cuh:133`), copied, returning the pass count.

    **THE INITIALIZATION IS INSIDE, AND OVER ALL `N`, NOT OVER THE BATCH.**
    Theirs calls `weak_cc_init_all_kernel` on every fit of every batch
    (`csr.cuh:149`), so each batch starts from a fresh labelling of the whole
    dataset and the caller merges the results afterwards. Reusing the
    previous batch's labels as the initial value and skipping the merge is
    the exact bug their comment at `runner.cuh:392-395` names, cuML issue
    #3094.

    The convergence flag comes back to the host every pass, which is theirs
    too: `raft::update_host(&host_m, state->m, 1, stream)` inside their
    `do { } while (host_m)`.
    """
    ctx.enqueue_function[weak_cc_init_kernel](
        labels.unsafe_ptr(),
        core.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=((n_rows + WEAK_CC_TPB - 1) // WEAK_CC_TPB, 1, 1),
        block_dim=(WEAK_CC_TPB, 1, 1),
    )
    ctx.synchronize()

    var passes = 0
    for _it in range(max_iterations):
        h_changed.unsafe_ptr().unsafe_store(0, Int32(0))
        ctx.enqueue_copy(dst_buf=d_changed, src_ptr=h_changed.unsafe_ptr())
        ctx.enqueue_function[weak_cc_label_kernel](
            labels.unsafe_ptr(),
            row_ind.unsafe_ptr(),
            col_ind.unsafe_ptr(),
            core.unsafe_ptr(),
            d_changed.unsafe_ptr(),
            Int32(start_vertex_id),
            Int32(batch_size),
            Int32(n_rows),
            grid_dim=((batch_size + WEAK_CC_TPB - 1) // WEAK_CC_TPB, 1, 1),
            block_dim=(WEAK_CC_TPB, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=h_changed.unsafe_ptr(), src_buf=d_changed)
        ctx.synchronize()
        passes += 1
        if h_changed.unsafe_ptr().unsafe_load(0) == Int32(0):
            break
    return passes
