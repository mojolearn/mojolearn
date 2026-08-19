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
    n_in: Int32,
):
    """`weak_cc_label_device`, copied including both propagation guards.

    One thread per vertex. Each pass pushes a smaller label to every
    neighbor it may propagate to, and pulls the smallest neighbor label it
    may accept. `changed` is their `*m`, and the host repeats until it stays
    zero.

    The asymmetry between the two arms is theirs and is load-bearing:
    pushing requires `ci_allow_prop` (the SOURCE must be core) while pulling
    requires `cj_allow_prop` (the NEIGHBOR must be core). A border point can
    therefore end up labelled by a core neighbor and still never join two
    clusters together.
    """
    var n = Int(n_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= n:
        return

    var start = Int(row_ind.unsafe_load(tid))
    var end = Int(row_ind.unsafe_load(tid + 1))

    var ci = labels.unsafe_load(tid)
    var ci_mod = False
    var ci_allow_prop = core.unsafe_load(tid) != 0

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
        _ = Atomic.min(labels.unsafe_offset(tid), ci)
        if ci_allow_prop:
            changed.unsafe_store(0, Int32(1))
