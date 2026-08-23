"""The callable surface over DBSCAN.

`dbscan/ported/dbscan/dbscan.mojo:132` already has `dbscan_fit_impl`,
mirroring `cuml/cpp/src/dbscan/dbscan.cuh:101`. It takes `DeviceBuffer`s, so
a caller holding a numpy array cannot reach it. This file is the same shape as
`neighbors/estimator.mojo` and `cluster/estimator.mojo`: host pointers in,
device work owned here, results read back.

**This one is thinner than the k-means surface, and the reason is worth
stating.** k-means needed the estimator to choose fixed-point accumulator
scales, which is a real policy decision with a wrong answer. DBSCAN has no
such knob: its labels are integers and its only tunables are eps and
min_samples, which the caller states. So there is less here, and the small
amount that IS here is listed below rather than being spread through the
code.

THE POLICY CHOICES
------------------

1. **THE LABEL CONVENTION IS scikit-learn's AND IT COMES FROM THE PORT, NOT
   FROM HERE.** Noise is `-1` and clusters are exactly `0..n_clusters-1`.
   That is what cuML's `final_relabel` plus `relabelForSkl`
   (`runner.cuh:410-416`) produce and what `dbscan/mojo_only/dbscan_check.mojo`
   asserts. This file does not renumber anything, and a caller can compare
   `labels_` against scikit-learn's directly.

2. **`eps_nn_method` DEFAULTS TO cuML's DEFAULT, `EPS_NN_RBC`**, the random
   ball cover, not brute force. It is left at their dispatch rather than
   pinned here, for the same reason `knn_search` does not override
   `KNN_METHOD_AUTO`: this file does not get an opinion about which kernel
   runs. `EPS_NN_BRUTE_FORCE` is reachable by passing it explicitly and is
   the arm `dbscan_brute` in the benchmark compares against.

3. **`max_mbytes_per_batch = 0` MEANS THEIR ESTIMATE, NOT "NO LIMIT".** Zero
   selects cuML's own `80% * total - dataset` rule
   (`dbscan.cuh:157-158`), which reads TOTAL device memory rather than free,
   because their own comment says the free figure cannot be relied on. Any
   other value is used as given and unvalidated, which is also theirs.

4. **THE RETURN IS THE BATCH COUNT**, which is how many passes the data was
   split into. It is surfaced rather than dropped because a run that batched
   is a run whose memory budget bound it, and that is worth knowing beside a
   timing number.

WHAT IS NOT HERE YET, NAMED SO IT IS NOT MISTAKEN FOR DONE
----------------------------------------------------------

- `sample_weight`. cuML supports it; nothing in this repository exercises it.
- Metrics other than L2. The ported kernels carry only that arm.
- `core_sample_indices_`. scikit-learn exposes it; the port does not compute
  it separately (`dbscan.cuh:171-173` notes theirs is not returned either).
- The CPython binding. This is the Mojo half only.
"""

from max.gpu.host import DeviceContext

from dbscan.ported.dbscan.dbscan import dbscan_fit_impl
from dbscan.ported.dbscan.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC


def dbscan_fit(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_samples: Int,
    n_features: Int,
    eps: Float64,
    min_samples: Int,
    out_labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    max_mbytes_per_batch: Int = 0,
    max_iterations: Int = 200,
    eps_nn_method: Int = EPS_NN_RBC,
) raises -> Int:
    """Cluster host-resident row-major data. Returns the PROPAGATION PASSES.

    THE RETURN VALUE WAS DOCUMENTED WRONG until 2026-08-23 ("Returns the
    batch count"). It is `dbscan_fit_impl`'s return, which that function's
    own docstring names correctly: the total number of `weak_cc` passes
    summed over the batches. The batch count is not returned by anything;
    it reaches the outside world only through the identity-trace header
    (`batches=`) and the `phase_timing` lines. The wrong sentence was
    load-bearing -- `check_dbscan_batch_count_invariance` named the value
    `batches` and printed it as one, so its diagnostics reported pass
    counts as batch counts and it never checked that its budgets moved the
    batch count at all.

    `x_ptr` is `n_samples x n_features`, row-major, float32.
    `out_labels_ptr` is `n_samples` Int32 and is written with scikit-learn's
    convention: `-1` for noise, `0..n_clusters-1` otherwise.

    Raises rather than clamping on a shape the kernels cannot serve, because
    a clamp returns a wrong answer quietly.
    """
    if n_samples < 1 or n_features < 1:
        raise Error(
            "dbscan_fit needs n_samples and n_features >= 1: got "
            + String(n_samples)
            + ", "
            + String(n_features)
        )
    if not (eps > 0.0):
        raise Error(
            "dbscan_fit needs eps > 0, got " + String(eps)
        )
    if min_samples < 1:
        raise Error(
            "dbscan_fit needs min_samples >= 1, got " + String(min_samples)
        )
    if eps_nn_method != EPS_NN_RBC and eps_nn_method != EPS_NN_BRUTE_FORCE:
        raise Error(
            "dbscan_fit: eps_nn_method must be EPS_NN_RBC (1) or"
            " EPS_NN_BRUTE_FORCE (0), got " + String(eps_nn_method)
        )

    var x = ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var labels = ctx.enqueue_create_buffer[DType.int32](n_samples)
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.synchronize()

    var passes = dbscan_fit_impl(
        ctx,
        x,
        labels,
        n_samples,
        n_features,
        eps,
        min_samples,
        max_mbytes_per_batch,
        max_iterations,
        eps_nn_method,
        False,
    )

    var hl = ctx.enqueue_create_host_buffer[DType.int32](n_samples)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    for i in range(n_samples):
        out_labels_ptr.unsafe_store(i, hl.unsafe_ptr().unsafe_load(i))

    return passes
