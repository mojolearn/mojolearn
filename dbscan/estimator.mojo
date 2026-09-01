# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The callable surface over DBSCAN.

`dbscan/impl/dbscan/dbscan.mojo:132` already has `dbscan_fit_impl`,
mirroring `cuml/cpp/src/dbscan/dbscan.cuh:101`. It takes `DeviceBuffer`s, so
a caller holding a numpy array cannot reach it. This file is the same shape as
`neighbors/estimator.mojo` and `cluster/estimator.mojo`: host pointers in,
device work owned here, results read back.

**This one is thinner than the k-means surface, and the reason is worth
stating.** k-means needed the estimator to choose fixed-point accumulator
scales, which is a real policy decision with a wrong answer. DBSCAN has no
such knob: its labels are integers and its tunables are eps, min_samples,
the metric and the per-sample weights, all of which the caller states. So
there is less here, and the small amount that IS here is listed below rather
than being spread through the code.

THE POLICY CHOICES
------------------

1. **THE LABEL CONVENTION IS scikit-learn's AND IT COMES FROM THE PORT, NOT
   FROM HERE.** Noise is `-1` and clusters are exactly `0..n_clusters-1`.
   That is what cuML's `final_relabel` plus `relabelForSkl`
   (`runner.cuh:410-416`) produce and what `dbscan/checks/dbscan_check.mojo`
   asserts. This file does not renumber anything, and a caller can compare
   `labels_` against scikit-learn's directly.

2. **`eps_nn_method` DEFAULTS TO `EPS_NN_RBC`, the random ball cover, AND
   THAT IS THIS PORT'S DEFAULT, NOT cuML's** -- DEVIATION 35 in
   `dbscan/impl/dbscan/runner.mojo`: cuML's Python default is
   `algorithm='brute'` and on an int32-label build their dispatch never
   reaches RBC. (This item used to say "cuML's default"; corrected
   2026-08-23.) The ball cover measured 2.7x-27x faster at 16k-200k rows
   and `check_dbscan_rbc_matches_brute` holds the two labellings identical
   point for point. This file does not add an opinion beyond the runner's
   default; `EPS_NN_BRUTE_FORCE` is reachable by passing it explicitly
   (`mojolearn.DBSCAN(algorithm='brute')`) and is the arm `dbscan_brute` in
   the benchmark compares against and the arm `E1U_RESULTS.md` certified.

3. **`max_mbytes_per_batch = 0` MEANS THEIR ESTIMATE, NOT "NO LIMIT".** Zero
   selects cuML's own `80% * total - dataset` rule
   (`dbscan.cuh:157-158`), which reads TOTAL device memory rather than free,
   because their own comment says the free figure cannot be relied on. Any
   other value is used as given and unvalidated, which is also theirs.

4. **THE RETURN IS THE PROPAGATION PASS COUNT**, summed over the batches
   -- NOT the batch count, which this item claimed until 2026-08-23 (the
   function docstring below records the history; the batch count reaches
   the outside only through the identity-trace header's `batches=`). It is
   surfaced because a pass count far above the batch count is a long
   label chain, and DEVIATION 519 is about exactly that.

5. **`sample_weight` IS HONORED AND ENTERS IN EXACTLY ONE PLACE** (added
   2026-09-01). A point is core when the SUM OF THE WEIGHTS in its
   eps-neighbourhood reaches `min_samples`, not when the COUNT does; that is
   cuML's `runner.cuh:300-306` and scikit-learn's `_dbscan.py:451-455`, and
   they agree. Nothing else in the fit reads it. The weight sum is a FLOAT
   reduction, so its fold is pinned exactly like every other reduction in
   this tree (DEVIATION 28, `dbscan/impl/dbscan/vertexdeg/algo.mojo`);
   an unpinned fold would put an AMD fit and a CUDA fit on opposite sides of
   `>= min_samples` for a point sitting on the threshold, which is a
   different MODEL and not a last-bit difference in a reported number.
   Passing all-ones reproduces the unweighted labels exactly, and
   `check_dbscan_uniform_weight_matches_unweighted` gates that.

6. **`metric` CARRIES AN L1 ARM AND IT IS ORIGINAL WORK** (added
   2026-09-01, DEVIATION 27). `metric = DBSCAN_METRIC_L2` is the ported
   arm. `DBSCAN_METRIC_L1` is not a port of anything: cuML's DBSCAN offers
   euclidean, cosine and precomputed and no Manhattan
   (`dbscan.pyx:110-115`), so there is no upstream eps-neighbourhood kernel
   to be faithful to and no `DERIVATION_MAP.tsv` row points anywhere for it.
   The per-pair arithmetic IS credited, to RAFT's `l1.cuh:49`. The one thing
   a reader must know before touching it: the L2 arm compares a SQUARED
   distance against `eps * eps`, and an L1 sum has no squared form, so the
   threshold is not squared on that arm. `dbscan_metric_threshold` is the
   only place that decides which.

WHAT IS NOT HERE YET, NAMED SO IT IS NOT MISTAKEN FOR DONE
----------------------------------------------------------

- The L1 arm on the BALL COVER. `neighbors/impl/neighbors/ball_cover/`
  computes Euclidean distances for its landmark radii and its three pruning
  bounds, so `metric='manhattan'` is served by `algorithm='brute'` and
  refused by name on `'rbc'`. The ball cover's pruning rests on the triangle
  inequality, which L1 satisfies, so this is reachable work in that lane and
  not a property of the algorithm.
- `metric='cosine'` and `metric='precomputed'`, cuML's other two
  (`dbscan.pyx:110-115`); the Cosine arm is two `matrixVectorOp` passes plus
  `eps2 = 2 * eps` around the same kernel (`algo.cuh:186-223`).
- `core_sample_indices_`. scikit-learn exposes it; the port does not compute
  it separately (`dbscan.cuh:171-173` notes theirs is not returned either).
- `eps_nn_method` and `max_iterations` cross the CPython binding since
  2026-08-23 (`bindings/_mojolearn_estimators.mojo`, slots 5 and 6;
  `python/mojolearn/density.py` `algorithm=` / `max_iterations=`).
  `metric` and `sample_weight` cross it since 2026-09-01 (slot 7 and the
  `weight_addr` argument).
"""

from max.gpu.host import DeviceContext

from dbscan.impl.dbscan.dbscan import dbscan_fit_impl_weighted
from dbscan.impl.dbscan.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC
from dbscan.impl.neighbors.epsilon_neighborhood import (
    DBSCAN_METRIC_L1,
    DBSCAN_METRIC_L2,
)


def dbscan_fit(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_samples: Int,
    n_features: Int,
    eps: Float64,
    min_samples: Int,
    out_labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    max_mbytes_per_batch: Int = 0,
    max_iterations: Int = 0,
    eps_nn_method: Int = EPS_NN_RBC,
    metric: Int = DBSCAN_METRIC_L2,
    sample_weight_addr: Int = 0,
) raises -> Int:
    """Cluster host-resident row-major data. Returns the PROPAGATION PASSES.

    `max_iterations <= 0` (THE DEFAULT since DEVIATION 519, 2026-08-23)
    means RUN TO THE FIXED POINT, which is cuML's behaviour: their
    `weak_cc_batched` and `MergeLabels` loops are `do { } while (host_m)`
    with no cap at all (`csr.cuh:133`; `csr.mojo`'s own docstring says so).
    The 200 that `dbscan_fit_impl` still defaults to is THIS PORT's number,
    not theirs, and on a 1,000-point chain at spacing 1/8, eps 3/16,
    min_samples 2 -- a thin trail of points, an ordinary input -- it
    returned SEVEN clusters where the fixed point is ONE (731 passes),
    silently, under FAST (`tools/e2u_matrix_fit.py`, `dbscan_chain_default`
    vs `dbscan_chain_iter5000`). Under IDENTICAL the same cap RAISES
    (DEVIATION 507), which is loud but is still a default that fails on an
    input cuML handles. The fixed point is reached in at most `n_samples`
    passes per call -- each pass moves the minimum label of a component at
    least one hop and no component has a path longer than `n_samples` --
    so the cap passed down is `n_samples + 1`, a bound and not a guess. An
    EXPLICIT positive cap is honoured as before, and under IDENTICAL a cap
    that binds still raises.

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
    if metric != DBSCAN_METRIC_L2 and metric != DBSCAN_METRIC_L1:
        raise Error(
            "dbscan_fit: metric must be DBSCAN_METRIC_L2 (0) or"
            " DBSCAN_METRIC_L1 (1), got " + String(metric)
        )

    # DEVIATION 519: the fixed point, bounded by the path length.
    var cap = max_iterations
    if cap <= 0:
        cap = n_samples + 1

    var x = ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var labels = ctx.enqueue_create_buffer[DType.int32](n_samples)
    # `sample_weight_addr == 0` IS THEIR `sample_weight == nullptr`. Mojo
    # has no null `DeviceBuffer`, so the address arrives as an Int, the
    # decision is made once, here, and everything downstream carries a
    # `has_weights` Bool beside a buffer that is a one-element placeholder
    # when there are none.
    var has_weights = sample_weight_addr != 0
    var w = ctx.enqueue_create_buffer[DType.float32](
        n_samples if has_weights else 1
    )
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    if has_weights:
        ctx.enqueue_copy(
            dst_buf=w,
            src_ptr=MutPointer[Float32, MutUntrackedOrigin](
                unsafe_from_address=sample_weight_addr
            ),
        )
    ctx.synchronize()

    var passes = dbscan_fit_impl_weighted(
        ctx,
        x,
        labels,
        w,
        n_samples,
        n_features,
        eps,
        min_samples,
        max_mbytes_per_batch,
        cap,
        eps_nn_method,
        False,
        metric,
        has_weights,
    )

    var hl = ctx.enqueue_create_host_buffer[DType.int32](n_samples)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    for i in range(n_samples):
        out_labels_ptr.unsafe_store(i, hl.unsafe_ptr().unsafe_load(i))

    return passes
