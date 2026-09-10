# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-list surface for the implemented metrics: what `bindings/` calls.

Shaped like `kde/estimator.mojo`, for the same reason: every entry here
takes HOST lists, uploads them, runs one implemented metric, downloads what the
caller needs and returns. No `DeviceBuffer` and no `DeviceContext` crosses
this boundary, so the CPython layer above it never has to know either type.
`metrics/README.md`'s HAND-OFF names the Python surface; this file is the
entry it reaches, and `python/mojolearn/_metrics_impl.py` is the mirror.

**THESE ENTRIES EMIT NO IDENTITY CARD, DELIBERATELY.** The metrics card is
`metrics/metrics_main.mojo`'s, its stage list is part of the build, and the
three-vendor result that is certified is that card's. A single-metric call
from Python that wrote a subset of those tags would produce a file that
looks like a metrics card and is not one, and `tools/identity_trace_diff.py`
would report a STRUCTURAL divergence against a real card for no reason
anybody could act on. If you want a card, run the driver:

    MOJOLEARN_IDENTITY_TRACE=/tmp/metrics.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . metrics/metrics_main.mojo

The UNTRACED implemented entry is therefore what each function below calls, and
`metrics_main.mojo` keeps sole ownership of the traced ones.

WHAT THIS FILE DOES NOT VALIDATE. The implemented entries do not scan their
device inputs, and neither does this file: a NaN in `X` reaches `sil_op` as
`+0.0` (DEVIATION 656), and a NaN in `y`/`y_hat` makes `r2_score` return the
canonical NaN `0x7fc00000` (DEVIATION 657). Per `metrics/README.md`'s
HAND-OFF ask 3, the finiteness checks live in the Python layer, where
scikit-learn's users expect `check_array` to raise. Everything the KERNELS
refuse -- an unimplemented silhouette metric, `2 * n_neighbors >= n`,
`n_neighbors + 1 > TRUST_MAX_K`, `n == 0`, `chunk < 1`, a label count
outside `[2, n_rows - 1]` -- is refused BY NAME below the line, by the
implemented code itself, and this file adds only the length and shape checks
that would otherwise read past the end of a list.

DEVIATIONS 890-899 are this surface's. It spends none of them on
arithmetic: nothing here computes, it only moves bytes and forwards.
"""

from max.gpu.host import DeviceContext
from std.math import isfinite
from metrics.impl.metrics.regression_errors import regression_error
from metrics.impl.metrics.log_loss import log_loss
from metrics.impl.metrics.binary_ranking import binary_ranking
from metrics.impl.metrics.classification import confusion_matrix, precision_recall_fscore, MAX_CONFUSION_CLASSES, MAX_PRF_CLASSES

from metrics.checks.device_io import download_f32, upload_f32, upload_i32
from metrics.impl.metrics.accuracy_score import accuracy_score_py
from metrics.impl.metrics.adjusted_rand_index import adjusted_rand_index
from metrics.impl.metrics.completeness_score import completeness_score
from metrics.impl.metrics.entropy import entropy
from metrics.impl.metrics.homogeneity_score import homogeneity_score
from metrics.impl.metrics.kl_divergence import kl_divergence
from metrics.impl.metrics.mutual_info_score import mutual_info_score
from metrics.impl.metrics.r2_score import r2_score_py
from metrics.impl.metrics.rand_index import rand_index
from metrics.impl.metrics.silhouette_score_batched_float import (
    silhouette_score,
)
from metrics.impl.metrics.trustworthiness import trustworthiness_score
from metrics.impl.metrics.v_measure import v_measure
from metrics.impl.stats.detail.batched.silhouette_score import (
    DISTANCE_L2_SQRT_UNEXPANDED,
)


def _check_pair(y_true: List[Int32], y_pred: List[Int32], n: Int) raises:
    """Both label arrays must hold at least `n` entries. A short list is a
    read past the end, which is a crash on a good day."""
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    if len(y_true) < n or len(y_pred) < n:
        raise Error(
            "metrics: labels_true holds " + String(len(y_true))
            + " and labels_pred holds " + String(len(y_pred))
            + " entries, both must hold at least n = " + String(n)
        )


def _check_float_pair(a: List[Float32], b: List[Float32], n: Int) raises:
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    if len(a) < n or len(b) < n:
        raise Error(
            "metrics: the two float arrays hold " + String(len(a)) + " and "
            + String(len(b)) + " entries, both must hold at least n = "
            + String(n)
        )


def _check_range(lower: Int32, upper: Int32) raises:
    if upper < lower:
        raise Error(
            "metrics: upper_class_range (" + String(upper)
            + ") is below lower_class_range (" + String(lower) + ")"
        )


# ===========================================================================
# Group A: the label metrics (int32 labels)
# ===========================================================================


def accuracy_score_host(
    y_true: List[Int32], y_pred: List[Int32], n: Int
) raises -> Float32:
    """`ML::Metrics::accuracy_score_py` (cuML `accuracy_score.cu`): the
    FRACTION of positions where the two label arrays agree.

    NOTE FOR A READER COMPARING SURFACES: cuML 26.08's own Python
    `accuracy_score` (`python/cuml/cuml/metrics/_classification.py:54`) is
    pure cupy and does not call this kernel at all. This entry is the C++
    one, which is what `metrics/` implemented."""
    _check_pair(y_true, y_pred, n)
    var ctx = DeviceContext()
    var dt = upload_i32(ctx, y_true)
    var dp = upload_i32(ctx, y_pred)
    var out = accuracy_score_py(ctx, dt, dp, n)
    _ = dt^
    _ = dp^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


def rand_score_host(
    y_true: List[Int32], y_pred: List[Int32], n: Int
) raises -> Float64:
    """`ML::Metrics::rand_index` (cuML `rand_index.cu`, DEVIATION 652).
    cuML's Python exports only `adjusted_rand_score`; the unadjusted index
    has a C++ entry and this surface exposes it under scikit-learn's name
    `rand_score`."""
    _check_pair(y_true, y_pred, n)
    var ctx = DeviceContext()
    var dt = upload_i32(ctx, y_true)
    var dp = upload_i32(ctx, y_pred)
    var out = rand_index(ctx, dt, dp, n)
    _ = dt^
    _ = dp^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


def adjusted_rand_score_host(
    y_true: List[Int32], y_pred: List[Int32], n: Int
) raises -> Float64:
    """`ML::Metrics::adjusted_rand_index` (cuML `adjusted_rand_index.cu`),
    the int32 instantiation. Takes RAW labels and builds its own
    contingency matrix over its own label range, exactly as theirs does;
    there is no `lower_class_range` on this entry."""
    _check_pair(y_true, y_pred, n)
    var ctx = DeviceContext()
    var dt = upload_i32(ctx, y_true)
    var dp = upload_i32(ctx, y_pred)
    var out = adjusted_rand_index(ctx, dt, dp, n)
    _ = dt^
    _ = dp^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


def entropy_host(
    labels: List[Int32], n: Int, lower: Int32, upper: Int32
) raises -> Float64:
    """`ML::Metrics::entropy` (cuML `entropy.cu`, DEVIATIONS 650/651), in
    NATS. `lower` and `upper` are the label range the histogram spans;
    cuML's `entropy.pyx:61-62` passes `min(labels)` and `max(labels)` with
    NO remapping, and the Python mirror does the same."""
    if n <= 0:
        raise Error("entropy: n must be positive, got " + String(n))
    if len(labels) < n:
        raise Error(
            "entropy: labels holds " + String(len(labels))
            + " entries, needs at least n = " + String(n)
        )
    _check_range(lower, upper)
    var ctx = DeviceContext()
    var dl = upload_i32(ctx, labels)
    var out = entropy(ctx, dl, n, lower, upper)
    _ = dl^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


def mutual_info_score_host(
    y_true: List[Int32],
    y_pred: List[Int32],
    n: Int,
    lower: Int32,
    upper: Int32,
) raises -> Float64:
    """`ML::Metrics::mutual_info_score` (DEVIATIONS 650/651), in NATS.

    `lower`/`upper` bound the SHARED label range: the contingency matrix is
    `(upper - lower + 1)` square and both arrays index into it. cuML 26.08's
    `prepare_cluster_metric_inputs` remaps both arrays onto
    `[0, n_classes - 1]` and passes `(0, n_classes - 1)`; the Python mirror
    does the same, which is what keeps that square small."""
    _check_pair(y_true, y_pred, n)
    _check_range(lower, upper)
    var ctx = DeviceContext()
    var dt = upload_i32(ctx, y_true)
    var dp = upload_i32(ctx, y_pred)
    var out = mutual_info_score(ctx, dt, dp, n, lower, upper)
    _ = dt^
    _ = dp^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


def homogeneity_score_host(
    y_true: List[Int32],
    y_pred: List[Int32],
    n: Int,
    lower: Int32,
    upper: Int32,
) raises -> Float64:
    """`ML::Metrics::homogeneity_score`: `MI(true, pred) / H(true)`."""
    _check_pair(y_true, y_pred, n)
    _check_range(lower, upper)
    var ctx = DeviceContext()
    var dt = upload_i32(ctx, y_true)
    var dp = upload_i32(ctx, y_pred)
    var out = homogeneity_score(ctx, dt, dp, n, lower, upper)
    _ = dt^
    _ = dp^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


def completeness_score_host(
    y_true: List[Int32],
    y_pred: List[Int32],
    n: Int,
    lower: Int32,
    upper: Int32,
) raises -> Float64:
    """`ML::Metrics::completeness_score`: RAFT computes it as
    `homogeneity_score` with the two arrays SWAPPED, so its mutual
    information folds the TRANSPOSED contingency matrix and is the same
    quantity in exact arithmetic but not necessarily the same Float32 bits
    (`metrics/metrics_main.mojo` records both). Implemented as theirs."""
    _check_pair(y_true, y_pred, n)
    _check_range(lower, upper)
    var ctx = DeviceContext()
    var dt = upload_i32(ctx, y_true)
    var dp = upload_i32(ctx, y_pred)
    var out = completeness_score(ctx, dt, dp, n, lower, upper)
    _ = dt^
    _ = dp^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


def v_measure_score_host(
    y_true: List[Int32],
    y_pred: List[Int32],
    n: Int,
    lower: Int32,
    upper: Int32,
    beta: Float64,
) raises -> Float64:
    """`ML::Metrics::v_measure` with `beta` honored (cuML `v_measure.cu`,
    RAFT `v_measure.cuh`): `(1 + beta) h c / (beta h + c)`."""
    _check_pair(y_true, y_pred, n)
    _check_range(lower, upper)
    var ctx = DeviceContext()
    var dt = upload_i32(ctx, y_true)
    var dp = upload_i32(ctx, y_pred)
    var out = v_measure(ctx, dt, dp, n, lower, upper, beta)
    _ = dt^
    _ = dp^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


# ===========================================================================
# Group B: r2 and KL divergence (float32)
# ===========================================================================


def r2_score_host(
    y_true: List[Float32], y_pred: List[Float32], n: Int
) raises -> Float32:
    """`ML::Metrics::r2_score_py` (float overload; the double overload is
    refused at the Python surface, no Float64 on device).

    DEVIATION 653 pins the three sums; DEVIATION 657 pins the epilogue,
    which is cuML's own Python surface's and scikit-learn's
    `force_finite=True`: `ssto == 0` returns `1.0` when `sse == 0` and
    `0.0` otherwise, and an overflow-scale `y` returns the canonical NaN
    `0x7fc00000` rather than a vendor payload. `force_finite=False` is
    therefore NOT reachable and the Python mirror refuses it by name."""
    _check_float_pair(y_true, y_pred, n)
    var ctx = DeviceContext()
    var dy = upload_f32(ctx, y_true)
    var dh = upload_f32(ctx, y_pred)
    var out = r2_score_py(ctx, dy, dh, n)
    _ = dy^
    _ = dh^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


def kl_divergence_host(
    p: List[Float32], q: List[Float32], n: Int
) raises -> Float32:
    """`ML::Metrics::kl_divergence` (float overload; DEVIATIONS 653, 658).
    `sum p_i (log p_i - log q_i)`, NOT normalized, exactly as theirs."""
    _check_float_pair(p, q, n)
    var ctx = DeviceContext()
    var dp = upload_f32(ctx, p)
    var dq = upload_f32(ctx, q)
    var out = kl_divergence(ctx, dp, dq, n)
    _ = dp^
    _ = dq^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out


# ===========================================================================
# Group C: silhouette (the batched float path cuML dispatches)
# ===========================================================================


def silhouette_host(
    x: List[Float32],
    labels: List[Int32],
    n_rows: Int,
    n_cols: Int,
    n_labels: Int,
    chunk: Int,
    mut scores: List[Float32],
) raises -> Float32:
    """`ML::Metrics::Batched::silhouette_score` (float, DEVIATIONS 654/656).

    `x` is row-major `n_rows x n_cols`. `labels` must already be mapped onto
    `[0, n_labels - 1]`; cuML's `silhouette_score.pyx:99-101` does that with
    `cp.unique(..., return_inverse=True)` and the Python mirror does the
    same. `scores` is CLEARED and refilled with the `n_rows` per-sample
    coefficients (cuML's `silhouette_samples`); the return is their mean.

    `chunk` is cuML's `chunksize`, default 40000. It is VALIDATED (>= 1) and
    is SCHEDULING ONLY here: this implementation materializes no distance tile, so
    there is nothing for it to size, and `check_silhouette_refusals` holds
    chunk 1 / 7 / 40000 to one byte pattern. `metric` is not a parameter of
    this entry at all -- only `L2SqrtUnexpanded` (cuML 'euclidean'/'l2') is
    implemented and the implemented kernel refuses any other DistanceType by name, so
    the Python mirror carries the refusal where the caller can read it."""
    if n_rows <= 0:
        raise Error("silhouette: n_rows must be positive, got " + String(n_rows))
    if n_cols <= 0:
        raise Error("silhouette: n_cols must be positive, got " + String(n_cols))
    if len(x) < n_rows * n_cols:
        raise Error(
            "silhouette: X holds " + String(len(x)) + " floats, needs "
            + String(n_rows * n_cols)
        )
    if len(labels) < n_rows:
        raise Error(
            "silhouette: labels holds " + String(len(labels))
            + " entries, needs n_rows = " + String(n_rows)
        )
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dl = upload_i32(ctx, labels)
    var ds = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.synchronize()
    var mean = silhouette_score(
        ctx, dx, n_rows, n_cols, dl, n_labels, ds, chunk,
        DISTANCE_L2_SQRT_UNEXPANDED,
    )
    var got = download_f32(ctx, ds, n_rows)
    scores.clear()
    for i in range(n_rows):
        scores.append(got[i])
    _ = dx^
    _ = dl^
    _ = ds^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return mean


# ===========================================================================
# Group D: trustworthiness
# ===========================================================================


def trustworthiness_host(
    mut x: List[Float32],
    mut x_embedded: List[Float32],
    n: Int,
    m: Int,
    d: Int,
    n_neighbors: Int,
    batch_size: Int,
) raises -> Float64:
    """`ML::Metrics::trustworthiness_score<float, L2SqrtUnexpanded>`
    (DEVIATION 655). `x` is row-major `n x m`, `x_embedded` row-major
    `n x d`; both stay on the HOST because the embedded k-NN goes through
    `neighbors/estimator.mojo::knn_search`, whose boundary is host pointers.

    Refused by the implemented entry, by name: `n_neighbors < 1`,
    `2 * n_neighbors >= n` (cuML `trustworthiness.pyx:114`, which keeps the
    closed form's denominator positive so no NaN reaches a recorded value),
    and `n_neighbors + 1 > TRUST_MAX_K`. `batch_size` is VALIDATED (>= 1)
    and sizes nothing: DEVIATION 655 counts ranks instead of sorting every
    row of the n x n distance matrix, so there is no batch to size."""
    if n <= 0 or m <= 0 or d <= 0:
        raise Error(
            "trustworthiness: n, n_features and n_components must all be"
            " positive, got " + String(n) + ", " + String(m) + ", " + String(d)
        )
    if len(x) < n * m:
        raise Error(
            "trustworthiness: X holds " + String(len(x)) + " floats, needs "
            + String(n * m)
        )
    if len(x_embedded) < n * d:
        raise Error(
            "trustworthiness: X_embedded holds " + String(len(x_embedded))
            + " floats, needs " + String(n * d)
        )
    var ctx = DeviceContext()
    return trustworthiness_score(
        ctx, x, x_embedded, n, m, d, n_neighbors, batch_size
    )


def regression_error_host[absolute: Bool = False, root: Bool = False](
    y_true: List[Float32], y_pred: List[Float32], n: Int
) raises -> Float32:
    """Validate/upload only; residuals, reduction and epilogue run on GPU."""
    _check_float_pair(y_true, y_pred, n)
    for i in range(n):
        if not isfinite(y_true[i]) or not isfinite(y_pred[i]):
            raise Error("regression_error: inputs must be finite Float32")
    var ctx = DeviceContext()
    var y = upload_f32(ctx, y_true)
    var prediction = upload_f32(ctx, y_pred)
    var result = regression_error[absolute, root](ctx, y, prediction, n)
    _ = y^
    _ = prediction^
    _ = ctx^
    return result


def _check_classification[matrix: Bool](y: List[Int32], p: List[Int32], n: Int, k: Int) raises:
    _check_pair(y,p,n)
    comptime cap = MAX_CONFUSION_CLASSES if matrix else MAX_PRF_CLASSES
    if n > 2147483647 or k <= 0 or k > cap:
        raise Error("classification metrics: count or class allocation bound exceeded")
    comptime minimum = -1 if matrix else 0
    for i in range(n):
        if Int(y[i]) < minimum or Int(y[i]) >= k or Int(p[i]) < minimum or Int(p[i]) >= k:
            raise Error("classification metrics: encoded label out of range")


def confusion_matrix_host[dtype: DType](
    y: List[Int32], p: List[Int32], n: Int, k: Int, normalization: Int,
) raises -> List[Scalar[dtype]]:
    _check_classification[True](y,p,n,k)
    var ctx = DeviceContext()
    var dy = upload_i32(ctx,y)
    var dp = upload_i32(ctx,p)
    var result = confusion_matrix[dtype](ctx,dy,dp,n,k,normalization)
    var host = List[Scalar[dtype]]()
    with result.map_to_host() as h:
        for i in range(k*k):
            host.append(h[i])
    _ = result^
    _ = dy^
    _ = dp^
    _ = ctx^
    return host^


def precision_recall_fscore_host(
    y: List[Int32], p: List[Int32], n: Int, k: Int, average: Int,
    positive: Int, zero: Int, selected: Int,
) raises -> List[Float32]:
    _check_classification[False](y,p,n,k)
    var ctx = DeviceContext()
    var dy = upload_i32(ctx,y)
    var dp = upload_i32(ctx,p)
    var result = precision_recall_fscore(ctx,dy,dp,n,k,average,positive,zero,selected)
    var width = selected if average == 0 else 1
    var host = download_f32(ctx,result,3*width+3)
    _ = result^
    _ = dy^
    _ = dp^
    _ = ctx^
    return host^


def log_loss_host(y: List[Int32], probability: List[Float32], n: Int, k: Int, normalize: Int) raises -> Float32:
    """Validate and upload; logarithms and the entire reduction execute on GPU."""
    if n <= 0 or n > 2147483647 or k < 2 or k > 2147483647 // n:
        raise Error("log_loss: invalid input dimensions")
    if len(y) < n or len(probability) < n*k or normalize < 0 or normalize > 1:
        raise Error("log_loss: invalid input length or normalization")
    for i in range(n):
        if Int(y[i]) < 0 or Int(y[i]) >= k:
            raise Error("log_loss: encoded label out of range")
    for i in range(n*k):
        if not isfinite(probability[i]) or probability[i] < 0 or probability[i] > 1:
            raise Error("log_loss: probabilities must be finite and within [0,1]")
    var ctx = DeviceContext()
    var dy = upload_i32(ctx,y)
    var dp = upload_f32(ctx,probability)
    var result = log_loss(ctx,dy,dp,n,k,normalize)
    _ = dy^
    _ = dp^
    _ = ctx^
    return result


def binary_ranking_host[curve: Bool](y: List[Int32], scores: List[Float32], n: Int) raises -> Tuple[List[Float32], Int]:
    if n <= 0 or n > 2147483647 or len(y) < n or len(scores) < n:
        raise Error("binary ranking: invalid input length")
    var has_zero = False
    var has_one = False
    for i in range(n):
        if y[i] == 0:
            has_zero = True
        elif y[i] == 1:
            has_one = True
        else:
            raise Error("binary ranking: labels must encode 0 or 1")
        if not isfinite(scores[i]):
            raise Error("binary ranking: scores must be finite")
    comptime if not curve:
        if not has_zero or not has_one:
            raise Error("roc_auc_score: both classes required")
    var ctx = DeviceContext()
    var dy = upload_i32(ctx,y)
    var ds = upload_f32(ctx,scores)
    var result = binary_ranking[curve](ctx,dy,ds,n)
    _ = dy^
    _ = ds^
    _ = ctx^
    return result^
