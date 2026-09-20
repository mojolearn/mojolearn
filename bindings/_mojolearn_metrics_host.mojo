# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_metrics` family: the label, regression
and silhouette metrics (workstream E batch 2, lane/cpu-training-e2,
2026-09-14; brief docs/lanes/BRIEF_cpu_training_2026-09-13.md section 1.1
"metrics" and 3.2).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`metrics/host/metrics_oracle.mojo`, the second spelling of the integer
kernels (a count, a histogram, a contingency matrix) as serial loops and
of DEVIATION 653's slab tree for every float sum; that file's header names
every original by file and line. The validation is the GPU entry's, in the
GPU entry's words (`metrics/estimator.mojo`'s length checks, then the
kernels' own refusals), so a bad call raises the same error and nothing is
written on a refusal.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers
(`bindings/_mojolearn_metrics.mojo`): `accuracy_score`,
`adjusted_rand_score`, `entropy`, `mutual_info_score`,
`homogeneity_score`, `completeness_score`, `v_measure_score`, `r2_score`
and `silhouette`, each with the SAME address contract and `params` list
(repeated in the docstrings below and mirrored in
`python/mojolearn/_metrics_impl.py`), so `mojolearn.metrics` runs
unchanged on a CPU-only install through `_backend._HOST_MODULES`
(`"_mojolearn_metrics": "_mojolearn_metrics_host"`). The read-backs
`metrics_vendor` ("cpu") and `metrics_numeric_mode` (1) are the ones
`_metrics_impl._get_binding` and `_backend` consult.

THE SPECTRAL CLUSTERING ENTRY (the spectral lane, same batch):
`spectral_fit_predict_dataset` under the GPU binding's name and eight-value
`params` list, over `spectral/host/spectral_oracle.mojo` (the host oracle
the device arm is gated against, moved there from spectral/checks/, plus
the k-NN graph, the symmetrize kernel and the k-means recluster restated
on the host); `python/mojolearn/_spectral_impl.py` runs unchanged. The
spectral-precomputed lane (2026-09-14) adds `spectral_fit_predict_graph`,
the GPU binding's precomputed affinity entry under its name and eight-value
`params` list, over `host_spectral_fit_predict_coo` (the same oracle, the
graph given as COO triples); the dense affinity's COO scan it needs,
`nonzero_f64_count` and `nonzero_f64_fill`, is in the core host binding.

THE UMAP ENTRIES (lane/cpu-training-umap-b, 2026-09-14):
`umap_fit_transform`, `umap_transform` and `umap_numeric_mode` under the
GPU binding's names, address contracts and 10/13 and 11/14 value `params`
lists, over `umap/host/umap_oracle.mojo` (the host k-NN, the fuzzy graph,
the spectral oracle for the initialization, and the IDENTICAL device epoch
fold restated vertex by vertex); `python/mojolearn/_umap_impl.py` runs
unchanged.

THE METRICS-CLASSIFICATION LANE (2026-09-14) adds the GPU binding's
remaining metric entries under their names and params lists: `rand_score`,
`roc_auc_score`, `precision_recall_curve`, `log_loss`, `confusion_matrix`,
`precision_recall_fscore`, `mean_squared_error`, `mean_absolute_error`,
`root_mean_squared_error`, `kl_divergence` and `trustworthiness`, over
`metrics/host/classification_oracle.mojo` (the log loss's probability
check, `probability_rows_f32`, is in the core host binding).

The GPU binding's OTHER entries (`graph_parallel_available`) are deliberately ABSENT here, so every lane that
reaches them keeps refusing BY NAME through `_HostBinding` until a lane
lands them.
"""
from std.math import isfinite
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, read_f32, read_i32
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from bindings.hostptr import i32_ptr
from spectral.host.spectral_oracle import (
    SPECTRAL_ORACLE_HOST_SABOTAGE,
    host_spectral_fit_predict_coo,
    host_spectral_fit_predict_coo_keep,
    host_spectral_fit_predict_dataset,
    host_spectral_fit_predict_dataset_keep,
)
from spectral.host.spectral_predict_host import (
    SPECTRAL_PREDICT_HOST_SABOTAGE,
    SpectralPredictionState,
    host_spectral_predict,
    spectral_predict_check_state,
)
from umap.host.umap_oracle import (
    UMAP_ORACLE_HOST_SABOTAGE,
    host_umap_fit_transform,
    host_umap_transform,
)
from umap.params import UMAPParams
from metrics.host.metrics_oracle import (
    DISTANCE_L2_SQRT_UNEXPANDED,
    METRICS_ORACLE_HOST_SABOTAGE,
    host_accuracy_score,
    host_accuracy_score_ptr,
    host_adjusted_rand_score,
    host_entropy,
    host_fowlkes_mallows,
    host_weighted_accuracy,
    host_weighted_accuracy_ptr,
    host_weighted_r2,
    host_homogeneity_score,
    host_mutual_info,
    host_mutual_info_ptr,
    host_r2_score,
    host_silhouette,
    host_v_measure,
)
from metrics.host.classification_oracle import (
    MAX_CONFUSION_CLASSES,
    MAX_PRF_CLASSES,
    host_binary_ranking,
    host_confusion_matrix_f32,
    host_confusion_matrix_i64,
    host_confusion_matrix_ptr,
    host_kl_divergence,
    host_log_loss,
    host_precision_recall_fscore,
    host_rand_score,
    host_regression_error,
    host_regression_error_ptr,
    host_trustworthiness,
)


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("metrics host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def _want(name: String, params: PythonObject, k: Int) raises:
    """`_want`, `bindings/_mojolearn_metrics.mojo`, verbatim."""
    if len(params) != k:
        raise Error(
            name + ": params must contain " + String(k) + " values, got "
            + String(len(params))
        )


def _check_pair(y_true: List[Int32], y_pred: List[Int32], n: Int) raises:
    """`_check_pair`, `metrics/estimator.mojo`, verbatim."""
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


def metrics_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def metrics_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def metrics_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_metrics_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "metrics host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_metrics_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `metrics_host_detected_column` read-back, for the reason
# 8d16ce2f removed it from the forest and byte LM host bindings: the detected
# column folds to the GPU of the machine that ran the build, so its name
# would land in the vendor-neutral binary. The comptime assert above is the
# check.


def metrics_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary was built with -D MOJOLEARN_HOST_SABOTAGE=1 (the
    gate's negative control): every slab tree's chunk boundaries shifted by
    one value, a perturbed value read by accuracy, the adjusted Rand index,
    entropy, mutual information, r2 and the silhouette
    (`metrics/host/metrics_oracle.mojo`, THE NEGATIVE CONTROL), and the spectral
    recluster seeded one draw off, every UMAP negative draw keyed one
    epoch late, and spectral predict's first non-trivial embedding column negated (also
    alone under -D MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE=1); refused outside
    the gate as one set."""
    return PythonObject(
        METRICS_ORACLE_HOST_SABOTAGE or SPECTRAL_ORACLE_HOST_SABOTAGE
        or UMAP_ORACLE_HOST_SABOTAGE or SPECTRAL_PREDICT_HOST_SABOTAGE
    )


# The GPU binding's names, same contract.


def metrics_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def metrics_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here, a host
    binding is IDENTICAL only (the build script refuses any other)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


# ===========================================================================
# Group A: the label metrics. int32 labels, one shape each.
# ===========================================================================


def accuracy_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::accuracy_score_py` on the host. `params`: `0 n`."""
    _want(String("accuracy_score"), params, 1)
    var n = _index(params[0])
    var yt = i32_ptr(_index(y_true_addr))
    var yp = i32_ptr(_index(y_pred_addr))
    var out = Float32(0.0)
    with GILReleased(Python()):
        out = host_accuracy_score_ptr(yt, yp, n)
    return PythonObject(Float64(out))


def adjusted_rand_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::adjusted_rand_index` on the host, RAW labels.
    `params`: `0 n`."""
    _want(String("adjusted_rand_score"), params, 1)
    var n = _index(params[0])
    var yt = read_i32(_index(y_true_addr), n)
    var yp = read_i32(_index(y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        _check_pair(yt, yp, n)
        out = host_adjusted_rand_score(yt, yp, n)
    return PythonObject(out)


def entropy_binding(
    labels_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::entropy` on the host, in NATS. `params`: `0 n, 1
    lower_class_range, 2 upper_class_range`."""
    _want(String("entropy"), params, 3)
    var n = _index(params[0])
    var lower = Int32(_index(params[1]))
    var upper = Int32(_index(params[2]))
    var lab = read_i32(_index(labels_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        if n <= 0:
            raise Error("entropy: n must be positive, got " + String(n))
        if len(lab) < n:
            raise Error(
                "entropy: labels holds " + String(len(lab))
                + " entries, needs at least n = " + String(n)
            )
        _check_range(lower, upper)
        out = host_entropy(lab, n, lower, upper)
    return PythonObject(out)


def mutual_info_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::mutual_info_score` on the host, in NATS. `params`:
    `0 n, 1 lower_class_range, 2 upper_class_range`."""
    _want(String("mutual_info_score"), params, 3)
    var n = _index(params[0])
    var lower = Int32(_index(params[1]))
    var upper = Int32(_index(params[2]))
    var yt = i32_ptr(_index(y_true_addr))
    var yp = i32_ptr(_index(y_pred_addr))
    var out = Float64(0.0)
    with GILReleased(Python()):
        if n <= 0:
            raise Error("metrics: n must be positive, got " + String(n))
        _check_range(lower, upper)
        out = host_mutual_info_ptr(yt, yp, n, lower, upper)
    return PythonObject(out)


def accuracy_score_weighted_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    w_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Weighted accuracy on the host. `params`: `0 n`."""
    _want(String("accuracy_score_weighted"), params, 1)
    var n = _index(params[0])
    var yt = i32_ptr(_index(y_true_addr))
    var yp = i32_ptr(_index(y_pred_addr))
    var w = f32_ptr(_index(w_addr))
    var out = Float32(0.0)
    with GILReleased(Python()):
        if n <= 0:
            raise Error("metrics: n must be positive, got " + String(n))
        out = host_weighted_accuracy_ptr(yt, yp, w, n)
    return PythonObject(Float64(out))


def r2_score_weighted_binding(
    y_addr: PythonObject,
    y_hat_addr: PythonObject,
    w_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Weighted R2 on the host (`force_finite=True`). `params`: `0 n`."""
    _want(String("r2_score_weighted"), params, 1)
    var n = _index(params[0])
    var y = read_f32(_index(y_addr), n)
    var yh = read_f32(_index(y_hat_addr), n)
    var w = read_f32(_index(w_addr), n)
    var out = Float32(0.0)
    with GILReleased(Python()):
        out = host_weighted_r2(y, yh, w, n)
    return PythonObject(Float64(out))


def fowlkes_mallows_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """scikit-learn `fowlkes_mallows_score` on the host. `params`: `0 n, 1
    lower_class_range, 2 upper_class_range`."""
    _want(String("fowlkes_mallows_score"), params, 3)
    var n = _index(params[0])
    var lower = Int32(_index(params[1]))
    var upper = Int32(_index(params[2]))
    var yt = read_i32(_index(y_true_addr), n)
    var yp = read_i32(_index(y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        _check_pair(yt, yp, n)
        _check_range(lower, upper)
        out = host_fowlkes_mallows(yt, yp, n, lower, upper)
    return PythonObject(out)


def homogeneity_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::homogeneity_score` on the host. `params`: `0 n, 1
    lower_class_range, 2 upper_class_range`."""
    _want(String("homogeneity_score"), params, 3)
    var n = _index(params[0])
    var lower = Int32(_index(params[1]))
    var upper = Int32(_index(params[2]))
    var yt = read_i32(_index(y_true_addr), n)
    var yp = read_i32(_index(y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        _check_pair(yt, yp, n)
        _check_range(lower, upper)
        out = host_homogeneity_score(yt, yp, n, lower, upper)
    return PythonObject(out)


def completeness_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::completeness_score` on the host, the homogeneity of the
    swapped arrays. `params`: `0 n, 1 lower_class_range, 2
    upper_class_range`."""
    _want(String("completeness_score"), params, 3)
    var n = _index(params[0])
    var lower = Int32(_index(params[1]))
    var upper = Int32(_index(params[2]))
    var yt = read_i32(_index(y_true_addr), n)
    var yp = read_i32(_index(y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        _check_pair(yt, yp, n)
        _check_range(lower, upper)
        out = host_homogeneity_score(yp, yt, n, lower, upper)
    return PythonObject(out)


def v_measure_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::v_measure` on the host with `beta` honored. `params`:
    `0 n, 1 lower_class_range, 2 upper_class_range, 3 beta (float)`."""
    _want(String("v_measure_score"), params, 4)
    var n = _index(params[0])
    var lower = Int32(_index(params[1]))
    var upper = Int32(_index(params[2]))
    var beta = Float64(py=params[3])
    var yt = read_i32(_index(y_true_addr), n)
    var yp = read_i32(_index(y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        _check_pair(yt, yp, n)
        _check_range(lower, upper)
        out = host_v_measure(yt, yp, n, lower, upper, beta)
    return PythonObject(out)


# ===========================================================================
# Group B: r2. float32 in, float32 out.
# ===========================================================================


def r2_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::r2_score_py` on the host, the float overload
    (DEVIATIONS 653, 657). `params`: `0 n`."""
    _want(String("r2_score"), params, 1)
    var n = _index(params[0])
    var y = read_f32(_index(y_true_addr), n)
    var yh = read_f32(_index(y_pred_addr), n)
    var out = Float32(0.0)
    with GILReleased(Python()):
        _check_float_pair(y, yh, n)
        out = host_r2_score(y, yh, n)
    return PythonObject(Float64(out))


# ===========================================================================
# Group C: silhouette. Writes n_rows per-sample scores, returns their mean.
# ===========================================================================


def silhouette_binding(
    x_addr: PythonObject,
    labels_addr: PythonObject,
    scores_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::Batched::silhouette_score` on the host (DEVIATIONS 654,
    656). Writes `n_rows` float32 per-sample coefficients to `scores_addr`
    and returns their mean. `params`: `0 n_rows, 1 n_cols, 2 n_labels
    (labels mapped onto [0, n_labels-1]), 3 chunksize (validated >= 1,
    scheduling only)`."""
    _want(String("silhouette"), params, 4)
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_labels = _index(params[2])
    var chunk = _index(params[3])
    var x = read_f32(_index(x_addr), n_rows * n_cols)
    var lab = read_i32(_index(labels_addr), n_rows)
    var sp = f32_ptr(_index(scores_addr))
    var scores = List[Float32]()
    var mean = Float32(0.0)
    with GILReleased(Python()):
        if n_rows <= 0:
            raise Error("silhouette: n_rows must be positive, got " + String(n_rows))
        if n_cols <= 0:
            raise Error("silhouette: n_cols must be positive, got " + String(n_cols))
        if len(x) < n_rows * n_cols:
            raise Error(
                "silhouette: X holds " + String(len(x)) + " floats, needs "
                + String(n_rows * n_cols)
            )
        if len(lab) < n_rows:
            raise Error(
                "silhouette: labels holds " + String(len(lab))
                + " entries, needs n_rows = " + String(n_rows)
            )
        mean = host_silhouette(
            x, lab, n_rows, n_cols, n_labels, chunk,
            DISTANCE_L2_SQRT_UNEXPANDED, scores,
        )
        for i in range(n_rows):
            sp.unsafe_store(i, scores[i])
    return PythonObject(Float64(mean))


# ===========================================================================
# Group E: spectral clustering on a dataset and on a precomputed graph.
# ===========================================================================


def _guard_spectral_outputs(
    labels: List[Int32],
    embedding: List[Float32],
    n_samples: Int,
    n_components: Int,
) raises:
    """`_guard_spectral_outputs`, `bindings/_mojolearn_metrics.mojo`: the
    two output buffers were sized by the Python caller; check what came
    back BEFORE writing a single element into them."""
    if len(labels) != n_samples:
        raise Error(
            "spectral clustering: the kernel returned " + String(len(labels))
            + " labels for " + String(n_samples)
            + " samples; the output buffer was sized for n_samples"
        )
    if len(embedding) != n_samples * n_components:
        raise Error(
            "spectral clustering: the kernel returned "
            + String(len(embedding)) + " embedding floats, but the output"
            " buffer was sized for n_samples * n_components = "
            + String(n_samples * n_components)
        )


def spectral_fit_predict_dataset_binding(
    x_addr: PythonObject,
    labels_addr: PythonObject,
    embedding_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`fit_predict` on a DATASET on the host (cuML
    `affinity='nearest_neighbors'`): kNN connectivity graph, Laplacian,
    thick-restart Lanczos, k-means. Writes `n_samples` int32 labels and
    the `n_samples x n_out` row-major embedding; returns `n_out`.
    `params`: `0 n_samples, 1 n_features, 2 n_clusters, 3 n_components,
    4 n_init, 5 n_neighbors, 6 eigen_tol (float), 7 seed`."""
    _want(String("spectral_fit_predict_dataset"), params, 8)
    var n_samples = _index(params[0])
    var n_features = _index(params[1])
    var n_clusters = _index(params[2])
    var n_components = _index(params[3])
    var n_init = _index(params[4])
    var n_neighbors = _index(params[5])
    var eigen_tol = Float32(Float64(py=params[6]))
    var seed = UInt64(_index(params[7]))
    var x = read_f32(_index(x_addr), n_samples * n_features)
    var lp = i32_ptr(_index(labels_addr))
    var ep = f32_ptr(_index(embedding_addr))
    var labels = List[Int32]()
    var embedding = List[Float32]()
    var n_out = 0
    with GILReleased(Python()):
        n_out = host_spectral_fit_predict_dataset(
            x, n_samples, n_features, n_clusters, n_components, n_init,
            n_neighbors, eigen_tol, seed, labels, embedding,
        )
        _guard_spectral_outputs(labels, embedding, n_samples, n_components)
        for i in range(n_samples):
            lp.unsafe_store(i, labels[i])
        for i in range(len(embedding)):
            ep.unsafe_store(i, embedding[i])
    return PythonObject(n_out)


def spectral_fit_predict_graph_binding(
    rows_addr: PythonObject,
    cols_addr: PythonObject,
    vals_addr: PythonObject,
    labels_addr: PythonObject,
    embedding_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`fit_predict` on a PRECOMPUTED connectivity graph on the host, given
    as COO triples (cuML `affinity='precomputed'`), the GPU binding's
    `spectral_fit_predict_graph` (`bindings/_mojolearn_metrics.mojo`). No
    k-NN runs, so `n_neighbors` is carried and read by nobody on this path.
    Writes `n_samples` int32 labels and the `n_samples x n_out` row-major
    embedding; returns `n_out`. `params`: `0 n_samples, 1 nnz (length of
    rows, cols and vals), 2 n_clusters, 3 n_components, 4 n_init, 5
    n_neighbors (carried, unused), 6 eigen_tol (float), 7 seed`."""
    _want(String("spectral_fit_predict_graph"), params, 8)
    var n_samples = _index(params[0])
    var nnz = _index(params[1])
    var n_clusters = _index(params[2])
    var n_components = _index(params[3])
    var n_init = _index(params[4])
    var n_neighbors = _index(params[5])
    var eigen_tol = Float32(Float64(py=params[6]))
    var seed = UInt64(_index(params[7]))
    var rows = read_i32(_index(rows_addr), max(0, nnz))
    var cols = read_i32(_index(cols_addr), max(0, nnz))
    var vals = read_f32(_index(vals_addr), max(0, nnz))
    var lp = i32_ptr(_index(labels_addr))
    var ep = f32_ptr(_index(embedding_addr))
    var labels = List[Int32]()
    var embedding = List[Float32]()
    var n_out = 0
    with GILReleased(Python()):
        n_out = host_spectral_fit_predict_coo(
            rows, cols, vals, n_samples, n_clusters, n_components, n_init,
            n_neighbors, eigen_tol, seed, labels, embedding,
        )
        _guard_spectral_outputs(labels, embedding, n_samples, n_components)
        for i in range(n_samples):
            lp.unsafe_store(i, labels[i])
        for i in range(len(embedding)):
            ep.unsafe_store(i, embedding[i])
    return PythonObject(n_out)


# lane/spectral-predict (2026-09-15, DEVIATION 2860): the GPU binding's
# spectral_fit_predict_dataset_state, spectral_fit_predict_graph_state and
# spectral_predict, same names, address contracts and params lists.


def _write_spectral_state(
    state: SpectralPredictionState,
    addrs: PythonObject,
    first: Int,
    n_samples: Int,
    n_components: Int,
    n_clusters: Int,
) raises:
    spectral_predict_check_state(state, n_samples, n_components, n_clusters)
    var ev = f32_ptr(_index(addrs[first]))
    var evec = f32_ptr(_index(addrs[first + 1]))
    var dg = f32_ptr(_index(addrs[first + 2]))
    var cent = f32_ptr(_index(addrs[first + 3]))
    for i in range(len(state.eigenvalues)):
        ev.unsafe_store(i, state.eigenvalues[i])
    for i in range(len(state.eigenvectors)):
        evec.unsafe_store(i, state.eigenvectors[i])
    for i in range(len(state.diag)):
        dg.unsafe_store(i, state.diag[i])
    for i in range(len(state.centroids)):
        cent.unsafe_store(i, state.centroids[i])


def spectral_fit_predict_dataset_state_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The GPU binding's `spectral_fit_predict_dataset_state` on the host.
    `addrs`: x, labels, embedding, eigenvalues, eigenvectors, diag,
    centroids; `params` the eight of `spectral_fit_predict_dataset`."""
    if len(addrs) != 7:
        raise Error("spectral_fit_predict_dataset_state: addrs must contain 7 addresses, got " + String(len(addrs)))
    _want(String("spectral_fit_predict_dataset_state"), params, 8)
    var n_samples = _index(params[0])
    var n_features = _index(params[1])
    var n_clusters = _index(params[2])
    var n_components = _index(params[3])
    var n_init = _index(params[4])
    var n_neighbors = _index(params[5])
    var eigen_tol = Float32(Float64(py=params[6]))
    var seed = UInt64(_index(params[7]))
    var x = read_f32(_index(addrs[0]), n_samples * n_features)
    var lp = i32_ptr(_index(addrs[1]))
    var ep = f32_ptr(_index(addrs[2]))
    var labels = List[Int32]()
    var embedding = List[Float32]()
    var state = SpectralPredictionState()
    var n_out = 0
    with GILReleased(Python()):
        n_out = host_spectral_fit_predict_dataset_keep(
            x, n_samples, n_features, n_clusters, n_components, n_init,
            n_neighbors, eigen_tol, seed, labels, embedding, state, True,
        )
    _guard_spectral_outputs(labels, embedding, n_samples, n_components)
    _write_spectral_state(state, addrs, 3, n_samples, n_components, n_clusters)
    for i in range(n_samples):
        lp.unsafe_store(i, labels[i])
    for i in range(len(embedding)):
        ep.unsafe_store(i, embedding[i])
    return PythonObject(n_out)


def spectral_fit_predict_graph_state_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The GPU binding's `spectral_fit_predict_graph_state` on the host.
    `addrs`: rows, cols, vals, labels, embedding, eigenvalues, eigenvectors,
    diag, centroids; `params` the eight of `spectral_fit_predict_graph`."""
    if len(addrs) != 9:
        raise Error("spectral_fit_predict_graph_state: addrs must contain 9 addresses, got " + String(len(addrs)))
    _want(String("spectral_fit_predict_graph_state"), params, 8)
    var n_samples = _index(params[0])
    var nnz = _index(params[1])
    var n_clusters = _index(params[2])
    var n_components = _index(params[3])
    var n_init = _index(params[4])
    var n_neighbors = _index(params[5])
    var eigen_tol = Float32(Float64(py=params[6]))
    var seed = UInt64(_index(params[7]))
    var rows = read_i32(_index(addrs[0]), max(0, nnz))
    var cols = read_i32(_index(addrs[1]), max(0, nnz))
    var vals = read_f32(_index(addrs[2]), max(0, nnz))
    var lp = i32_ptr(_index(addrs[3]))
    var ep = f32_ptr(_index(addrs[4]))
    var labels = List[Int32]()
    var embedding = List[Float32]()
    var state = SpectralPredictionState()
    var n_out = 0
    with GILReleased(Python()):
        n_out = host_spectral_fit_predict_coo_keep(
            rows, cols, vals, n_samples, n_clusters, n_components, n_init,
            n_neighbors, eigen_tol, seed, labels, embedding, state, True,
        )
    _guard_spectral_outputs(labels, embedding, n_samples, n_components)
    _write_spectral_state(state, addrs, 5, n_samples, n_components, n_clusters)
    for i in range(n_samples):
        lp.unsafe_store(i, labels[i])
    for i in range(len(embedding)):
        ep.unsafe_store(i, embedding[i])
    return PythonObject(n_out)


def spectral_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The GPU binding's `spectral_predict` on the host by
    `host_spectral_predict` (no fit code on this path). `addrs`: input,
    training rows (0 for precomputed), eigenvalues, eigenvectors, diag,
    centroids, out labels, out embedding. `params`: n_train, n_queries,
    n_features, n_components, n_clusters, n_neighbors, affinity. Returns 0."""
    if len(addrs) != 8:
        raise Error("spectral_predict: addrs must contain 8 addresses, got " + String(len(addrs)))
    _want(String("spectral_predict"), params, 7)
    var n_train = _index(params[0])
    var n_queries = _index(params[1])
    var n_features = _index(params[2])
    var k = _index(params[3])
    var n_clusters = _index(params[4])
    var n_neighbors = _index(params[5])
    var affinity = _index(params[6])
    var width = n_train if affinity == 1 else n_features
    var input = read_f32(_index(addrs[0]), max(0, n_queries * width))
    var train_x = List[Float32]()
    if affinity == 0:
        train_x = read_f32(_index(addrs[1]), max(0, n_train * n_features))
    var state = SpectralPredictionState()
    state.eigenvalues = read_f32(_index(addrs[2]), max(0, k))
    state.eigenvectors = read_f32(_index(addrs[3]), max(0, n_train * k))
    state.diag = read_f32(_index(addrs[4]), max(0, n_train))
    state.centroids = read_f32(_index(addrs[5]), max(0, n_clusters * k))
    var olp = i32_ptr(_index(addrs[6]))
    var oep = f32_ptr(_index(addrs[7]))
    with GILReleased(Python()):
        var out = host_spectral_predict(
            input, train_x, n_train, n_queries, n_features, k, n_clusters,
            n_neighbors, affinity, state,
        )
        for i in range(n_queries):
            olp.unsafe_store(i, out.labels[i])
        for i in range(n_queries * k):
            oep.unsafe_store(i, out.embedding[i])
    return PythonObject(0)


# ===========================================================================
# The metrics-classification lane (2026-09-14): the GPU binding's remaining
# metric entries, same names, same address contracts and params lists, over
# metrics/host/classification_oracle.mojo. Validation is the GPU binding's
# and metrics/estimator.mojo's, in their order and words.
# ===========================================================================


def rand_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::rand_index` on the host (DEVIATION 652). `params`:
    `0 n`."""
    _want(String("rand_score"), params, 1)
    var n = _index(params[0])
    var yt = read_i32(_index(y_true_addr), max(0, n))
    var yp = read_i32(_index(y_pred_addr), max(0, n))
    var out = Float64(0.0)
    with GILReleased(Python()):
        _check_pair(yt, yp, n)
        out = host_rand_score(yt, yp, n)
    return PythonObject(out)


def _regression_error_entry(
    y_true_addr: PythonObject, y_pred_addr: PythonObject, params: PythonObject,
    absolute: Bool, root: Bool,
) raises -> PythonObject:
    _want(String("regression_error"), params, 1)
    var n = _index(params[0])
    var y = f32_ptr(_index(y_true_addr))
    var prediction = f32_ptr(_index(y_pred_addr))
    var result = Float32(0.0)
    with GILReleased(Python()):
        if n <= 0:
            raise Error("metrics: n must be positive, got " + String(n))
        for i in range(n):
            if not isfinite(y.unsafe_load(i)) or not isfinite(prediction.unsafe_load(i)):
                raise Error("regression_error: inputs must be finite Float32")
        result = host_regression_error_ptr(y, prediction, n, absolute, root)
    return PythonObject(Float64(result))


def mean_squared_error_binding(
    y_true_addr: PythonObject, y_pred_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`regression_error[False, False]` on the host. `params`: `0 n`."""
    return _regression_error_entry(y_true_addr, y_pred_addr, params, False, False)


def mean_absolute_error_binding(
    y_true_addr: PythonObject, y_pred_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`regression_error[True, False]` on the host. `params`: `0 n`."""
    return _regression_error_entry(y_true_addr, y_pred_addr, params, True, False)


def root_mean_squared_error_binding(
    y_true_addr: PythonObject, y_pred_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`regression_error[False, True]` on the host. `params`: `0 n`."""
    return _regression_error_entry(y_true_addr, y_pred_addr, params, False, True)


def _binary_ranking_checked(
    y: List[Int32], scores: List[Float32], n: Int, curve: Bool,
) raises -> Tuple[List[Float32], Int]:
    """`binary_ranking_host[curve]`, `metrics/estimator.mojo`: the length,
    label and finiteness checks, the both-classes refusal for the AUC."""
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
    if not curve:
        if not has_zero or not has_one:
            raise Error("roc_auc_score: both classes required")
    return host_binary_ranking(y, scores, n, curve)


def roc_auc_score_binding(
    true_addr: PythonObject, score_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Binary ROC AUC on the host. `params`: `0 n`; one float32 written."""
    _want(String("roc_auc_score"), params, 1)
    var n = _index(params[0])
    if n <= 0 or n > 2147483647:
        raise Error("roc_auc_score: invalid n")
    var y = read_i32(_index(true_addr), n)
    var scores = read_f32(_index(score_addr), n)
    var output = f32_ptr(_index(out_addr))
    with GILReleased(Python()):
        var result = _binary_ranking_checked(y, scores, n, False)
        output[0] = result[0][0]
    return PythonObject(1)


def precision_recall_curve_binding(
    true_addr: PythonObject, score_addr: PythonObject, precision_addr: PythonObject,
    recall_addr: PythonObject, threshold_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Binary PR curve on the host. `params`: `0 n`; writes `m + 1`
    precisions and recalls and `m` thresholds, returns `m`."""
    _want(String("precision_recall_curve"), params, 1)
    var n = _index(params[0])
    if n <= 0 or n > 2147483647:
        raise Error("precision_recall_curve: invalid n")
    var y = read_i32(_index(true_addr), n)
    var scores = read_f32(_index(score_addr), n)
    var precision = f32_ptr(_index(precision_addr))
    var recall = f32_ptr(_index(recall_addr))
    var thresholds = f32_ptr(_index(threshold_addr))
    var m = 0
    with GILReleased(Python()):
        var result = _binary_ranking_checked(y, scores, n, True)
        m = result[1]
        for i in range(m + 1):
            precision[i] = result[0][i]
            recall[i] = result[0][n + 1 + i]
        for i in range(m):
            thresholds[i] = result[0][2 * (n + 1) + i]
    return PythonObject(m)


def log_loss_binding(
    true_addr: PythonObject, probabilities_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Log loss on the host. `params`: `0 n, 1 k, 2 normalize`; one float32
    written."""
    _want(String("log_loss"), params, 3)
    var n = _index(params[0])
    var k = _index(params[1])
    var normalize = _index(params[2])
    if n <= 0 or n > 2147483647 or k < 2 or k > 2147483647 // n:
        raise Error("log_loss: invalid input dimensions")
    var y = read_i32(_index(true_addr), n)
    var probability = read_f32(_index(probabilities_addr), n * k)
    var out = f32_ptr(_index(out_addr))
    with GILReleased(Python()):
        if len(y) < n or len(probability) < n * k or normalize < 0 or normalize > 1:
            raise Error("log_loss: invalid input length or normalization")
        for i in range(n):
            if Int(y[i]) < 0 or Int(y[i]) >= k:
                raise Error("log_loss: encoded label out of range")
        for i in range(n * k):
            if not isfinite(probability[i]) or probability[i] < 0 or probability[i] > 1:
                raise Error("log_loss: probabilities must be finite and within [0,1]")
        out[0] = host_log_loss(y, probability, n, k, normalize)
    return PythonObject(1)


def _check_classification(y: List[Int32], p: List[Int32], n: Int, k: Int, matrix: Bool) raises:
    """`_check_classification[matrix]`, `metrics/estimator.mojo`."""
    _check_pair(y, p, n)
    var cap = MAX_CONFUSION_CLASSES if matrix else MAX_PRF_CLASSES
    if n > 2147483647 or k <= 0 or k > cap:
        raise Error("classification metrics: count or class allocation bound exceeded")
    var minimum = -1 if matrix else 0
    for i in range(n):
        if Int(y[i]) < minimum or Int(y[i]) >= k or Int(p[i]) < minimum or Int(p[i]) >= k:
            raise Error("classification metrics: encoded label out of range")


def _check_classification_ptr(
    y: MutPointer[Int32, MutUntrackedOrigin],
    p: MutPointer[Int32, MutUntrackedOrigin],
    n: Int, k: Int, matrix: Bool,
) raises:
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    var cap = MAX_CONFUSION_CLASSES if matrix else MAX_PRF_CLASSES
    if n > 2147483647 or k <= 0 or k > cap:
        raise Error("classification metrics: count or class allocation bound exceeded")
    var minimum = -1 if matrix else 0
    for i in range(n):
        if Int(y.unsafe_load(i)) < minimum or Int(y.unsafe_load(i)) >= k or Int(p.unsafe_load(i)) < minimum or Int(p.unsafe_load(i)) >= k:
            raise Error("classification metrics: encoded label out of range")


def confusion_matrix_binding(
    true_addr: PythonObject, pred_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Confusion counts on the host. `params`: `0 n, 1 n_classes,
    2 normalization`; normalization 0 writes Int64 counts, 1 (true), 2
    (pred) and 3 (all) write Float32 ratios. Returns `k * k`."""
    _want(String("confusion_matrix"), params, 3)
    var n = _index(params[0])
    var k = _index(params[1])
    var normalization = _index(params[2])
    var address = _index(out_addr)
    if address == 0:
        raise Error("confusion_matrix: null output")
    var y = i32_ptr(_index(true_addr))
    var p = i32_ptr(_index(pred_addr))
    if normalization == 0:
        var output = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
        with GILReleased(Python()):
            _check_classification_ptr(y, p, n, k, True)
            var values = host_confusion_matrix_ptr(y, p, n, k, normalization)
            for i in range(len(values[0])):
                output[i] = values[0][i]
    else:
        var output = f32_ptr(address)
        with GILReleased(Python()):
            _check_classification_ptr(y, p, n, k, True)
            var values = host_confusion_matrix_ptr(y, p, n, k, normalization)
            for i in range(len(values[1])):
                output[i] = values[1][i]
    return PythonObject(k * k)


def precision_recall_fscore_binding(
    true_addr: PythonObject, pred_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Precision, recall and F1 on the host. `params`: `0 n, 1 k,
    2 average (0 None, 1 binary, 2 micro, 3 macro, 4 weighted), 3 pos_idx,
    4 zero_division, 5 n_selected`; writes `3 * width + 3` float32 values
    and returns that count."""
    _want(String("precision_recall_fscore"), params, 6)
    var n = _index(params[0])
    var k = _index(params[1])
    var average = _index(params[2])
    var positive = _index(params[3])
    var zero = _index(params[4])
    var selected = _index(params[5])
    var y = read_i32(_index(true_addr), max(0, n))
    var p = read_i32(_index(pred_addr), max(0, n))
    var output = f32_ptr(_index(out_addr))
    var written = 0
    with GILReleased(Python()):
        _check_classification(y, p, n, k, False)
        var values = host_precision_recall_fscore(y, p, n, k, average, positive, zero, selected)
        for i in range(len(values)):
            output[i] = values[i]
        written = len(values)
    return PythonObject(written)


def kl_divergence_binding(
    p_addr: PythonObject,
    q_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::kl_divergence` on the host, the float overload
    (DEVIATIONS 653, 658), not normalized. `params`: `0 n`."""
    _want(String("kl_divergence"), params, 1)
    var n = _index(params[0])
    var p = read_f32(_index(p_addr), max(0, n))
    var q = read_f32(_index(q_addr), max(0, n))
    var out = Float32(0.0)
    with GILReleased(Python()):
        _check_float_pair(p, q, n)
        out = host_kl_divergence(p, q, n)
    return PythonObject(Float64(out))


def trustworthiness_binding(
    x_addr: PythonObject,
    x_embedded_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::trustworthiness_score<float, L2SqrtUnexpanded>` on the
    host (DEVIATION 655). `params`: `0 n, 1 m, 2 d, 3 n_neighbors,
    4 batch_size`."""
    _want(String("trustworthiness"), params, 5)
    var n = _index(params[0])
    var m = _index(params[1])
    var d = _index(params[2])
    var n_neighbors = _index(params[3])
    var batch_size = _index(params[4])
    var x = read_f32(_index(x_addr), max(0, n * m))
    var emb = read_f32(_index(x_embedded_addr), max(0, n * d))
    var out = Float64(0.0)
    with GILReleased(Python()):
        if n <= 0 or m <= 0 or d <= 0:
            raise Error(
                "trustworthiness: n, n_features and n_components must all be"
                " positive, got " + String(n) + ", " + String(m) + ", " + String(d)
            )
        out = host_trustworthiness(x, emb, n, m, d, n_neighbors, batch_size)
    return PythonObject(out)


# ===========================================================================
# Group F: UMAP fit_transform and transform.
# ===========================================================================


def _umap_float32(value: PythonObject) raises -> Float32:
    return Float32(Float64(py=value))


def umap_fit_transform_binding(
    x_addr: PythonObject,
    embedding_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`umap_fit_transform_binding`, `bindings/_mojolearn_metrics.mojo`, on
    the host: the same refusals in the same order, then
    `host_umap_fit_transform`. Writes the `n_samples x n_components`
    row-major embedding; returns `n_components`.

    `params` (mirrored in `_umap_impl.py`): `0 n_samples, 1 n_features,
    2 n_neighbors, 3 n_components, 4 n_epochs, 5 min_dist, 6 spread,
    7 set_op_mix_ratio, 8 local_connectivity, 9 random_state`, then
    optionally `10 learning_rate, 11 repulsion_strength,
    12 negative_sample_rate`."""
    if len(params) != 10:
        _want(String("umap_fit_transform"), params, 13)
    var n = _index(params[0])
    var d = _index(params[1])
    var seed = _index(params[9])
    if d < 1 or seed < 0:
        raise Error("UMAP requires positive features and a nonnegative seed")
    var config = UMAPParams(
        n_neighbors=_index(params[2]), n_components=_index(params[3]),
        n_epochs=_index(params[4]), min_dist=_umap_float32(params[5]),
        spread=_umap_float32(params[6]),
        set_op_mix_ratio=_umap_float32(params[7]),
        local_connectivity=_umap_float32(params[8]),
        random_seed=UInt64(seed),
    )
    if len(params) == 13:
        config.learning_rate = _umap_float32(params[10])
        config.repulsion_strength = _umap_float32(params[11])
        config.negative_sample_rate = _index(params[12])
    config.validate(n)
    if (config.n_components != 2 and config.n_components != 3) or (
        n < 2 * config.n_components + 4
    ):
        raise Error("UMAP requires 2D/3D output and enough samples for spectral init")
    var x = read_f32(_index(x_addr), n * d)
    var output = f32_ptr(_index(embedding_addr))
    var embedding = List[Float32]()
    with GILReleased(Python()):
        embedding = host_umap_fit_transform(x, n, d, config)
    if len(embedding) != n * config.n_components:
        raise Error("UMAP returned an unexpected embedding shape")
    for value in embedding:
        if not isfinite(value):
            raise Error("UMAP returned a non-finite embedding")
    for i in range(len(embedding)):
        output.unsafe_store(i, embedding[i])
    return PythonObject(config.n_components)


def umap_transform_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`umap_transform_binding`, `bindings/_mojolearn_metrics.mojo`, on the
    host. Addresses: training X, frozen embedding, query X, output. Scalars:
    `n_train, n_queries, n_features`, then the eight legacy fit parameters
    and optionally `learning_rate, repulsion_strength,
    negative_sample_rate`. Only the output is written."""
    _want(String("umap_transform addresses"), addrs, 4)
    if len(params) != 11:
        _want(String("umap_transform parameters"), params, 14)
    var n = _index(params[0])
    var rows = _index(params[1])
    var d = _index(params[2])
    var seed = _index(params[10])
    if n < 2 or rows < 1 or d < 1 or seed < 0:
        raise Error("UMAP transform requires positive dimensions and a nonnegative seed")
    var config = UMAPParams(
        n_neighbors=_index(params[3]), n_components=_index(params[4]),
        n_epochs=_index(params[5]), min_dist=_umap_float32(params[6]),
        spread=_umap_float32(params[7]),
        set_op_mix_ratio=_umap_float32(params[8]),
        local_connectivity=_umap_float32(params[9]), random_seed=UInt64(seed),
    )
    if len(params) == 14:
        config.learning_rate = _umap_float32(params[11])
        config.repulsion_strength = _umap_float32(params[12])
        config.negative_sample_rate = _index(params[13])
    config.validate(n)
    if config.n_components != 2 and config.n_components != 3:
        raise Error("UMAP transform supports only 2D or 3D")
    var training = read_f32(_index(addrs[0]), n * d)
    var fitted = read_f32(_index(addrs[1]), n * config.n_components)
    var queries = read_f32(_index(addrs[2]), rows * d)
    var output = f32_ptr(_index(addrs[3]))
    var embedding = List[Float32]()
    with GILReleased(Python()):
        embedding = host_umap_transform(training, fitted, queries, n, rows, d, config)
    if len(embedding) != rows * config.n_components:
        raise Error("UMAP transform returned an unexpected shape")
    for value in embedding:
        if not isfinite(value):
            raise Error("UMAP transform returned a non-finite embedding")
    for i in range(len(embedding)):
        output.unsafe_store(i, embedding[i])
    return PythonObject(config.n_components)


@export
def PyInit__mojolearn_metrics_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_metrics_host")
        module.def_function[metrics_host_numeric_mode_binding]("metrics_host_numeric_mode")
        module.def_function[metrics_host_vendor_binding]("metrics_host_vendor")
        module.def_function[metrics_host_column_binding]("metrics_host_column")
        module.def_function[metrics_host_sabotage_binding]("metrics_host_sabotage")
        module.def_function[metrics_vendor_binding]("metrics_vendor")
        module.def_function[metrics_numeric_mode_binding]("metrics_numeric_mode")
        module.def_function[accuracy_score_binding]("accuracy_score")
        module.def_function[adjusted_rand_score_binding]("adjusted_rand_score")
        module.def_function[entropy_binding]("entropy")
        module.def_function[mutual_info_score_binding]("mutual_info_score")
        module.def_function[fowlkes_mallows_score_binding]("fowlkes_mallows_score")
        module.def_function[accuracy_score_weighted_binding]("accuracy_score_weighted")
        module.def_function[r2_score_weighted_binding]("r2_score_weighted")
        module.def_function[homogeneity_score_binding]("homogeneity_score")
        module.def_function[completeness_score_binding]("completeness_score")
        module.def_function[v_measure_score_binding]("v_measure_score")
        module.def_function[r2_score_binding]("r2_score")
        module.def_function[silhouette_binding]("silhouette")
        module.def_function[rand_score_binding]("rand_score")
        module.def_function[mean_squared_error_binding]("mean_squared_error")
        module.def_function[mean_absolute_error_binding]("mean_absolute_error")
        module.def_function[root_mean_squared_error_binding]("root_mean_squared_error")
        module.def_function[roc_auc_score_binding]("roc_auc_score")
        module.def_function[precision_recall_curve_binding]("precision_recall_curve")
        module.def_function[log_loss_binding]("log_loss")
        module.def_function[confusion_matrix_binding]("confusion_matrix")
        module.def_function[precision_recall_fscore_binding]("precision_recall_fscore")
        module.def_function[kl_divergence_binding]("kl_divergence")
        module.def_function[trustworthiness_binding]("trustworthiness")
        module.def_function[spectral_fit_predict_dataset_binding]("spectral_fit_predict_dataset")
        module.def_function[spectral_fit_predict_graph_binding]("spectral_fit_predict_graph")
        module.def_function[spectral_fit_predict_dataset_state_binding]("spectral_fit_predict_dataset_state")
        module.def_function[spectral_fit_predict_graph_state_binding]("spectral_fit_predict_graph_state")
        module.def_function[spectral_predict_binding]("spectral_predict")
        module.def_function[umap_fit_transform_binding]("umap_fit_transform")
        module.def_function[umap_transform_binding]("umap_transform")
        module.def_function[metrics_numeric_mode_binding]("umap_numeric_mode")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_metrics_host: ", error))
