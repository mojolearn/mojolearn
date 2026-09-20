# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the verified metrics kernels and for spectral clustering.

Kept in a SEPARATE extension, like `_mojolearn_estimators.mojo`, so an
independently changing binding does not become a merge point. Arrays cross
as borrowed NumPy addresses; every device buffer and context lives for one
call and no pointer is retained past the call that was handed it.

`metrics/` and `spectral/` both got
their Python surface in the same round and neither is large enough to earn a
build script of its own. They share nothing but this file and
`bindings/build_metrics.sh`.
UMAP shares this extension because its pipeline uses the same neighbor and
spectral primitives. Its source fixture evidence is recorded in umap/README.md;
its Python surface is gated separately by tests/test_umap_surface.py.

WHY THE SCALARS ARRIVE AS ONE LIST. `PythonModuleBuilder.def_function`
infers its signature from arity and stops working above roughly nine
arguments. Buffer addresses go positionally, every scalar goes in one
`params` list, and **THE ORDER OF THAT LIST IS WRITTEN OUT IN A COMMENT ON
BOTH SIDES IN THE SAME WORDS** -- here and in
`python/mojolearn/_metrics_impl.py` / `_spectral_impl.py`. A silent
reordering here is a WRONG ANSWER rather than a failure, so every entry
below also checks `len(params)` and names the count it wanted.

THE CERTIFICATION STATUS OF THE TWO LANES IS NOT THE SAME, and a reader of
this file should know which is which before believing a number that comes
out of it:

  * `metrics/` is CERTIFIED bit-identical Apple M4 <-> NVIDIA H100 <-> AMD
    MI325X at leg 11 (`archive/evidence/E3_RESULTS.md` round 11, commit 144aa5b, section 7,
    34 stages), on the 34-stage card of that commit. The card has since
    grown to 61 stages on Apple only; the three-vendor leg on the GROWN card
    is OWED (`metrics/README.md` Status).
  * `spectral/` has run on ONE Apple M4 and NOWHERE ELSE. Its contract says
    so in section 10: "no cross-vendor result of any kind". It is not in
    `tools/e1_bootstrap.sh` phase 8, it has no card in either leg-11 lane
    directory, and `tools/e3_round_judge.sh` section 7 does not name it.

Neither of those sentences may be softened in this file or in the Python
wrappers without a leg to point at.
"""

# DEVIATION 2486: shared byte-preserving host copies.
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from max.gpu.host import DeviceContext
from std.math import isfinite
from umap.estimator import fit_transform as umap_fit_transform
from umap.transform import transform as umap_transform
from umap.params import UMAPParams

from metrics.estimator import (
    accuracy_score_host,
    accuracy_score_weighted_host,
    adjusted_rand_score_host,
    completeness_score_host,
    entropy_host,
    fowlkes_mallows_score_host,
    homogeneity_score_host,
    kl_divergence_host,
    mutual_info_score_host,
    r2_score_host,
    r2_score_weighted_host,
    regression_error_host,
    log_loss_host,
    binary_ranking_host,
    confusion_matrix_host,
    precision_recall_fscore_host,
    rand_score_host,
    silhouette_host,
    trustworthiness_host,
    v_measure_score_host,
)
from spectral.estimator import (
    spectral_embedding_dataset_host,
    spectral_embedding_graph_host,
    spectral_fit_predict_dataset_host,
    spectral_fit_predict_dataset_host_keep,
    spectral_fit_predict_graph_host,
    spectral_fit_predict_graph_host_keep,
    spectral_predict_host,
)
from spectral.host.spectral_predict_host import (
    SpectralPredictionState,
    spectral_predict_check_state,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _load_i32(addr: Int, n: Int) raises -> List[Int32]:
    return read_i32(addr, max(0, n))


def _load_f32(addr: Int, n: Int) raises -> List[Float32]:
    return read_f32(addr, max(0, n))


def _want(name: String, params: PythonObject, k: Int) raises:
    if len(params) != k:
        raise Error(
            name + ": params must contain " + String(k) + " values, got "
            + String(len(params))
        )


# ===========================================================================
# Group A: the label metrics. int32 labels, one shape each.
# ===========================================================================


def accuracy_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::accuracy_score_py`: the fraction of agreeing positions.

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
    """
    _want(String("accuracy_score"), params, 1)
    var n = Int(py=params[0])
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var out = Float32(0.0)
    with GILReleased(Python()):
        out = accuracy_score_host(yt, yp, n)
    return PythonObject(Float64(out))


def rand_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::rand_index` (DEVIATION 652).

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
    """
    _want(String("rand_score"), params, 1)
    var n = Int(py=params[0])
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = rand_score_host(yt, yp, n)
    return PythonObject(out)


def adjusted_rand_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::adjusted_rand_index`, the int32 instantiation. Takes
    RAW labels: this entry has no label range, because theirs has none.

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
    """
    _want(String("adjusted_rand_score"), params, 1)
    var n = Int(py=params[0])
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = adjusted_rand_score_host(yt, yp, n)
    return PythonObject(out)


def entropy_binding(
    labels_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::entropy`, in NATS (DEVIATIONS 650, 651).

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
        1  lower_class_range
        2  upper_class_range
    """
    _want(String("entropy"), params, 3)
    var n = Int(py=params[0])
    var lower = Int32(Int(py=params[1]))
    var upper = Int32(Int(py=params[2]))
    var lab = _load_i32(Int(py=labels_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = entropy_host(lab, n, lower, upper)
    return PythonObject(out)


def mutual_info_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::mutual_info_score`, in NATS (DEVIATIONS 650, 651).

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
        1  lower_class_range
        2  upper_class_range
    """
    _want(String("mutual_info_score"), params, 3)
    var n = Int(py=params[0])
    var lower = Int32(Int(py=params[1]))
    var upper = Int32(Int(py=params[2]))
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = mutual_info_score_host(yt, yp, n, lower, upper)
    return PythonObject(out)


def accuracy_score_weighted_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    w_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Weighted accuracy (`metrics/impl/weighted_scores.mojo`), scikit-learn's
    `np.average(y_true == y_pred, weights=w)` in Float32.

    `params`, in this exact order (matched in
    `python/mojolearn/_metrics_impl.py`):

        0  n
    """
    _want(String("accuracy_score_weighted"), params, 1)
    var n = Int(py=params[0])
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var w = _load_f32(Int(py=w_addr), n)
    var out = Float32(0.0)
    with GILReleased(Python()):
        out = accuracy_score_weighted_host(yt, yp, w, n)
    return PythonObject(Float64(out))


def r2_score_weighted_binding(
    y_addr: PythonObject,
    y_hat_addr: PythonObject,
    w_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Weighted R2 (`metrics/impl/weighted_scores.mojo`), scikit-learn's
    weighted `r2_score` with `force_finite=True`, Float32.

    `params`, in this exact order (matched in
    `python/mojolearn/_metrics_impl.py`):

        0  n
    """
    _want(String("r2_score_weighted"), params, 1)
    var n = Int(py=params[0])
    var y = _load_f32(Int(py=y_addr), n)
    var yh = _load_f32(Int(py=y_hat_addr), n)
    var w = _load_f32(Int(py=w_addr), n)
    var out = Float32(0.0)
    with GILReleased(Python()):
        out = r2_score_weighted_host(y, yh, w, n)
    return PythonObject(Float64(out))


def fowlkes_mallows_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """scikit-learn `fowlkes_mallows_score` (cuML has none) over the device
    contingency matrix (`metrics/impl/fowlkes_mallows.mojo`).

    `params`, in this exact order (matched in
    `python/mojolearn/_metrics_impl.py`):

        0  n
        1  lower_class_range
        2  upper_class_range
    """
    _want(String("fowlkes_mallows_score"), params, 3)
    var n = Int(py=params[0])
    var lower = Int32(Int(py=params[1]))
    var upper = Int32(Int(py=params[2]))
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = fowlkes_mallows_score_host(yt, yp, n, lower, upper)
    return PythonObject(out)


def homogeneity_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::homogeneity_score`.

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
        1  lower_class_range
        2  upper_class_range
    """
    _want(String("homogeneity_score"), params, 3)
    var n = Int(py=params[0])
    var lower = Int32(Int(py=params[1]))
    var upper = Int32(Int(py=params[2]))
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = homogeneity_score_host(yt, yp, n, lower, upper)
    return PythonObject(out)


def completeness_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::completeness_score` (RAFT's homogeneity with the two
    arrays swapped, so the TRANSPOSED fold).

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
        1  lower_class_range
        2  upper_class_range
    """
    _want(String("completeness_score"), params, 3)
    var n = Int(py=params[0])
    var lower = Int32(Int(py=params[1]))
    var upper = Int32(Int(py=params[2]))
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = completeness_score_host(yt, yp, n, lower, upper)
    return PythonObject(out)


def v_measure_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::v_measure` with `beta` honored.

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
        1  lower_class_range
        2  upper_class_range
        3  beta               (float)
    """
    _want(String("v_measure_score"), params, 4)
    var n = Int(py=params[0])
    var lower = Int32(Int(py=params[1]))
    var upper = Int32(Int(py=params[2]))
    var beta = Float64(py=params[3])
    var yt = _load_i32(Int(py=y_true_addr), n)
    var yp = _load_i32(Int(py=y_pred_addr), n)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = v_measure_score_host(yt, yp, n, lower, upper, beta)
    return PythonObject(out)


# ===========================================================================
# Group B: r2 and KL divergence. float32 in, float32 out.
# ===========================================================================


def r2_score_binding(
    y_true_addr: PythonObject,
    y_pred_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::r2_score_py`, the float overload (DEVIATIONS 653, 657).

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
    """
    _want(String("r2_score"), params, 1)
    var n = Int(py=params[0])
    var y = _load_f32(Int(py=y_true_addr), n)
    var yh = _load_f32(Int(py=y_pred_addr), n)
    var out = Float32(0.0)
    with GILReleased(Python()):
        out = r2_score_host(y, yh, n)
    return PythonObject(Float64(out))



def regression_error_binding[absolute: Bool = False, root: Bool = False](
    y_true_addr: PythonObject, y_pred_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    # params[0] = n; finite 1-D Float32, unweighted. Entire reduction is GPU.
    _want(String("regression_error"), params, 1)
    var n = Int(py=params[0])
    var y = _load_f32(Int(py=y_true_addr), n)
    var prediction = _load_f32(Int(py=y_pred_addr), n)
    var result = Float32(0.0)
    with GILReleased(Python()):
        result = regression_error_host[absolute, root](y, prediction, n)
    return PythonObject(Float64(result))




def roc_auc_score_binding(true_addr: PythonObject, score_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    _want(String("roc_auc_score"),params,1)
    var n = Int(py=params[0])
    if n <= 0 or n > 2147483647:
        raise Error("roc_auc_score: invalid n")
    var y = _load_i32(Int(py=true_addr),n)
    var scores = _load_f32(Int(py=score_addr),n)
    var output = _f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var result = binary_ranking_host[False](y,scores,n)
        output.unsafe_store(0,result[0][0])
    return PythonObject(1)


def precision_recall_curve_binding(true_addr: PythonObject, score_addr: PythonObject, precision_addr: PythonObject, recall_addr: PythonObject, threshold_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    _want(String("precision_recall_curve"),params,1)
    var n = Int(py=params[0])
    if n <= 0 or n > 2147483647:
        raise Error("precision_recall_curve: invalid n")
    var y = _load_i32(Int(py=true_addr),n)
    var scores = _load_f32(Int(py=score_addr),n)
    var precision = _f32_ptr(Int(py=precision_addr))
    var recall = _f32_ptr(Int(py=recall_addr))
    var thresholds = _f32_ptr(Int(py=threshold_addr))
    var m = 0
    with GILReleased(Python()):
        var result = binary_ranking_host[True](y,scores,n)
        m = result[1]
        for i in range(m+1):
            precision.unsafe_store(i,result[0][i])
            recall.unsafe_store(i,result[0][n+1+i])
        for i in range(m):
            thresholds.unsafe_store(i,result[0][2*(n+1)+i])
    return PythonObject(m)


def log_loss_binding(
    true_addr: PythonObject, probabilities_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    _want(String("log_loss"), params, 3)
    var n = Int(py=params[0])
    var k = Int(py=params[1])
    var normalize = Int(py=params[2])
    if n <= 0 or n > 2147483647 or k < 2 or k > 2147483647 // n:
        raise Error("log_loss: invalid input dimensions")
    var y = _load_i32(Int(py=true_addr), n)
    var probability = _load_f32(Int(py=probabilities_addr), n*k)
    var out = _f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        out.unsafe_store(0, log_loss_host(y,probability,n,k,normalize))
    return PythonObject(1)


def confusion_matrix_binding(
    true_addr: PythonObject, pred_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    # params=[n,n_classes,normalization]; norm0:Int64,1true/2pred/3all:Float32.
    _want(String("confusion_matrix"),params,3)
    var n = Int(py=params[0])
    var k = Int(py=params[1])
    var normalization = Int(py=params[2])
    var address = Int(py=out_addr)
    if address == 0:
        raise Error("confusion_matrix: null output")
    var y = _load_i32(Int(py=true_addr),n)
    var p = _load_i32(Int(py=pred_addr),n)
    if normalization == 0:
        var output = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
        with GILReleased(Python()):
            var values = confusion_matrix_host[DType.int64](y,p,n,k,normalization)
            for i in range(len(values)):
                output.unsafe_store(i,values[i])
    else:
        var output = _f32_ptr(address)
        with GILReleased(Python()):
            var values = confusion_matrix_host[DType.float32](y,p,n,k,normalization)
            for i in range(len(values)):
                output.unsafe_store(i,values[i])
    return PythonObject(k*k)


def precision_recall_fscore_binding(
    true_addr: PythonObject, pred_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    # params=[n,k,average,pos_idx,zero_division,n_selected]. The selected
    # classes are first in the encoding; all remaining labels STILL count.
    # avg0None/1binary/2micro/3macro/4weighted. Output3*w+3 Float32:
    # precision,recall,F1 rows; three trailing undefined flags.
    _want(String("precision_recall_fscore"),params,6)
    var n = Int(py=params[0])
    var k = Int(py=params[1])
    var average = Int(py=params[2])
    var positive = Int(py=params[3])
    var zero = Int(py=params[4])
    var selected = Int(py=params[5])
    var y = _load_i32(Int(py=true_addr),n)
    var p = _load_i32(Int(py=pred_addr),n)
    var output = _f32_ptr(Int(py=out_addr))
    var written = 0
    with GILReleased(Python()):
        var values = precision_recall_fscore_host(y,p,n,k,average,positive,zero,selected)
        for i in range(len(values)):
            output.unsafe_store(i,values[i])
        written = len(values)
    return PythonObject(written)


def kl_divergence_binding(
    p_addr: PythonObject,
    q_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::kl_divergence`, the float overload (DEVIATIONS 653,
    658). NOT normalized, exactly as theirs.

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n
    """
    _want(String("kl_divergence"), params, 1)
    var n = Int(py=params[0])
    var p = _load_f32(Int(py=p_addr), n)
    var q = _load_f32(Int(py=q_addr), n)
    var out = Float32(0.0)
    with GILReleased(Python()):
        out = kl_divergence_host(p, q, n)
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
    """`ML::Metrics::Batched::silhouette_score`, the float batched entry
    cuML's Python dispatches (DEVIATIONS 654, 656). Writes `n_rows` float32
    per-sample coefficients to `scores_addr` (cuML's `silhouette_samples`)
    and returns their mean (cuML's `silhouette_score`).

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n_rows
        1  n_cols
        2  n_labels    (labels must already be mapped onto [0, n_labels-1])
        3  chunksize   (cuML's chunk, default 40000; validated >= 1,
                        SCHEDULING only -- no distance tile is materialized)

    There is no `metric` slot: only `DistanceType::L2SqrtUnexpanded` (cuML
    'euclidean'/'l2') is implemented and the Python wrapper refuses every other
    name before reaching here.
    """
    _want(String("silhouette"), params, 4)
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_labels = Int(py=params[2])
    var chunk = Int(py=params[3])
    var x = _load_f32(Int(py=x_addr), n_rows * n_cols)
    var lab = _load_i32(Int(py=labels_addr), n_rows)
    var sp = _f32_ptr(Int(py=scores_addr))
    var scores = List[Float32]()
    var mean = Float32(0.0)
    with GILReleased(Python()):
        mean = silhouette_host(x, lab, n_rows, n_cols, n_labels, chunk, scores)
    for i in range(n_rows):
        sp.unsafe_store(i, scores[i])
    return PythonObject(Float64(mean))


# ===========================================================================
# Group D: trustworthiness.
# ===========================================================================


def trustworthiness_binding(
    x_addr: PythonObject,
    x_embedded_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ML::Metrics::trustworthiness_score<float, L2SqrtUnexpanded>`
    (DEVIATION 655).

    `params`, in this exact order (mirrored in
    `python/mojolearn/_metrics_impl.py`):

        0  n            (rows of X and of X_embedded)
        1  m            (columns of X)
        2  d            (columns of X_embedded)
        3  n_neighbors
        4  batch_size   (cuML's batchSize; validated >= 1, sizes nothing
                         because DEVIATION 655 counts ranks instead of
                         sorting the n x n distance matrix)
    """
    _want(String("trustworthiness"), params, 5)
    var n = Int(py=params[0])
    var m = Int(py=params[1])
    var d = Int(py=params[2])
    var n_neighbors = Int(py=params[3])
    var batch_size = Int(py=params[4])
    var x = _load_f32(Int(py=x_addr), n * m)
    var emb = _load_f32(Int(py=x_embedded_addr), n * d)
    var out = Float64(0.0)
    with GILReleased(Python()):
        out = trustworthiness_host(x, emb, n, m, d, n_neighbors, batch_size)
    return PythonObject(out)


# ===========================================================================
# spectral: cuML's `ML::SpectralClustering::fit_predict`, two overloads.
# ===========================================================================


def _guard_spectral_outputs(
    labels: List[Int32],
    embedding: List[Float32],
    n_samples: Int,
    n_components: Int,
) raises:
    """The two output buffers were sized by the Python caller from
    `n_samples` and `n_components`. Check what came back BEFORE writing a
    single element into them.

    `n_out == n_components` holds on the clustering path because cuVS's
    clustering overload sets `drop_first = false`, so this cannot fire
    today. It is here because the day it CAN fire -- somebody threading
    `drop_first` through, say -- the alternative is a silent heap overrun
    in a NumPy array, which is the worst failure this boundary can have."""
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
    """`fit_predict` on a DATASET (cuML `affinity='nearest_neighbors'`):
    kNN connectivity graph, Laplacian, thick-restart Lanczos, k-means.

    Writes `n_samples` int32 labels to `labels_addr` and the `n_samples x
    n_out` row-major embedding to `embedding_addr`; returns `n_out`, which
    equals `n_components` on this path because cuVS's clustering overload
    sets `drop_first = false`. The caller must have sized `embedding_addr`
    for `n_samples * n_components` floats.

    `params`, in this exact order (mirrored in
    `python/mojolearn/_spectral_impl.py`):

        0  n_samples
        1  n_features
        2  n_clusters
        3  n_components
        4  n_init
        5  n_neighbors
        6  eigen_tol      (float; cuVS's plumbed `tolerance`, default 1e-5)
        7  seed           (cuVS's `rng_state` seed; the no-seed arm is
                           REFUSED by DEVIATION 772, so there is always one)
    """
    _want(String("spectral_fit_predict_dataset"), params, 8)
    var n_samples = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_clusters = Int(py=params[2])
    var n_components = Int(py=params[3])
    var n_init = Int(py=params[4])
    var n_neighbors = Int(py=params[5])
    var eigen_tol = Float32(Float64(py=params[6]))
    var seed = UInt64(Int(py=params[7]))
    var x = _load_f32(Int(py=x_addr), n_samples * n_features)
    var lp = _i32_ptr(Int(py=labels_addr))
    var ep = _f32_ptr(Int(py=embedding_addr))
    var labels = List[Int32]()
    var embedding = List[Float32]()
    var n_out = 0
    with GILReleased(Python()):
        n_out = spectral_fit_predict_dataset_host(
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
    """`fit_predict` on a PRECOMPUTED connectivity graph given as COO
    triples (cuML `affinity='precomputed'`). No kNN runs, so `n_neighbors`
    is carried in the config and read by nobody on this path.

    Writes `n_samples` int32 labels and the `n_samples x n_out` row-major
    embedding; returns `n_out`.

    `params`, in this exact order (mirrored in
    `python/mojolearn/_spectral_impl.py`):

        0  n_samples
        1  nnz            (length of rows, cols and vals)
        2  n_clusters
        3  n_components
        4  n_init
        5  n_neighbors    (carried, unused on this path)
        6  eigen_tol      (float)
        7  seed
    """
    _want(String("spectral_fit_predict_graph"), params, 8)
    var n_samples = Int(py=params[0])
    var nnz = Int(py=params[1])
    var n_clusters = Int(py=params[2])
    var n_components = Int(py=params[3])
    var n_init = Int(py=params[4])
    var n_neighbors = Int(py=params[5])
    var eigen_tol = Float32(Float64(py=params[6]))
    var seed = UInt64(Int(py=params[7]))
    var rows = _load_i32(Int(py=rows_addr), nnz)
    var cols = _load_i32(Int(py=cols_addr), nnz)
    var vals = _load_f32(Int(py=vals_addr), nnz)
    var lp = _i32_ptr(Int(py=labels_addr))
    var ep = _f32_ptr(Int(py=embedding_addr))
    var labels = List[Int32]()
    var embedding = List[Float32]()
    var n_out = 0
    with GILReleased(Python()):
        n_out = spectral_fit_predict_graph_host(
            rows, cols, vals, n_samples, n_clusters, n_components, n_init,
            n_neighbors, eigen_tol, seed, labels, embedding,
        )
    _guard_spectral_outputs(labels, embedding, n_samples, n_components)
    for i in range(n_samples):
        lp.unsafe_store(i, labels[i])
    for i in range(len(embedding)):
        ep.unsafe_store(i, embedding[i])
    return PythonObject(n_out)


# SpectralEmbedding (lane/expose-spectral-embedding, 2026-09-20): cuML's
# `ML::SpectralEmbedding::transform`, the dataset and the COO overloads.


def _guard_embedding_output(
    embedding: List[Float32], n_samples: Int, n_out: Int, n_cols: Int
) raises:
    """The output buffer was sized by the Python caller for `n_samples x
    n_cols`; check what came back before writing into it."""
    if n_out != n_cols or len(embedding) != n_samples * n_cols:
        raise Error(
            "spectral embedding: the kernel returned " + String(len(embedding))
            + " floats in " + String(n_out) + " columns, but the output buffer"
            " was sized for " + String(n_samples) + " x " + String(n_cols)
        )


def spectral_embedding_dataset_binding(
    x_addr: PythonObject,
    embedding_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SpectralEmbedding` on a DATASET (`affinity='nearest_neighbors'`).
    Writes the `n_samples x n_cols` row-major embedding; returns `n_out`.

    `params`, in this exact order (matched in
    `python/mojolearn/_spectral_impl.py`):

        0  n_samples
        1  n_features
        2  n_lanczos      (n_components, plus one when drop_first)
        3  n_cols         (columns the output buffer was sized for)
        4  n_neighbors
        5  norm_laplacian (0 or 1)
        6  drop_first     (0 or 1)
        7  seed
    """
    _want(String("spectral_embedding_dataset"), params, 8)
    var n_samples = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_lanczos = Int(py=params[2])
    var n_cols = Int(py=params[3])
    var n_neighbors = Int(py=params[4])
    var norm_laplacian = Int(py=params[5]) != 0
    var drop_first = Int(py=params[6]) != 0
    var seed = UInt64(Int(py=params[7]))
    var x = _load_f32(Int(py=x_addr), n_samples * n_features)
    var ep = _f32_ptr(Int(py=embedding_addr))
    var embedding = List[Float32]()
    var n_out = 0
    with GILReleased(Python()):
        n_out = spectral_embedding_dataset_host(
            x, n_samples, n_features, n_lanczos, n_neighbors, norm_laplacian,
            drop_first, seed, embedding,
        )
    _guard_embedding_output(embedding, n_samples, n_out, n_cols)
    for i in range(len(embedding)):
        ep.unsafe_store(i, embedding[i])
    return PythonObject(n_out)


def spectral_embedding_graph_binding(
    rows_addr: PythonObject,
    cols_addr: PythonObject,
    vals_addr: PythonObject,
    embedding_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SpectralEmbedding` on a PRECOMPUTED affinity given as COO triples.
    Writes the `n_samples x n_cols` row-major embedding; returns `n_out`.

    `params`, in this exact order (matched in
    `python/mojolearn/_spectral_impl.py`):

        0  n_samples
        1  nnz            (length of rows, cols and vals)
        2  n_lanczos      (n_components, plus one when drop_first)
        3  n_cols         (columns the output buffer was sized for)
        4  norm_laplacian (0 or 1)
        5  drop_first     (0 or 1)
        6  seed
    """
    _want(String("spectral_embedding_graph"), params, 7)
    var n_samples = Int(py=params[0])
    var nnz = Int(py=params[1])
    var n_lanczos = Int(py=params[2])
    var n_cols = Int(py=params[3])
    var norm_laplacian = Int(py=params[4]) != 0
    var drop_first = Int(py=params[5]) != 0
    var seed = UInt64(Int(py=params[6]))
    var rows = _load_i32(Int(py=rows_addr), nnz)
    var cols = _load_i32(Int(py=cols_addr), nnz)
    var vals = _load_f32(Int(py=vals_addr), nnz)
    var ep = _f32_ptr(Int(py=embedding_addr))
    var embedding = List[Float32]()
    var n_out = 0
    with GILReleased(Python()):
        n_out = spectral_embedding_graph_host(
            rows, cols, vals, n_samples, n_lanczos, norm_laplacian,
            drop_first, seed, embedding,
        )
    _guard_embedding_output(embedding, n_samples, n_out, n_cols)
    for i in range(len(embedding)):
        ep.unsafe_store(i, embedding[i])
    return PythonObject(n_out)


# lane/spectral-predict (2026-09-15, DEVIATION 2860): the fit entries that
# keep the prediction data, and SpectralClustering.predict.


def _write_spectral_state(
    state: SpectralPredictionState,
    addrs: PythonObject,
    first: Int,
    n_samples: Int,
    n_components: Int,
    n_clusters: Int,
) raises:
    """Copy the kept state into the caller's four arrays, `addrs[first..]`:
    eigenvalues (k), eigenvectors (n x k), diag (n), centroids
    (n_clusters x k), after checking every length."""
    spectral_predict_check_state(state, n_samples, n_components, n_clusters)
    var ev = _f32_ptr(Int(py=addrs[first]))
    var evec = _f32_ptr(Int(py=addrs[first + 1]))
    var dg = _f32_ptr(Int(py=addrs[first + 2]))
    var cent = _f32_ptr(Int(py=addrs[first + 3]))
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
    """`spectral_fit_predict_dataset` that also writes the prediction data
    for `SpectralClustering(prediction_data=True)`: COPIES of the Ritz
    values, the undivided Ritz vectors, the degree scaling and the final
    centroids (`spectral/host/spectral_predict_host.mojo`). Labels and
    embedding are the same call's.

    `addrs`, in this exact order (matched in `_spectral_impl.py`): x,
    labels, embedding, eigenvalues (k f32), eigenvectors (n x k f32), diag
    (n f32), centroids (n_clusters x k f32). `params`: the eight of
    `spectral_fit_predict_dataset`. Returns `n_out`."""
    if len(addrs) != 7:
        raise Error("spectral_fit_predict_dataset_state: addrs must contain 7 addresses, got " + String(len(addrs)))
    _want(String("spectral_fit_predict_dataset_state"), params, 8)
    var n_samples = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_clusters = Int(py=params[2])
    var n_components = Int(py=params[3])
    var n_init = Int(py=params[4])
    var n_neighbors = Int(py=params[5])
    var eigen_tol = Float32(Float64(py=params[6]))
    var seed = UInt64(Int(py=params[7]))
    var x = _load_f32(Int(py=addrs[0]), n_samples * n_features)
    var lp = _i32_ptr(Int(py=addrs[1]))
    var ep = _f32_ptr(Int(py=addrs[2]))
    var labels = List[Int32]()
    var embedding = List[Float32]()
    var state = SpectralPredictionState()
    var n_out = 0
    with GILReleased(Python()):
        n_out = spectral_fit_predict_dataset_host_keep(
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
    """`spectral_fit_predict_graph` that also writes the prediction data.
    `addrs`: rows, cols, vals, labels, embedding, eigenvalues, eigenvectors,
    diag, centroids. `params`: the eight of `spectral_fit_predict_graph`.
    Returns `n_out`."""
    if len(addrs) != 9:
        raise Error("spectral_fit_predict_graph_state: addrs must contain 9 addresses, got " + String(len(addrs)))
    _want(String("spectral_fit_predict_graph_state"), params, 8)
    var n_samples = Int(py=params[0])
    var nnz = Int(py=params[1])
    var n_clusters = Int(py=params[2])
    var n_components = Int(py=params[3])
    var n_init = Int(py=params[4])
    var n_neighbors = Int(py=params[5])
    var eigen_tol = Float32(Float64(py=params[6]))
    var seed = UInt64(Int(py=params[7]))
    var rows = _load_i32(Int(py=addrs[0]), nnz)
    var cols = _load_i32(Int(py=addrs[1]), nnz)
    var vals = _load_f32(Int(py=addrs[2]), nnz)
    var lp = _i32_ptr(Int(py=addrs[3]))
    var ep = _f32_ptr(Int(py=addrs[4]))
    var labels = List[Int32]()
    var embedding = List[Float32]()
    var state = SpectralPredictionState()
    var n_out = 0
    with GILReleased(Python()):
        n_out = spectral_fit_predict_graph_host_keep(
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
    """`SpectralClustering.predict` (DEVIATION 2860; the rule is stated in
    `spectral/host/spectral_predict_host.mojo`).

    `addrs`, in this exact order (matched in `_spectral_impl.py`): input
    (queries n_queries x n_features, or the precomputed affinity n_queries x
    n_train), training rows (n_train x n_features; 0 for precomputed),
    eigenvalues (k), eigenvectors (n_train x k), diag (n_train), centroids
    (n_clusters x k), out labels (int32 n_queries), out embedding (float32
    n_queries x k). `params`: n_train, n_queries, n_features (0 for
    precomputed), n_components, n_clusters, n_neighbors, affinity (0
    nearest_neighbors, 1 precomputed). Returns 0."""
    if len(addrs) != 8:
        raise Error("spectral_predict: addrs must contain 8 addresses, got " + String(len(addrs)))
    _want(String("spectral_predict"), params, 7)
    var n_train = Int(py=params[0])
    var n_queries = Int(py=params[1])
    var n_features = Int(py=params[2])
    var k = Int(py=params[3])
    var n_clusters = Int(py=params[4])
    var n_neighbors = Int(py=params[5])
    var affinity = Int(py=params[6])
    var width = n_train if affinity == 1 else n_features
    var input = _load_f32(Int(py=addrs[0]), n_queries * width)
    var train_x = List[Float32]()
    if affinity == 0:
        train_x = _load_f32(Int(py=addrs[1]), n_train * n_features)
    var state = SpectralPredictionState()
    state.eigenvalues = _load_f32(Int(py=addrs[2]), k)
    state.eigenvectors = _load_f32(Int(py=addrs[3]), n_train * k)
    state.diag = _load_f32(Int(py=addrs[4]), n_train)
    state.centroids = _load_f32(Int(py=addrs[5]), n_clusters * k)
    var olp = _i32_ptr(Int(py=addrs[6]))
    var oep = _f32_ptr(Int(py=addrs[7]))
    with GILReleased(Python()):
        var out = spectral_predict_host(
            input, train_x, n_train, n_queries, n_features, k, n_clusters,
            n_neighbors, affinity, state,
        )
        if len(out.labels) != n_queries or len(out.embedding) != n_queries * k:
            raise Error("spectral_predict: the kernel returned the wrong number of outputs")
        for i in range(n_queries):
            olp.unsafe_store(i, out.labels[i])
        for i in range(n_queries * k):
            oep.unsafe_store(i, out.embedding[i])
    return PythonObject(0)


def umap_fit_transform_binding(
    x_addr: PythonObject,
    embedding_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Borrow input/output buffers for the supported UMAP slice.

    params order (mirrored in _umap_impl.py): n_samples, n_features,
    n_neighbors, n_components, n_epochs, min_dist, spread,
    set_op_mix_ratio, local_connectivity, random_state, then optional
    learning_rate, repulsion_strength, negative_sample_rate.
    """
    if len(params) != 10:
        _want(String("umap_fit_transform"), params, 13)
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var seed = Int(py=params[9])
    if d < 1 or seed < 0:
        raise Error("UMAP requires positive features and a nonnegative seed")
    var config = UMAPParams(
        n_neighbors=Int(py=params[2]), n_components=Int(py=params[3]),
        n_epochs=Int(py=params[4]), min_dist=Float32(Float64(py=params[5])),
        spread=Float32(Float64(py=params[6])),
        set_op_mix_ratio=Float32(Float64(py=params[7])),
        local_connectivity=Float32(Float64(py=params[8])),
        random_seed=UInt64(seed),
    )
    if len(params) == 13:
        config.learning_rate = Float32(Float64(py=params[10]))
        config.repulsion_strength = Float32(Float64(py=params[11]))
        config.negative_sample_rate = Int(py=params[12])
    config.validate(n)
    if (config.n_components != 2 and config.n_components != 3) or (
        n < 2 * config.n_components + 4
    ):
        raise Error("UMAP requires 2D/3D output and enough samples for spectral init")
    var x = _load_f32(Int(py=x_addr), n * d)
    var output = _f32_ptr(Int(py=embedding_addr))
    var embedding = List[Float32]()
    with GILReleased(Python()):
        var ctx = DeviceContext()
        embedding = umap_fit_transform(ctx, x, n, d, config)
    if len(embedding) != n * config.n_components:
        raise Error("UMAP returned an unexpected embedding shape")
    for value in embedding:
        if not isfinite(value):
            raise Error("UMAP returned a non-finite embedding")
    for i in range(len(embedding)):
        output.unsafe_store(i, embedding[i])
    return PythonObject(config.n_components)


def umap_transform_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Addresses: training X, frozen embedding, query X, output.

    Scalars: n_train, n_queries, n_features, then the eight legacy fit parameters
    and optional learning_rate, repulsion_strength, negative_sample_rate.
    The training arrays are read-only; only output is written.
    """
    _want(String("umap_transform addresses"), addrs, 4)
    if len(params) != 11:
        _want(String("umap_transform parameters"), params, 14)
    var n = Int(py=params[0])
    var rows = Int(py=params[1])
    var d = Int(py=params[2])
    var seed = Int(py=params[10])
    if n < 2 or rows < 1 or d < 1 or seed < 0:
        raise Error("UMAP transform requires positive dimensions and a nonnegative seed")
    var config = UMAPParams(
        n_neighbors=Int(py=params[3]), n_components=Int(py=params[4]),
        n_epochs=Int(py=params[5]), min_dist=Float32(Float64(py=params[6])),
        spread=Float32(Float64(py=params[7])),
        set_op_mix_ratio=Float32(Float64(py=params[8])),
        local_connectivity=Float32(Float64(py=params[9])), random_seed=UInt64(seed),
    )
    if len(params) == 14:
        config.learning_rate = Float32(Float64(py=params[11]))
        config.repulsion_strength = Float32(Float64(py=params[12]))
        config.negative_sample_rate = Int(py=params[13])
    config.validate(n)
    if config.n_components != 2 and config.n_components != 3:
        raise Error("UMAP transform supports only 2D or 3D")
    var training = _load_f32(Int(py=addrs[0]), n * d)
    var fitted = _load_f32(Int(py=addrs[1]), n * config.n_components)
    var queries = _load_f32(Int(py=addrs[2]), rows * d)
    var output = _f32_ptr(Int(py=addrs[3]))
    var embedding = List[Float32]()
    with GILReleased(Python()):
        with DeviceContext() as ctx:
            embedding = umap_transform(ctx, training, fitted, queries, n, rows, d, config)
    if len(embedding) != rows * config.n_components:
        raise Error("UMAP transform returned an unexpected shape")
    for value in embedding:
        if not isfinite(value):
            raise Error("UMAP transform returned a non-finite embedding")
    for i in range(len(embedding)):
        output.unsafe_store(i, embedding[i])
    return PythonObject(config.n_components)


def umap_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def metrics_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_metrics() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_metrics")
        m.def_function[graph_parallel_available_binding]("graph_parallel_available")
        m.def_function[metrics_vendor_binding]("metrics_vendor")
        m.def_function[umap_fit_transform_binding]("umap_fit_transform")
        m.def_function[umap_transform_binding]("umap_transform")
        m.def_function[umap_numeric_mode_binding]("umap_numeric_mode")
        m.def_function[accuracy_score_binding]("accuracy_score")
        m.def_function[rand_score_binding]("rand_score")
        m.def_function[adjusted_rand_score_binding]("adjusted_rand_score")
        m.def_function[entropy_binding]("entropy")
        m.def_function[mutual_info_score_binding]("mutual_info_score")
        m.def_function[fowlkes_mallows_score_binding]("fowlkes_mallows_score")
        m.def_function[accuracy_score_weighted_binding]("accuracy_score_weighted")
        m.def_function[r2_score_weighted_binding]("r2_score_weighted")
        m.def_function[homogeneity_score_binding]("homogeneity_score")
        m.def_function[completeness_score_binding]("completeness_score")
        m.def_function[v_measure_score_binding]("v_measure_score")
        m.def_function[r2_score_binding]("r2_score")
        m.def_function[roc_auc_score_binding]("roc_auc_score")
        m.def_function[precision_recall_curve_binding]("precision_recall_curve")
        m.def_function[log_loss_binding]("log_loss")
        m.def_function[confusion_matrix_binding]("confusion_matrix")
        m.def_function[precision_recall_fscore_binding]("precision_recall_fscore")
        m.def_function[regression_error_binding[False, False]]("mean_squared_error")
        m.def_function[regression_error_binding[True, False]]("mean_absolute_error")
        m.def_function[regression_error_binding[False, True]]("root_mean_squared_error")
        m.def_function[umap_numeric_mode_binding]("metrics_numeric_mode")
        m.def_function[kl_divergence_binding]("kl_divergence")
        m.def_function[silhouette_binding]("silhouette")
        m.def_function[trustworthiness_binding]("trustworthiness")
        m.def_function[spectral_fit_predict_dataset_binding](
            "spectral_fit_predict_dataset"
        )
        m.def_function[spectral_fit_predict_graph_binding](
            "spectral_fit_predict_graph"
        )
        m.def_function[spectral_fit_predict_dataset_state_binding](
            "spectral_fit_predict_dataset_state"
        )
        m.def_function[spectral_fit_predict_graph_state_binding](
            "spectral_fit_predict_graph_state"
        )
        m.def_function[spectral_predict_binding]("spectral_predict")
        m.def_function[spectral_embedding_dataset_binding]("spectral_embedding_dataset")
        m.def_function[spectral_embedding_graph_binding]("spectral_embedding_graph")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_metrics: ", e))


def graph_parallel_available_binding() raises -> PythonObject:
    return PythonObject(1)
