"""CPython boundary for the verified metrics kernels and for spectral clustering.

Kept in a SEPARATE extension, like `_mojolearn_estimators.mojo`, so an
independently changing binding does not become a merge point. Arrays cross
as borrowed NumPy addresses; every device buffer and context lives for one
call and no pointer is retained past the call that was handed it.

TWO LANES, ONE EXTENSION, DELIBERATELY. `metrics/` and `spectral/` both got
their Python surface in the same round and neither is large enough to earn a
build script of its own. They share nothing but this file and
`bindings/build_metrics.sh`.

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
    MI325X at leg 11 (`E3_RESULTS.md` round 11, commit 144aa5b, section 7,
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

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from metrics.estimator import (
    accuracy_score_host,
    adjusted_rand_score_host,
    completeness_score_host,
    entropy_host,
    homogeneity_score_host,
    kl_divergence_host,
    mutual_info_score_host,
    r2_score_host,
    rand_score_host,
    silhouette_host,
    trustworthiness_host,
    v_measure_score_host,
)
from spectral.estimator import (
    spectral_fit_predict_dataset_host,
    spectral_fit_predict_graph_host,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def _load_i32(addr: Int, n: Int) raises -> List[Int32]:
    var p = _i32_ptr(addr)
    var out = List[Int32]()
    for i in range(n):
        out.append(p.unsafe_load(i))
    return out^


def _load_f32(addr: Int, n: Int) raises -> List[Float32]:
    var p = _f32_ptr(addr)
    var out = List[Float32]()
    for i in range(n):
        out.append(p.unsafe_load(i))
    return out^


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
    'euclidean'/'l2') is ported and the Python wrapper refuses every other
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


@export
def PyInit__mojolearn_metrics() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_metrics")
        m.def_function[accuracy_score_binding]("accuracy_score")
        m.def_function[rand_score_binding]("rand_score")
        m.def_function[adjusted_rand_score_binding]("adjusted_rand_score")
        m.def_function[entropy_binding]("entropy")
        m.def_function[mutual_info_score_binding]("mutual_info_score")
        m.def_function[homogeneity_score_binding]("homogeneity_score")
        m.def_function[completeness_score_binding]("completeness_score")
        m.def_function[v_measure_score_binding]("v_measure_score")
        m.def_function[r2_score_binding]("r2_score")
        m.def_function[kl_divergence_binding]("kl_divergence")
        m.def_function[silhouette_binding]("silhouette")
        m.def_function[trustworthiness_binding]("trustworthiness")
        m.def_function[spectral_fit_predict_dataset_binding](
            "spectral_fit_predict_dataset"
        )
        m.def_function[spectral_fit_predict_graph_binding](
            "spectral_fit_predict_graph"
        )
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_metrics: ", e))
