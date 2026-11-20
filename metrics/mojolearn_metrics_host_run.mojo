# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN pieces of `bindings/_mojolearn_metrics_host.mojo`: the class-range
check, the spectral output guards and copies, and the typed bodies of
`homogeneity_completeness_v_measure`, `silhouette`,
`precision_recall_curve`, `confusion_matrix` and `spectral_predict` on the
host. Typed Mojo only, no Python import (tools/python_free_dirs.sh). The
door keeps the argument reads and releases the GIL around each body exactly
where it did before; the bodies are moved unchanged, so the bits are
unchanged (lane rs-metrics-estimators, 2026-11-12)."""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr
from metrics.impl.stats.detail.v_measure_compose import v_measure_from
from metrics.host.metrics_oracle import DISTANCE_L2_SQRT_UNEXPANDED, host_homogeneity_score, host_silhouette
from metrics.host.metrics_host_entries_impl import (
    binary_ranking_checked, check_classification_ptr, metrics_check_pair as _check_pair,
)
from metrics.host.classification_oracle import host_confusion_matrix_ptr
from spectral.host.spectral_predict_host import SpectralPredictionState, host_spectral_predict


def _check_range(lower: Int32, upper: Int32) raises:
    if upper < lower:
        raise Error(
            "metrics: upper_class_range (" + String(upper)
            + ") is below lower_class_range (" + String(lower) + ")"
        )


def homogeneity_completeness_v_measure_run(
    yt: List[Int32], yp: List[Int32], n: Int, lower: Int32, upper: Int32, beta: Float64,
) raises -> Tuple[Float64, Float64, Float64]:
    # Called with the GIL released by the door (homogeneity_completeness_v_measure_binding).
    _check_pair(yt, yp, n)
    _check_range(lower, upper)
    var h = host_homogeneity_score(yt, yp, n, lower, upper)
    var c = host_homogeneity_score(yp, yt, n, lower, upper)
    var v = v_measure_from(h, c, beta)
    return (h, c, v)


def silhouette_run(
    x: List[Float32], lab: List[Int32], n_rows: Int, n_cols: Int, n_labels: Int, chunk: Int,
    sp: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Float32:
    # Called with the GIL released by the door (silhouette_binding).
    var scores = List[Float32]()
    var mean = Float32(0.0)
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
    if n_rows > 0:
        unsafe_memcpy(dest=sp, src=scores.unsafe_ptr(), count=n_rows)
    _ = len(scores)
    return mean


def precision_recall_curve_run(
    y: List[Int32], scores: List[Float32], n: Int,
    precision: MutPointer[Float32, MutUntrackedOrigin], recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    # Called with the GIL released by the door (precision_recall_curve_binding).
    var m = 0
    var result = binary_ranking_checked(y, scores, n, True)
    m = result[1]
    var curve = result[0].unsafe_ptr()
    unsafe_memcpy(dest=precision, src=curve, count=m + 1)
    unsafe_memcpy(dest=recall, src=curve + (n + 1), count=m + 1)
    unsafe_memcpy(dest=thresholds, src=curve + 2 * (n + 1), count=m)
    _ = len(result[0])
    return m


def confusion_matrix_run(
    y: MutPointer[Int32, MutUntrackedOrigin], p: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int, normalization: Int, address: Int,
) raises:
    # Called with the GIL released by the door (confusion_matrix_binding);
    # normalization 0 writes Int64 counts, otherwise Float32 ratios.
    if normalization == 0:
        var output = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
        check_classification_ptr(y, p, n, k, True)
        var values = host_confusion_matrix_ptr(y, p, n, k, normalization)
        unsafe_memcpy(dest=output, src=values[0].unsafe_ptr(), count=len(values[0]))
        _ = len(values[0])
    else:
        var output = f32_ptr(address)
        check_classification_ptr(y, p, n, k, True)
        var values = host_confusion_matrix_ptr(y, p, n, k, normalization)
        unsafe_memcpy(dest=output, src=values[1].unsafe_ptr(), count=len(values[1]))
        _ = len(values[1])


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


def _spectral_write_outputs(
    labels: List[Int32], embedding: List[Float32], lp: MutPointer[Int32, MutUntrackedOrigin], ep: MutPointer[Float32, MutUntrackedOrigin], n_samples: Int,
):
    # The spectral doors' label and embedding copies, after their guard.
    unsafe_memcpy(dest=lp, src=labels.unsafe_ptr(), count=n_samples)
    unsafe_memcpy(dest=ep, src=embedding.unsafe_ptr(), count=len(embedding))
    _ = len(labels)
    _ = len(embedding)


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


def _spectral_write_embedding(embedding: List[Float32], ep: MutPointer[Float32, MutUntrackedOrigin]):
    # The spectral embedding doors' copy, after their guard.
    unsafe_memcpy(dest=ep, src=embedding.unsafe_ptr(), count=len(embedding))
    _ = len(embedding)


def _spectral_state_copy(
    state: SpectralPredictionState, ev: MutPointer[Float32, MutUntrackedOrigin], evec: MutPointer[Float32, MutUntrackedOrigin],
    dg: MutPointer[Float32, MutUntrackedOrigin], cent: MutPointer[Float32, MutUntrackedOrigin],
):
    # `_write_spectral_state`'s copies, after its state check and address reads.
    unsafe_memcpy(dest=ev, src=state.eigenvalues.unsafe_ptr(), count=len(state.eigenvalues))
    unsafe_memcpy(dest=evec, src=state.eigenvectors.unsafe_ptr(), count=len(state.eigenvectors))
    unsafe_memcpy(dest=dg, src=state.diag.unsafe_ptr(), count=len(state.diag))
    unsafe_memcpy(dest=cent, src=state.centroids.unsafe_ptr(), count=len(state.centroids))


def spectral_predict_run(
    input: List[Float32], train_x: List[Float32], n_train: Int, n_queries: Int,
    n_features: Int, k: Int, n_clusters: Int, n_neighbors: Int, affinity: Int,
    state: SpectralPredictionState, olp: MutPointer[Int32, MutUntrackedOrigin], oep: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    # Called with the GIL released by the door (spectral_predict_binding).
    var out = host_spectral_predict(
        input, train_x, n_train, n_queries, n_features, k, n_clusters,
        n_neighbors, affinity, state,
    )
    unsafe_memcpy(dest=olp, src=out.labels.unsafe_ptr(), count=n_queries)
    unsafe_memcpy(dest=oep, src=out.embedding.unsafe_ptr(), count=n_queries * k)
    _ = len(out.labels)
    _ = len(out.embedding)
