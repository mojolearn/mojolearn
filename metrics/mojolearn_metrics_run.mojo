# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN pieces of `bindings/_mojolearn_metrics.mojo`: the typed address casts
and loads, the spectral output guards, the output copies of the spectral
doors, and the typed bodies of `precision_recall_curve`, `confusion_matrix`
and `spectral_predict`. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). The door keeps the argument reads and releases
the GIL around each body exactly where it did before; the bodies are moved
unchanged, so the bits are unchanged (lane rs-metrics-estimators,
2026-11-12)."""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from metrics.estimator import binary_ranking_host, confusion_matrix_host
from spectral.estimator import spectral_predict_host
from spectral.impl.spectral_predict_common import SpectralPredictionState


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _load_i32(addr: Int, n: Int) raises -> List[Int32]:
    return read_i32(addr, max(0, n))


def _load_f32(addr: Int, n: Int) raises -> List[Float32]:
    return read_f32(addr, max(0, n))


def precision_recall_curve_run(
    y: List[Int32], scores: List[Float32], n: Int,
    precision: MutPointer[Float32, MutUntrackedOrigin], recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    # Called with the GIL released by the door (precision_recall_curve_binding).
    var m = 0
    var result = binary_ranking_host[True](y,scores,n)
    m = result[1]
    var curve = result[0].unsafe_ptr()
    unsafe_memcpy(dest=precision, src=curve, count=m + 1)
    unsafe_memcpy(dest=recall, src=curve + (n + 1), count=m + 1)
    unsafe_memcpy(dest=thresholds, src=curve + 2 * (n + 1), count=m)
    _ = len(result[0])
    return m


def confusion_matrix_run(
    y: List[Int32], p: List[Int32], n: Int, k: Int, normalization: Int, address: Int,
) raises:
    # Called with the GIL released by the door (confusion_matrix_binding);
    # normalization 0 writes Int64 counts, otherwise Float32.
    if normalization == 0:
        var output = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
        var values = confusion_matrix_host[DType.int64](y,p,n,k,normalization)
        unsafe_memcpy(dest=output, src=values.unsafe_ptr(), count=len(values))
        _ = len(values)
    else:
        var output = _f32_ptr(address)
        var values = confusion_matrix_host[DType.float32](y,p,n,k,normalization)
        unsafe_memcpy(dest=output, src=values.unsafe_ptr(), count=len(values))
        _ = len(values)


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
    var out = spectral_predict_host(
        input, train_x, n_train, n_queries, n_features, k, n_clusters,
        n_neighbors, affinity, state,
    )
    if len(out.labels) != n_queries or len(out.embedding) != n_queries * k:
        raise Error("spectral_predict: the kernel returned the wrong number of outputs")
    unsafe_memcpy(dest=olp, src=out.labels.unsafe_ptr(), count=n_queries)
    unsafe_memcpy(dest=oep, src=out.embedding.unsafe_ptr(), count=n_queries * k)
    _ = len(out.labels)
    _ = len(out.embedding)
