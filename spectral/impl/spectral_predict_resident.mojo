# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The resident prediction state of `SpectralClustering.predict` (lane
ne1-a-cluster-spectral, owed row N326; plan workstream A).

Every predict used to copy `_fit_X`, the eigenvalues, the eigenvectors, the
degrees and the centroids into host Lists and upload the last four to the
device. The first predict after a fit now takes them ONCE
(`spectral_resident_prepare`): the degrees, the eigenvectors, `mu` (from
the eigenvalues, refused by name as before) and the centroids are uploaded
to the device, and the training rows are kept as one host copy for the
nearest-neighbors search (`knn_search` reads them from a host pointer).
Later predicts upload only their queries (`spectral_resident_predict`).

The registry follows bindings/ivf_host_search.mojo: entries behind
reference-counted pointers, a lock word around every dictionary step, and a
predict copies the pointer under the lock and runs with the GIL released, so
a release during a predict drops only the registry's reference.

WHAT DOES NOT CHANGE. A resident predict runs `spectral_predict_run`
(spectral/impl/spectral_predict.mojo), the very body the per-call
`spectral_predict_device` runs after its own uploads, over the same bytes on
the same context slot: the bits are the same."""
from std.ffi import _Global
from std.atomic import Atomic, Ordering
from std.memory import ArcPointer

from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from spectral.checks.device_io import upload_f32
from spectral.impl.spectral_predict import spectral_predict_run
from spectral.impl.spectral_predict_common import (
    SPECTRAL_AFFINITY_NEAREST_NEIGHBORS,
    SpectralPrediction,
    SpectralPredictionState,
    spectral_predict_check_state,
    spectral_predict_mu,
    spectral_predict_validate,
)

# The context slot of spectral/estimator.mojo (`spectral_predict_host`), so a
# resident predict runs where the per-call one does.
comptime _DEVCTX_SLOT = "MojolearnMetricsContextIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnMetricsContextFastV2"


struct SpectralResident(Movable):
    """One fitted model's prediction state: the device copies and the host
    training rows."""

    var ctx: DeviceContext
    var d_diag: DeviceBuffer[DType.float32]
    var d_vecs: DeviceBuffer[DType.float32]
    var d_mu: DeviceBuffer[DType.float32]
    var d_cent: DeviceBuffer[DType.float32]
    var train_x: List[Float32]
    var n_train: Int
    var n_features: Int
    var k: Int
    var n_clusters: Int
    var affinity: Int

    def __init__(out self, var train_x: List[Float32], n_train: Int, n_features: Int, k: Int, n_clusters: Int,
                 affinity: Int, state: SpectralPredictionState) raises:
        # The per-call refusals, in their order: the state, then `mu`.
        spectral_predict_check_state(state, n_train, k, n_clusters)
        if affinity == SPECTRAL_AFFINITY_NEAREST_NEIGHBORS and len(train_x) < n_train * n_features:
            raise Error("spectral_predict: a buffer is shorter than its shape; refused by name")
        var mu = spectral_predict_mu(state.eigenvalues)
        var ctx = process_ctx[_DEVCTX_SLOT]()
        self.d_diag = upload_f32(ctx, state.diag)
        self.d_vecs = upload_f32(ctx, state.eigenvectors)
        self.d_mu = upload_f32(ctx, mu)
        self.d_cent = upload_f32(ctx, state.centroids)
        self.train_x = train_x^
        self.n_train = n_train
        self.n_features = n_features
        self.k = k
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.d_diag^
        _ = self.d_vecs^
        _ = self.d_mu^
        _ = self.d_cent^
        _ = self.train_x^
        _ = self.ctx^


struct SpectralResidentRegistry(Movable):
    var entries: Dict[Int, ArcPointer[SpectralResident]]
    var next_id: Int
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ArcPointer[SpectralResident]]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime SPECTRAL_RESIDENT_REGISTRY = _Global[
    StorageType=SpectralResidentRegistry,
    name=(
        "MojoSpectralPredictResidentIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else
        "MojoSpectralPredictResidentFast"
    ),
    init_fn=SpectralResidentRegistry.__init__,
]


def _registry() raises -> UnsafePointer[SpectralResidentRegistry, MutAnyOrigin]:
    return SPECTRAL_RESIDENT_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _lock(state: UnsafePointer[SpectralResidentRegistry, MutAnyOrigin]):
    # Non-raising (lesson 6).
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _unlock(state: UnsafePointer[SpectralResidentRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def spectral_resident_prepare(var train_x: List[Float32], n_train: Int, n_features: Int, k: Int, n_clusters: Int,
                              affinity: Int, state: SpectralPredictionState) raises -> Int:
    """Take the prediction state once; the handle every later predict names."""
    var entry = ArcPointer[SpectralResident](
        SpectralResident(train_x^, n_train, n_features, k, n_clusters, affinity, state)
    )
    var reg = _registry()
    _lock(reg)
    try:
        if reg[].next_id == 9223372036854775807:
            raise Error("resident spectral prediction handle space exhausted")
        var handle = reg[].next_id
        reg[].next_id += 1
        reg[].entries[handle] = entry^
        _unlock(reg)
        return handle
    except error:
        _unlock(reg)
        raise error^


def spectral_resident_release(handle: Int) raises:
    """Drop the registry's reference; quiet for a released handle."""
    var reg = _registry()
    _lock(reg)
    try:
        if handle not in reg[].entries:
            _unlock(reg)
            return
        var popped = reg[].entries.pop(handle)
        _unlock(reg)
        # Dropped outside the lock: a running predict holds its own reference.
        _ = popped^
    except error:
        _unlock(reg)
        raise error^


def spectral_resident_count() raises -> Int:
    """The number of live resident spectral handles (tests only)."""
    var reg = _registry()
    _lock(reg)
    var n = len(reg[].entries)
    _unlock(reg)
    return n


def _view(handle: Int) raises -> ArcPointer[SpectralResident]:
    var reg = _registry()
    _lock(reg)
    try:
        if handle not in reg[].entries:
            raise Error("unknown or released resident spectral prediction handle")
        var view = reg[].entries[handle].copy()
        _unlock(reg)
        return view^
    except error:
        _unlock(reg)
        raise error^


def spectral_resident_predict(
    handle: Int, input: List[Float32], n_train: Int, n_queries: Int, n_features: Int, k: Int, n_clusters: Int,
    n_neighbors: Int, affinity: Int,
) raises -> SpectralPrediction:
    """`spectral_predict_device` on the handle's state: the call's shape
    words are checked against the handle's, then `spectral_predict_run`."""
    var e = _view(handle)
    if (e[].n_train != n_train or e[].n_features != n_features or e[].k != k or e[].n_clusters != n_clusters
            or e[].affinity != affinity):
        raise Error("spectral_predict: the resident handle was taken at another shape or affinity")
    spectral_predict_validate(n_train, n_queries, n_features, k, n_clusters, n_neighbors, affinity)
    # The handles' own copies (reference counted; the memory is shared), so
    # the run's `mut` centroid argument never aliases the registry's entry.
    var ctx = e[].ctx.copy()
    var d_diag = e[].d_diag.copy()
    var d_vecs = e[].d_vecs.copy()
    var d_mu = e[].d_mu.copy()
    var d_cent = e[].d_cent.copy()
    var out = spectral_predict_run(
        ctx, input, e[].train_x, n_train, n_queries, n_features, k, n_clusters, n_neighbors, affinity,
        d_diag, d_vecs, d_mu, d_cent,
    )
    _ = d_diag^
    _ = d_vecs^
    _ = d_mu^
    _ = d_cent^
    _ = ctx^
    _ = e^
    return out^
