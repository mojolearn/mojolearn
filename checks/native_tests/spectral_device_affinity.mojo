# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The spectral embedding's generated affinities on the metrics image's device
(lane vdf-misc3, 2026-11-24): `spectral_embedding_rbf_host` and
`spectral_embedding_knn_host` build the n x n affinity on the device and embed
from that device buffer. Each cell compares, word for word, with the former
two-image route rebuilt on the host column:

  rbf   the x_decomp cells in their three-launch order on the host
        (`sqdist_cell`, `ew_cell(OP_SCALE, s=-gamma)`, `ew_cell(OP_EXP)`), the
        published affinity equal to it, and the embedding equal to
        `spectral_embedding_dense_host` on that host affinity;
  knn   the base image's host body `knn_affinity_f32`, the published affinity
        and the embedding as above; a negative distance refuses with the same
        3 status words and SPECTRAL_KNN_REFUSED before any embedding work.

Lane vdf-misc4 (2026-11-24): the same two routes on SpectralClustering's fit
(`spectral_fit_predict_rbf_host`, `spectral_fit_predict_knn_host`): labels,
embedding and published affinity equal, word for word, to
`spectral_fit_predict_dense_host_keep` on the host-column affinity.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import require
from x_decomp.cells import F32Ptr, OP_EXP, OP_SCALE, ew_cell, sqdist_cell
from core.dense_coo_host import knn_affinity_f32, F32P, I32P
from core.spectral_native_api import SPECTRAL_KNN_REFUSED
from spectral.estimator import (
    spectral_embedding_dense_host, spectral_embedding_rbf_host, spectral_embedding_knn_host,
    spectral_fit_predict_dense_host_keep, spectral_fit_predict_rbf_host, spectral_fit_predict_knn_host,
)
from spectral.impl.spectral_predict_common import SpectralPredictionState

comptime _N = 24
comptime _D = 3
comptime _K = 4
comptime _LANCZOS = 3


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": word " + String(i) + " differs")


def _x() -> List[Float32]:
    var x = List[Float32](length=_N * _D, fill=Float32(0))
    for i in range(_N):
        for j in range(_D):
            x[i * _D + j] = Float32(((i * 37 + j * 11) % 17)) * Float32(0.125) - Float32(1)
    return x^


def _host_rbf(x: List[Float32], gamma: Float64) -> List[Float32]:
    var xp = F32Ptr(unsafe_from_address=Int(x.unsafe_ptr()))
    var out = List[Float32](length=_N * _N, fill=Float32(0))
    for i in range(_N):
        for j in range(_N):
            var sq = sqdist_cell(xp, xp, i, j, _D)
            var scaled = ew_cell(OP_SCALE, sq, Float32(0), Float32(0), Float32(-gamma))
            out[i * _N + j] = ew_cell(OP_EXP, scaled, Float32(0), Float32(0), Float32(0))
    return out^


def test_rbf_device_affinity() raises:
    var x = _x()
    var gamma = Float64(0.5)
    var host_aff = _host_rbf(x, gamma)
    var host_emb = List[Float32]()
    var host_cols = spectral_embedding_dense_host(
        Int(host_aff.unsafe_ptr()), _N, _LANCZOS, True, True, UInt64(3), host_emb
    )
    var dev_aff = List[Float32](length=_N * _N, fill=Float32(-7))
    var dev_emb = List[Float32]()
    var dev_cols = spectral_embedding_rbf_host(
        Int(x.unsafe_ptr()), _N, _D, gamma, Int(dev_aff.unsafe_ptr()), _LANCZOS, True, True, UInt64(3), dev_emb
    )
    require(dev_cols == host_cols, "rbf: column counts differ")
    _same(dev_aff, host_aff, "rbf: published affinity vs the x_decomp cells")
    _same(dev_emb, host_emb, "rbf: embedding vs the dense route on the host affinity")
    # the standalone function publishes nothing
    var bare_emb = List[Float32]()
    _ = spectral_embedding_rbf_host(Int(x.unsafe_ptr()), _N, _D, gamma, 0, _LANCZOS, True, True, UInt64(3), bare_emb)
    _ = x^
    _same(bare_emb, host_emb, "rbf: embedding without a published affinity")


def _distances(negative: Bool) -> List[Float32]:
    var dist = List[Float32](length=_N * _N, fill=Float32(0))
    for i in range(_N):
        for j in range(_N):
            if i != j:
                var gap = i - j if i > j else j - i
                dist[i * _N + j] = Float32(gap) + Float32((i * 7 + j * 3) % 5) * Float32(0.01)
    if negative:
        dist[2 * _N + 5] = Float32(-1)
    return dist^


def _knn_cell(negative: Bool) raises:
    var dist = _distances(negative)
    var host_aff = List[Float32](length=_N * _N, fill=Float32(0))
    var host_status = List[Int32](length=3, fill=Int32(0))
    var none_i = I32P(unsafe_from_address=0)
    var none_f = F32P(unsafe_from_address=0)
    knn_affinity_f32(F32P(unsafe_from_address=Int(dist.unsafe_ptr())), none_i, none_i, none_f, 0, False,
                     _N, _K, F32P(unsafe_from_address=Int(host_aff.unsafe_ptr())),
                     I32P(unsafe_from_address=Int(host_status.unsafe_ptr())))
    var dev_aff = List[Float32](length=_N * _N, fill=Float32(0))
    var dev_status = List[Int32](length=3, fill=Int32(9))
    var dev_emb = List[Float32]()
    var dev_cols = spectral_embedding_knn_host(
        Int(dist.unsafe_ptr()), 0, 0, 0, 0, _N, _K, Int(dev_status.unsafe_ptr()), Int(dev_aff.unsafe_ptr()),
        _LANCZOS, True, True, UInt64(3), dev_emb,
    )
    _ = dist^
    for w in range(3):
        require(dev_status[w] == host_status[w], "knn: status word " + String(w) + " differs")
    if negative:
        require(host_status[0] != Int32(0), "knn: the host body did not refuse a negative distance")
        require(dev_cols == SPECTRAL_KNN_REFUSED and len(dev_emb) == 0, "knn: the device route did not refuse first")
        return
    var host_emb = List[Float32]()
    var host_cols = spectral_embedding_dense_host(
        Int(host_aff.unsafe_ptr()), _N, _LANCZOS, True, True, UInt64(3), host_emb
    )
    require(dev_cols == host_cols, "knn: column counts differ")
    _same(dev_aff, host_aff, "knn: published affinity vs the base host body")
    _same(dev_emb, host_emb, "knn: embedding vs the dense route on the host affinity")


def test_knn_device_affinity() raises:
    _knn_cell(False)
    _knn_cell(True)


comptime _CLUSTERS = 3
comptime _COMPONENTS = 3
comptime _INIT = 2


def _same_i(a: List[Int32], b: List[Int32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):
        require(a[i] == b[i], what + ": label " + String(i) + " differs")


def _host_fit(aff: List[Float32], mut labels: List[Int32], mut emb: List[Float32]) raises -> Int:
    var state = SpectralPredictionState()
    return spectral_fit_predict_dense_host_keep(
        Int(aff.unsafe_ptr()), _N, _CLUSTERS, _COMPONENTS, _INIT, _K, Float32(1e-5), UInt64(3),
        labels, emb, state, False,
    )


def test_rbf_device_fit() raises:
    var x = _x()
    var gamma = Float64(0.5)
    var host_aff = _host_rbf(x, gamma)
    var host_labels = List[Int32]()
    var host_emb = List[Float32]()
    var host_cols = _host_fit(host_aff, host_labels, host_emb)
    var dev_aff = List[Float32](length=_N * _N, fill=Float32(-7))
    var dev_labels = List[Int32]()
    var dev_emb = List[Float32]()
    var state = SpectralPredictionState()
    var dev_cols = spectral_fit_predict_rbf_host(
        Int(x.unsafe_ptr()), _N, _D, gamma, Int(dev_aff.unsafe_ptr()), _CLUSTERS, _COMPONENTS, _INIT, _K,
        Float32(1e-5), UInt64(3), dev_labels, dev_emb, state, False,
    )
    _ = x^
    require(dev_cols == host_cols, "rbf fit: column counts differ")
    _same(dev_aff, host_aff, "rbf fit: published affinity vs the x_decomp cells")
    _same(dev_emb, host_emb, "rbf fit: embedding vs the dense fit on the host affinity")
    _same_i(dev_labels, host_labels, "rbf fit: labels vs the dense fit on the host affinity")


def _knn_fit_cell(negative: Bool) raises:
    var dist = _distances(negative)
    var host_aff = List[Float32](length=_N * _N, fill=Float32(0))
    var host_status = List[Int32](length=3, fill=Int32(0))
    var none_i = I32P(unsafe_from_address=0)
    var none_f = F32P(unsafe_from_address=0)
    knn_affinity_f32(F32P(unsafe_from_address=Int(dist.unsafe_ptr())), none_i, none_i, none_f, 0, False,
                     _N, _K, F32P(unsafe_from_address=Int(host_aff.unsafe_ptr())),
                     I32P(unsafe_from_address=Int(host_status.unsafe_ptr())))
    var dev_aff = List[Float32](length=_N * _N, fill=Float32(0))
    var dev_status = List[Int32](length=3, fill=Int32(9))
    var dev_labels = List[Int32]()
    var dev_emb = List[Float32]()
    var state = SpectralPredictionState()
    var dev_cols = spectral_fit_predict_knn_host(
        Int(dist.unsafe_ptr()), 0, 0, 0, 0, _N, _K, Int(dev_status.unsafe_ptr()), Int(dev_aff.unsafe_ptr()),
        _CLUSTERS, _COMPONENTS, _INIT, _K, Float32(1e-5), UInt64(3), dev_labels, dev_emb, state, False,
    )
    _ = dist^
    for w in range(3):
        require(dev_status[w] == host_status[w], "knn fit: status word " + String(w) + " differs")
    if negative:
        require(host_status[0] != Int32(0), "knn fit: the host body did not refuse a negative distance")
        require(dev_cols == SPECTRAL_KNN_REFUSED and len(dev_labels) == 0, "knn fit: the device route did not refuse first")
        return
    var host_labels = List[Int32]()
    var host_emb = List[Float32]()
    var host_cols = _host_fit(host_aff, host_labels, host_emb)
    require(dev_cols == host_cols, "knn fit: column counts differ")
    _same(dev_aff, host_aff, "knn fit: published affinity vs the base host body")
    _same(dev_emb, host_emb, "knn fit: embedding vs the dense fit on the host affinity")
    _same_i(dev_labels, host_labels, "knn fit: labels vs the dense fit on the host affinity")


def test_knn_device_fit() raises:
    _knn_fit_cell(False)
    _knn_fit_cell(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
