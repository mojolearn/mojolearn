# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The 'rbf' affinity of SpectralEmbedding / SpectralClustering over a fold already on the device (lane
vdf-scope-cluster8: spectral/impl/rbf_affinity_device.mojo `rbf_affinity_device` reading a NEGATIVE X word, a slot of
the metrics image's resident store, in place; the member's `_store_affinity` route in core/spectral_member.mojo).

Claims (device column; the host image builds no device affinity and has no store):
  case 0: `spectral_embedding_rbf_host` over the slot word == over the same rows as host words: the embedding's
          float32 bits, the column count and the published n x n affinity's bits.
  case 1: `spectral_fit_predict_rbf_host` over the slot word == over the host words: labels, the embedding's
          bits, the published affinity's bits and the kept prediction state (eigenvalues, centroids).
The host-word route itself is checked against the x_decomp cells (the CPU column's spelling) by
spectral_device_affinity.mojo; equal bits here carry that claim to the slot route.
Data: a SplitMix64 stream (uniform [0, 4)), 48 rows x 3; gamma 0.7; 2 components, 2 clusters, n_init 2, seed 5.
Sabotage: in `rbf_affinity_device` read slot `fold_x_slot(x_addr) + 1` -> both cases fail (or refuse a dead slot).
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from core.metrics_store import METRICS_STORE, metrics_store_ctx
from core.fold_x_word import fold_x_word
from spectral.estimator import spectral_embedding_rbf_host, spectral_fit_predict_rbf_host
from spectral.impl.spectral_predict_common import SpectralPredictionState

comptime _N = 48
comptime _D = 3


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _bits(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits differ at " + String(i))


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var seed = UInt64(20261124)
        var x = List[Float32](capacity=_N * _D)
        for _ in range(_N * _D):  # small-loop(fixture words)
            x.append(Float32(Float64(_next(seed) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
        var ctx = metrics_store_ctx()
        var store = METRICS_STORE.get_or_create_ptr()
        var slot = store[].put(ctx, Int(x.unsafe_ptr()), _N * _D)
        var failure = String("")
        try:
            var aff_host = List[Float32](length=_N * _N, fill=0.0)
            var aff_slot = List[Float32](length=_N * _N, fill=0.0)
            if which == 0:
                var e_host = List[Float32]()
                var e_slot = List[Float32]()
                var a = spectral_embedding_rbf_host(Int(x.unsafe_ptr()), _N, _D, 0.7, Int(aff_host.unsafe_ptr()), 3,
                                                    True, True, UInt64(5), e_host)
                var b = spectral_embedding_rbf_host(fold_x_word(slot), _N, _D, 0.7, Int(aff_slot.unsafe_ptr()), 3,
                                                    True, True, UInt64(5), e_slot)
                require(a == b, "embedding column counts")
                _bits(e_host, e_slot, "embedding")
            else:
                var l_host = List[Int32]()
                var l_slot = List[Int32]()
                var e_host = List[Float32]()
                var e_slot = List[Float32]()
                var s_host = SpectralPredictionState()
                var s_slot = SpectralPredictionState()
                var a = spectral_fit_predict_rbf_host(Int(x.unsafe_ptr()), _N, _D, 0.7, Int(aff_host.unsafe_ptr()),
                                                      2, 2, 2, 8, Float32(1e-5), UInt64(5), l_host, e_host, s_host,
                                                      True)
                var b = spectral_fit_predict_rbf_host(fold_x_word(slot), _N, _D, 0.7, Int(aff_slot.unsafe_ptr()),
                                                      2, 2, 2, 8, Float32(1e-5), UInt64(5), l_slot, e_slot, s_slot,
                                                      True)
                require(a == b, "fit column counts")
                require(len(l_host) == len(l_slot), "label lengths")
                for i in range(len(l_host)):  # small-loop(compared labels)
                    require(l_host[i] == l_slot[i], "label differs at " + String(i))
                _bits(e_host, e_slot, "fit embedding")
                _bits(s_host.eigenvalues, s_slot.eigenvalues, "eigenvalues")
                _bits(s_host.centroids, s_slot.centroids, "centroids")
            _bits(aff_host, aff_slot, "published affinity")
        except error:
            failure = String(error)
        store[].free(ctx, slot)
        _ = x^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("spectral_rbf_slot device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if not CPU_COLUMN_BUILD:
        suite.test[test_device_fixtures]()
    suite^.run()
