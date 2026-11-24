# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""SpectralEmbedding over rows already on the device (lane vdf-scope-cluster6: spectral/estimator.mojo
`spectral_embedding_dataset_resident`, spectral/impl/preprocessing/detail/spectral_embedding.mojo
`transform_dataset_resident`, the SPECTRAL_DATASET_STORE embed route of core/spectral_native_device.mojo, the member's
`_fit_se` on a model-selection fold taken into the metrics store).

Claims (device column; the host image has no store and refuses the route):
  case 0 (drop_first, normalized Laplacian), case 1 (not dropped, unnormalized): the embedding of the rows in a
  metrics-store slot == the embedding of the same rows as host words (float32 bits of every column), and the same
  column count.
Data: a SplitMix64 stream (uniform [0, 4)), 80 rows x 3; 2 components, n_neighbors 8, seed 7, tolerance 1e-5.
Sabotage: in spectral/impl/preprocessing/detail/spectral_embedding.mojo `transform_dataset_resident` pass
`n_neighbors - 1` to the graph -> case 0 fails.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from core.metrics_store import METRICS_STORE, metrics_store_ctx, metrics_store_buffer
from spectral.estimator import spectral_embedding_dataset_host, spectral_embedding_dataset_resident

comptime _ROWS = 80
comptime _D = 3


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var state = UInt64(20261124)
        var x = List[Float32](capacity=_ROWS * _D)
        for _ in range(_ROWS * _D):  # small-loop(fixture words)
            x.append(Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
        var drop = case == 0
        var lanczos = 3 if drop else 2
        var ctx = metrics_store_ctx()
        var store = METRICS_STORE.get_or_create_ptr()
        var slot = store[].put(ctx, Int(x.unsafe_ptr()), _ROWS * _D)
        var failure = String("")
        try:
            var host = List[Float32]()
            var resident = List[Float32]()
            var a = spectral_embedding_dataset_host(x, _ROWS, _D, lanczos, 8, drop, drop, UInt64(7), host)
            var data = metrics_store_buffer(slot, _ROWS * _D)
            var b = spectral_embedding_dataset_resident(data, _ROWS, _D, lanczos, 8, drop, drop, UInt64(7), resident)
            _ = data^
            require(a == b, "column counts")
            require(len(host) == len(resident), "embedding lengths")
            for i in range(len(host)):  # small-loop(compared words)
                require(bitcast[DType.uint32](host[i]) == bitcast[DType.uint32](resident[i]),
                        "embedding bits differ at " + String(i))
        except error:
            failure = String(error)
        store[].free(ctx, slot)
        _ = x^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("spectral_embedding_resident device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if not CPU_COLUMN_BUILD:
        suite.test[test_device_fixtures]()
    suite^.run()
