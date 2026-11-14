# SPDX-License-Identifier: Apache-2.0
"""The device resampling entries (resample/estimator.mojo, the
`_mojolearn_resample` entries), fixtures in order: they share the binding's
process context; every call keeps its own kernel parallelism."""
from resample.estimator import (
    bootstrap_host, bootstrap_unpaired_host, permutation_samples_host, permutation_test_host, resample_indices_host,
)
from checks.native_tests.core_ne2_rounds_cases import (
    NE2_FIXTURES, ResampleCalls, require, run_ne2_fixture, METHOD_PERCENTILE, ALT_TWO_SIDED, STAT_DIFF_MEANS,
)


struct DeviceResample(ResampleCalls):
    @staticmethod
    def indices(n: Int, count: Int, replace: Bool, seed: UInt64) raises -> List[Int32]:
        return resample_indices_host(n, count, replace, seed)

    @staticmethod
    def bootstrap(x: List[Float32], n: Int, statistic: Int, n_resamples: Int, seed: UInt64,
                  r_first: Int) raises -> Tuple[List[Float32], List[Float32]]:
        var r = bootstrap_host(x, n, 1, statistic, n_resamples, seed, METHOD_PERCENTILE, Float32(0.95),
                               ALT_TWO_SIDED, Float32(0.5), r_first)
        return (r.distribution.copy(), r.sorted_distribution.copy())

    @staticmethod
    def bootstrap_unpaired(x: List[Float32], y: List[Float32], n_resamples: Int,
                           seed: UInt64) raises -> Tuple[List[Float32], List[Float32]]:
        var r = bootstrap_unpaired_host(x, len(x), y, len(y), n_resamples, seed, METHOD_PERCENTILE,
                                        Float32(0.95), ALT_TWO_SIDED, 0)
        return (r.distribution.copy(), r.sorted_distribution.copy())

    @staticmethod
    def permutation(x: List[Float32], y: List[Float32], samples: Bool, n_resamples: Int, seed: UInt64,
                    r_first: Int) raises -> List[Float32]:
        if samples:
            return permutation_samples_host(x, y, True, n_resamples, seed, ALT_TWO_SIDED, r_first).null_distribution.copy()
        return permutation_test_host(x, y, STAT_DIFF_MEANS, n_resamples, seed, ALT_TWO_SIDED, r_first).null_distribution.copy()


def run_checks() raises:
    var failed = 0
    for unit in range(NE2_FIXTURES):
        try:
            run_ne2_fixture[DeviceResample](unit)
        except error:
            failed += 1
            print("FAIL core_ne2_rounds_gpu fixture", unit, String(error))
    require(failed == 0, "core_ne2_rounds_gpu failed " + String(failed) + " fixtures")
    print("PASS core_ne2_rounds_gpu:", NE2_FIXTURES, "resampling invariants, ordered on the device")
