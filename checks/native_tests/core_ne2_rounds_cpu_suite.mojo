# SPDX-License-Identifier: Apache-2.0
"""The host resampling column (resample/host/resample_host.mojo, the
`_mojolearn_resample_host` entries), fixtures run across host cores."""
from core.host_parallel import host_parallelize
from resample.host.resample_host import (
    host_bootstrap, host_bootstrap_unpaired, host_permutation_samples, host_permutation_test, host_resample_indices,
)
from checks.native_tests.core_ne2_rounds_cases import (
    NE2_FIXTURES, ResampleCalls, require, run_ne2_fixture, METHOD_PERCENTILE, ALT_TWO_SIDED, STAT_DIFF_MEANS,
)


struct HostResample(ResampleCalls):
    @staticmethod
    def indices(n: Int, count: Int, replace: Bool, seed: UInt64) raises -> List[Int32]:
        return host_resample_indices(n, count, replace, seed)

    @staticmethod
    def bootstrap(x: List[Float32], n: Int, statistic: Int, n_resamples: Int, seed: UInt64,
                  r_first: Int) raises -> Tuple[List[Float32], List[Float32]]:
        var r = host_bootstrap(x, n, 1, statistic, n_resamples, seed, METHOD_PERCENTILE, Float32(0.95),
                               ALT_TWO_SIDED, Float32(0.5), r_first)
        return (r.distribution.copy(), r.sorted_distribution.copy())

    @staticmethod
    def bootstrap_unpaired(x: List[Float32], y: List[Float32], n_resamples: Int,
                           seed: UInt64) raises -> Tuple[List[Float32], List[Float32]]:
        var r = host_bootstrap_unpaired(x, len(x), y, len(y), n_resamples, seed, METHOD_PERCENTILE,
                                        Float32(0.95), ALT_TWO_SIDED, 0)
        return (r.distribution.copy(), r.sorted_distribution.copy())

    @staticmethod
    def permutation(x: List[Float32], y: List[Float32], samples: Bool, n_resamples: Int, seed: UInt64,
                    r_first: Int) raises -> List[Float32]:
        if samples:
            return host_permutation_samples(x, y, True, n_resamples, seed, ALT_TWO_SIDED, r_first).null_distribution.copy()
        return host_permutation_test(x, y, STAT_DIFF_MEANS, n_resamples, seed, ALT_TWO_SIDED, r_first).null_distribution.copy()


def run_checks() raises:
    var failures = List[Int](length=NE2_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            run_ne2_fixture[HostResample](unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL core_ne2_rounds_cpu fixture", unit, String(error))
    host_parallelize(task, NE2_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_ne2_rounds_cpu failed " + String(failed) + " fixtures")
    print("PASS core_ne2_rounds_cpu:", NE2_FIXTURES, "resampling invariants, parallel")
