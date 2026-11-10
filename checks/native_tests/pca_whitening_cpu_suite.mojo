# SPDX-License-Identifier: Apache-2.0
"""Importable native CPU PCA whitening checks; independent fixture owners."""
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from core.estimators_host_impl import pca_fit_host_run, pca_whiten_host_run
from checks.native_tests.pca_whitening_cases import PCA_WHITENING_FIXTURES, PcaWhiteningCalls, fixture, require


struct HostPcaWhitening(PcaWhiteningCalls):
    @staticmethod
    def fit(x: Int, components: Int, mean: Int, explained: Int, ratio: Int,
        singular: Int, rows: Int) raises:
        _ = pca_fit_host_run(x, f32_ptr(components), f32_ptr(mean), f32_ptr(explained),
            f32_ptr(ratio), f32_ptr(singular), rows, 2, 2)

    @staticmethod
    def query(x: Int, mean: Int, components: Int, singular: Int, dst: Int,
        rows: Int, fit_rows: Int, inverse: Bool) raises:
        pca_whiten_host_run(x, mean, components, singular, f32_ptr(dst), rows, 2, 2, fit_rows,
            rows * 2, rows * 2, inverse)


def run_checks() raises:
    var failures = List[Int](length=PCA_WHITENING_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            fixture[HostPcaWhitening](fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL pca_whitening_cpu fixture", fixture_id, String(error))
    host_parallelize(task, PCA_WHITENING_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "pca_whitening_cpu failed " + String(failed) + " fixtures")
    print("PASS pca_whitening_cpu:", PCA_WHITENING_FIXTURES, "native whitening state fixtures")
