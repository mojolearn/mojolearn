# SPDX-License-Identifier: Apache-2.0
"""Independent native CPU linear fits/scores use the existing host pool."""
from core.host_parallel import host_parallelize
from glm.host.linear_fit_native import linear_fit_host_v1
from glm.host.linear_predict_native import linear_predict_host_v1
from x_linear.ridge_descriptor_host import ridge_fit_host_v1
from x_decomp.descriptor_host import x_decomp_host_v1
from x_metrics.descriptor_host import metrics_program_host_v1
from checks.native_tests.linear_owner_cases import LinearTestAPI, LINEAR_OWNER_FIXTURES, require, run_linear_fixture


def main() raises:
    var api = LinearTestAPI(linear_fit_host_v1, ridge_fit_host_v1,
        linear_predict_host_v1, x_decomp_host_v1, metrics_program_host_v1)
    require(api.vendor == 0, "CPU fixture connected a device descriptor")
    var failures = List[Int](length=LINEAR_OWNER_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm api, imm fp}:
        try:
            run_linear_fixture(api, fixture_id)
        except error:
            print("FAIL linear_owner_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, LINEAR_OWNER_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "linear_owner_cpu failed " + String(failed) + " fixtures")
    print("PASS linear_owner_cpu:", LINEAR_OWNER_FIXTURES, "native fit/predict/score fixtures including explicit wide-OLS refusal, parallel")
