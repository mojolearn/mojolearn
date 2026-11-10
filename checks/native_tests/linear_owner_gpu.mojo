# SPDX-License-Identifier: Apache-2.0
"""Real device descriptors invoke native linear RUNs and existing kernels.

The product entries share process_ctx and the decomposition device pool, so
fixture graphs submit in order. Source preparation uses host_parallelize and
product kernels retain their native parallelism; no concurrency claim.
"""
from glm.linear_fit_device import linear_fit_device_v1
from glm.linear_predict_device import linear_predict_device_v1
from x_linear.ridge_descriptor_device import ridge_fit_device_v1
from x_decomp.descriptor_device import x_decomp_device_v1
from x_metrics.descriptor_device import metrics_program_device_v1
from checks.native_tests.linear_owner_cases import LinearTestAPI, LINEAR_OWNER_FIXTURES, require, run_linear_fixture


def main() raises:
    var api = LinearTestAPI(linear_fit_device_v1, ridge_fit_device_v1,
        linear_predict_device_v1, x_decomp_device_v1, metrics_program_device_v1)
    require(api.vendor != 0, "GPU fixture connected a host descriptor")
    var failed = 0
    for fixture_id in range(LINEAR_OWNER_FIXTURES):
        try:
            run_linear_fixture(api, fixture_id)
        except error:
            print("FAIL linear_owner_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "linear_owner_gpu failed " + String(failed) + " fixtures")
    print("PASS linear_owner_gpu:", LINEAR_OWNER_FIXTURES, "native fit/predict/score fixtures including wide minimum norm; parallel kernels")
