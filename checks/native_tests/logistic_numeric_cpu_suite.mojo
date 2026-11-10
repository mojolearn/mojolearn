# SPDX-License-Identifier: Apache-2.0
"""Importable native CPU LogisticRegression numerical suite."""
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from core.host_helpers_impl import argmax_rows_f32
from core.logistic_native_api import QnNativeDriver, QnQueryRequest
from glm.host.logistic_native_host import qn_host_native_v1
from checks.native_tests.logistic_numeric_cases import LOGISTIC_NUMERIC_FIXTURES, LogisticPrediction, fixture
from checks.native_tests.logistic_numeric_fixtures import require


struct HostPrediction(LogisticPrediction):
    @staticmethod
    def codes(driver: QnNativeDriver, request: QnQueryRequest, scores: Int) raises:
        # Actual host public fallback, whose first-maximum scan is tested
        # against the independent Float64 oracle in the common fixture.
        argmax_rows_f32(f32_ptr(scores), request.rows, request.targets,
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=request.dst))


def run_checks() raises:
    var failures = List[Int](length=LOGISTIC_NUMERIC_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            fixture[HostPrediction](QnNativeDriver(qn_host_native_v1), fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL logistic_numeric_cpu fixture", fixture_id, String(error))
    host_parallelize(task, LOGISTIC_NUMERIC_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "logistic_numeric_cpu failed " + String(failed) + " fixtures")
    print("PASS logistic_numeric_cpu:", LOGISTIC_NUMERIC_FIXTURES, "native fit/query/state fixtures")
