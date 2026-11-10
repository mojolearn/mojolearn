# SPDX-License-Identifier: Apache-2.0
"""Ten independent typed host OOB fixtures run on the existing host pool."""
from core.host_parallel import host_parallelize
from checks.native_tests.oob_exact_sum_cases import OOB_EXACT_FIXTURES, require, run_oob_fixture


def main() raises:
    var failures = List[Int](length=OOB_EXACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            run_oob_fixture[False](fixture_id)
        except error:
            print("FAIL oob_exact_sum_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, OOB_EXACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "oob_exact_sum_cpu failed " + String(failed) + " fixtures")
    print("PASS oob_exact_sum_cpu:", OOB_EXACT_FIXTURES, "native finite/cancellation/zero-count fixtures, parallel")
