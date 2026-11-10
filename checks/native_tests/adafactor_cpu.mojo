# SPDX-License-Identifier: Apache-2.0
"""Independent host matrix/vector Adafactor fixtures run in parallel."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from checks.native_tests.adafactor_cases import ADAFACTOR_FIXTURES, require, run_adafactor_fixture


def main() raises:
    var failures = List[Int](length=ADAFACTOR_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_adafactor_fixture(ex, fixture_id)
        except error:
            print("FAIL adafactor_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, ADAFACTOR_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "adafactor_cpu failed " + String(failed) + " fixtures")
    print("PASS adafactor_cpu:", ADAFACTOR_FIXTURES, "parallel matrix/vector fixtures")
