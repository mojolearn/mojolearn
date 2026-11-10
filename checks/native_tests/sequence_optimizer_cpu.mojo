# SPDX-License-Identifier: Apache-2.0
"""Typed sequence optimizer checks; independent host fixtures run in parallel.

Each reference cell uses the host pool; six optimizer steps are dependent.
"""
from sequence.exec import HostExec
from checks.native_tests.sequence_optimizer_cases import SEQUENCE_OPTIMIZER_FIXTURES, require, run_sequence_optimizer_fixture
from core.host_parallel import host_parallelize


def main() raises:
    var failures = List[Int](length=SEQUENCE_OPTIMIZER_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_sequence_optimizer_fixture(ex, fixture_id)
        except error:
            print("FAIL sequence_optimizer_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, SEQUENCE_OPTIMIZER_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "sequence_optimizer_cpu failed " + String(failed) + " fixtures")
    print("PASS sequence_optimizer_cpu:", SEQUENCE_OPTIMIZER_FIXTURES, "fixtures; parallel host")
