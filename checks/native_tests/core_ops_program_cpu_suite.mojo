# SPDX-License-Identifier: Apache-2.0
"""Captured programs against the eager session: each fixture on its own two host sessions, the fixtures run
across host cores with a private failure slot each."""
from autograd.ops_api import OpsSession
from checks.native_tests.ops_higher_cases import require
from checks.native_tests.core_ops_program_cases import CORE_OPS_PROGRAM_FIXTURES, run_core_ops_program_fixture, core_ops_program_records
from core.host_parallel import host_parallelize
from sequence.exec import HostExec


def run_checks() raises:
    var failures = List[Int](length=CORE_OPS_PROGRAM_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var record = core_ops_program_records(fixture_id)
            var s = OpsSession[HostExec](HostExec(), record)
            var again = OpsSession[HostExec](HostExec(), record)
            try:
                run_core_ops_program_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            print("FAIL core_ops_program_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, CORE_OPS_PROGRAM_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_ops_program_cpu failed " + String(failed) + " of " + String(CORE_OPS_PROGRAM_FIXTURES))
    print("PASS core_ops_program_cpu:", CORE_OPS_PROGRAM_FIXTURES, "fixtures, parallel")
