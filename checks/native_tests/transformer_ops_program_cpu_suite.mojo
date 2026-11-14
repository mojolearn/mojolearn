# SPDX-License-Identifier: Apache-2.0
"""The transformer-primitive session fixtures, each on its own recording host
session, run across host cores."""
from autograd.ops_api import OpsSession
from checks.native_tests.transformer_ops_program_cases import (
    TRANSFORMER_OPS_PROGRAM_FIXTURES, run_transformer_ops_program_fixture,
)
from checks.native_tests.ops_higher_cases import require
from core.host_parallel import host_parallelize
from sequence.exec import HostExec


def run_checks() raises:
    var failures = List[Int](length=TRANSFORMER_OPS_PROGRAM_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var s = OpsSession[HostExec](HostExec(), True)
            try:
                run_transformer_ops_program_fixture(s, fixture_id)
            except error:
                s.close()
                raise error
            s.close()
        except error:
            print("FAIL transformer_ops_program_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, TRANSFORMER_OPS_PROGRAM_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "transformer_ops_program_cpu failed " + String(failed) + " of " + String(TRANSFORMER_OPS_PROGRAM_FIXTURES))
    print("PASS transformer_ops_program_cpu:", TRANSFORMER_OPS_PROGRAM_FIXTURES, "transformer primitive fixtures, parallel")
