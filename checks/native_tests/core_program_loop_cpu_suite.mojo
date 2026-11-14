# SPDX-License-Identifier: Apache-2.0
"""The bounded program loop on host sessions.

Serial by nature: control_prepare/control_run reach a session through the
process-wide session table (autograd/ops_api.mojo _OPS_SESSIONS), whose
open/close append to and pop from unlocked lists, so concurrent sessions from
host tasks would race on that table. The fixtures are tiny (1 x 1 and 1 x 2
states, at most 10 iterations) and run one after another, each on its own
session.
"""
from autograd.ops_api import ops_session_open, ops_session_close, OPS_HOST
from checks.native_tests.ops_higher_cases import require
from checks.native_tests.core_program_loop_cases import CORE_PROGRAM_LOOP_FIXTURES, run_core_program_loop_fixture
from sequence.exec import HostExec


def run_checks() raises:
    var failed = 0
    for fixture_id in range(CORE_PROGRAM_LOOP_FIXTURES):
        try:
            var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
            try:
                run_core_program_loop_fixture[HostExec, OPS_HOST](sid, fixture_id)
            except error:
                ops_session_close[HostExec, OPS_HOST](sid)
                raise error
            ops_session_close[HostExec, OPS_HOST](sid)
        except error:
            print("FAIL core_program_loop_cpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "core_program_loop_cpu failed " + String(failed) + " of " + String(CORE_PROGRAM_LOOP_FIXTURES))
    print("PASS core_program_loop_cpu:", CORE_PROGRAM_LOOP_FIXTURES, "fixtures (session table: serial)")
