# SPDX-License-Identifier: Apache-2.0
"""Partitioned programs on host sessions, devices (0,).

Serial by nature: the session and the partition reach their owners through
the process-wide tables (autograd/ops_api.mojo _OPS_SESSIONS,
autograd/partition_entries.mojo _PARTITIONS), whose open/close append to and
pop from unlocked lists, so concurrent fixtures from host tasks would race on
those tables. Each partitioned program runs its own ranks in parallel
(PartitionedProgram.wave); the four tiny fixtures run one after another.
"""
from autograd.ops_api import ops_session_open, ops_session_close, OPS_HOST
from checks.native_tests.ops_higher_cases import require
from checks.native_tests.core_partitioned_program_cases import (
    CORE_PARTITIONED_PROGRAM_FIXTURES, run_core_partitioned_program_fixture,
)
from sequence.exec import HostExec


def run_checks() raises:
    var failed = 0
    for fixture_id in range(CORE_PARTITIONED_PROGRAM_FIXTURES):
        try:
            var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
            try:
                var executors = List[HostExec]()
                executors.append(HostExec())
                var devices: List[Int] = [0]
                run_core_partitioned_program_fixture[HostExec, OPS_HOST](sid, fixture_id, devices, executors^)
            except error:
                ops_session_close[HostExec, OPS_HOST](sid)
                raise error
            ops_session_close[HostExec, OPS_HOST](sid)
        except error:
            print("FAIL core_partitioned_program_cpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "core_partitioned_program_cpu failed " + String(failed) + " of " + String(CORE_PARTITIONED_PROGRAM_FIXTURES))
    print("PASS core_partitioned_program_cpu:", CORE_PARTITIONED_PROGRAM_FIXTURES, "fixtures (owner tables: serial)")
