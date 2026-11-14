# SPDX-License-Identifier: Apache-2.0
"""Partitioned programs on device sessions, devices (0,).

Serial by nature: the session and the partition reach their owners through
the process-wide tables (autograd/ops_api.mojo _OPS_SESSIONS,
autograd/partition_entries.mojo _PARTITIONS), whose open/close append to and
pop from unlocked lists, so concurrent fixtures from host tasks would race on
those tables. Each partitioned program runs its own ranks in parallel
(PartitionedProgram.wave); the four tiny fixtures run one after another.
"""
from autograd.ops_api import ops_session_open, ops_session_close, OPS_DEVICE
from checks.native_tests.ops_higher_cases import require
from checks.native_tests.core_partitioned_program_cases import (
    CORE_PARTITIONED_PROGRAM_FIXTURES, run_core_partitioned_program_fixture,
)
from std.os import getenv
from sequence.exec_device import DeviceExec
from sequence.x_sequence_impl import device_executors


def _devices() raises -> List[Int]:
    """MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES (comma-separated physical ids,
    the former test's multi-device qualification), default (0,)."""
    var text = getenv("MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES", "0")
    var out = List[Int]()
    for part in text.split(","):
        out.append(Int(String(part)))
    return out^


def run_checks() raises:
    var failed = 0
    for fixture_id in range(CORE_PARTITIONED_PROGRAM_FIXTURES):
        try:
            var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
            try:
                var devices = _devices()
                var executors = device_executors(devices)
                run_core_partitioned_program_fixture[DeviceExec, OPS_DEVICE](sid, fixture_id, devices, executors^)
            except error:
                ops_session_close[DeviceExec, OPS_DEVICE](sid)
                raise error
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
        except error:
            print("FAIL core_partitioned_program_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "core_partitioned_program_gpu failed " + String(failed) + " of " + String(CORE_PARTITIONED_PROGRAM_FIXTURES))
    print("PASS core_partitioned_program_gpu:", CORE_PARTITIONED_PROGRAM_FIXTURES, "fixtures (owner tables: serial)")
