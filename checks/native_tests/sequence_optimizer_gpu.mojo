# SPDX-License-Identifier: Apache-2.0
"""Typed sequence optimizer checks; GPU shared-context submission is ordered.

DeviceExec shares sequence_ctx/pool. Kernels and independent oracle cells remain parallel. This does not prove concurrent GPU session ownership.
"""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_optimizer_cases import SEQUENCE_OPTIMIZER_FIXTURES, require, run_sequence_optimizer_fixture


def main() raises:
    var failed = 0
    for fixture_id in range(SEQUENCE_OPTIMIZER_FIXTURES):
        try:
            var ex = DeviceExec()
            run_sequence_optimizer_fixture(ex, fixture_id)
        except error:
            print("FAIL sequence_optimizer_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "sequence_optimizer_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_optimizer_gpu:", SEQUENCE_OPTIMIZER_FIXTURES, "fixtures; shared-context ordered")
