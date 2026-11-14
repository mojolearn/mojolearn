# SPDX-License-Identifier: Apache-2.0
"""Samba surface schedules and random stream: the independent fixtures run
across host cores with a private failure slot each. Host only: a schedule
value is host scalar code on every column, and the stream is checked
through the training host image's entry (the device kernel evaluates the
same per-element functions of core/philox_neural.mojo)."""
from core.host_parallel import host_parallelize
from checks.native_tests.sequence_port_common import require, count_failures
from checks.native_tests.sequence_samba_surface_cases import SAMBA_SURFACE_FIXTURES, run_samba_surface_fixture


def run_checks() raises:
    var failures = List[Int](length=SAMBA_SURFACE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            run_samba_surface_fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL sequence_samba_surface_cpu fixture", unit, String(error))
    host_parallelize(task, SAMBA_SURFACE_FIXTURES)
    var failed = count_failures(failures)
    require(failed == 0, "sequence_samba_surface_cpu failed " + String(failed) + " fixtures")
    print("PASS sequence_samba_surface_cpu:", SAMBA_SURFACE_FIXTURES, "fixtures")
