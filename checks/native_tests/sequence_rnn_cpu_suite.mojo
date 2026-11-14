# SPDX-License-Identifier: Apache-2.0
"""Recurrent models against the float64 PyTorch cell equations, and their loss going down: the independent fixtures run across host cores, each on its own
HostExec with a private failure slot."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from checks.native_tests.sequence_port_common import require, count_failures
from checks.native_tests.sequence_rnn_cases import RNN_FIXTURES, run_rnn_fixture


def run_checks() raises:
    var failures = List[Int](length=RNN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var ex = HostExec()
            run_rnn_fixture(ex, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL sequence_rnn_cpu fixture", unit, String(error))
    host_parallelize(task, RNN_FIXTURES)
    var failed = count_failures(failures)
    require(failed == 0, "sequence_rnn_cpu failed " + String(failed) + " fixtures")
    print("PASS sequence_rnn_cpu:", RNN_FIXTURES, "fixtures")
