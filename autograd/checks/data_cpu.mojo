# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.data_cases import run_data_checks


def run_checks() raises:
    var ex = HostExec()
    run_data_checks(ex)
