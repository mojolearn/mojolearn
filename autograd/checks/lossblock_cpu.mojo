# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.lossblock_cases import run_lossblock_cases


def run_checks() raises:
    var ex = HostExec()
    run_lossblock_cases(ex)
