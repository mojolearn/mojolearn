# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.pass1_cases import run_pass1_cases


def run_checks() raises:
    var ex = DeviceExec()
    run_pass1_cases(ex)
