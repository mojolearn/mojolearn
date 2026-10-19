# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.fused_cases import run_fused_cases


def main() raises:
    var ex = DeviceExec()
    run_fused_cases(ex)
