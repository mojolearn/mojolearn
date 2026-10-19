# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.lossblock_cases import run_lossblock_cases


def main() raises:
    var ex = DeviceExec()
    run_lossblock_cases(ex)
