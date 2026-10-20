# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.rankn_conv_cases import run_rankn_conv_checks


def main() raises:
    var ex = DeviceExec()
    run_rankn_conv_checks(ex)
