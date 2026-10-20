# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.higher import run_higher_checks


def main() raises:
    var ex = DeviceExec()
    run_higher_checks(ex)
